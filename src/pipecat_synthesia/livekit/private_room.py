"""Relayed join: the avatar renders in a private room that the plugin creates and owns.

`launch()` creates the room over LiveKit's room API, connects the bot to it as a standard participant,
and hands both to `LiveKitAvatarSession.launch()` in a `RoomHandle(owns_room=True)`. `AvatarTrackCapture`
then turns the avatar's tracks into Pipecat frames.

With the default timeouts, a room nobody joined closes 120 s after its creation, and after a bot crash
the room closes about 20 s after livekit-server notices the bot is gone (see *agent-kind requirement*).

*Room ownership handoff*: everything before `LiveKitAvatarSession.launch()` is rolled back here, including
a cancellation between creating the room and connecting to it, which would otherwise leave a room that
nothing deletes. From then on the room belongs to the session, which releases it on every path, so this
module never releases it a second time.

Every wait is capped by what is left of the *launch timeout*, because neither SDK call bounds itself the
way its arguments suggest: `create_room` retries a LiveKit Cloud host past its timeout (livekit-api
1.2.1), and `Room.connect` has its own 5 s default.

The room is connected once per `launch()`; there is no reconnect. A dead room is a dead session, and the
service launches a new one.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import os
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiohttp
from pipecat.utils.errors import ErrorCategory

from pipecat_synthesia._async import await_cancel_proof, wait_bounded
from pipecat_synthesia._redact import describe_exception
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._rtc import lk_api, lk_rtc
from pipecat_synthesia.livekit._types import (
    ROOM_CONNECT_SECS,
    LiveKitCredentials,
    MintedAvatarToken,
    RoomHandle,
)
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import (
    DEFAULT_AVATAR_IDENTITY,
    DEFAULT_AVATAR_NAME,
    DEFAULT_BOT_IDENTITY,
    DEFAULT_BOT_NAME,
    DEFAULT_TOKEN_TTL_SECONDS,
    checked_livekit_url,
    mint_bot_token,
)
from pipecat_synthesia.livekit.track_capture import AvatarTrackCapture
from pipecat_synthesia.room import AvatarRoom, JoinMode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import loguru

    from pipecat_synthesia.room import AvatarSession, LaunchContext

__all__ = ["PrivateLiveKitRoom"]


class PrivateLiveKitRoom(AvatarRoom):
    """Creates its own LiveKit room, captures the avatar's tracks and re-emits them as Pipecat frames.

    ``connect_timeout`` bounds the bot's connect to the private room. That connect is never cancelled
    (cancelling the LiveKit SDK's ``Room.connect()`` can end the bot process), so a launch cancelled while
    the bot is connecting returns once the connect has ended and the bot has left the room: up to
    ``connect_timeout`` plus about six seconds. The service waits for a cancelled launch, and logs a
    warning if that takes over 30 s.
    """

    join_mode = JoinMode.RELAYED

    def __init__(
        self,
        *,
        # env LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        room_name: str | None = None,  # default f"synthesia-avatar-{uuid4().hex[:12]}"
        bot_identity: str = DEFAULT_BOT_IDENTITY,
        avatar_identity: str = DEFAULT_AVATAR_IDENTITY,
        avatar_name: str = DEFAULT_AVATAR_NAME,
        empty_timeout_secs: int = 120,
        departure_timeout_secs: int = 20,
        delete_room_on_close: bool = True,
        connect_timeout: float = ROOM_CONNECT_SECS,
        room_factory=None,
        api_factory=None,  # test seams (default lk_rtc.Room / lk_api.LiveKitAPI)
    ) -> None:
        # Normalised here, so a ws:// or host-less URL fails at startup (*misuse vs session errors*). The
        # avatar worker is told the URL we connect with; `LiveKitAPI` accepts the ws URL as it is.
        self._ws_url = checked_livekit_url(
            _required(url, "url", "LIVEKIT_URL"),
            source="url" if url is not None else "LIVEKIT_URL",
        )
        self._credentials = LiveKitCredentials(
            api_key=_required(api_key, "api_key", "LIVEKIT_API_KEY"),
            api_secret=_required(api_secret, "api_secret", "LIVEKIT_API_SECRET"),
        )
        self._room_name = room_name or f"synthesia-avatar-{uuid4().hex[:12]}"
        self._bot_identity = bot_identity
        self._avatar_identity = avatar_identity
        self._avatar_name = avatar_name
        self._empty_timeout_secs = empty_timeout_secs
        self._departure_timeout_secs = departure_timeout_secs
        self._delete_room_on_close = delete_room_on_close
        self._connect_timeout = connect_timeout
        self._room_factory = room_factory or lk_rtc.Room
        self._api_factory = api_factory or lk_api.LiveKitAPI

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def room_name(self) -> str:
        """The private room's name: generated at construction unless one was given."""
        return self._room_name

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        # First, before anything exists: a misconfigured relayed join must fail before it creates a room
        # and a billed Synthesia session, and with nothing to roll back. The service sets both media
        # callbacks for a room whose join mode is relayed; a context built by hand, or built for a room
        # whose join mode says direct, has neither. A check, not an assert: under `python -O` an assert
        # would let the launch succeed with an avatar whose video and audio go nowhere.
        on_video, on_audio = ctx.callbacks.on_video_frame, ctx.callbacks.on_audio_frame
        if on_video is None or on_audio is None:
            raise SynthesiaError(
                "a relayed join needs the launch context's on_video_frame and on_audio_frame "
                "callbacks, which the service sets for a room whose join_mode is RELAYED",
                type=ErrorType.UNKNOWN,
                category=ErrorCategory.APPLICATION,
            )
        log = ctx.log.bind(room=self._room_name)
        # Cap every wait by what is left of the *launch timeout*, so a failure here gets its own error.
        deadline = monotonic() + ctx.launch_timeout

        # ① The room API client, on the URL normalised at construction.
        ws_url = self._ws_url
        lkapi = self._api_factory(
            ws_url,
            self._credentials.api_key,
            self._credentials.api_secret,
            timeout=aiohttp.ClientTimeout(total=_types.ROOM_CREATE_SECS),
        )

        async def close_api_only() -> None:
            await lkapi.aclose()

        # ② Create the room explicitly rather than relying on LiveKit's auto-create for a room-join grant:
        # auto-create leaves `empty_timeout` at the server default, so an abandoned private room would
        # linger far longer. That same `empty_timeout` is the backstop for the one case nothing here can
        # clean up: a cancel inside `create_room` cannot know whether the server created the room, so it
        # releases the client only and lets the room expire on its own.
        create_budget = _budget(deadline, _types.ROOM_CREATE_SECS)
        try:
            await asyncio.wait_for(
                lkapi.room.create_room(
                    lk_api.CreateRoomRequest(
                        name=self._room_name,
                        empty_timeout=self._empty_timeout_secs,
                        departure_timeout=self._departure_timeout_secs,
                    )
                ),
                create_budget,
            )
        except BaseException as exc:
            # Through `_release` (*cancel-proof cleanup*), so a second cancel cannot abort it and leak the
            # client's aiohttp session.
            interrupted = await _release(close_api_only, log)
            mapped = _create_room_failed(exc, create_budget)
            if interrupted and not isinstance(exc, asyncio.CancelledError):
                # Cancelled while cleaning up after some other failure: the caller's cancellation
                # wins, even over a failure that passes through unchanged, and the failure it
                # replaces stays on the chain.
                raise asyncio.CancelledError() from mapped
            if mapped is exc:
                raise
            raise mapped from exc
        log.info(f"private room created (empty_timeout={self._empty_timeout_secs}s)")

        # ③ The room-owning teardown, defined once: every failure below and the handle itself use it,
        # so there is exactly one place that knows how this room and this client are released.
        async def delete_room() -> None:
            try:
                await lkapi.room.delete_room(lk_api.DeleteRoomRequest(room=self._room_name))
                log.info("private room deleted")
            except lk_api.TwirpError as exc:
                # Already gone (`empty_timeout` fired, or someone deleted it): the end state we want.
                if exc.code != _NOT_FOUND:
                    log.warning(f"could not delete the private room: {exc}")
            except Exception as exc:
                log.warning(f"could not delete the private room: {describe_exception(exc)}")
            finally:
                await lkapi.aclose()

        release = delete_room if self._delete_room_on_close else close_api_only

        # ④ Connect the bot. BaseException, not Exception: a cancellation here (the service's launch
        # timeout) must still release the room we just created.
        connect_budget = _budget(deadline, self._connect_timeout)
        leaving: asyncio.Task[None] | None = None
        try:
            bot_token = mint_bot_token(
                credentials=self._credentials,
                room_name=self._room_name,
                bot_identity=self._bot_identity,
                bot_name=DEFAULT_BOT_NAME,
                ttl_seconds=DEFAULT_TOKEN_TTL_SECONDS,
            )
            room = self._room_factory()
            # The SDK's own deadline, inside ours, is what normally ends a connect that hangs.
            sdk_timeout = max(_MIN_BUDGET_SECS, connect_budget - _SDK_CONNECT_GRACE_SECS)
            options = lk_rtc.RoomOptions(auto_subscribe=True, connect_timeout=sdk_timeout)
            # Nothing may cancel an in-flight `Room.connect()`: the SDK would wait for an event that never
            # comes, and when its native side panics it sends the bot process SIGTERM (livekit rtc
            # 1.1.19-1.1.21). So the connect runs in a task of its own, which `wait_bounded` never
            # cancels. On a cancel, or past our budget, `leaving` takes the bot out of the room once the
            # connect ends; the release below deletes the room, which makes a pending connect fail fast,
            # and then waits for `leaving`, never for the connect itself.
            connect = asyncio.ensure_future(room.connect(ws_url, bot_token, options=options))
            try:
                connected = await wait_bounded(connect, connect_budget)
            except asyncio.CancelledError:
                leaving = _disconnect_once_connected(connect, room, sdk_timeout, log)
                raise
            if not connected:
                leaving = _disconnect_once_connected(connect, room, sdk_timeout, log)
                # Converted here rather than in `_connect_failed`: on 3.11 a socket-level ETIMEDOUT
                # is also a `TimeoutError`, and only this call site knows our budget is what expired.
                raise SynthesiaError(
                    f"the bot did not connect to the private room within {connect_budget:.1f}s",
                    type=ErrorType.CONNECTION,
                ) from TimeoutError()
            connected_at = monotonic()
        except BaseException as exc:
            interrupted = await _release(release, log)
            if leaving is not None:
                interrupted = await await_cancel_proof(leaving) or interrupted
            mapped = _connect_failed(exc)
            if interrupted and not isinstance(exc, asyncio.CancelledError):
                raise asyncio.CancelledError() from mapped  # as after a create failure
            if mapped is exc:
                raise
            raise mapped from exc
        log.info(f"bot {self._bot_identity} connected to the private room")

        # ⑤ From here the room is the session's (*room ownership handoff*): its launch() releases it on
        # every failure path, so nothing below adds a second cleanup.
        handle = RoomHandle(
            room=room,
            bot_identity=self._bot_identity,
            livekit_url=ws_url,
            room_name=self._room_name,
            owns_room=True,
            release_room=release,
            connected_at=connected_at,
        )
        session = await LiveKitAvatarSession.launch(
            handle,
            token_source=MintedAvatarToken(self._credentials),
            avatar_identity=self._avatar_identity,
            avatar_name=self._avatar_name,
            ctx=ctx,
        )

        # ⑥ Capture starts at connect, not at the first TTS: the worker renders an idle avatar from
        # the moment it joins, and a pipeline that only sees video once the bot speaks looks frozen.
        # Wrapped because launch() must leave nothing running when it raises, and past step ⑤ only we can
        # close the session, and the room with it.
        try:
            capture = AvatarTrackCapture(
                room,
                avatar_identity=self._avatar_identity,
                audio_sample_rate=ctx.audio_out_sample_rate,
                audio_channels=ctx.audio_out_channels,
                on_video_frame=on_video,
                on_audio_frame=on_audio,
                log=log,
            )
            # Hook before start: `start()` installs a listener and can create a consumer task before it
            # raises, and `capture.aclose()` running as the session's close hook is the only thing that
            # can stop either of them.
            session.add_close_hook(capture.aclose)
            capture.start()
        except BaseException:
            await session.aclose()  # idempotent, bounded, and it owns the room now
            raise
        return session


# --------------------------------------------------------------------------- module helpers

#: `ServerErrorCode.NOT_FOUND`: the room is already gone, which livekit-agents' own avatar code also treats
#: as success.
_NOT_FOUND = "not_found"

#: Floor under any budget derived from `ctx.launch_timeout`: a budget that has already run out should
#: still make one bounded attempt and fail with our message, not with `wait_for(..., 0)`.
_MIN_BUDGET_SECS = 0.5

#: How much earlier than our own timeout the SDK's `connect_timeout` fires (see the `RoomOptions` comment).
_SDK_CONNECT_GRACE_SECS = 1.0


def _disconnect_once_connected(
    connect: asyncio.Future[None], room: lk_rtc.Room, sdk_timeout: float, log: loguru.Logger
) -> asyncio.Task[None]:
    """Start the task that leaves ``room`` once a connect launch() stopped waiting for has ended.

    Both its steps are bounded (*bounded waits*): it waits for the connect up to the SDK's own
    ``sdk_timeout`` + 1 s, by which the SDK has normally ended it, and disconnects a room that connected
    within ``ROOM_DISCONNECT_SECS``. A failed connect is logged. Past the first bound, one warning, and
    the connect is let go (`_after_let_go`). The task never raises.
    """
    wait_secs = sdk_timeout + 1.0

    async def leave() -> None:
        done, _ = await asyncio.wait({connect}, timeout=wait_secs)
        if not done:
            log.warning(
                f"the bot's connect to the private room did not end within {wait_secs:.1f}s; letting it go"
            )
            _hold(connect).add_done_callback(lambda _: _after_let_go(connect, room, log))
            return
        if connect.cancelled():
            return
        if (exc := connect.exception()) is not None:
            log.info(f"the bot's connect to the private room failed: {describe_exception(exc)}")
            return
        await _disconnect(room, log)

    return asyncio.ensure_future(leave())


#: Let-go connects, and the tasks that disconnect them, until they finish: nothing else of ours holds
#: them once the launch has ended, and the event loop keeps only weak references to tasks.
_let_go: set[asyncio.Future[Any]] = set()


def _hold(task: asyncio.Future[Any]) -> asyncio.Future[Any]:
    """Hold ``task`` in `_let_go` until it finishes, and return it."""
    _let_go.add(task)
    task.add_done_callback(_let_go.discard)
    return task


def _after_let_go(connect: asyncio.Future[None], room: lk_rtc.Room, log: loguru.Logger) -> None:
    """When a let-go connect lands: log a failure (retrieving it, or asyncio reports it at ERROR when the
    task is collected), or leave a room that connected after all."""
    if connect.cancelled():
        return
    if (exc := connect.exception()) is not None:
        log.warning(f"the bot's let-go connect to the private room failed: {describe_exception(exc)}")
        return
    _hold(connect.get_loop().create_task(_disconnect(room, log)))


async def _disconnect(room: lk_rtc.Room, log: loguru.Logger) -> None:
    """Disconnect ``room`` within ``ROOM_DISCONNECT_SECS``; a failure is logged, never raised."""
    try:
        await asyncio.wait_for(room.disconnect(), _types.ROOM_DISCONNECT_SECS)
    except Exception as exc:
        log.warning(f"could not disconnect the bot from the private room: {describe_exception(exc)}")


def _required(value: str | None, parameter: str, env_var: str) -> str:
    """The value, the environment fallback, or a `ValueError` naming both ways to supply it."""
    resolved = (value if value is not None else os.environ.get(env_var, "")).strip()
    if not resolved:
        raise ValueError(f"{parameter} is required: pass {parameter}= or set {env_var}")
    return resolved


def _budget(deadline: float, cap: float) -> float:
    """What is left of the launch timeout, never more than `cap`, never below the floor."""
    return max(_MIN_BUDGET_SECS, min(cap, deadline - monotonic()))


async def _release(release: Callable[[], Awaitable[None]], log: loguru.Logger) -> bool:
    """Run ``release`` once, bounded, as *cancel-proof cleanup*; return whether we were cancelled."""
    return await await_cancel_proof(asyncio.ensure_future(_bounded_release(release, log)))


async def _bounded_release(release: Callable[[], Awaitable[None]], log: loguru.Logger) -> None:
    """Best effort, never raises: a teardown failure must not replace the error that caused it."""
    try:
        await asyncio.wait_for(release(), _types.ROOM_DELETE_SECS)
    except Exception as exc:
        log.warning(f"could not release the private room: {describe_exception(exc)}")


def _create_room_failed(exc: BaseException, budget: float) -> BaseException:
    """Map a `create_room` failure; anything that is not ours is returned unchanged."""
    if isinstance(exc, lk_api.TwirpError):
        if exc.status in (401, 403):
            return SynthesiaError(
                "LiveKit rejected the API key/secret used to create the private room",
                type=ErrorType.LIVEKIT_CREDENTIALS_REJECTED,
            )
        return SynthesiaError(f"could not create the private room: {exc}", type=ErrorType.CONNECTION)
    if isinstance(exc, TimeoutError):  # our `wait_for`, or the client's own total timeout
        return SynthesiaError(
            f"creating the private room did not finish within {budget:.0f}s",
            type=ErrorType.CONNECTION,
        )
    if isinstance(exc, SynthesiaError) or not isinstance(exc, Exception):
        return exc
    return SynthesiaError(
        f"could not create the private room: {describe_exception(exc)}", type=ErrorType.CONNECTION
    )


def _connect_failed(exc: BaseException) -> BaseException:
    """Map a bot-join failure; `CancelledError` (and anything else not ours) passes through.

    The budget case is converted at the call site, the only place that knows our own budget is what
    ran out — `TimeoutError` alone does not say that on 3.11.
    """
    if isinstance(exc, lk_rtc.ConnectError):
        return SynthesiaError(f"could not join the private room: {exc}", type=ErrorType.CONNECTION)
    if isinstance(exc, SynthesiaError) or not isinstance(exc, Exception):
        return exc
    return SynthesiaError(
        f"could not join the private room: {describe_exception(exc)}", type=ErrorType.CONNECTION
    )
