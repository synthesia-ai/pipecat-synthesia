"""The LiveKit avatar session: launch, leave detection and close, shared by both join modes.

**Launch.** In two orders, over the same steps: take the room (both playback RPC handlers, then the room
listeners, then whatever is already in it), send the session request, then wait for the 201 and the avatar
participant and read its readiness.

- `launch()`, for a room the bot is already connected to (a relayed join, or a direct join whose bot token
  names no room): the room, then the request.
- `launch_while_connecting()`, a direct join's *connect overlap*: the request, then the bot's connect, a
  check that the bot connected as the identity and to the room the avatar token names, then the room.

The session owns the request from the moment it starts, so `_roll_back_launch()` can cancel it. The avatar
worker reaches us only through synchronous room listeners (rtc rejects coroutine callbacks), so each
listener changes state first and delivers the service's callback in a callback task of its own
(*state-before-await*).

**Leave detection.** Three room events can mean the avatar left: the avatar participant disconnecting,
its video being unpublished, and the bot's own room connection ending. A disconnect with no stated reason
(or a duplicate identity), or a video unpublish, may be a same-identity replacement, so it starts the
*replacement grace*, which checks the room again when it ends. Any other disconnect, or the room ending,
is a leave at once. Either way the leave is declared once (`_declare_left`), and the service decides what
it means. When humans still appear present, it asks `wait_no_humans_left()` first (*no-humans-left window*).

**Close.** `_roll_back_launch()` undoes a launch that failed or was cancelled, cancelling the session
request if it is still in flight; `aclose()` is the normal close. Both release an owned room through
`_release_owned_room()` (*cancel-proof cleanup*, *single flight*), because cancelling `room.disconnect()`
wedges the LiveKit room: every later call on it blocks for its whole timeout.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import dataclasses
import secrets
from collections.abc import Awaitable, Callable, Coroutine
from time import monotonic
from typing import TYPE_CHECKING, Any

from pipecat_synthesia._async import CallbackTasks, await_cancel_proof
from pipecat_synthesia._redact import describe_exception, redact_jwt
from pipecat_synthesia.config import normalize_avatar_id
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._rtc import lk_rtc
from pipecat_synthesia.livekit.readiness import AvatarReadiness
from pipecat_synthesia.livekit.rpc import PlaybackRpcHandlers, perform_clear_buffer, perform_swap_avatar
from pipecat_synthesia.livekit.utterance_streamer import DEFAULT_IDLE_CLOSE_SECS, UtteranceStreamer
from pipecat_synthesia.protocol import (
    AVATAR_STATE_READY,
    ERROR_CODE_SERVICE_UNAVAILABLE,
    SWAP_DEFAULT,
    TOKEN_ATTR_LAUNCH,
)
from pipecat_synthesia.room import AvatarSession, LaunchTimings

if TYPE_CHECKING:
    import loguru

    from pipecat_synthesia.api import SessionCreated
    from pipecat_synthesia.livekit._rtc_protocols import LocalParticipantLike, RoomLike
    from pipecat_synthesia.livekit._types import AvatarTokenSource, RoomHandle
    from pipecat_synthesia.room import LaunchContext

# The leave reasons that `AvatarCallbacks.on_avatar_left` documents. Each comes from a LiveKit room event:
# participant_disconnected, track_unpublished (video), and the room's own disconnected.
AVATAR_LEFT_ROOM = "avatar_left_room"
AVATAR_VIDEO_STOPPED = "avatar_video_stopped"
ROOM_CONNECTION_LOST = "room_connection_lost"

#: How informative each reason is, for `_start_grace` when two signals describe one leave.
#: "the participant left" explains "its video went away"; not the other way round.
_LEFT_REASON_RANK = {
    AVATAR_VIDEO_STOPPED: 0,
    AVATAR_LEFT_ROOM: 1,
    ROOM_CONNECTION_LOST: 2,
}

#: Disconnect reasons that may mean "replaced" rather than "gone", and so start the *replacement grace*.
#: The replaced client itself sees `DUPLICATE_IDENTITY`, but everyone else, us included, sees no reason:
#: livekit-server sends `UNKNOWN_REASON`, which rtc surfaces as `None` (livekit-server 1.9.1, livekit rtc
#: 1.1.19). Both spellings are listed, because a fake or a future rtc may report the enum.
_REPLACEMENT_REASONS = (
    None,
    lk_rtc.DisconnectReason.UNKNOWN_REASON,
    lk_rtc.DisconnectReason.DUPLICATE_IDENTITY,
)


#: Added to a direct join's 503 `service_unavailable`. The avatar worker's failure to join the room, a wrong
#: LiveKit secret or a URL that resolves to a private address, reaches the bot only as that generic 503.
_DIRECT_JOIN_503_HINT = (
    "If this persists, the avatar worker most likely could not join your LiveKit room: credentials from a "
    "different LiveKit project, or a URL that resolves to a private address."
)


class LiveKitAvatarSession(AvatarSession):
    """A live Synthesia avatar in a LiveKit room: audio out, playback RPCs, readiness and teardown."""

    avatar_participant: lk_rtc.RemoteParticipant | None
    # Set by `_attach()`, once the bot is connected. Two views of one object: `handle.room` for the concrete
    # rtc types the public attributes carry, `_room` for the narrow Protocol the fakes satisfy (and whose
    # `on()` takes any event name).
    _room: RoomLike
    _participant: LocalParticipantLike

    def __init__(
        self,
        *,
        owns_room: bool,
        avatar_identity: str,
        avatar_name: str,
        ctx: LaunchContext,
        log,
    ) -> None:
        """Built by `launch()` and `launch_while_connecting()` only; not usable until one has returned it.

        The room arrives later, in `_attach()`: in a *connect overlap* the session request goes out before
        the bot is connected. Whether the room is owned is known from the start.
        """
        self.avatar_participant = None
        self._handle: RoomHandle | None = None
        self._owns_room = owns_room
        self._avatar_identity = avatar_identity
        self._avatar_name = avatar_name
        self._ctx = ctx
        # Every logger derived from this one, including those made before the 201 (the audio stream's, after
        # an *early start*), carries the session id once the 201 has set it.
        self._log = log.patch(self._add_session_id)
        self._handlers: PlaybackRpcHandlers | None = None
        self._listeners: list[tuple[str, Callable[..., Any]]] = []
        # The session request, owned from the moment it starts: `_roll_back_launch()` cancels it.
        self._request: asyncio.Task[SessionCreated] | None = None
        self._requested_at: float | None = None
        self._created_at: float | None = None
        self._session_id: str | None = None
        self._timings: LaunchTimings | None = None
        self._readiness: AvatarReadiness | None = None
        self._streamer: UtteranceStreamer | None = None
        self._active = ctx.avatar_ids[0]
        self._joined: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        # when the avatar participant was first seen in the room, which `_joined` resolves on: the launch may
        # go on waiting for its video or the 201 well after that, so the join is timed here
        self._joined_at: float | None = None
        self._alive = True
        self._left = False
        # the leave reason `_declare_left` reported; None until the avatar has left
        self._declared_reason: str | None = None
        self._grace_timer: asyncio.Task[None] | None = None
        #: Reason the pending grace timer will report; sharpened by a later, more specific signal.
        self._left_reason = AVATAR_VIDEO_STOPPED
        # The one owned-room disconnect+delete, shared by `_roll_back_launch()` and `aclose()`.
        self._releasing: asyncio.Task[None] | None = None
        self._closing: asyncio.Future[None] | None = None
        self._hooks: list[Callable[[], Awaitable[None]]] = []
        self._callbacks = CallbackTasks("avatar callback")
        self._warned_dead_send = False
        self._ready_logged = False
        # *Early start*: a random id on the avatar token we mint, and whether the avatar that carries it has
        # published its video. Only that avatar is known to be this launch's before the 201.
        self._launch_id = secrets.token_hex(8)
        self._own_video: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._started_early = False
        # Once the request is retried, the avatar in the room may be replaced by the retry's worker, which
        # carries the same token: no early start from then, and after one, sending is held until the request
        # settles.
        self._request_retried = False
        self._sending = asyncio.Event()
        self._sending.set()

    @classmethod
    async def launch(
        cls,
        handle: RoomHandle,
        *,
        token_source: AvatarTokenSource,
        avatar_identity: str,
        avatar_name: str,
        ctx: LaunchContext,
    ) -> LiveKitAvatarSession:
        """Launch an avatar session in a room the bot is already connected to, and return it once the avatar
        has joined.

        The relayed join's order, and a direct join's when the bot token names no room: the playback RPC
        handlers are registered, and the room listeners added, before the session request goes out. On any
        failure or cancellation, `_roll_back_launch()` undoes what was done (*cancel-proof cleanup*), and the
        error is re-raised with the session id stamped on it.
        """
        session = cls(
            owns_room=handle.owns_room,
            avatar_identity=avatar_identity,
            avatar_name=avatar_name,
            ctx=ctx,
            log=ctx.log.bind(room=handle.room_name),
        )
        return await session._run_launch(session._connected_steps(handle, token_source))

    @classmethod
    async def launch_while_connecting(
        cls,
        connect: Callable[[], Awaitable[RoomHandle]],
        *,
        bot_identity: str,
        room_name: str,
        livekit_url: str,
        token_source: AvatarTokenSource,
        avatar_identity: str,
        avatar_name: str,
        ctx: LaunchContext,
    ) -> LiveKitAvatarSession:
        """Launch an avatar session while the bot is still connecting (*connect overlap*), and return it once
        the avatar has joined.

        A direct join only, so the room is borrowed. The session request goes out first, for the bot
        identity and room the bot token names; ``connect()`` then returns the connected room, where the bot
        must be that identity in that room. Failure and cancellation roll back as in `launch()`; a failed
        connect first goes through `_after_failed_connect()`.
        """
        session = cls(
            owns_room=False,
            avatar_identity=avatar_identity,
            avatar_name=avatar_name,
            ctx=ctx,
            log=ctx.log.bind(room=room_name),
        )
        steps = session._overlap_steps(
            connect, token_source, bot_identity=bot_identity, room_name=room_name, livekit_url=livekit_url
        )
        return await session._run_launch(steps)

    # ------------------------------------------------------------------ launch internals

    async def _run_launch(self, steps: Coroutine[Any, Any, None]) -> LiveKitAvatarSession:
        """Run one of the launch orders below, and roll back on any failure or cancellation."""
        try:
            await steps
        except BaseException as first:  # including the CancelledError of the service's launch timeout
            interrupted = await await_cancel_proof(asyncio.ensure_future(self._roll_back_launch()))
            if isinstance(first, SynthesiaError):
                first.with_session(self._session_id)
            if interrupted and not isinstance(first, asyncio.CancelledError):
                raise asyncio.CancelledError() from first
            raise
        return self

    async def _connected_steps(self, handle: RoomHandle, token_source: AvatarTokenSource) -> None:
        """`launch()`'s order: the room, then the session request, then the 201 and the join."""
        self._attach(handle)
        self._request_session(
            token_source,
            bot_identity=handle.bot_identity,
            room_name=handle.room_name,
            livekit_url=handle.livekit_url,
        )
        await self._finish_launch()

    async def _overlap_steps(
        self,
        connect: Callable[[], Awaitable[RoomHandle]],
        token_source: AvatarTokenSource,
        *,
        bot_identity: str,
        room_name: str,
        livekit_url: str,
    ) -> None:
        """`launch_while_connecting()`'s order (*connect overlap*): the session request, the bot's connect,
        then the room, then the 201 and the join."""
        self._request_session(
            token_source, bot_identity=bot_identity, room_name=room_name, livekit_url=livekit_url
        )
        try:
            handle = await connect()
            _check_connected_as(handle, bot_identity=bot_identity, room_name=room_name)
        except Exception:
            # Not the launch timeout's CancelledError: that goes straight to the rollback.
            await self._after_failed_connect()
            raise
        self._attach(handle)
        video_already_published = self._video_already_published()
        await self._finish_launch()
        # Reported only now: until the 201 or this launch's video, an avatar already in the room may be a
        # stale one.
        if video_already_published and not self._left:
            self._callbacks.schedule(self._ctx.callbacks.on_track_published("video"), log=self._log)

    def _attach(self, handle: RoomHandle) -> None:
        """Take the connected room: both playback handlers, then the listeners, then what is already there."""
        self._handle = handle
        self._room = handle.room
        # Both join modes hand over a connected room, so `local_participant` is set.
        self._participant = participant = handle.room.local_participant
        # Both playback handlers before anything else: without `lk.playback_finished`, the worker ends its
        # session after the first utterance. Registered before the session is active, which is early enough:
        # no utterance is sent before then (*connect overlap*).
        self._handlers = PlaybackRpcHandlers(
            participant,
            expected_caller=self._avatar_identity,
            on_started=self._on_playback_started,
            on_finished=self._on_playback_finished,
            log=self._log,
        )
        self._handlers.register()
        # Listeners first, then the existing-participant check: the avatar may be in the room already,
        # because a retry with the same token replaces a stalled worker, or, in a *connect overlap*, because
        # it joined while the bot was connecting.
        self._listen("participant_connected", self._on_participant_connected)
        self._listen("track_published", self._on_track_published)
        existing = handle.room.remote_participants.get(self._avatar_identity)
        if existing is not None and not self._joined.done():
            self._joined_at = monotonic()
            self._joined.set_result(existing)

    def _video_already_published(self) -> bool:
        """Whether the avatar's video was published before the `track_published` listener existed.

        Only a *connect overlap* can find it there: the avatar may join and publish while the bot connects.
        rtc lists the tracks already in the room at connect without a `track_published` for them, and one
        that arrives just after (the avatar's video did, against livekit-server 1.9.1) reaches no listener
        while Pipecat finishes its own connect by publishing the bot's microphone. `_overlap_steps` then
        reports it late rather than never, for the `avatar_video_published` metric.
        """
        avatar = self.handle.room.remote_participants.get(self._avatar_identity)
        return avatar is not None and any(
            pub.kind == lk_rtc.TrackKind.KIND_VIDEO for pub in avatar.track_publications.values()
        )

    def _request_session(
        self, token_source: AvatarTokenSource, *, bot_identity: str, room_name: str, livekit_url: str
    ) -> None:
        """Resolve the avatar token, and send the session request in a task the session owns from here on."""
        # Minted or validated, once per launch: a retry with a new identity would take a second
        # concurrency slot.
        token = token_source.resolve(
            bot_identity=bot_identity,
            room_name=room_name,
            avatar_identity=self._avatar_identity,
            avatar_name=self._avatar_name,
            client=self._ctx.client,
            launch_id=self._launch_id,
        )
        self._log.debug(f"avatar token {redact_jwt(token)}")
        self._log.info("avatar token minted/validated")
        self._request = asyncio.get_running_loop().create_task(
            self._create_session(livekit_url, token), name="synthesia-session-request"
        )
        # a retry's hold ends when the request settles, whatever its outcome and whoever waits for it
        self._request.add_done_callback(lambda _request: self._sending.set())

    async def _create_session(self, livekit_url: str, token: str) -> SessionCreated:
        """The session request, timed here rather than where it is awaited: in a *connect overlap* the 201
        can arrive before the bot has connected. The API client retries inside `create_session`, so a retry
        never moves `session_requested`. The 201 is reported to the service at once, with no `await` in
        between: the session bills and can be looked up from here, even if the launch fails later."""
        self._requested_at = monotonic()
        ctx = self._ctx
        try:
            created = await ctx.api.create_session(
                avatar_ids=ctx.avatar_ids,
                livekit_url=livekit_url,
                livekit_token=token,
                on_retry=self._on_request_retry,
            )
        except SynthesiaError as err:
            # A direct join only: a relayed join created its room with these credentials, which proves them,
            # while a borrowed room's credentials are the developer's and unchecked.
            if err.code == ERROR_CODE_SERVICE_UNAVAILABLE and not self._owns_room:
                err.with_hint(_DIRECT_JOIN_503_HINT)
            raise
        self._created_at = monotonic()
        ctx.callbacks.on_session_created(created.session_id, created.request_id, self._created_at)
        self._take_created(created)
        return created

    async def _finish_launch(self) -> None:
        """The 201, the join wait, then readiness and the listeners that last for the whole session.

        *Early start*: if this launch's avatar (the one carrying our ``synthesia.launch`` id) publishes its
        video before the 201 arrives, the launch goes on without the 201, which typically trails the video by
        about half a second; the request stamps it on the session when it lands (`_take_created`). Any other
        avatar, including a stale one with the same identity, or one joined with a caller-provided token,
        waits for the 201 as before.
        """
        handle, ctx = self.handle, self._ctx
        request = self._request
        assert request is not None  # both launch orders send the request before they get here
        avatar_now = handle.room.remote_participants.get(self._avatar_identity)
        if self._is_this_launch(avatar_now) and self._video_already_published():
            self._mark_own_video()
        # the 201 means the worker's tracks are published, not that it is rendering
        await asyncio.wait({request, self._own_video}, return_when=asyncio.FIRST_COMPLETED)
        if request.done() or self._request_retried:
            await request  # the 201, which stamps itself on the session (`_take_created`), or its failure
        else:
            self._started_early = True
            self._log.info("this launch's avatar published its video before the 201: starting early")
        assert self._requested_at is not None  # stamped by the request, which has started
        log = self._log

        # the join wait, bounded by the smaller of its backstop and the *launch timeout*
        wait_secs = min(_types.POST_CREATE_JOIN_WAIT_SECS, ctx.launch_timeout)
        avatar = await self._wait_for_avatar(wait_secs)
        # the join itself, not now: after an early start the launch got here at the video, and otherwise at
        # the 201, both later than the join
        avatar_joined = self._joined_at if self._joined_at is not None else monotonic()
        self.avatar_participant = avatar
        self._timings = LaunchTimings(
            room_connected=handle.connected_at,
            session_requested=self._requested_at,
            session_created=self._created_at,
            avatar_joined=avatar_joined,
        )
        since_request = avatar_joined - self._requested_at
        if since_request >= 0:
            log.info(f"avatar joined {since_request * 1000:.0f} ms after the session request")
        else:
            log.info("avatar was already in the room when the session was requested")

        # readiness from the attributes already on the participant, then the listeners that last for the
        # whole session
        self._readiness = AvatarReadiness(ctx.avatar_ids, log)
        if self._readiness.update(avatar.attributes):
            self._deliver_status()
        self._listen("participant_attributes_changed", self._on_attributes_changed)
        self._listen("participant_disconnected", self._on_participant_disconnected)
        self._listen("track_unpublished", self._on_track_unpublished)
        self._listen("disconnected", self._on_room_disconnected)
        self._streamer = UtteranceStreamer(
            self._participant,
            destination_identity=self._avatar_identity,
            idle_close_secs=DEFAULT_IDLE_CLOSE_SECS,
            log=log,
        )

    def _take_created(self, created: SessionCreated) -> None:
        """Stamp the 201 on the session as it lands, whether or not anything waits for it then: its session
        id (and with it every log record's), and its time on the timings if the launch has returned."""
        self._session_id = created.session_id
        self._log.info(f"session created {created.session_id} ({created.request_id})")
        if self._timings is not None and self._timings.session_created is None:
            self._timings = dataclasses.replace(self._timings, session_created=self._created_at)

    def _add_session_id(self, record: loguru.Record) -> None:
        if self._session_id is not None:
            record["extra"].setdefault("session_id", self._session_id)

    def _is_this_launch(self, participant: lk_rtc.RemoteParticipant | None) -> bool:
        """Whether ``participant`` joined with the avatar token this launch minted (*early start*)."""
        return participant is not None and participant.attributes.get(TOKEN_ATTR_LAUNCH) == self._launch_id

    def _on_request_retry(self) -> None:
        """The session request is being retried, with the same token. The failed attempt may have launched
        the avatar now in the room, and the retry may launch a second worker as the same identity that
        replaces it, so that avatar is no longer known to be the one that stays: no early start from here.

        After an early start, sending is held (`sending` False) from here until the request settles: its
        201, which follows the worker that stays, or its failure. So nothing taken from the send queue after
        the retry goes to an avatar the retry may replace (at most one chunk, whose write was already under
        way): seen live, a reply sent in that window was never heard. The service's *launch timeout* bounds
        the hold through its failure path, and during a graceful stop the drain timeout does: a request still
        out then loses the goodbye. What the avatar was sent before the retry, often the greeting, is still
        cut off if it is replaced (accepted, as rare). Held audio shares the pre-launch cap
        (`prelaunch_max_seconds`). When the API deduplicates retries, the hold only costs one attempt's
        wait, and should go.
        """
        if self._request_retried:
            return
        self._request_retried = True
        if self._started_early and self._session_id is None:
            self._sending.clear()
            self._log.warning(
                "the session request is being retried after an early start: holding the bot's audio until "
                "the request settles, in case the retry's avatar worker replaces this one"
            )

    def _mark_own_video(self) -> None:
        if not self._own_video.done():
            self._own_video.set_result(None)

    async def _after_failed_connect(self) -> None:
        """The bot's connect failed, or put it somewhere the avatar token does not name, after the session
        request had gone out: give up on the request, and say so.

        The request carries the avatar's own token, so the avatar worker joins and publishes, and billing
        starts, whether or not the bot ever arrives. Nothing here can end that session: there is no
        end-session API, and the room is borrowed (*borrowed-room rule*). The worker ends it itself once the
        bot fails to appear, about 30 s after the avatar started, when the worker's agent-join watchdog runs
        out. So one warning says a session may exist, naming it if the 201 has already arrived (the connect
        error then carries its id too), and the request is abandoned: `_roll_back_launch()`, which runs
        next, cancels it. That leaves nothing running, but cannot stop the worker.

        Called from the connect-failure path only. The launch timeout's cancellation goes straight to the
        rollback, which cancels the request the same way, because there is no time left to wait for it.
        """
        request = self._request
        if request is None:
            return
        if not request.done():
            self._log.warning(
                "the session request went out, but the bot did not connect as the avatar token expects: a "
                "session may exist, and the avatar worker will end it within about 30 s"
            )
        elif not request.cancelled() and request.exception() is None:
            self._session_id = request.result().session_id
            self._log.warning(
                f"session {self._session_id} was created, but the bot did not connect as the avatar token "
                "expects: the avatar worker will end it within about 30 s"
            )

    async def _wait_for_avatar(self, wait_secs: float) -> lk_rtc.RemoteParticipant:
        """Wait for the avatar participant and return the object rtc currently holds for it.

        The participant that resolved the future is never used: it may already have left again, and
        rtc replaces the object on every join. If the re-fetch finds nothing, wait again on a fresh
        future with whatever time is left.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_secs
        while True:
            try:
                await asyncio.wait_for(self._joined, max(0.0, deadline - loop.time()))
            except TimeoutError as exc:
                # `wait_for` cancels the future on timeout, so a join event that lands in the same
                # loop iteration can no longer resolve it; the room itself is the source of truth.
                present = self.handle.room.remote_participants.get(self._avatar_identity)
                if present is not None:
                    self._log.debug("avatar participant was in the room as the join wait expired")
                    return present
                raise SynthesiaError(
                    f"avatar did not join within {wait_secs}s",
                    type=ErrorType.TIMEOUT,
                    session_id=self._session_id,
                ) from exc
            avatar = self.handle.room.remote_participants.get(self._avatar_identity)
            if avatar is not None:
                return avatar
            self._log.debug("avatar participant left again before it was observed; waiting for a re-join")
            self._joined = loop.create_future()
            self._joined_at = None

    def _listen(self, event: str, callback: Callable[..., Any]) -> None:
        self._room.on(event, callback)
        self._listeners.append((event, callback))

    # ------------------------------------------------------------------ room listeners (all sync)

    def _on_participant_connected(self, participant: lk_rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity:
            return
        if self._timings is None:
            # Every join until the launch times it, not only the first: a stale avatar with the same identity
            # seen at attach can be replaced by this launch's, and the launch uses the participant in the room
            # now (`_wait_for_avatar`), so the latest join is its join.
            self._joined_at = monotonic()
        if not self._joined.done():
            self._joined.set_result(participant)
        rejoin = self._grace_timer is not None
        self._cancel_grace()
        self.avatar_participant = participant
        if self._readiness is not None and self._readiness.update(participant.attributes):
            self._deliver_status()
        if rejoin:
            self._log.info("avatar participant re-joined (same-identity replacement)")
        else:
            self._log.info("avatar participant joined")

    def _on_track_published(self, publication: Any, participant: lk_rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity or self._left:
            return
        kind = _track_kind_name(publication.kind)
        if kind == "video" and self._is_this_launch(participant):
            self._mark_own_video()
        if kind is not None:
            self._callbacks.schedule(self._ctx.callbacks.on_track_published(kind), log=self._log)

    def _on_attributes_changed(self, changed: Any, participant: lk_rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity or self._readiness is None:
            return
        # The worker writes its whole attribute map back on every change, so read the map.
        if self._readiness.update(participant.attributes):
            self._deliver_status()

    def _on_participant_disconnected(self, participant: lk_rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity:
            return
        self.avatar_participant = None
        reason = participant.disconnect_reason
        if reason in _REPLACEMENT_REASONS:
            # possibly a same-identity replacement (see `_REPLACEMENT_REASONS`)
            self._log.info(
                f"avatar participant may have been replaced (disconnect_reason={reason}); "
                f"waiting {_types.REPLACEMENT_GRACE_SECS}s"
            )
            if not self._left and self._closing is None:
                # whatever it was playing has stopped, and no worker will finish it
                self._callbacks.schedule(self._ctx.callbacks.on_playback_lost(), log=self._log)
            self._start_grace(AVATAR_LEFT_ROOM)
        else:
            self._declare_left(AVATAR_LEFT_ROOM)

    def _on_track_unpublished(self, publication: Any, participant: lk_rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity:
            return
        # Never `_declare_left` directly: in a replacement this precedes `participant_disconnected`.
        if publication.kind == lk_rtc.TrackKind.KIND_VIDEO:
            self._start_grace(AVATAR_VIDEO_STOPPED)

    def _on_room_disconnected(self, reason: Any) -> None:
        self._declare_left(ROOM_CONNECTION_LOST)

    # ------------------------------------------------------------------ leave detection

    def _start_grace(self, reason: str) -> None:
        """Start the *replacement grace*, or sharpen the reason of the one already running.

        The deadline belongs to the first signal: a later one never extends the grace, or a chatty leave
        could postpone the report indefinitely. The reason is kept from the most specific signal: LiveKit
        sends `track_unpublished` (video) before `participant_disconnected` for the same leave (measured
        against livekit-server 1.9.1), so keeping the first would report "the avatar stopped publishing
        video" when the truth is "the avatar left the room". `_LEFT_REASON_RANK` decides. None starts once
        the close has begun: a close is not a leave.
        """
        if self._left or self._closing is not None:
            return
        self._left_reason = max((self._left_reason, reason), key=lambda name: _LEFT_REASON_RANK.get(name, -1))
        if self._grace_timer is None:
            self._grace_timer = self._callbacks.track(
                asyncio.get_running_loop().create_task(self._grace_expired())
            )

    async def _grace_expired(self) -> None:
        # Read from the module at call time, so tests can shrink it.
        await asyncio.sleep(_types.REPLACEMENT_GRACE_SECS)
        if self._grace_timer is asyncio.current_task():
            self._grace_timer = None
        avatar = self.handle.room.remote_participants.get(self._avatar_identity)
        if avatar is not None and any(
            pub.kind == lk_rtc.TrackKind.KIND_VIDEO for pub in avatar.track_publications.values()
        ):
            self._log.info("avatar back after replacement")
            self._left_reason = AVATAR_VIDEO_STOPPED
            self.avatar_participant = avatar
            return
        # Read now, not when the grace started: a later signal may have sharpened it (see `_start_grace`).
        self._declare_left(self._left_reason)

    def _cancel_grace(self) -> None:
        """Drop the pending grace timer and forget the reason it had accumulated.

        The reason belongs to the leave that armed the timer. A re-join that cancels it must not
        leave "avatar_left_room" behind for a later, unrelated video unpublish to report.
        """
        timer, self._grace_timer = self._grace_timer, None
        self._left_reason = AVATAR_VIDEO_STOPPED
        if timer is not None and not timer.done():
            timer.cancel()

    def _declare_left(self, reason: str) -> None:
        """Declare that the avatar has left: at most once, with the flags set before any await.

        LiveKit emits `track_unpublished` and `participant_disconnected` for the same leave in the same
        loop iteration; the `_left` guard makes "at most once" true. Never once the close has begun: a close
        is not a leave. Does not call `aclose()`: the service decides what a leave means.
        """
        if self._left or self._closing is not None:
            return
        self._left = True
        self._alive = False
        self._declared_reason = reason
        self._cancel_grace()
        humans = self._humans_present()
        self._log.info(f"avatar left: {reason} (humans_present={humans})")
        self._callbacks.track(asyncio.get_running_loop().create_task(self._deliver_left(reason, humans)))

    async def _deliver_left(self, reason: str, humans: bool | None) -> None:
        if self._streamer is not None:
            await self._streamer.abort()
        await self._callbacks.guard(self._ctx.callbacks.on_avatar_left(reason, humans), log=self._log)

    def _humans_present(self) -> bool | None:
        """In a direct join, whether a human participant remains (rtc has already removed the one leaving)."""
        if self._owns_room:
            return None
        return any(
            participant.kind == lk_rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
            for participant in self.handle.room.remote_participants.values()
        )

    async def wait_no_humans_left(self) -> bool:
        """Wait up to the *no-humans-left window* for the last human participant to leave; return whether none
        remains. False at once in a relayed join, where the session cannot tell, and after the bot lost the
        room (`room_connection_lost`), where it can no longer hear anyone leave.

        The service calls it once it has started the close, which removes the session's listeners, so the
        wait listens for itself. The answer is read when the wait ends, not when it starts.
        """
        if self._owns_room:
            return False
        if self._declared_reason == ROOM_CONNECTION_LOST:
            # The bot's own connection is gone, so its view of the room means nothing, and no
            # participant_disconnected can arrive: rtc does not clear remote_participants on `disconnected`
            # (livekit rtc 1.1.19). Waiting would only delay the error.
            return False
        if not self._humans_present():
            return True
        started = monotonic()
        gone = asyncio.get_running_loop().create_future()

        def on_participant_disconnected(participant: lk_rtc.RemoteParticipant) -> None:
            if not gone.done() and not self._humans_present():
                gone.set_result(None)

        self._room.on("participant_disconnected", on_participant_disconnected)
        try:
            # Read from the module at call time, so tests can change it.
            await asyncio.wait({gone}, timeout=_types.NO_HUMANS_LEFT_WINDOW_SECS)
        finally:
            self._room.off("participant_disconnected", on_participant_disconnected)
        if self._humans_present():
            return False
        self._log.info(f"the last human participant left {monotonic() - started:.1f}s after the avatar")
        return True

    # ------------------------------------------------------------------ callbacks

    async def _on_playback_started(self) -> None:
        # Already inside the handler's own guarded task (rpc.py); nothing after `on_avatar_left`.
        if self._left:
            return
        await self._ctx.callbacks.on_playback_started()

    async def _on_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        if self._left:
            return
        await self._ctx.callbacks.on_playback_finished(playback_position, interrupted)

    def _deliver_status(self) -> None:
        readiness = self._readiness
        if readiness is None or self._left:
            return
        status, all_ready = readiness.status, readiness.all_ready
        if all_ready and not self._ready_logged:
            self._ready_logged = True
            self._log.info("avatar ready")
        self._callbacks.schedule(self._ctx.callbacks.on_avatar_status(status, all_ready), log=self._log)

    # ------------------------------------------------------------------ AvatarSession interface

    @property
    def handle(self) -> RoomHandle:
        """The room the session runs in; known once the bot is connected."""
        if self._handle is None:
            raise RuntimeError("avatar session is not connected")
        return self._handle

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("avatar session is not connected")
        return self._session_id

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def room_name(self) -> str | None:
        return self.handle.room_name

    @property
    def active_avatar_id(self) -> str:
        return self._active

    @property
    def avatar_status(self) -> dict[str, str]:
        return self._readiness.status if self._readiness is not None else {}

    @property
    def all_ready(self) -> bool:
        return self._readiness is not None and self._readiness.all_ready

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def timings(self) -> LaunchTimings:
        if self._timings is None:
            raise RuntimeError("avatar session is not connected")
        return self._timings

    async def wait_session_created(self) -> None:
        if self._session_id is not None:
            return
        request = self._request
        assert request is not None  # launch() returned, so the request went out
        try:
            # shielded: cancelling the caller must not cancel the request; aclose() owns it from here on
            await asyncio.shield(request)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if request.cancelled() and not (current is not None and current.cancelling()):
                raise SynthesiaError(
                    "the avatar session closed before the Synthesia API returned the session",
                    type=ErrorType.CONNECTION,
                ) from None
            raise

    @property
    def sending(self) -> bool:
        return self._sending.is_set()

    async def wait_sending(self) -> None:
        await self._sending.wait()

    async def send_audio(self, pcm_s16le: bytes, sample_rate: int, num_channels: int) -> None:
        streamer = self._streamer
        if not self._alive or streamer is None:
            if not self._warned_dead_send:
                self._warned_dead_send = True
                self._log.warning("avatar session is no longer alive; dropping audio")
            return
        self._log.trace("send_audio {} bytes at {} Hz x{}", len(pcm_s16le), sample_rate, num_channels)
        await streamer.write(pcm_s16le, sample_rate, num_channels)

    async def end_utterance(self) -> None:
        if self._streamer is not None:
            await self._streamer.end()

    async def interrupt(self) -> None:
        """Send clear buffer (bounded, never raises), then end the current utterance.

        Clear buffer first, so the worker drops the audio it already holds; then the close ends the
        utterance. Once the avatar has left, the RPC could only time out, so only the close remains.
        """
        if self._alive:
            await perform_clear_buffer(
                self._participant, avatar_identity=self._avatar_identity, log=self._log
            )
        else:
            self._log.debug("interrupt after the avatar left: skipping clear_buffer")
        if self._streamer is not None:
            await self._streamer.abort()

    async def swap_avatar(self, avatar_id: str, *, timeout: float) -> str:  # noqa: ASYNC109
        try:
            return await self._swap(avatar_id, budget_secs=timeout)
        except SynthesiaError as err:
            err.with_session(self._session_id)
            raise

    async def _swap(self, avatar_id: str, *, budget_secs: float) -> str:
        ctx = self._ctx
        prefixed: str | None
        if avatar_id == SWAP_DEFAULT:
            prefixed = ctx.avatar_ids[0]
        else:
            try:
                prefixed = normalize_avatar_id(avatar_id)
            except ValueError:
                prefixed = None
        if prefixed is None or prefixed not in ctx.avatar_ids:
            raise SynthesiaError(
                f"avatar {avatar_id} is not one of this session's avatar_ids: {list(ctx.avatar_ids)}",
                type=ErrorType.UNKNOWN_AVATAR,
                session_id=self._session_id,
            )
        deadline = monotonic() + budget_secs
        readiness = self._readiness
        # `state_of is None` means no status has been seen at all: do not wait on a signal that may
        # never come (an older worker); the RPC itself reports readiness problems.
        if readiness is not None and readiness.state_of(prefixed) not in (None, AVATAR_STATE_READY):
            self._log.info(f"avatar {prefixed} is not ready yet; waiting before the swap")
            await readiness.wait_ready(prefixed, timeout=max(0.0, deadline - monotonic()))
        new = await perform_swap_avatar(
            self._participant,
            avatar_identity=self._avatar_identity,
            avatar_id=prefixed,
            timeout=max(0.5, deadline - monotonic()),
        )
        self._active = new
        self._log.info(f"avatar swapped to {new}")
        return new

    def add_close_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        self._hooks.append(hook)

    async def aclose(self) -> None:
        """Close the session, once (*single flight*), with every wait bounded.

        The close runs in a task of its own that every caller joins (*cancel-proof cleanup*), so a
        cancelled caller neither aborts the close nor marks it finished.
        """
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._close_everything())
        if await await_cancel_proof(self._closing):
            raise asyncio.CancelledError()

    async def _close_everything(self) -> None:
        log = self._log
        # First, synchronously: a `send_audio` that races the close drops instead of reopening a stream, and
        # a replacement grace already running cannot report a leave during the awaits below.
        self._alive = False
        self._sending.set()  # nothing waits on a hold once the session is closing
        self._cancel_grace()
        # after an *early start* the 201 may still be on its way; nothing is left waiting for it
        await self._cancel_request()
        streamer = self._streamer
        if streamer is not None:
            try:
                # The close and the reap after it are each bounded by STREAM_CLOSE_SECS. The wait for the
                # streamer's lock before them is not: it queues behind a write in flight, which ends only
                # when rtc answers it. The service's SESSION_ACLOSE_BACKSTOP_SECS bounds its wait for the
                # whole close.
                await streamer.aclose()
            except Exception as exc:
                log.warning(f"aclose: closing the audio stream failed: {describe_exception(exc)}")
        # Listeners off before the owned-room disconnect, or `disconnected` would report a leave.
        self._remove_listeners()
        self._unregister_handlers()
        for hook in list(self._hooks):
            try:
                await asyncio.wait_for(hook(), _types.CLOSE_HOOK_SECS)
            except Exception as exc:
                log.warning(f"aclose: close hook {hook!r} failed: {describe_exception(exc)}")
        await self._release_owned_room()
        log.info("avatar session closed")

    # ------------------------------------------------------------------ teardown internals

    async def _roll_back_launch(self) -> None:
        """Undo whatever the launch had done so far. Never raises; nothing ever cancels it."""
        try:
            self._alive = False
            self._unregister_handlers()
            self._remove_listeners()
            self._cancel_grace()
            await self._cancel_request()
            await self._release_owned_room()
        except Exception:
            self._log.exception("rolling back a failed launch raised")

    async def _cancel_request(self) -> None:
        """Cancel the session request if it is still in flight, and wait for it to stop (*bounded waits*)."""
        request = self._request
        if request is None:
            return
        if not request.done():
            request.cancel()
            # `asyncio.wait` never raises on behalf of what it waits for, so the CancelledError of the
            # request we just cancelled cannot be mistaken for our own, as `await_cancel_proof` would.
            await asyncio.wait({request}, timeout=_types.TASK_CANCEL_SECS)
        if not request.done():
            self._log.warning(f"the session request did not stop within {_types.TASK_CANCEL_SECS}s")
            # When it does, its outcome is retrieved, so the event loop does not report it as unhandled.
            request.add_done_callback(lambda task: task.cancelled() or task.exception())
        elif not request.cancelled():
            # Retrieved, so the event loop does not report it again: the launch has failed anyway, with this
            # error or one that replaced it.
            request.exception()

    def _unregister_handlers(self) -> None:
        if self._handlers is None:
            return
        try:
            self._handlers.unregister()
        except Exception as exc:
            self._log.warning(f"could not unregister the playback RPC handlers: {describe_exception(exc)}")

    def _remove_listeners(self) -> None:
        listeners, self._listeners = self._listeners, []
        for event, callback in listeners:
            try:
                self._room.off(event, callback)
            except Exception as exc:
                self._log.warning(f"could not remove the {event} listener: {describe_exception(exc)}")

    async def _release_owned_room(self) -> None:
        """Release an owned room, once, as *cancel-proof cleanup*. A borrowed room is never touched
        (*borrowed-room rule*)."""
        if not self._owns_room:
            return
        if self._releasing is None:
            self._releasing = asyncio.ensure_future(self._disconnect_and_delete())
        if await await_cancel_proof(self._releasing):
            raise asyncio.CancelledError()

    async def _disconnect_and_delete(self) -> None:
        """One owned-room teardown: each step bounded, each in its own try/except, never raises."""
        log = self._log
        try:
            await asyncio.wait_for(self._room.disconnect(), _types.ROOM_DISCONNECT_SECS)
        except TimeoutError:
            # The timeout cancelled the disconnect and wedged the room, so nothing below touches it again;
            # the delete is a server-side call.
            log.warning(
                f"room disconnect did not finish within {_types.ROOM_DISCONNECT_SECS}s; "
                "leaving the room object alone"
            )
        except Exception as exc:
            log.warning(f"room disconnect failed: {describe_exception(exc)}")
        delete = self.handle.release_room
        if delete is not None:
            try:
                await asyncio.wait_for(delete(), _types.ROOM_DELETE_SECS)
            except Exception as exc:
                log.warning(f"room delete failed: {describe_exception(exc)}")


def _check_connected_as(handle: RoomHandle, *, bot_identity: str, room_name: str) -> None:
    """Fail unless the bot connected as ``bot_identity`` to ``room_name``: what the avatar token names.

    The avatar worker binds to the bot identity in the avatar token, so a bot that connected as anyone else,
    or to another room, would see the session end silently, about 30 s after the avatar started, when the
    worker's agent-join watchdog runs out (*connect overlap*). It cannot happen with Pipecat's transport,
    which takes both from the token, and it costs one comparison.
    """
    if (handle.bot_identity, handle.room_name) != (bot_identity, room_name):
        raise SynthesiaError(
            f"the bot connected as {handle.bot_identity!r} to room {handle.room_name!r}, but its token, "
            f"which the avatar token was minted from, names {bot_identity!r} in room {room_name!r}",
            type=ErrorType.INVALID_ROOM_TOKEN,
        )


def _track_kind_name(kind: Any) -> str | None:
    if kind == lk_rtc.TrackKind.KIND_VIDEO:
        return "video"
    if kind == lk_rtc.TrackKind.KIND_AUDIO:
        return "audio"
    return None
