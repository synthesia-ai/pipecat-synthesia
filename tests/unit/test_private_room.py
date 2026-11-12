"""`PrivateLiveKitRoom` on the `room_factory` and `api_factory` seams.

What these protect: the claims of both tokens a relayed join mints (*agent-kind requirement*); a create
or join failure is CONNECTION and keeps its cause, and a `SynthesiaError` passes through untouched;
nothing this adapter creates outlives a failure, including a cancellation between `create_room` and
`Room.connect`, and a cancel during the release after a failure wins over it; and the *room ownership
handoff*: once `LiveKitAvatarSession.launch` has the handle, the room is deleted by the session and by
nobody else.

Only the seams are patched (`room_factory`, `api_factory` and the `launch` classmethod), so the adapter
runs its real code against real `api.TwirpError`s and real minted tokens. No duration is asserted: the
cancellation tests cancel at a gate; they do not race a sleep.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import Any, cast

import aiohttp
import jwt
import pytest
from livekit import rtc as lk_rtc

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.errors import ErrorType, SynthesiaError, error_from_problem
from pipecat_synthesia.livekit import _types, private_room
from pipecat_synthesia.livekit import token as livekit_token
from pipecat_synthesia.livekit._types import MintedAvatarToken, RoomHandle
from pipecat_synthesia.livekit.private_room import PrivateLiveKitRoom
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import (
    DEFAULT_AVATAR_IDENTITY,
    DEFAULT_AVATAR_NAME,
    DEFAULT_BOT_IDENTITY,
    DEFAULT_BOT_NAME,
    decode_unsigned,
)
from pipecat_synthesia.protocol import (
    TOKEN_ATTR_CLIENT,
    TOKEN_ATTR_LAUNCH,
    TOKEN_ATTR_PROTOCOL,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
)
from pipecat_synthesia.room import JoinMode, LaunchContext
from tests.fakes.fake_lk_api import LiveKitApiFactory, RoomFactory, server_error
from tests.fakes.fake_rtc import FakeRemoteParticipant
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.support.callbacks import make_launch_context
from tests.support.secrets import DEFAULT_SECRETS, LK_API_SECRET, assert_no_secrets
from tests.support.waiting import until

URL = "wss://livekit.example"
API_KEY = "APIlivekit"
ROOM = "private-room"
ENV = {"LIVEKIT_URL": URL, "LIVEKIT_API_KEY": API_KEY, "LIVEKIT_API_SECRET": LK_API_SECRET}


class MediaRecorder:
    """The two relayed-join media callbacks, which `CallbackRecorder` leaves `None`."""

    def __init__(self) -> None:
        self.video: list[tuple[bytes, int, int]] = []
        self.audio: list[tuple[bytes, int, int]] = []

    async def on_video_frame(self, data: bytes, width: int, height: int) -> None:
        self.video.append((data, width, height))

    async def on_audio_frame(self, data: bytes, sample_rate: int, channels: int) -> None:
        self.audio.append((data, sample_rate, channels))


class StubSession:
    """The `AvatarSession` the patched `launch` hands back: close hooks and nothing else."""

    def __init__(self, *, add_hook_error: BaseException | None = None) -> None:
        self.hooks: list[Any] = []
        self.aclose_calls = 0
        self.add_hook_error = add_hook_error

    def add_close_hook(self, hook: Any) -> None:
        if self.add_hook_error is not None:
            raise self.add_hook_error
        self.hooks.append(hook)

    async def aclose(self) -> None:
        self.aclose_calls += 1
        for hook in list(self.hooks):
            await hook()


class SessionSpy:
    """Stands in for `LiveKitAvatarSession.launch` and records what the relayed join hands it.

    `cleanup_on_cancel` mirrors what the real `launch()` does when it is cancelled after taking
    ownership of the handle (`session.py::_release_owned_room`): disconnect the owned room, then call
    `handle.release_room`. The ownership-handoff tests check that `PrivateLiveKitRoom` does not repeat it.
    """

    def __init__(
        self,
        *,
        gate: asyncio.Event | None = None,
        error: BaseException | None = None,
        cleanup_on_cancel: bool = False,
        add_hook_error: BaseException | None = None,
    ) -> None:
        self.gate = gate
        self.error = error
        self.cleanup_on_cancel = cleanup_on_cancel
        self.calls: list[dict[str, Any]] = []
        self.session = StubSession(add_hook_error=add_hook_error)
        self.started = asyncio.Event()

    @property
    def handle(self) -> RoomHandle:
        assert len(self.calls) == 1, f"expected one launch(), got {len(self.calls)}"
        return cast(RoomHandle, self.calls[0]["handle"])

    async def __call__(
        self,
        handle: RoomHandle,
        *,
        token_source: Any,
        avatar_identity: str,
        avatar_name: str,
        ctx: LaunchContext,
    ) -> StubSession:
        self.calls.append(
            {
                "handle": handle,
                "token_source": token_source,
                "avatar_identity": avatar_identity,
                "avatar_name": avatar_name,
                "ctx": ctx,
            }
        )
        self.started.set()
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                if self.cleanup_on_cancel:
                    await handle.room.disconnect()
                    if handle.release_room is not None:
                        await handle.release_room()
                raise
        if self.error is not None:
            raise self.error
        return self.session


class Rig:
    """A `PrivateLiveKitRoom` wired to both seams, plus the context `launch()` takes."""

    def __init__(
        self,
        *,
        room_factory: RoomFactory | None = None,
        api_factory: LiveKitApiFactory | None = None,
        audio_out_sample_rate: int = 24000,
        launch_timeout: float = 5.0,
        url: str = URL,
        **room_kwargs: Any,
    ) -> None:
        self.rooms = room_factory or RoomFactory()
        self.apis = api_factory or LiveKitApiFactory()
        self.room = PrivateLiveKitRoom(
            url=url,
            api_key=API_KEY,
            api_secret=LK_API_SECRET,
            room_name=room_kwargs.pop("room_name", ROOM),
            room_factory=self.rooms,
            api_factory=self.apis,
            **room_kwargs,
        )
        self.api = FakeSessionApiClient()
        self.recorder = MediaRecorder()
        ctx = make_launch_context(
            self.api, launch_timeout=launch_timeout, audio_out_sample_rate=audio_out_sample_rate
        )
        self.ctx = dataclasses.replace(
            ctx,
            callbacks=dataclasses.replace(
                ctx.callbacks,
                on_video_frame=self.recorder.on_video_frame,
                on_audio_frame=self.recorder.on_audio_frame,
            ),
        )

    def launch_task(self) -> asyncio.Task[Any]:
        return asyncio.ensure_future(self.room.launch(self.ctx))


def patch_launch(monkeypatch: pytest.MonkeyPatch, spy: SessionSpy) -> SessionSpy:
    monkeypatch.setattr(LiveKitAvatarSession, "launch", spy)
    return spy


# ------------------------------------------------------------------------------ construction


def test_ctor_env_fallbacks_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every credential falls back to its environment variable; a missing one names both ways to set it."""
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    room = PrivateLiveKitRoom()
    assert room.avatar_identity == DEFAULT_AVATAR_IDENTITY
    assert room.join_mode is JoinMode.RELAYED
    assert re.fullmatch(r"synthesia-avatar-[0-9a-f]{12}", room.room_name), room.room_name

    # An explicit argument wins over the environment, and a given room name is kept verbatim.
    explicit = PrivateLiveKitRoom(url="wss://other.example", room_name=ROOM)
    assert explicit.room_name == ROOM

    for missing, parameter in (
        ("LIVEKIT_URL", "url"),
        ("LIVEKIT_API_KEY", "api_key"),
        ("LIVEKIT_API_SECRET", "api_secret"),
    ):
        monkeypatch.delenv(missing, raising=False)
        with pytest.raises(ValueError) as caught:
            PrivateLiveKitRoom()
        assert parameter in str(caught.value) and missing in str(caught.value)
        monkeypatch.setenv(missing, ENV[missing])


@pytest.mark.parametrize(
    "url", ["ws://livekit.example", "http://livekit.example", "https://", "ftp://livekit.example"]
)
def test_ctor_rejects_a_url_the_launch_would_refuse(url: str) -> None:
    """An insecure or host-less LiveKit URL is a misuse error at construction, not a `SynthesiaError`
    seconds into the call (*misuse vs session errors*); the message names the argument and the scheme
    the Synthesia API accepts."""
    with pytest.raises(ValueError) as caught:
        PrivateLiveKitRoom(url=url, api_key=API_KEY, api_secret=LK_API_SECRET)

    assert not isinstance(caught.value, SynthesiaError)
    assert str(caught.value).startswith("url ")
    assert "the Synthesia API accepts only wss://" in str(caught.value)


def test_ctor_url_error_names_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LIVEKIT_URL", "ws://livekit.example")

    with pytest.raises(ValueError, match="^LIVEKIT_URL "):
        PrivateLiveKitRoom()


async def test_launch_uses_the_url_normalised_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """`https://` becomes `wss://` once, at construction, and every launch step uses that URL; a `ws://`
    URL is accepted only through the test seam the integration suite's local LiveKit server needs."""
    rig = Rig(url="https://livekit.example/")
    spy = patch_launch(monkeypatch, SessionSpy())

    await rig.room.launch(rig.ctx)

    assert rig.apis.api.url == "wss://livekit.example"
    assert rig.rooms.room.connect_calls[0]["url"] == "wss://livekit.example"
    assert spy.handle.livekit_url == "wss://livekit.example"

    monkeypatch.setattr(livekit_token, "_ALLOW_INSECURE_WS", True)
    PrivateLiveKitRoom(url="ws://localhost:7880", api_key=API_KEY, api_secret=LK_API_SECRET)


def test_ctor_error_names_only_the_first_missing_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank environment variable counts as missing (a `.env` with `LIVEKIT_URL=` is common)."""
    monkeypatch.setenv("LIVEKIT_URL", "   ")
    monkeypatch.setenv("LIVEKIT_API_KEY", API_KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", LK_API_SECRET)
    with pytest.raises(ValueError, match="LIVEKIT_URL"):
        PrivateLiveKitRoom()


# ------------------------------------------------------------------------------ the happy path


async def test_launch_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launch order: create_room -> agent-kind bot token -> auto-subscribing connect -> owned handle
    -> capture."""
    rig = Rig()
    spy = patch_launch(monkeypatch, SessionSpy())

    session = await rig.room.launch(rig.ctx)

    # The LiveKit API client takes the wss:// URL exactly as it was normalised; the SDK rewrites the scheme.
    api = rig.apis.api
    assert api.url == URL
    assert api.api_key == API_KEY
    assert api.timeout is not None and api.timeout.total == _types.ROOM_CREATE_SECS

    # The room is created explicitly, with the adapter's default empty and departure timeouts.
    assert api.room.created_names == [ROOM]
    request = api.room.create_calls[0]
    assert (request.empty_timeout, request.departure_timeout) == (120, 20)

    # The bot joins as a standard participant, so that its private room stays open while it is there
    # (*agent-kind requirement*): neither the agent kind nor the `agent` grant.
    connect_call = rig.rooms.room.connect_calls[0]
    assert connect_call["url"] == URL
    options = connect_call["options"]
    assert options.auto_subscribe is True
    # The SDK gets a deadline of its own, strictly inside what is left of `ctx.launch_timeout`: a join
    # our own `wait_for` kills leaks the FFI queue subscription `Room.connect` takes out before its
    # only await, while one the SDK kills raises `ConnectError` and unsubscribes.
    assert options.connect_timeout is not None
    assert 0 < options.connect_timeout < rig.ctx.launch_timeout
    claims = decode_unsigned(connect_call["token"])
    assert claims.kind is None
    assert "agent" not in claims.video
    assert claims.identity == DEFAULT_BOT_IDENTITY
    assert claims.room == ROOM

    # The handle says "mine": the session is what disconnects and deletes it.
    handle = spy.handle
    assert handle.owns_room is True
    assert handle.release_room is not None
    assert handle.connected_at is not None
    assert (handle.bot_identity, handle.livekit_url, handle.room_name) == (
        DEFAULT_BOT_IDENTITY,
        URL,
        ROOM,
    )
    assert handle.room is rig.rooms.room
    call = spy.calls[0]
    assert isinstance(call["token_source"], MintedAvatarToken)
    assert (call["avatar_identity"], call["avatar_name"]) == (DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME)
    assert call["ctx"] is rig.ctx

    # Capture is running and closes with the session; nothing has been torn down yet.
    assert "track_subscribed" in rig.rooms.room.listeners
    assert len(spy.session.hooks) == 1
    assert rig.rooms.room.listeners["track_subscribed"]
    assert api.room.delete_calls == []
    assert api.close_count == 0
    assert session is spy.session


async def test_the_bot_and_the_avatar_worker_tokens_are_minted_for_the_private_room() -> None:
    """Token claims are a locked contract. The real session runs, so neither token is rebuilt by the test."""
    rig = Rig()

    async def avatar_joins(*_args: Any) -> None:
        room = rig.rooms.room
        room.emit(
            "participant_connected", room.add_participant(FakeRemoteParticipant(DEFAULT_AVATAR_IDENTITY))
        )

    rig.api.on_create = avatar_joins
    session = await rig.room.launch(rig.ctx)
    try:
        bot = jwt.decode(rig.rooms.room.connect_calls[0]["token"], options={"verify_signature": False})
        assert len(rig.api.calls) == 1
        avatar = jwt.decode(rig.api.calls[0]["livekit_token"], options={"verify_signature": False})
    finally:
        await session.aclose()

    # The grant sets themselves are pinned in test_token.py; this is what the relayed join feeds the mints.
    assert bot["video"]["room"] == ROOM
    assert (bot.get("kind"), bot["sub"], bot["name"], bot["iss"]) == (
        None,
        DEFAULT_BOT_IDENTITY,
        DEFAULT_BOT_NAME,
        API_KEY,
    )
    assert not bot.get("attributes")

    assert avatar["video"]["room"] == ROOM
    assert (avatar["kind"], avatar["sub"], avatar["name"], avatar["iss"]) == (
        "agent",
        DEFAULT_AVATAR_IDENTITY,
        DEFAULT_AVATAR_NAME,
        API_KEY,
    )
    launch_id = avatar["attributes"].pop(TOKEN_ATTR_LAUNCH)  # random per launch (*early start*)
    assert re.fullmatch(r"[0-9a-f]{16}", launch_id)
    assert avatar["attributes"] == {
        TOKEN_ATTR_PUBLISH_ON_BEHALF: DEFAULT_BOT_IDENTITY,
        TOKEN_ATTR_PROTOCOL: "1",
        TOKEN_ATTR_CLIENT: build_client_info().token_client_attr,
    }


# ------------------------------------------------------------------------------ create_room failures


@pytest.mark.parametrize("status", [401, 403])
async def test_create_room_401_maps_to_credentials_rejected(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, status: int
) -> None:
    """A rejected key/secret is its own error type, and the LiveKit API client is still closed."""
    rig = Rig(api_factory=LiveKitApiFactory(create_error=server_error("unauthenticated", status=status)))
    patch_launch(monkeypatch, SessionSpy())

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value.type is ErrorType.LIVEKIT_CREDENTIALS_REJECTED
    assert rig.apis.api.close_count == 1
    assert rig.rooms.rooms == [], "no room should be joined after create_room failed"
    assert_no_secrets(str(caught.value), repr(caught.value), caplog_loguru.text)


@pytest.mark.parametrize(
    ("make_root", "fragment"),
    [
        pytest.param(lambda: server_error("internal", status=500, message="boom"), "boom", id="twirp-500"),
        pytest.param(
            lambda: aiohttp.ClientConnectionError("connection reset by peer"),
            "connection reset by peer",
            id="transport",
        ),
    ],
)
async def test_create_room_other_error_maps_to_connection(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, make_root: Any, fragment: str
) -> None:
    """Any other room-service failure is CONNECTION, caused by the original; the LiveKit API client is
    closed on that path too."""
    root = make_root()
    rig = Rig(api_factory=LiveKitApiFactory(create_error=root))
    patch_launch(monkeypatch, SessionSpy())

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value.type is ErrorType.CONNECTION
    assert caught.value.__cause__ is root
    assert fragment in str(caught.value), "the error names the original"
    assert rig.apis.api.close_count == 1
    assert rig.rooms.rooms == [], "no room is joined after the create failed"
    assert_no_secrets(str(caught.value), repr(caught.value), caplog_loguru.text)


# ------------------------------------------------------------------------------ join failures


@pytest.mark.parametrize(
    "root",
    [
        pytest.param(lk_rtc.ConnectError("no route"), id="connect-error"),
        pytest.param(OSError("network is unreachable"), id="os-error"),
    ],
)
async def test_connect_failure_deletes_room(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, root: BaseException
) -> None:
    """A room we created but could not join is deleted before the error reaches the caller, which is
    CONNECTION, caused by and naming the original, and carries neither the secret nor the bot token."""
    rig = Rig(room_factory=RoomFactory(connect_error=root))
    patch_launch(monkeypatch, SessionSpy())

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value.type is ErrorType.CONNECTION
    assert caught.value.__cause__ is root
    assert str(root) in str(caught.value)
    api = rig.apis.api
    assert api.room.deleted_names == [ROOM]
    assert api.close_count == 1
    bot_token = rig.rooms.room.connect_calls[0]["token"]
    secrets = (*DEFAULT_SECRETS, bot_token)
    assert_no_secrets(str(caught.value), repr(caught.value), caplog_loguru.text, secrets=secrets)


async def test_connect_timeout_maps_to_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A join that never completes is bounded by `connect_timeout` and deletes the room.

    The timeout is an input (0.01 s through the constructor), never an assertion.
    """
    gate = asyncio.Event()
    rig = Rig(room_factory=RoomFactory(connect_gate=gate), connect_timeout=0.01)
    patch_launch(monkeypatch, SessionSpy())

    task = rig.launch_task()
    await until(lambda: bool(rig.apis.instances) and bool(rig.apis.api.room.delete_calls), what="the delete")
    gate.set()  # the launch waits for the connect it never cancels
    with pytest.raises(SynthesiaError) as caught:
        await task

    assert caught.value.type is ErrorType.CONNECTION
    assert isinstance(caught.value.__cause__, TimeoutError), "the error says it was our budget"
    assert rig.apis.api.room.deleted_names == [ROOM]
    assert rig.apis.api.close_count == 1


async def test_cancel_between_create_and_connect_deletes_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancel while the join is in flight still deletes the room and closes the LiveKit API client."""
    gate = asyncio.Event()
    rooms = RoomFactory(connect_gate=gate)
    rig = Rig(room_factory=rooms)
    patch_launch(monkeypatch, SessionSpy())

    task = rig.launch_task()
    await until(lambda: bool(rooms.rooms) and rooms.room.connect_started.is_set(), what="the join")
    task.cancel()
    gate.set()  # the launch waits for the connect it never cancels
    with pytest.raises(asyncio.CancelledError):
        await task

    api = rig.apis.api
    assert api.room.deleted_names == [ROOM], "the room we created outlived a cancelled launch()"
    assert api.close_count == 1


@pytest.mark.parametrize("step", ["create", "join"])
async def test_a_synthesia_error_from_either_step_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """The same object, with its own cause untouched."""
    root = ValueError("the underlying failure")
    original = SynthesiaError("already a session error", type=ErrorType.TIMEOUT)
    original.__cause__ = root
    if step == "create":
        rig = Rig(api_factory=LiveKitApiFactory(create_error=original))
    else:
        rig = Rig(room_factory=RoomFactory(connect_error=original))
    patch_launch(monkeypatch, SessionSpy())

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value is original
    assert caught.value.type is ErrorType.TIMEOUT
    assert caught.value.__cause__ is root
    assert rig.apis.api.close_count == 1
    if step == "join":
        assert rig.apis.api.room.deleted_names == [ROOM]


async def test_a_cancel_during_the_create_reaches_the_caller_and_closes_the_livekit_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apis = LiveKitApiFactory(create_gate=asyncio.Event())  # never set: the create never answers
    rig = Rig(api_factory=apis)
    patch_launch(monkeypatch, SessionSpy())

    task = rig.launch_task()
    await until(lambda: bool(apis.instances) and bool(apis.api.room.create_calls), what="the create")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert apis.api.close_count == 1, "the LiveKit API client outlived a cancelled launch"
    assert rig.rooms.rooms == []


@pytest.mark.parametrize(
    ("step", "failure"),
    [
        ("create", "mapped"),
        ("join", "mapped"),
        ("create", "pass-through"),
        ("join", "pass-through"),
        ("join", "timeout"),
    ],
)
async def test_a_cancel_during_the_release_after_a_failure_wins(
    monkeypatch: pytest.MonkeyPatch, step: str, failure: str
) -> None:
    """The caller's cancel is what propagates, whatever the failure it interrupted, and that failure
    stays on the chain: one this adapter maps to CONNECTION, a `SynthesiaError` that passes through
    unchanged, and the bot-connect timeout. The release closes the client last, so holding the close
    holds the release open."""
    gate = asyncio.Event()  # never set in the timeout case: the connect never answers
    passed_through = SynthesiaError("already a session error", type=ErrorType.TIMEOUT)
    error: BaseException | None
    if failure == "pass-through":
        error = passed_through
    elif failure == "mapped":
        error = (
            server_error("internal", status=500, message="boom")
            if step == "create"
            else lk_rtc.ConnectError("no route")
        )
    else:
        error = None
    if step == "create":
        rig = Rig(api_factory=LiveKitApiFactory(create_gate=gate, create_error=error))
    elif failure == "timeout":
        rig = Rig(room_factory=RoomFactory(connect_gate=gate), connect_timeout=0.05)
    else:
        rig = Rig(room_factory=RoomFactory(connect_gate=gate, connect_error=error))
    patch_launch(monkeypatch, SessionSpy())

    task = rig.launch_task()
    if step == "create":
        await until(
            lambda: bool(rig.apis.instances) and bool(rig.apis.api.room.create_calls), what="the create"
        )
    else:
        await until(
            lambda: bool(rig.rooms.rooms) and rig.rooms.room.connect_started.is_set(), what="the join"
        )
    api = rig.apis.api
    api.close_gate = asyncio.Event()
    if failure != "timeout":
        gate.set()  # the step now fails
    await until(api.close_started.is_set, what="the release after the failure")
    task.cancel()
    api.close_gate.set()
    gate.set()  # in the timeout case, the launch then waits for the connect it never cancels

    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    cause = caught.value.__cause__
    if failure == "pass-through":
        assert cause is passed_through, repr(cause)
    else:
        assert isinstance(cause, SynthesiaError), repr(cause)
        assert cause.type is ErrorType.CONNECTION
    if failure == "timeout":
        assert isinstance(cause.__cause__, TimeoutError), repr(cause.__cause__)
    assert api.close_count == 1
    if step == "join":
        assert api.room.deleted_names == [ROOM]


# ------------------------------------------------------------------------------ the delete hook


async def test_release_room_hook_swallows_not_found(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The handle's `release_room` is idempotent against a room LiveKit has already reaped."""
    rig = Rig(
        api_factory=LiveKitApiFactory(
            delete_error=server_error("not_found", status=404, message="room not found")
        )
    )
    spy = patch_launch(monkeypatch, SessionSpy())
    await rig.room.launch(rig.ctx)

    delete = spy.handle.release_room
    assert delete is not None
    await delete()  # must not raise: the room being gone is the end state we wanted

    api = rig.apis.api
    assert api.room.deleted_names == [ROOM]
    assert api.close_count == 1, "the LiveKit API client is released even when the delete failed"
    assert not [
        record
        for record in caplog_loguru.records
        if record.levelname == "WARNING" and "delete" in record.getMessage()
    ], "an already-reaped room is the end state we wanted, not something to warn about"


async def test_delete_room_on_close_false_keeps_the_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """`delete_room_on_close=False` leaves the room for the developer, but still closes the LiveKit API
    client."""
    rig = Rig(delete_room_on_close=False)
    spy = patch_launch(monkeypatch, SessionSpy())
    await rig.room.launch(rig.ctx)

    delete = spy.handle.release_room
    assert delete is not None
    await delete()

    api = rig.apis.api
    assert api.room.delete_calls == []
    assert api.close_count == 1


async def test_delete_room_on_close_false_still_releases_after_a_failed_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same choice holds on the failure path: the room survives, the aiohttp session does not."""
    rig = Rig(
        room_factory=RoomFactory(connect_error=lk_rtc.ConnectError("no route")),
        delete_room_on_close=False,
    )
    patch_launch(monkeypatch, SessionSpy())

    with pytest.raises(SynthesiaError):
        await rig.room.launch(rig.ctx)

    assert rig.apis.api.room.delete_calls == []
    assert rig.apis.api.close_count == 1


# ------------------------------------------------------------------------------- ownership handoff


async def test_launch_cancelled_after_session_launch_is_cleaned_by_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the session has the handle, the room is the session's (*room ownership handoff*): a cancel
    deletes it once, and this adapter does not delete it again."""
    rig = Rig()
    spy = patch_launch(monkeypatch, SessionSpy(gate=asyncio.Event(), cleanup_on_cancel=True))

    task = rig.launch_task()
    await until(spy.started.is_set, what="LiveKitAvatarSession.launch")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    api = rig.apis.api
    assert api.room.deleted_names == [ROOM], "the session's cleanup path should have deleted the room"
    assert len(api.room.delete_calls) == 1, "PrivateLiveKitRoom deleted a room it no longer owns"
    assert api.close_count == 1
    assert rig.rooms.room.disconnect_calls == 1


async def test_launch_failure_after_session_launch_is_not_cleaned_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same handoff for an ordinary failure: `LiveKitAvatarSession.launch()` cleans up, and
    `PrivateLiveKitRoom.launch()` re-raises untouched."""
    rig = Rig()
    failure = SynthesiaError("worker never joined", type=ErrorType.TIMEOUT)
    spy = patch_launch(monkeypatch, SessionSpy(error=failure))

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value is failure
    # The spy is not the real `LiveKitAvatarSession.launch()`, so nothing was deleted, which is the
    # point: this adapter added no cleanup of its own past the handoff.
    assert rig.apis.api.room.delete_calls == []
    assert spy.calls


async def test_capture_setup_failure_closes_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """`launch()` leaves nothing running when it raises, including past the handoff, where closing the
    session is the only way to release the room."""
    rig = Rig()
    boom = RuntimeError("close hooks are full")
    spy = patch_launch(monkeypatch, SessionSpy(add_hook_error=boom))

    with pytest.raises(RuntimeError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value is boom
    assert spy.session.aclose_calls == 1, "a session launch() built and then abandoned"


# ------------------------------------------------------------------------------ capture wiring


class CaptureRecorder:
    """Stands in for `AvatarTrackCapture`, recording the arguments `launch()` builds it with.

    Patched onto the `private_room` module rather than injected, because the adapter has no
    capture-factory seam. It is the only way to see the wiring between `LaunchContext` and the capture,
    which nothing else asserts.
    """

    instances: list[CaptureRecorder] = []

    def __init__(self, room: Any, **kwargs: Any) -> None:
        self.room = room
        self.kwargs = kwargs
        self.start_calls = 0
        self.aclose_calls = 0
        self.start_error: BaseException | None = None
        CaptureRecorder.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    async def aclose(self) -> None:
        self.aclose_calls += 1


@pytest.fixture
def capture_recorder(monkeypatch: pytest.MonkeyPatch) -> type[CaptureRecorder]:
    CaptureRecorder.instances = []
    monkeypatch.setattr(private_room, "AvatarTrackCapture", CaptureRecorder)
    return CaptureRecorder


async def test_capture_is_wired_to_the_contexts_audio_format(
    monkeypatch: pytest.MonkeyPatch, capture_recorder: type[CaptureRecorder]
) -> None:
    """The capture takes the pipeline's rate and channel count from the context.

    The rate here is deliberately not the 24 kHz default, so a hard-coded rate fails as loudly as a
    missing one.
    """
    rig = Rig(audio_out_sample_rate=16000)
    patch_launch(monkeypatch, SessionSpy())

    await rig.room.launch(rig.ctx)

    assert len(capture_recorder.instances) == 1
    capture = capture_recorder.instances[0]
    assert capture.room is rig.rooms.room
    assert capture.kwargs["audio_sample_rate"] == rig.ctx.audio_out_sample_rate == 16000
    assert capture.kwargs["audio_channels"] == rig.ctx.audio_out_channels
    assert capture.kwargs["avatar_identity"] == DEFAULT_AVATAR_IDENTITY
    assert capture.kwargs["on_video_frame"] is rig.ctx.callbacks.on_video_frame
    assert capture.kwargs["on_audio_frame"] is rig.ctx.callbacks.on_audio_frame
    assert capture.start_calls == 1


async def test_capture_start_failure_closes_the_capture_and_the_session(
    monkeypatch: pytest.MonkeyPatch, capture_recorder: type[CaptureRecorder]
) -> None:
    """A capture whose `start()` fails is closed, and so is the session: `launch()` leaves nothing
    running when it raises.

    `start()` can install its room listener and create a consumer task before it fails, and only
    `capture.aclose()` stops either, so the close hook has to be registered first.
    """
    rig = Rig()
    spy = patch_launch(monkeypatch, SessionSpy())
    boom = RuntimeError("the room rejected the listener")

    def failing_start(self: CaptureRecorder) -> None:
        self.start_calls += 1
        raise boom

    monkeypatch.setattr(CaptureRecorder, "start", failing_start)

    with pytest.raises(RuntimeError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value is boom
    capture = capture_recorder.instances[0]
    assert spy.session.aclose_calls == 1
    assert capture.aclose_calls == 1, "the capture outlived the launch() that built it"


# ------------------------------------------------------------------------------ the second cancel


async def test_second_cancel_during_create_room_still_closes_the_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cleanup survives a second cancel (*cancel-proof cleanup*): the service cancels `launch()` at
    its launch timeout and then bounds that cancellation with a timeout of its own.

    `LiveKitAPI` builds its `aiohttp.ClientSession` eagerly, so a cleanup aborted here leaks a
    connector and an "Unclosed client session" at GC.
    """
    apis = LiveKitApiFactory(create_gate=asyncio.Event())  # never set: the call never answers
    rig = Rig(api_factory=apis)
    patch_launch(monkeypatch, SessionSpy())

    task = rig.launch_task()
    await until(lambda: bool(apis.instances) and bool(apis.api.room.create_calls), what="create_room")
    api = apis.api
    api.close_gate = asyncio.Event()  # hold the close open so the second cancel lands inside it

    task.cancel()
    await until(api.close_started.is_set, what="the cleanup to reach the close")
    task.cancel()  # the second cancel, with the cleanup half way through
    api.close_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert api.close_count == 1, "the aiohttp session outlived a twice-cancelled launch()"


async def test_second_cancel_during_connect_still_deletes_the_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same guarantee on the path that owns a room as well as a LiveKit API client."""
    gate = asyncio.Event()
    rooms = RoomFactory(connect_gate=gate)
    rig = Rig(room_factory=rooms)
    patch_launch(monkeypatch, SessionSpy())

    task = rig.launch_task()
    await until(lambda: bool(rooms.rooms) and rooms.room.connect_started.is_set(), what="the join")
    api = rig.apis.api
    # The delete releases the LiveKit API client in its `finally`, so holding the close holds the tail of the
    # teardown open, which is exactly where a second cancel would do the damage.
    api.close_gate = asyncio.Event()

    task.cancel()
    await until(api.close_started.is_set, what="the teardown to reach the close")
    task.cancel()  # the second cancel, with the teardown half way through
    api.close_gate.set()
    gate.set()  # the launch waits for the connect it never cancels

    with pytest.raises(asyncio.CancelledError):
        await task

    assert api.room.deleted_names == [ROOM], "the room outlived a twice-cancelled launch()"
    assert api.close_count == 1, "the aiohttp session outlived a twice-cancelled launch()"


async def test_create_room_timeout_is_bounded_and_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """`create_room` is bounded by what is left of `ctx.launch_timeout`, not left to the SDK.

    The LiveKit API client's `ClientTimeout` bounds one HTTP attempt, and the Twirp client retries a LiveKit
    Cloud host up to three times whenever that timeout is 5 s or more (livekit-api 1.2.1), so without a wait
    of our own a dead region would sit here for three times the nominal budget. `launch_timeout` is the
    input.
    """
    apis = LiveKitApiFactory(create_gate=asyncio.Event())  # never set: the call never answers
    rig = Rig(api_factory=apis, launch_timeout=0.6)
    patch_launch(monkeypatch, SessionSpy())

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value.type is ErrorType.CONNECTION
    assert "private room" in str(caught.value)
    assert apis.api.close_count == 1
    assert rig.rooms.rooms == [], "no room should be joined after create_room timed out"


# ------------------------------------------------------------------------------ the Synthesia API's errors


async def test_service_unavailable_carries_no_room_hint() -> None:
    """A relayed join's 503 `service_unavailable` is passed through as the Synthesia API client raised it:
    this adapter created the room with these credentials, so a credentials problem would already have failed
    at `create_room`, and a hint blaming them would mislead. The real session runs here, not the spy."""
    rig = Rig()
    detail = "The session could not be started. Retry shortly."
    original = error_from_problem(503, {"code": "service_unavailable", "detail": detail}, {})
    rig.api.error = original

    with pytest.raises(SynthesiaError) as caught:
        await rig.room.launch(rig.ctx)

    assert caught.value is original
    assert detail in str(caught.value)
    assert "could not join your LiveKit room" not in str(caught.value)
    assert caught.value.type is ErrorType.CONNECTION
    assert rig.apis.api.room.deleted_names == [ROOM]
