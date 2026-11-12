"""*Connect overlap*: in a direct join, the session request goes out while Pipecat's transport is still
connecting.

These drive the real `LiveKitTransportRoom` and `LiveKitAvatarSession` over a `FakeLiveKitTransport` that
has not connected yet, which is the normal case: the service starts its launch in `setup()`, concurrently
with Pipecat's own connect. The test connects it (`Rig.connect`), so every ordering here is decided by the
test, not by the event loop. `FakeSessionApiClient(gate=...)` holds the 201 open.

What they protect: the request does not wait for the connect, and is made for the bot identity and room the
bot token names; the playback handlers still answer before the session is active; an avatar that joined
while the bot was connecting is seen at once, video included; a bot that connected elsewhere fails the
launch; a failed connect or a launch timeout leaves no request running, and a failed connect says that a
session may exist; an expired bot token never gets as far as a request; a bot token without a room claim
falls back to requesting after the connect.

"Nothing outlives the launch" is asserted as the tasks still running in the loop, minus those running before
the test started it: no name of ours is involved, so a refactor of the session's internals cannot make the
check pass vacuously.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any, cast

import pytest
from livekit import api as lk_api

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._types import LiveKitCredentials
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, decode_unsigned
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from pipecat_synthesia.protocol import (
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
)
from tests.fakes.fake_rtc import FakeLocalParticipant, FakeRemoteParticipant, FakeRoom
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.fakes.fake_transport import FakeLiveKitTransport
from tests.support.callbacks import CallbackRecorder, make_launch_context
from tests.support.secrets import LK_API_SECRET, LK_TOKEN_MARKER
from tests.support.waiting import until
from tests.unit._session_rig import make_avatar, settle

AVATAR = DEFAULT_AVATAR_IDENTITY
BOT = "pipecat-bot"
ROOM = "fake-room"
SESSION_ID = "ses_overlap"
CREDENTIALS = LiveKitCredentials(api_key="fake-lk-key", api_secret=LK_API_SECRET)
#: A join wait far longer than `until()`'s backstop, so "launch() returned" can only mean it did not wait it
#: out. Durations are inputs: the only upper bound asserted is "much less than the injected duration".
LONG_JOIN_WAIT = _types.POST_CREATE_JOIN_WAIT_SECS
#: The stable keyword of the one warning a failed connect logs; the rest of its wording is free.
ABANDONED = "the avatar worker will end it"


def bot_token(*, room: str | None = ROOM, ttl: timedelta = timedelta(hours=1)) -> str:
    """An agent-kind bot token for `BOT`, naming `room`, or no room at all when `room` is None.

    livekit-api refuses a room-join grant without a room, so a room-less token carries no room-join grant
    either: the shape a token minted elsewhere could have, which the constructor accepts.
    """
    grants = (
        lk_api.VideoGrants(room_join=True, room=room, agent=True)
        if room is not None
        else lk_api.VideoGrants(agent=True)
    )
    return (
        lk_api.AccessToken(CREDENTIALS.api_key, LK_API_SECRET)
        .with_identity(BOT)
        .with_name(f"{BOT} {LK_TOKEN_MARKER}")
        .with_kind("agent")
        .with_grants(grants)
        .with_ttl(ttl)
        .to_jwt()
    )


def running_tasks() -> set[asyncio.Task[Any]]:
    """Every task still running in the loop, whoever started it."""
    return {task for task in asyncio.all_tasks() if not task.done()}


def warnings_with(caplog: pytest.LogCaptureFixture, keyword: str) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and keyword in record.getMessage()
    ]


class Rig:
    """A bot transport that has not connected yet, its room, a fake API, and the direct join under test."""

    def __init__(
        self,
        *,
        token: str | None = None,
        room_name: str = ROOM,
        connected: bool = False,
        avatar_present: bool = False,
        api: FakeSessionApiClient | None = None,
        launch_timeout: float = 5.0,
        **adapter_kwargs: Any,
    ) -> None:
        self.participant = FakeLocalParticipant(BOT)
        self.room = FakeRoom(room_name, local_participant=self.participant)
        self.transport = FakeLiveKitTransport(
            room=self.room, token=token or bot_token(), participant_id=BOT if connected else ""
        )
        self.api = api or FakeSessionApiClient(session_id=SESSION_ID)
        self.rec = CallbackRecorder()
        self.ctx = make_launch_context(
            self.api, avatar_ids=("av_a",), launch_timeout=launch_timeout, recorder=self.rec
        )
        self.avatar: FakeRemoteParticipant | None = None
        if avatar_present:
            self.avatar = self.room.add_participant(make_avatar())
        self.adapter = LiveKitTransportRoom(
            cast(Any, self.transport), credentials=CREDENTIALS, **adapter_kwargs
        )

    def launch_task(self) -> asyncio.Task[LiveKitAvatarSession]:
        """A direct join always yields the LiveKit session; the cast lets a test read `.timings`."""
        return cast(
            "asyncio.Task[LiveKitAvatarSession]", asyncio.ensure_future(self.adapter.launch(self.ctx))
        )

    async def connect(self, identity: str = BOT) -> None:
        """What Pipecat's own connect does: set the bot identity, then fire `on_connected`."""
        self.transport.set_connected(identity)
        await self.transport.fire("on_connected")

    def join(self) -> FakeRemoteParticipant:
        """The avatar joins after the bot: into the room, then the event, in rtc's order."""
        self.avatar = self.room.add_participant(make_avatar())
        self.room.emit("participant_connected", self.avatar)
        return self.avatar

    def assert_torn_down(self) -> None:
        """Nothing of ours is left on the borrowed room, its participant, or the transport."""
        assert self.room.disconnect_calls == 0, "a borrowed room was disconnected (borrowed-room rule)"
        assert self.room.listeners == {}, f"room listeners left behind: {self.room.listeners}"
        assert self.participant.rpc_methods == {}, "playback RPC handlers left registered"
        assert self.transport.handlers_for("on_connected") == [], "the connect wait left its handler"


async def launch_through_the_join(rig: Rig) -> LiveKitAvatarSession:
    """The ordinary overlap, start to finish: the request, then the connect, then the 201 and the join."""
    task = rig.launch_task()
    try:
        await until(lambda: len(rig.api.calls) == 1, what="the session request, before the connect")
        await rig.connect()
        await until(lambda: rig.api.returned == 1, what="the 201")
        rig.join()
        await until(task.done, what="launch() to return once the avatar joined")
    finally:
        await settle(task)
    return task.result()


# ---------------------------------------------------------------------------------------- the overlap


async def test_the_session_request_goes_out_while_the_bot_connects() -> None:
    """The session request does not wait for Pipecat's own connect: it goes out before the transport has
    connected, and before anything has read its room, for the bot identity and room its token names. The
    launch completes once the bot connects and the 201 arrives, and the metrics show the overlap.

    It is also the unexpired-token half of the expired-bot-token test below: this token launches."""
    gate = asyncio.Event()
    rig = Rig(api=FakeSessionApiClient(session_id=SESSION_ID, gate=gate))
    task = rig.launch_task()
    try:
        await until(lambda: len(rig.api.calls) == 1, what="the session request, before the connect")
        assert rig.transport.room_reads == 0, "the transport's room was read before the connect"
        claims = decode_unsigned(rig.api.calls[0]["livekit_token"])
        assert claims.room == ROOM, "the avatar token's room must come from the bot token"
        assert claims.attributes[TOKEN_ATTR_PUBLISH_ON_BEHALF] == BOT, "and its publish-on-behalf too"
        assert not task.done()

        await rig.connect()
        gate.set()
        await until(lambda: rig.api.returned == 1, what="the 201")
        rig.join()
        await until(task.done, what="launch() to return once the avatar joined")
    finally:
        await settle(task)
    session = task.result()
    try:
        timings = session.timings
        assert timings.room_connected is not None
        assert timings.session_requested <= timings.room_connected, "the request waited for the connect"
        assert timings.session_requested <= timings.session_created <= timings.avatar_joined
    finally:
        await session.aclose()
    rig.assert_torn_down()


async def test_the_playback_handlers_answer_before_the_launch_returns() -> None:
    """In a *connect overlap* the playback handlers are registered once the bot has connected, not before the
    request, and that is early enough: by the time `launch()` returns, and so before the session is active,
    both answer the avatar worker. Without `lk.playback_finished`, the worker ends its session after the
    first utterance."""
    rig = Rig()
    session = await launch_through_the_join(rig)
    try:
        assert rig.api.calls[0]["livekit_url"] == "wss://fake.livekit"
        assert rig.api.calls[0]["avatar_ids"] == ("av_a",)
        participant = rig.participant
        assert await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR) == RPC_HANDLER_RESPONSE
        payload = json.dumps({"playback_position": 1.5, "interrupted": False})
        assert (
            await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=payload)
            == RPC_HANDLER_RESPONSE
        )
        await until(
            lambda: rig.rec.started == 1 and rig.rec.finished == [(1.5, False)], what="both callbacks"
        )
    finally:
        await session.aclose()
    rig.assert_torn_down()


async def test_an_avatar_already_publishing_at_connect_is_seen_at_once() -> None:
    """The avatar can join, and publish, while the bot is still connecting. By the time the session can
    listen, it is in the room with its video, and neither `participant_connected` nor `track_published`
    will come for it: rtc emits none for what it lists at connect, and anything that arrived while Pipecat
    finished its own connect reached no listener. The join wait ends at once, and the video still reaches
    `on_track_published`, so `avatar_video_published` is stamped late rather than never. `session_created`
    is when the 201 arrived, before the connect, not when the launch read it.

    The join wait is long (an input), so "launch() returned" can only mean it did not wait it out."""
    rig = Rig(avatar_present=True, launch_timeout=LONG_JOIN_WAIT)
    task = rig.launch_task()
    try:
        await until(lambda: rig.api.returned == 1, what="the 201, before the bot connects")
        await rig.connect()
        await until(task.done, what="launch() to return without waiting out the join wait")
    finally:
        await settle(task)
    session = task.result()
    try:
        assert session.avatar_participant is rig.avatar
        timings = session.timings
        assert timings.room_connected is not None
        assert timings.session_created <= timings.room_connected, "the 201 was timed when read, not received"
        await until(lambda: "video" in rig.rec.tracks, what="the video published before the connect")
    finally:
        await session.aclose()
    rig.assert_torn_down()


async def test_a_launch_that_fails_reports_no_video_from_an_avatar_already_there() -> None:
    """An avatar already in the room when the bot connects may be a stale one, left by an earlier launch
    with the same identity. Its video is reported only once the 201 shows this launch has a session: a
    launch that fails reports none, so its metrics carry no `avatar_video_published`."""
    failure = SynthesiaError("unauthenticated", type=ErrorType.AUTH, status=401)
    rig = Rig(avatar_present=True, api=FakeSessionApiClient(error=failure))
    task = rig.launch_task()
    try:
        await until(lambda: len(rig.api.calls) == 1, what="the session request")
        await rig.connect()
        await until(task.done, what="the launch to fail")
    finally:
        await settle(task)

    assert task.exception() is failure
    await asyncio.sleep(0)  # a delivery scheduled before the failure would have run by now
    assert rig.rec.tracks == [], "a failed launch reported the video of an avatar that was already there"
    rig.assert_torn_down()


# -------------------------------------------------------------------------------- when the bot fails


@pytest.mark.parametrize(
    "identity, room_name", [("another-bot", ROOM), (BOT, "another-room")], ids=["identity", "room"]
)
async def test_a_bot_connected_elsewhere_fails_the_launch(
    identity: str, room_name: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The avatar worker binds to the bot identity in the avatar token, in the room it names. A bot that
    connected as anyone else, or to another room, fails the launch at once, naming both sides, rather than
    losing the session silently about 30 s later, when the worker's agent-join watchdog runs out. For the
    worker it is the same as a failed connect, so it is handled as one: the request still in flight is
    cancelled and waited for, and one warning says the avatar worker will end any session it made."""
    before = running_tasks()
    gate = asyncio.Event()
    rig = Rig(room_name=room_name, api=FakeSessionApiClient(gate=gate))
    task = rig.launch_task()
    try:
        await until(lambda: len(rig.api.calls) == 1, what="the session request")
        await rig.connect(identity)
        await until(task.done, what="the launch to fail without waiting for the 201")
    finally:
        await settle(task)

    err = task.exception()
    assert isinstance(err, SynthesiaError), repr(err)
    assert err.type is ErrorType.INVALID_ROOM_TOKEN
    message = str(err)
    for value in (identity, room_name, BOT, ROOM):
        assert repr(value) in message, message
    assert running_tasks() - before == set(), "a task outlived the failed launch"
    gate.set()
    await asyncio.sleep(0)
    assert rig.api.returned == 0, "the session request was left to finish"
    assert len(warnings_with(caplog_loguru, ABANDONED)) == 1, caplog_loguru.text
    rig.assert_torn_down()


@pytest.mark.parametrize("request_state", ["in-flight", "created", "failed"])
async def test_a_failed_connect_abandons_the_session_request(
    request_state: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The bot's connect times out after the session request went out. The connect error is raised, as it
    was before the overlap, and nothing is left running. The request carried the avatar's own token, so the
    avatar worker may already be in the room for a bot that will never arrive, and it ends that session
    itself: one warning says so, naming the session once the 201 has arrived, and the error then carries its
    id too. A request that failed made no session, so there is nothing to warn about.

    `connect_timeout` is shrunk because the test is about it running out. An unheld fake API answers in
    the same loop iteration, far inside it."""
    gate = asyncio.Event()
    api = {
        "in-flight": FakeSessionApiClient(session_id=SESSION_ID, gate=gate),
        "created": FakeSessionApiClient(session_id=SESSION_ID),
        "failed": FakeSessionApiClient(
            error=SynthesiaError("unauthenticated", type=ErrorType.AUTH, status=401)
        ),
    }[request_state]
    before = running_tasks()
    rig = Rig(api=api, connect_timeout=0.05)

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.adapter.launch(rig.ctx)

    err = excinfo.value
    assert err.type is ErrorType.CONNECTION
    assert "did not connect within" in str(err)
    assert running_tasks() - before == set(), "a task outlived the failed launch"
    abandoned = warnings_with(caplog_loguru, ABANDONED)
    if request_state == "failed":
        assert abandoned == [], "no session was created, so there is none to warn about"
        assert err.session_id is None
    else:
        assert len(abandoned) == 1, caplog_loguru.text
    if request_state == "created":
        assert SESSION_ID in abandoned[0].getMessage()
        assert err.session_id == SESSION_ID
    if request_state == "in-flight":
        gate.set()
        await asyncio.sleep(0)
        assert api.returned == 0, "the session request was left to finish"
    rig.assert_torn_down()


class SlowToCancel(FakeSessionApiClient):
    """A request whose cancellation takes a moment to finish, as an aiohttp request does while it closes its
    connection: what lets a second cancel land while the rollback is waiting for it."""

    def __init__(self) -> None:
        super().__init__(gate=asyncio.Event())
        self.cancelling = asyncio.Event()
        self.finish_cancel = asyncio.Event()

    async def create_session(self, **kwargs: Any) -> Any:
        try:
            return await super().create_session(**kwargs)
        except asyncio.CancelledError:
            self.cancelling.set()
            await self.finish_cancel.wait()
            raise


async def test_a_launch_timeout_during_the_overlap_cancels_the_session_request(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """The service's launch timeout cancels the launch while both the request and the connect are in
    flight. The rollback cancels the request and waits for it to stop, absorbing the second cancel Pipecat's
    `cancel_task` can send (*cancel-proof cleanup*), and only then does the cancellation propagate. Nothing
    outlives the launch. This is not a failed connect: there is no time left to wait for the request, and no
    warning about it."""
    before = running_tasks()
    api = SlowToCancel()
    rig = Rig(api=api)
    task = rig.launch_task()
    try:
        await until(lambda: len(rig.api.calls) == 1, what="the session request")
        task.cancel()
        await until(api.cancelling.is_set, what="the rollback to be waiting for the cancelled request")
        task.cancel()  # cancel_task's own timeout, from the inside
        await asyncio.sleep(0)
        assert not task.done(), "the rollback must absorb the second cancel"
        api.finish_cancel.set()
        await until(task.done, what="the rollback to finish")
    finally:
        # Whatever failed above: a request still holding its cancel would stop the event loop closing.
        api.finish_cancel.set()
        await settle(task)

    assert task.cancelled(), "the cancellation must propagate once the rollback has finished"
    assert running_tasks() - before == set(), "a task outlived the cancelled launch"
    assert api.gate is not None
    api.gate.set()
    await asyncio.sleep(0)
    assert api.returned == 0
    assert warnings_with(caplog_loguru, ABANDONED) == [], caplog_loguru.text
    rig.assert_torn_down()


# ---------------------------------------------------------------------------------- the bot's token


@pytest.mark.parametrize("room", [ROOM, None], ids=["room-claim", "no-room-claim"])
async def test_an_expired_bot_token_fails_the_launch_before_any_request(room: str | None) -> None:
    """An expired bot token cannot connect, while the avatar worker's join, on its own token, would still
    succeed: a session nobody can use. So the launch fails before the session request, naming the bot's
    token, whichever order the launch would have taken. The constructor accepts the token, because it can
    expire between startup and launch."""
    rig = Rig(token=bot_token(room=room, ttl=timedelta(seconds=-60)))
    task = rig.launch_task()
    try:
        await until(task.done, what="the launch to fail without waiting for the connect")
    finally:
        await settle(task)

    err = task.exception()
    assert isinstance(err, SynthesiaError), repr(err)
    assert err.type is ErrorType.INVALID_ROOM_TOKEN
    assert "the LiveKitTransport's token is expired" in str(err)
    assert rig.api.calls == [], "a session was requested with an expired bot token"
    rig.assert_torn_down()


async def test_an_expired_bot_token_does_not_matter_once_the_bot_has_connected() -> None:
    """LiveKit checks the bot token only when the bot joins. A transport already connected with it stays
    connected after the token expires, so a launch that starts then goes ahead."""
    rig = Rig(token=bot_token(ttl=timedelta(seconds=-60)), connected=True, avatar_present=True)
    session = await rig.adapter.launch(rig.ctx)
    try:
        assert session.is_alive
        assert len(rig.api.calls) == 1
    finally:
        await session.aclose()
    rig.assert_torn_down()


async def test_a_bot_token_without_a_room_requests_the_session_after_the_connect() -> None:
    """Without a room claim, only the connected room can say which room the avatar token is for. So the
    launch falls back to the order of a room the bot is already connected to: the connect, then the
    request, with the avatar token minted for the room the bot is in."""
    rig = Rig(token=bot_token(room=None))
    task = rig.launch_task()
    try:
        await until(
            lambda: rig.transport.handlers_for("on_connected") != [], what="launch() to wait for connect"
        )
        assert rig.api.calls == [], "the session request went out before the connect"
        await rig.connect()
        await until(lambda: rig.api.returned == 1, what="the 201")
        rig.join()
        await until(task.done, what="launch() to return once the avatar joined")
    finally:
        await settle(task)
    session = task.result()
    try:
        assert decode_unsigned(rig.api.calls[0]["livekit_token"]).room == ROOM
    finally:
        await session.aclose()
    rig.assert_torn_down()
