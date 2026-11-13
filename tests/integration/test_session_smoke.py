"""`LiveKitAvatarSession.launch()` against a real `livekit-server` and `FakeAvatarWorker`.

`launch()` is the first place the token, the RPC handlers, the byte stream and the room watchers meet
real rtc together, and both room adapters wrap it. No room adapter, no Synthesia API: a
hand-built `RoomHandle(owns_room=False)` on a real borrowed room stands in for a direct join, and a
`FakeSessionApiClient` whose `create_session` *launches the fake worker* stands in for the API client
and the `Launcher`. The fake is started with the **very token the session minted**, exactly as the
Synthesia API hands it to the worker, so `lk.publish_on_behalf` is verified for real, not restated from
the test's setup.

`launch_while_connecting()`, a direct join's *connect overlap*, gets one test here too: the order where the
avatar worker has joined and published before the bot connects. The unit fakes cannot model what real rtc
lists in the room at connect, and that is what the overlap relies on.

Every network-observable wait goes through `until()` with a generous backstop; the real 1 s
`REPLACEMENT_GRACE_SECS` stays real in this tier.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from time import monotonic

import pytest
from livekit import rtc

from pipecat_synthesia.api import SessionCreated
from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._types import LiveKitCredentials, MintedAvatarToken, RoomHandle
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME
from pipecat_synthesia.protocol import STREAM_ATTR_NUM_CHANNELS, STREAM_ATTR_SAMPLE_RATE
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.integration.fake_worker import FakeAvatarWorker
from tests.support.callbacks import CallbackRecorder, make_launch_context
from tests.support.livekit_dev import LiveKitDev, connect_room, mint_dev_token
from tests.support.waiting import until

pytestmark = pytest.mark.integration

BOT = "bot"
AVATAR_IDS = ("av_a", "av_b")
CHUNK = b"\x01\x02" * 960  # 1920 bytes = 40 ms of 24 kHz mono s16le
ATTRIBUTES = {STREAM_ATTR_SAMPLE_RATE: "24000", STREAM_ATTR_NUM_CHANNELS: "1"}
#: Every network-observable wait. A backstop, not a budget.
WAIT = 10.0


class Launcher:
    """`create_session` that starts a `FakeAvatarWorker` with the token it was given: the integration
    `Launcher` in miniature."""

    def __init__(self, url: str, **options: object) -> None:
        self.url = url
        self.options = options
        self.fakes: list[FakeAvatarWorker] = []

    async def __call__(
        self, avatar_ids: Sequence[str], livekit_url: str, livekit_token: str
    ) -> SessionCreated:
        fake = FakeAvatarWorker(livekit_url, livekit_token, avatar_ids=list(avatar_ids), **self.options)  # type: ignore[arg-type]
        self.fakes.append(fake)
        # `start()` returns once both tracks are published, as the real API's 201 promises.
        await fake.start()
        return SessionCreated(session_id="ses_smoke", request_id="req_smoke")

    async def stop_all(self) -> None:
        for fake in self.fakes:
            await fake.stop()


async def test_launch_send_interrupt_swap_close(livekit_server: LiveKitDev, room_name: str) -> None:
    """The whole session surface once, on a borrowed real room, against the fake worker.

    Launch, one utterance, readiness, interruption, swap and close each work against real rtc, and the
    borrowed room is still connected afterwards (*borrowed-room rule*).
    """
    dev = livekit_server
    bot_token = mint_dev_token(dev, identity=BOT, room=room_name, kind="agent")
    launcher = Launcher(dev.url, status_delay=0.5)

    async with connect_room(dev, bot_token) as room:
        handle = RoomHandle(
            room=room,
            bot_identity=BOT,
            livekit_url=dev.url,
            room_name=room_name,
            owns_room=False,
            connected_at=monotonic(),
        )
        token_source = MintedAvatarToken(LiveKitCredentials(dev.api_key, dev.api_secret))
        rec = CallbackRecorder()
        ctx = make_launch_context(
            FakeSessionApiClient(on_create=launcher),
            avatar_ids=AVATAR_IDS,
            launch_timeout=30,
            audio_out_sample_rate=24000,
            recorder=rec,
        )
        session: LiveKitAvatarSession | None = None
        try:
            session = await LiveKitAvatarSession.launch(
                handle,
                token_source=token_source,
                avatar_identity=DEFAULT_AVATAR_IDENTITY,
                avatar_name=DEFAULT_AVATAR_NAME,
                ctx=ctx,
            )
            fake = launcher.fakes[0]

            # --- the token the session minted is the one the worker is bound to ----------------
            assert fake.token_violations == [], fake.debug_dump()
            assert fake.publish_on_behalf == BOT
            # the worker is started by the session request, so it joins after it; the 201 may come either side
            assert session.timings.session_requested <= session.timings.avatar_joined
            assert session.avatar_identity == fake.identity
            assert session.session_id == "ses_smoke"

            # --- one utterance: one stream, byte-exact attributes, both playback callbacks -------
            await session.send_audio(CHUNK, 24000, 1)
            await session.send_audio(CHUNK, 24000, 1)
            await session.end_utterance()
            await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="one stream at the worker")
            assert fake.streams[0]["attributes"] == ATTRIBUTES, fake.debug_dump()
            assert fake.streams[0]["bytes"] == CHUNK * 2
            await until(lambda: rec.started == 1, timeout=WAIT, what="lk.playback_started")
            await until(lambda: len(rec.finished) == 1, timeout=WAIT, what="lk.playback_finished")
            assert rec.finished[0][1] is False
            assert fake.rejected_stream_identities == [], fake.debug_dump()
            assert fake.overlapping_streams == 0, fake.debug_dump()

            # --- readiness arrives through the participant attributes ----------------------------
            # Waited on the RECORDER, not on `session.all_ready`: readiness is folded in synchronously
            # and the callback is delivered in a task of its own (*state-before-await*), so the flag
            # flips first and a test that waits on it can read `rec.status` before the "ready" delivery
            # has landed.
            await until(
                lambda: any(all_ready for _, all_ready in rec.status),
                timeout=WAIT,
                what="the all-ready status callback",
            )
            assert session.all_ready is True
            assert session.avatar_status == {"av_a": "ready", "av_b": "ready"}

            # --- interruption: clear_buffer FIRST, then the close ---------------------------------
            await session.send_audio(CHUNK, 24000, 1)
            # The worker sends `lk.playback_started` from the first chunk it actually reads; waiting
            # for it says "the audio is in" without sleeping.
            await until(lambda: rec.started == 2, timeout=WAIT, what="playback_started for utterance 2")
            await session.interrupt()
            await until(lambda: fake.clear_buffer_calls == 1, timeout=WAIT, what="one clear_buffer")
            await until(lambda: fake.open_streams == 0, timeout=WAIT, what="every stream closed")

            # --- swap: the wire carries the unprefixed id; the confirmed id comes back prefixed ----
            new = await session.swap_avatar(ctx.avatar_ids[1], timeout=5)
            assert new == ctx.avatar_ids[1]
            assert fake.swap_calls[-1]["avatar_id"] == "b", fake.debug_dump()
            assert session.active_avatar_id == "av_b"

            # --- close: the borrowed room is untouched, the worker stays, a second close is a no-op
            await session.aclose()
            assert room.isconnected() is True, "borrowed-room rule: a borrowed room is never disconnected"
            await session.aclose()
            assert fake.connected is True, fake.debug_dump()
            assert session.is_alive is False
            assert rec.left == [], "a normal close is not a leave"
        finally:
            if session is not None:
                await session.aclose()  # idempotent; leaves nothing running when an assertion failed
            await launcher.stop_all()


async def test_avatar_leaves_reports_once_with_no_humans(
    livekit_server: LiveKitDev, room_name: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """On real rtc, the worker leaving is reported once, as `on_avatar_left` with no humans present.

    After the leave, audio sent to the dead session is dropped with a single WARNING, and closing the
    session leaves the borrowed room connected.
    """
    dev = livekit_server
    bot_token = mint_dev_token(dev, identity=BOT, room=room_name, kind="agent")
    launcher = Launcher(dev.url, leave_after=1.0)

    async with connect_room(dev, bot_token) as room:
        handle = RoomHandle(
            room=room,
            bot_identity=BOT,
            livekit_url=dev.url,
            room_name=room_name,
            owns_room=False,
            connected_at=monotonic(),
        )
        rec = CallbackRecorder()
        ctx = make_launch_context(
            FakeSessionApiClient(on_create=launcher), avatar_ids=AVATAR_IDS, launch_timeout=30, recorder=rec
        )
        session: LiveKitAvatarSession | None = None
        try:
            session = await LiveKitAvatarSession.launch(
                handle,
                token_source=MintedAvatarToken(LiveKitCredentials(dev.api_key, dev.api_secret)),
                avatar_identity=DEFAULT_AVATAR_IDENTITY,
                avatar_name=DEFAULT_AVATAR_NAME,
                ctx=ctx,
            )
            assert session.is_alive is True

            await until(lambda: len(rec.left) >= 1, timeout=WAIT, what="on_avatar_left")
            assert rec.left == [("avatar_left_room", False)]
            assert session.is_alive is False
            assert room.remote_participants.get(DEFAULT_AVATAR_IDENTITY) is None

            await session.send_audio(CHUNK, 24000, 1)
            await session.send_audio(CHUNK, 24000, 1)
            dropped = [
                record
                for record in caplog_loguru.records
                if record.levelno == logging.WARNING and "dropping audio" in record.getMessage()
            ]
            assert len(dropped) == 1

            await session.aclose()
            assert rec.left == [("avatar_left_room", False)], "reported exactly once"
            assert room.isconnected() is True
            assert isinstance(room, rtc.Room)
        finally:
            if session is not None:
                await session.aclose()  # idempotent; leaves nothing running when an assertion failed
            await launcher.stop_all()


async def test_an_avatar_already_publishing_when_the_bot_connects(
    livekit_server: LiveKitDev, room_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """*Connect overlap* against real rtc: the session request goes out before the bot is connected, and the
    avatar worker joins and publishes first. The launch sees the avatar in the room as soon as the bot has
    connected, without waiting for a join event, and still reports its video, so `avatar_video_published`
    is stamped.

    This is the one place that checks what the overlap relies on in rtc itself. Measured against
    livekit-server 1.9.1 and rtc 1.1.19: a participant already in the room is listed when `connect()`
    returns, with no `participant_connected` for it, and so is its audio, with no `track_published`; its
    video arrives as a `track_published` a moment later. `connect()` below returns only once that video is
    listed, as Pipecat's own connect does in effect, because it publishes the bot's microphone before
    `on_connected`: the event has gone by before the session can listen, and only the late report can
    deliver it.

    The bot connects only after the 201, which the fake worker sends once both its tracks are published, so
    the order is decided by the test, not by the network. The join wait is raised far past `WAIT` (an
    input), so "the launch returned" can only mean it did not wait it out.
    """
    monkeypatch.setattr(_types, "POST_CREATE_JOIN_WAIT_SECS", 300.0)
    dev = livekit_server
    bot_token = mint_dev_token(dev, identity=BOT, room=room_name, kind="agent")
    launcher = Launcher(dev.url)
    api = FakeSessionApiClient(on_create=launcher)
    rec = CallbackRecorder()
    ctx = make_launch_context(
        api, avatar_ids=AVATAR_IDS, launch_timeout=300, audio_out_sample_rate=24000, recorder=rec
    )

    async with contextlib.AsyncExitStack() as stack:

        async def connect() -> RoomHandle:
            await until(lambda: api.returned == 1, timeout=WAIT, what="the 201: the avatar has published")
            room = await stack.enter_async_context(connect_room(dev, bot_token))

            def avatar_video_listed() -> bool:
                avatar = room.remote_participants.get(DEFAULT_AVATAR_IDENTITY)
                return avatar is not None and any(
                    pub.kind == rtc.TrackKind.KIND_VIDEO for pub in avatar.track_publications.values()
                )

            await until(avatar_video_listed, timeout=WAIT, what="the avatar's video to be listed")
            return RoomHandle(
                room=room,
                bot_identity=BOT,
                livekit_url=dev.url,
                room_name=room_name,
                owns_room=False,
                connected_at=monotonic(),
            )

        launch = asyncio.ensure_future(
            LiveKitAvatarSession.launch_while_connecting(
                connect,
                bot_identity=BOT,
                room_name=room_name,
                livekit_url=dev.url,
                token_source=MintedAvatarToken(LiveKitCredentials(dev.api_key, dev.api_secret)),
                avatar_identity=DEFAULT_AVATAR_IDENTITY,
                avatar_name=DEFAULT_AVATAR_NAME,
                ctx=ctx,
            )
        )
        session: LiveKitAvatarSession | None = None
        try:
            await until(launch.done, timeout=WAIT, what="the launch to find the avatar already in the room")
            session = launch.result()
            fake = launcher.fakes[0]
            assert fake.token_violations == [], fake.debug_dump()
            assert fake.publish_on_behalf == BOT
            assert session.avatar_identity == fake.identity
            timings = session.timings
            assert timings.room_connected is not None
            assert timings.session_requested <= timings.session_created <= timings.room_connected
            await until(
                lambda: "video" in rec.tracks, timeout=WAIT, what="the video published before the connect"
            )
        finally:
            if not launch.done():
                launch.cancel()
            await asyncio.wait({launch})
            if session is not None:
                await session.aclose()
            elif not launch.cancelled():
                launch.exception()
            await launcher.stop_all()
