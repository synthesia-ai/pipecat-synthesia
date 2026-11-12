"""Fake-vs-real conformance for the wire layer (`UtteranceStreamer`, the playback RPCs, clear buffer, swap).

Everything in `tests/unit/test_utterance_streamer.py`, `test_rpc.py` and `test_readiness.py` is proved
against `tests/fakes/fake_rtc.py`, which models real rtc from reading its source. These tests prove the
model is right, by running the same sequence against real rtc and against the fake. They should take
under 30 s in total.

Tokens come from `tests.support.livekit_dev.mint_dev_token`, not from `pipecat_synthesia.livekit.token`,
so the wire layer is checked independently of the product's token recipe.

Any divergence found here is fixed in `fake_rtc.py` *and* recorded in its module docstring.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from livekit import rtc as lk_rtc
from loguru import logger

from pipecat_synthesia.livekit.rpc import PlaybackRpcHandlers, perform_clear_buffer, perform_swap_avatar
from pipecat_synthesia.livekit.utterance_streamer import UtteranceStreamer
from pipecat_synthesia.protocol import (
    ATTR_AVATAR_READY,
    AUDIO_STREAM_NAME_PREFIX,
    AUDIO_STREAM_TOPIC,
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    RPC_SWAP_AVATAR,
    STREAM_ATTR_NUM_CHANNELS,
    STREAM_ATTR_SAMPLE_RATE,
)
from tests.fakes.fake_rtc import FakeLocalParticipant, FakeRoom, reply_after
from tests.support.waiting import until

# Skips rather than failing collection if `fake_worker` or `livekit_dev` cannot be imported.
_fake_worker = pytest.importorskip(
    "tests.integration.fake_worker", reason="needs FakeAvatarWorker and the livekit_server fixture"
)
_livekit_dev = pytest.importorskip("tests.support.livekit_dev", reason="needs the dev-token and room helpers")
FakeAvatarWorker = _fake_worker.FakeAvatarWorker
connect_room = _livekit_dev.connect_room
mint_dev_token = _livekit_dev.mint_dev_token

pytestmark = pytest.mark.integration

BOT = "bot"
CHUNK = b"\x01\x02" * 960  # 1920 bytes = 40 ms of 24 kHz mono s16le
ATTRIBUTES = {STREAM_ATTR_SAMPLE_RATE: "24000", STREAM_ATTR_NUM_CHANNELS: "1"}
LOG = logger.bind(component="conformance")


class Recorder:
    """Playback callbacks with the signatures `LiveKitAvatarSession` gives `PlaybackRpcHandlers`."""

    def __init__(self) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []

    async def on_started(self) -> None:
        self.started += 1

    async def on_finished(self, position: float, interrupted: bool) -> None:
        self.finished.append((position, interrupted))


async def test_streamer_and_handlers_against_real_rtc(livekit_server, room_name) -> None:
    """Playback handlers, stream attributes, one open stream and clear buffer against a real SFU, then
    the same sequence against `FakeLocalParticipant`.

    The two recorded `(stream count, attributes, started count, finished count)` tuples must be equal:
    that equality is what lets the unit tier stand in for rtc.
    """
    dev = livekit_server
    token = mint_dev_token(dev, identity=BOT, room=room_name, kind="agent")

    async with connect_room(dev, token) as room:
        fake = FakeAvatarWorker.for_bot(dev, room_name=room_name, bot_identity=BOT)
        await fake.start()
        try:
            # `room.local_participant` is a real rtc.LocalParticipant, which is exactly the
            # `LocalParticipantLike` this package's classes take — no adapter in between.
            participant = room.local_participant
            rec = Recorder()
            handlers = PlaybackRpcHandlers(
                participant,
                expected_caller=fake.identity,
                on_started=rec.on_started,
                on_finished=rec.on_finished,
                log=LOG,
            )
            handlers.register()
            streamer = UtteranceStreamer(
                participant, destination_identity=fake.identity, idle_close_secs=10, log=LOG
            )

            # --- one utterance -------------------------------------------------------------------
            await streamer.write(CHUNK, 24000, 1)
            await streamer.write(CHUNK, 24000, 1)
            await streamer.end()

            await until(lambda: len(fake.streams) == 1, timeout=10, what="one stream at the worker")
            assert fake.streams[0]["attributes"] == ATTRIBUTES
            assert fake.streams[0]["bytes"] == CHUNK * 2
            await until(lambda: rec.started == 1, timeout=10, what="lk.playback_started")
            await until(lambda: len(rec.finished) == 1, timeout=10, what="lk.playback_finished")
            assert rec.finished[0][1] is False
            assert fake.overlapping_streams == 0
            assert fake.fatal_unsupported_playback_finished is False
            assert fake.attribute_violations == []

            # --- interruption: clear_buffer FIRST, then close ------------------------------------
            await streamer.write(CHUNK, 24000, 1)
            # The worker sends `lk.playback_started` from the first chunk it actually reads, and sends
            # no `playback_finished` at all if clear_buffer arrives before any audio played. Waiting for
            # the second `started` is how the test says "the audio is in" without sleeping, and it is
            # the same signal the service's speaking state uses.
            await until(lambda: rec.started == 2, timeout=10, what="playback_started for utterance 2")
            assert await perform_clear_buffer(participant, avatar_identity=fake.identity, log=LOG) is True
            await streamer.abort()

            await until(lambda: fake.clear_buffer_calls == 1, timeout=10, what="one clear_buffer")
            await until(lambda: len(rec.finished) == 2, timeout=10, what="the interrupted finish")
            assert rec.finished[1][1] is True
            await until(lambda: fake.open_streams == 0, timeout=10, what="every stream closed")

            # Recorded here, before the cancelled-open step: a cancelled open is by definition a
            # stream the *sender* requested and the *receiver* may never see, so the two sides count
            # different things past this point. Each side asserts the invariant in its own terms below.
            real = (len(fake.streams), fake.streams[0]["attributes"], rec.started, len(rec.finished))

            # --- one-open-stream rule: an orphaned open is still closed --------------------------
            task = asyncio.ensure_future(streamer.write(CHUNK, 24000, 1))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await streamer.abort()

            await until(
                lambda: fake.open_streams == 0,
                timeout=5,
                what="the cancelled-open stream to be closed too",
            )
            assert fake.overlapping_streams == 0
            handlers.unregister()
        finally:
            await fake.stop()

    assert real == await _same_sequence_against_the_fake(), (
        "fake_rtc.py and real rtc disagreed: fix the fake and record it in its module docstring"
    )


async def _same_sequence_against_the_fake() -> tuple[int, dict[str, str], int, int]:
    """Drive the identical sequence against `FakeLocalParticipant`, which stands in for rtc *and* for
    the worker: `invoke()` is how the worker's playback RPCs arrive."""
    participant = FakeLocalParticipant(BOT)
    avatar = "synthesia-avatar-agent"
    rec = Recorder()
    handlers = PlaybackRpcHandlers(
        participant,
        expected_caller=avatar,
        on_started=rec.on_started,
        on_finished=rec.on_finished,
        log=LOG,
    )
    handlers.register()
    streamer = UtteranceStreamer(participant, destination_identity=avatar, idle_close_secs=10, log=LOG)

    await streamer.write(CHUNK, 24000, 1)
    await streamer.write(CHUNK, 24000, 1)
    await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=avatar)
    await streamer.end()

    await participant.invoke(
        RPC_PLAYBACK_FINISHED,
        caller_identity=avatar,
        payload=json.dumps({"playback_position": 0.08, "interrupted": False}),
    )

    participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    await streamer.write(CHUNK, 24000, 1)
    await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=avatar)
    await perform_clear_buffer(participant, avatar_identity=avatar, log=LOG)
    await streamer.abort()
    await participant.invoke(
        RPC_PLAYBACK_FINISHED,
        caller_identity=avatar,
        payload=json.dumps({"playback_position": 0.04, "interrupted": True}),
    )

    await until(lambda: len(rec.finished) == 2, what="both finishes")
    recorded = (
        len(participant.streams),
        participant.streams[0].attributes,
        rec.started,
        len(rec.finished),
    )

    participant.open_delay = 0.2
    task = asyncio.ensure_future(streamer.write(CHUNK, 24000, 1))
    await until(lambda: participant.open_stream_count == 1, what="the open to be requested")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    participant.open_delay = 0.0
    await streamer.abort()

    assert participant.unclosed_streams() == [], "the cancelled open was never closed"
    assert participant.max_concurrent_open_streams == 1
    handlers.unregister()
    return recorded


async def test_fake_room_event_semantics_match_rtc() -> None:
    """`on()` rejects coroutine functions and `emit()` trims arity identically in `FakeRoom` and rtc.

    No connect and no server needed: the emitter is `rtc.Room`'s own base class, `EventEmitter`. It is
    `async` only because `rtc.Room()` calls `asyncio.get_event_loop()` in its constructor and so cannot be
    built outside a running loop, a divergence from `FakeRoom` recorded in `fake_rtc.py`.
    """
    real = lk_rtc.Room()
    fake = FakeRoom("conformance")

    async def async_callback(participant) -> None:
        """Never called; registering it is the assertion."""

    for room in (real, fake):
        with pytest.raises(ValueError, match="async callback"):
            room.on("participant_connected", async_callback)

    def record_into(seen: list[tuple]):
        def zero() -> None:
            seen.append(())

        def one(a) -> None:
            seen.append((a,))

        def two(a, b) -> None:
            seen.append((a, b))

        return zero, one, two

    real_seen: list[tuple] = []
    fake_seen: list[tuple] = []
    for room, seen in ((real, real_seen), (fake, fake_seen)):
        for callback in record_into(seen):
            room.on("participant_connected", callback)

    real.emit("participant_connected", "a", "b")
    fake.emit("participant_connected", "a", "b")

    # Sorted: real rtc keeps listeners in a `set`, so invocation order is undefined by construction.
    assert sorted(fake_seen, key=len) == sorted(real_seen, key=len)
    assert sorted(fake_seen, key=len) == [(), ("a",), ("a", "b")]


async def test_fake_writer_double_close_matches_rtc(livekit_server, room_name) -> None:
    """A second `aclose()` on a byte-stream writer raises `RuntimeError`, in rtc and in the fake alike."""
    dev = livekit_server
    token = mint_dev_token(dev, identity=BOT, room=room_name, kind="agent")

    async with connect_room(dev, token) as room:
        fake = FakeAvatarWorker.for_bot(dev, room_name=room_name, bot_identity=BOT)
        await fake.start()
        try:
            writer = await room.local_participant.stream_bytes(
                AUDIO_STREAM_NAME_PREFIX + "conformance",
                topic=AUDIO_STREAM_TOPIC,
                destination_identities=[fake.identity],
                attributes=dict(ATTRIBUTES),
            )
            await writer.write(CHUNK)
            await writer.aclose()
            with pytest.raises(RuntimeError, match="already closed"):
                await writer.aclose()
        finally:
            await fake.stop()

    participant = FakeLocalParticipant(BOT)
    fake_writer = await participant.stream_bytes(
        AUDIO_STREAM_NAME_PREFIX + "conformance",
        topic=AUDIO_STREAM_TOPIC,
        destination_identities=["synthesia-avatar-agent"],
        attributes=dict(ATTRIBUTES),
    )
    await fake_writer.write(CHUNK)
    await fake_writer.aclose()
    with pytest.raises(RuntimeError, match="already closed"):
        await fake_writer.aclose()


#: How long the worker takes to swap in `test_swap_reply_window_matches_rtc`: past the 1 s rtc gives the
#: worker when an RPC passes its timeout as both values, well inside the swap's own timeout.
SWAP_SECS = 2.5
SWAP_TIMEOUT_SECS = 15.0


async def swap_outcomes(participant, *, avatar_identity: str) -> tuple[str, str]:
    """`perform_swap_avatar`, then a bare `perform_rpc` with the timeout as both values: what each gave."""
    outcomes: list[str] = []
    try:
        outcomes.append(
            await perform_swap_avatar(
                participant, avatar_identity=avatar_identity, avatar_id="av_b", timeout=SWAP_TIMEOUT_SECS
            )
        )
    except Exception as exc:
        outcomes.append(repr(exc))
    try:
        await participant.perform_rpc(
            destination_identity=avatar_identity,
            method=RPC_SWAP_AVATAR,
            payload=json.dumps({"avatar_id": "b"}),
            response_timeout=SWAP_TIMEOUT_SECS,
            max_round_trip_latency=SWAP_TIMEOUT_SECS,
        )
        outcomes.append("answered")
    except lk_rtc.RpcError as exc:
        outcomes.append(f"RpcError {int(exc.code)}")
    return outcomes[0], outcomes[1]


async def test_swap_reply_window_matches_rtc(livekit_server, room_name) -> None:
    """A worker that takes 2.5 s to swap is heard by `perform_swap_avatar` with a 15 s timeout, while an RPC
    that gives rtc 15 s as both `response_timeout` and `max_round_trip_latency` gets `RESPONSE_TIMEOUT`, in
    rtc and in the fake alike.

    The second half is the rule `reply_after` models: rtc gives the worker's handler
    max(response_timeout - max_round_trip_latency, 1 s).
    """
    dev = livekit_server
    token = mint_dev_token(dev, identity=BOT, room=room_name, kind="agent")

    async with connect_room(dev, token) as room:
        fake = FakeAvatarWorker.for_bot(
            dev, room_name=room_name, bot_identity=BOT, avatar_ids=["av_a", "av_b"], swap_delay=SWAP_SECS
        )
        await fake.start()
        try:

            def ready() -> bool:
                participant = room.remote_participants.get(fake.identity)
                return participant is not None and participant.attributes.get(ATTR_AVATAR_READY) == "true"

            await until(ready, timeout=10, what="avatar_ready=true seen from the bot's room")
            real = await swap_outcomes(room.local_participant, avatar_identity=fake.identity)
            assert fake.swap_calls == [{"avatar_id": "b"}, {"avatar_id": "b"}], fake.debug_dump()
        finally:
            await fake.stop()

    participant = FakeLocalParticipant(BOT)
    participant.script_rpc(
        RPC_SWAP_AVATAR, reply_after(SWAP_SECS, json.dumps({"status": "ok", "avatar_id": "b"}))
    )
    fake_outcomes = await swap_outcomes(participant, avatar_identity="synthesia-avatar-agent")

    assert real == fake_outcomes
    assert real == ("av_b", f"RpcError {int(lk_rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT)}")
