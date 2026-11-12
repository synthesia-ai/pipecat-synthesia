"""`UtteranceStreamer`: the byte stream's attributes and name, failing writes and closes, and the
*one-open-stream rule*.

Nothing here times an operation; every asynchronous outcome is awaited with `until()`.
`idle_close_secs=10` everywhere except the tests about the idle close: a 50 ms idle close on a loaded
runner would split an utterance and fail a test that expects its writes on one stream for the wrong
reason.
`_types.STREAM_CLOSE_SECS` is shrunk only in the tests about a close overrunning it.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import random
import re
from typing import Any

import pytest
from loguru import logger

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit.utterance_streamer import UtteranceStreamer
from pipecat_synthesia.protocol import (
    AUDIO_STREAM_NAME_PREFIX,
    AUDIO_STREAM_TOPIC,
    STREAM_ATTR_NUM_CHANNELS,
    STREAM_ATTR_SAMPLE_RATE,
)
from tests.fakes.fake_rtc import FakeByteStreamWriter, FakeLocalParticipant
from tests.support.waiting import until

AVATAR = "synthesia-avatar-agent"
CHUNK = b"\x01\x02" * 480  # 480 s16le mono samples = 20 ms at 24 kHz
IDLE = 0.05
CLOSE_BUDGET = 0.05


def make_streamer(participant: FakeLocalParticipant, *, idle_close_secs: float = 10.0) -> UtteranceStreamer:
    return UtteranceStreamer(
        participant,
        destination_identity=AVATAR,
        idle_close_secs=idle_close_secs,
        log=logger.bind(component="test"),
    )


def attributes(sample_rate: int, num_channels: int) -> dict[str, str]:
    return {STREAM_ATTR_SAMPLE_RATE: str(sample_rate), STREAM_ATTR_NUM_CHANNELS: str(num_channels)}


def warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


def dropped_chunk_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [message for message in warnings(caplog) if "dropped" in message]


def hold_the_close(writer: FakeByteStreamWriter) -> tuple[asyncio.Event, asyncio.Event]:
    """Make one writer's `aclose()` block until released; returns (entered, release)."""
    entered, release = asyncio.Event(), asyncio.Event()
    original = writer.aclose

    async def slow_aclose() -> None:
        entered.set()
        await release.wait()
        await original()

    writer.aclose = slow_aclose  # type: ignore[method-assign]
    return entered, release


def closes_end_with(participant: FakeLocalParticipant, outcome: BaseException) -> None:
    """Every writer the participant opens from now on records its close, then raises `outcome`."""
    open_stream = participant.stream_bytes

    async def stream_bytes(*args: Any, **kwargs: Any) -> FakeByteStreamWriter:
        writer = await open_stream(*args, **kwargs)
        close = writer.aclose

        async def aclose() -> None:
            await close()
            raise outcome

        writer.aclose = aclose  # type: ignore[method-assign]
        return writer

    participant.stream_bytes = stream_bytes  # type: ignore[method-assign]


async def test_lazy_open_sends_the_exact_attributes_and_name_prefix() -> None:
    """The first chunk opens the stream, with exactly these attributes. `sample_rate` and `num_channels`
    are decimal strings: a missing or int-typed attribute is fatal for the worker's session."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)

    assert participant.streams == [], "no stream may be opened before the first chunk"
    assert streamer.is_open is False

    await streamer.write(CHUNK, 24000, 1)

    assert len(participant.streams) == 1
    stream = participant.streams[0]
    assert stream.attributes == {STREAM_ATTR_SAMPLE_RATE: "24000", STREAM_ATTR_NUM_CHANNELS: "1"}
    for key, value in stream.attributes.items():
        assert isinstance(value, str), f"{key} must be a str, got {type(value).__name__}"
    assert stream.name.startswith(AUDIO_STREAM_NAME_PREFIX)
    assert stream.topic == AUDIO_STREAM_TOPIC
    assert stream.destination_identities == [AVATAR]
    assert stream.data == CHUNK
    assert streamer.is_open is True

    await streamer.aclose()


async def test_end_closes_the_stream_and_a_later_write_opens_a_new_one() -> None:
    """`end()` closes the stream, the end-of-utterance marker the worker waits for; the next write opens
    a new one."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)
    await streamer.end()

    assert participant.streams[0].closed is True
    assert streamer.is_open is False

    await streamer.write(CHUNK, 24000, 1)

    assert len(participant.streams) == 2
    assert participant.streams[1].closed is False
    assert participant.max_concurrent_open_streams == 1, "the first stream must close before the second"

    await streamer.aclose()


@pytest.mark.parametrize(
    ("sample_rate", "num_channels"),
    [
        pytest.param(16000, 1, id="sample-rate-only"),
        pytest.param(24000, 2, id="channels-only"),
        pytest.param(16000, 2, id="both"),
    ],
)
async def test_a_format_change_mid_utterance_starts_a_new_byte_stream(
    sample_rate: int, num_channels: int
) -> None:
    """Attributes are per-stream, so one changed value cannot be expressed on the open stream; an
    unchanged format keeps writing to it."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)
    await streamer.write(CHUNK, 24000, 1)
    assert len(participant.streams) == 1, "an unchanged format keeps writing to the open stream"

    await streamer.write(CHUNK, sample_rate, num_channels)

    assert len(participant.streams) == 2
    first, second = participant.streams
    assert first.closed is True
    assert first.data == CHUNK * 2
    assert second.attributes == attributes(sample_rate, num_channels)
    assert second.data == CHUNK
    assert participant.max_concurrent_open_streams == 1

    await streamer.aclose()


async def test_every_byte_stream_is_named_like_the_livekit_plugins() -> None:
    """`AUDIO_` plus 12 hex digits, fresh per utterance: the LiveKit plugin's wire format."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)
    await streamer.end()
    await streamer.write(CHUNK, 24000, 1)
    await streamer.aclose()

    names = [stream.name for stream in participant.streams]
    assert len(names) == 2
    for name in names:
        assert re.fullmatch(re.escape(AUDIO_STREAM_NAME_PREFIX) + r"[0-9a-f]{12}", name), name
    assert names[0] != names[1]


@pytest.mark.parametrize("finish", ["end", "abort"])
async def test_the_utterance_after_an_end_or_interruption_carries_its_own_format(finish: str) -> None:
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)
    await getattr(streamer, finish)()
    await streamer.write(CHUNK, 16000, 2)
    await streamer.write(CHUNK, 16000, 2)

    assert len(participant.streams) == 2
    assert participant.streams[0].closed is True
    assert participant.streams[1].attributes == attributes(16000, 2)
    assert participant.streams[1].data == CHUNK * 2

    await streamer.aclose()


async def test_the_idle_close_ends_an_utterance_nobody_ended() -> None:
    """The *idle close*; the next chunk then opens a new byte stream."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant, idle_close_secs=IDLE)

    await streamer.write(CHUNK, 24000, 1)
    await until(lambda: participant.streams[0].closed, what="the idle close")
    assert streamer.is_open is False

    await streamer.write(CHUNK, 24000, 1)

    assert len(participant.streams) == 2
    assert participant.streams[1].data == CHUNK
    assert participant.max_concurrent_open_streams == 1

    await streamer.aclose()


async def test_ending_an_utterance_disarms_its_idle_close(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """A stale idle close would close nothing here, but it would start the next utterance's failure report
    over, so a second WARNING is how it would show. The sleep lets the first utterance's idle deadline pass
    with a 10x margin; it synchronises with nothing.

    This relies on a failed write not arming the idle close of its own: if it did, that timer would fire
    during the sleep and start the report over too, and the test would fail on correct code. So a
    regression that makes failed writes arm the timer shows here as a failure, not as a pass."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant, idle_close_secs=IDLE)

    await streamer.write(CHUNK, 24000, 1)
    await streamer.end()

    participant.fail_on_write = RuntimeError("transport gone")
    await streamer.write(CHUNK, 24000, 1)
    await asyncio.sleep(IDLE * 10)
    await streamer.write(CHUNK, 24000, 1)

    assert len(dropped_chunk_warnings(caplog_loguru)) == 1, caplog_loguru.text

    await streamer.aclose()


async def test_end_twice_is_safe() -> None:
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)
    await streamer.end()
    await streamer.end()

    assert participant.streams[0].close_count == 1
    assert participant.unclosed_streams() == []


async def test_failed_write_drops_the_chunk_warns_once_and_reopens(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A failed write drops the chunk, warns once, and lets the next chunk reopen a stream."""
    participant = FakeLocalParticipant()
    participant.fail_on_write = RuntimeError("stream is broken")
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)  # must not raise into the caller

    assert participant.streams[0].data == b""
    assert streamer.is_open is False
    dropped = [
        record
        for record in caplog_loguru.records
        if record.levelno == logging.WARNING and "dropped" in record.getMessage()
    ]
    assert len(dropped) == 1, caplog_loguru.text

    # The broken stream is still open on the wire, so the streamer must close it rather than forget it:
    # an unclosed stream wedges the worker's receiver (*one-open-stream rule*).
    await until(lambda: participant.unclosed_streams() == [], what="the broken stream to be closed")

    participant.fail_on_write = None
    await streamer.write(CHUNK, 24000, 1)

    assert len(participant.streams) == 2
    assert participant.streams[1].data == CHUNK
    assert participant.max_concurrent_open_streams == 1

    await streamer.aclose()
    assert participant.unclosed_streams() == []


async def test_a_failing_utterance_is_warned_about_once_and_each_broken_stream_closed_once(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """One WARNING per utterance, however many of its chunks fail; the next utterance gets its own."""
    participant = FakeLocalParticipant()
    participant.fail_on_write = RuntimeError("transport gone")
    streamer = make_streamer(participant)

    for _ in range(3):
        await streamer.write(CHUNK, 24000, 1)  # never raises into the caller
    assert len(dropped_chunk_warnings(caplog_loguru)) == 1, caplog_loguru.text

    await streamer.end()
    await streamer.write(CHUNK, 24000, 1)  # the next utterance, failing too
    assert len(dropped_chunk_warnings(caplog_loguru)) == 2, caplog_loguru.text

    await streamer.aclose()
    await until(lambda: participant.unclosed_streams() == [], what="every broken stream to be closed")
    assert [stream.close_count for stream in participant.streams] == [1] * len(participant.streams)


async def test_end_after_failed_write_is_safe() -> None:
    """A failed write must not turn every later teardown call into an exception."""
    participant = FakeLocalParticipant()
    participant.fail_on_write = OSError("transport gone")
    streamer = make_streamer(participant)

    await streamer.write(CHUNK, 24000, 1)

    await streamer.end()
    await streamer.abort()
    await streamer.aclose()

    assert streamer.is_open is False
    assert participant.unclosed_streams() == []
    assert participant.max_concurrent_open_streams == 1


async def test_close_cancelled_mid_flight_still_closes_the_stream() -> None:
    """A close whose caller is cancelled still closes the stream (a *detached close*).

    `SynthesiaVideoService` cancels its send and interrupt tasks with a 1 s bound while
    `STREAM_CLOSE_SECS` is 2 s, so a slow `aclose()` is cancelled in production. Cancelling `end()` may
    abandon the wait, never the close itself.
    """
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)
    entered, release = hold_the_close(participant.streams[0])

    closing = asyncio.ensure_future(streamer.end())
    await entered.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert participant.unclosed_streams() != [], "the writer has not finished closing yet"

    release.set()
    await until(lambda: participant.unclosed_streams() == [], what="the detached close to land")


async def test_aclose_reaps_a_close_that_outlived_its_caller(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """`aclose()` is the last word: it waits out a close that another caller's cancel detached, and a
    close that has finished by then is nothing to warn about."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)
    entered, release = hold_the_close(participant.streams[0])

    closing = asyncio.ensure_future(streamer.end())
    await entered.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()

    await streamer.aclose()

    assert participant.unclosed_streams() == [], "aclose() returned with a close still in flight"
    assert warnings(caplog_loguru) == []


async def test_a_byte_stream_someone_else_closed_is_not_an_error(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """rtc answers a second close with "Stream already closed"; the end state is the one we wanted."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)
    await participant.streams[0].aclose()  # closed behind the streamer's back

    await streamer.end()

    assert streamer.is_open is False
    assert warnings(caplog_loguru) == []

    await streamer.aclose()


async def test_a_byte_stream_that_fails_to_close_is_reported(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """Only "already closed" is benign: any other close failure is a WARNING, and still never raises."""
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)

    async def broken_close() -> None:
        raise RuntimeError("transport gone")

    participant.streams[0].aclose = broken_close  # type: ignore[method-assign]

    await streamer.end()

    assert streamer.is_open is False
    assert warnings(caplog_loguru) != []

    await streamer.aclose()


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(RuntimeError("transport gone"), id="fails"),
        pytest.param(asyncio.CancelledError(), id="cancelled"),
    ],
)
async def test_a_detached_close_that_does_not_finish_cleanly_is_consumed(outcome: BaseException) -> None:
    """Otherwise the event loop reports it ("Task exception was never retrieved", or an exception in a
    done callback). The loop's exception handler is the recorder; `gc.collect()` makes a task whose
    exception nobody took report it now rather than at exit."""
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        participant = FakeLocalParticipant()
        participant.fail_on_write = RuntimeError("stream is broken")
        closes_end_with(participant, outcome)
        streamer = make_streamer(participant)

        await streamer.write(CHUNK, 24000, 1)  # dropped; the broken stream is closed detached
        await until(lambda: participant.unclosed_streams() == [], what="the detached close")
        await streamer.aclose()
        gc.collect()
    finally:
        loop.set_exception_handler(previous)

    assert reported == [], [context.get("message") for context in reported]


async def test_a_close_that_overruns_its_budget_still_finishes_and_aclose_waits_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", CLOSE_BUDGET)
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)
    entered, release = hold_the_close(participant.streams[0])

    await streamer.end()  # returns although the close is held

    assert entered.is_set()
    assert participant.unclosed_streams() != [], "the held close cannot have finished yet"

    release.set()
    await streamer.aclose()

    assert participant.unclosed_streams() == [], "aclose() returned with a close still in flight"


async def test_aclose_warns_about_a_close_still_running_past_its_budget(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", CLOSE_BUDGET)
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)
    _, release = hold_the_close(participant.streams[0])
    await streamer.end()  # the budget is spent; the close carries on detached
    caplog_loguru.clear()

    await streamer.aclose()

    assert warnings(caplog_loguru) != [], "aclose() gave up on a close without saying so"

    release.set()
    await until(lambda: participant.unclosed_streams() == [], what="the held close to finish")


async def test_a_write_that_wins_the_race_is_not_ended_by_the_stale_idle_timer() -> None:
    """The idle task cannot be cancelled once it is running, so it must re-check before it closes.

    Without that re-check, a write that takes the lock first has its fresh stream closed by the timer
    it just re-armed: the worker sees end-of-utterance mid-sentence, reports playback finished, and
    the service emits BotStoppedSpeaking while the rest of the utterance goes out on a second stream.

    The race is built rather than waited for: a write is held inside `writer.write` across the idle
    deadline, so the timer fires and queues its close while the lock is taken, and the write re-arms
    the timer before the queued close can run.
    """
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant, idle_close_secs=0.02)
    await streamer.write(CHUNK, 24000, 1)

    writer = participant.streams[0]
    release = asyncio.Event()
    original_write = writer.write

    async def slow_write(data: bytes) -> None:
        await release.wait()
        await original_write(data)

    writer.write = slow_write  # type: ignore[method-assign]

    held = asyncio.ensure_future(streamer.write(CHUNK, 24000, 1))
    await until(lambda: release._waiters, what="the write to take the lock")  # noqa: SLF001
    await asyncio.sleep(0.05)  # the idle deadline passes while the lock is held
    release.set()
    await held

    for _ in range(5):  # let the queued stale close run
        await asyncio.sleep(0)

    assert writer.closed is False, "a stale idle close ended an utterance that had been re-armed"
    assert len(participant.streams) == 1
    assert writer.data == CHUNK * 2

    await streamer.aclose()


async def test_write_cancelled_during_open_then_close_closes_orphan() -> None:
    """An *orphaned open* is closed at the next close: rtc issues the FFI open before awaiting its
    callback, so a cancelled open may still have put a header on the wire (livekit rtc 1.1.19)."""
    participant = FakeLocalParticipant(open_delay=0.5)
    streamer = make_streamer(participant)

    task = asyncio.ensure_future(streamer.write(CHUNK, 24000, 1))
    await until(lambda: participant.open_stream_count == 1, what="the open to be requested")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(participant.streams) == 1
    assert participant.streams[0].closed is False, "the writer had not been returned yet"

    await streamer.abort()

    assert participant.streams[0].closed is True
    assert participant.unclosed_streams() == []


async def test_write_after_cancelled_open_closes_orphan_first() -> None:
    """The next write, not only a close, closes an orphaned open, and never overlaps it with its own
    stream."""
    participant = FakeLocalParticipant(open_delay=0.5)
    streamer = make_streamer(participant)

    task = asyncio.ensure_future(streamer.write(CHUNK, 24000, 1))
    await until(lambda: participant.open_stream_count == 1, what="the open to be requested")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    participant.open_delay = 0.0
    await streamer.write(CHUNK, 24000, 1)

    assert len(participant.streams) == 2
    assert participant.streams[0].closed is True, "the orphan is closed before the new stream opens"
    assert participant.streams[1].data == CHUNK
    assert participant.max_concurrent_open_streams == 1

    await streamer.aclose()


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
async def test_random_ops_never_overlap(seed: int) -> None:
    """The *one-open-stream rule* holds under arbitrary interleavings: 200 ops from each of five fixed
    seeds.

    The invariant is enforced by the fake at the moment it would be violated, so a failure names the
    seed and the counter rather than a symptom three operations later.
    """
    rng = random.Random(seed)
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    ops = (
        "write",
        "write",
        "write",
        "end",
        "abort",
        "cancel_mid_open",
        "cancel_mid_close",
        "failing_write",
        "format_change",
    )

    for index in range(200):
        op = rng.choice(ops)
        if op == "write":
            await streamer.write(CHUNK, 24000, 1)
        elif op == "end":
            await streamer.end()
        elif op == "abort":
            await streamer.abort()
        elif op == "format_change":
            await streamer.write(CHUNK, rng.choice((8000, 16000, 24000)), rng.choice((1, 2)))
        elif op == "failing_write":
            # A transport failure mid-utterance: the chunk is dropped, but the stream it was writing
            # to is still open and must not be left behind.
            participant.fail_on_write = RuntimeError(f"seed={seed} op#{index}")
            await streamer.write(CHUNK, 24000, 1)
            participant.fail_on_write = None
        elif op == "cancel_mid_close":
            # The service cancels its send and interrupt tasks with a 1 s bound, which is shorter than
            # STREAM_CLOSE_SECS: a close can be cancelled while it is in flight, and the stream must
            # still end up closed.
            await streamer.write(CHUNK, 24000, 1)
            closing = asyncio.ensure_future(streamer.end())
            await asyncio.sleep(0)
            closing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await closing
        else:
            participant.open_delay = 0.05
            before = len(participant.streams)
            task = asyncio.ensure_future(streamer.write(CHUNK, 24000, 1))
            # Either the write reached the open (a genuine orphan to cancel) or it had an open stream
            # to reuse and simply finished; both are interleavings worth exercising.
            await until(
                lambda before=before, task=task: len(participant.streams) > before or task.done(),
                what="the write to reach the open or finish",
            )
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            participant.open_delay = 0.0
        assert participant.max_concurrent_open_streams <= 1, (
            f"seed={seed} op#{index}={op}: two streams were open at once; {participant!r}"
        )

    await streamer.aclose()

    await until(
        lambda: participant.unclosed_streams() == [],
        what=f"seed={seed}: every stream to be closed after aclose()",
    )
    assert participant.unclosed_streams() == [], (
        f"seed={seed}: {len(participant.unclosed_streams())} stream(s) left open after aclose(); "
        f"each one wedges the avatar worker's receiver forever"
    )
