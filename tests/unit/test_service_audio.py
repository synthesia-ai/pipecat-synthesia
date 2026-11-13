"""The audio path: consuming TTS audio, pre-launch audio, interruption, terminal states, graceful stop.

Audio reaches the fake session only through the service's own send task; every assertion here is on the
frames the sink saw or the calls the fake recorded, in order and eventually (`until`).

`stop_drain_timeout` is shrunk to 0.2 s in the tests that are not about the drain: the fake session never
reports playback, so `stop()` would otherwise wait the full default before tearing down. Shrinking it is
safe because reaching the timeout, not how long it takes, is the outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import Harness, bidirectional_pipeline, make_service, make_setup
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, assert_ordered, count, frames_of
from tests.support.waiting import assert_no_dangling, until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, records_at, tts_audio

UP = FrameDirection.UPSTREAM
_SESSION_WORK = frozenset({"interrupt", "end_utterance", "send_audio"})


def _work(session: FakeAvatarSession) -> list[tuple]:
    return [c for c in session.calls if c[0] in _SESSION_WORK]


def _sent(session: FakeAvatarSession) -> list[bytes]:
    return [call[1] for call in session.send_audio_calls]


# --- consume, not forward ------------------------------------------------------------------------


@pytest.mark.unit
async def test_consume_not_forward() -> None:
    """In SESSION_ACTIVE `TTSAudioRawFrame` is consumed (it reaches the session, never the sink); the
    utterance boundaries and unrelated frames are forwarded in order (*frame contract*)."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.queue(TextFrame("after"))
        await until(lambda: len(room.session.send_audio_calls) == 1, what="the chunk at the session")
        await until(lambda: room.session.end_utterance_count == 1, what="end_utterance at the session")
        await h.finish()

    assert_absent(h.down, TTSAudioRawFrame, where="downstream of the service in SESSION_ACTIVE")
    assert_ordered(h.down, [TTSStartedFrame, TTSStoppedFrame, TextFrame])
    assert room.session.send_audio_calls[0][1:] == (AUDIO_1, 24000, 1)


# --- pre-launch audio -----------------------------------------------------------------------------


# `prelaunch_max_seconds=0.01` is a cap of 480 bytes at the pipeline's 24 kHz mono 16-bit output.
CAP_SECONDS = 0.01
CAP_BYTES = 480


def _pcm(tag: int, size: int) -> bytes:
    return bytes([tag]) * size


async def _utterance(h: Harness, context_id: str, *chunks: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    for audio in chunks:
        await h.queue(tts_audio(context_id, audio))
    await h.queue(TTSStoppedFrame(context_id=context_id))


def _audio_work(session: FakeAvatarSession) -> list[tuple[str, int | None]]:
    """The session's audio calls in order, as ("send_audio", first byte) or ("end_utterance", None)."""
    return [
        (c[0], c[1][0] if c[0] == "send_audio" else None)
        for c in session.calls
        if c[0] in ("send_audio", "end_utterance")
    ]


@pytest.mark.unit
async def test_pre_launch_audio_is_delivered_in_order_once_the_session_is_active() -> None:
    """Nothing reaches the session before it is active; then each chunk and utterance end once, in order."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        assert service.state is ServiceState.LAUNCHING
        await _utterance(h, "c1", _pcm(1, 240), _pcm(2, 240))
        await _utterance(h, "c2", _pcm(3, 240))
        await processed(h, "after the pre-launch audio")
        assert room.session.calls == []

        gate.set()
        await rec.wait("on_avatar_connected")
        await until(lambda: room.session.end_utterance_count == 2, what="both utterances delivered")
        await h.finish()

    assert _audio_work(room.session)[:5] == [
        ("send_audio", 1),
        ("send_audio", 2),
        ("end_utterance", None),
        ("send_audio", 3),
        ("end_utterance", None),
    ]
    assert service.metrics.prelaunch_dropped_bytes == 0
    assert_absent(h.down, TTSAudioRawFrame)


@pytest.mark.unit
async def test_prelaunch_bounded(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """`prelaunch_max_seconds=0.01` (the config seam; 480 bytes at 24 kHz) against three 4 KB chunks: the
    oldest audio is dropped, the drop is counted in the metrics, and exactly one WARNING is logged."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, prelaunch_max_seconds=0.01, stop_drain_timeout=0.2)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        for _ in range(3):
            await h.queue(tts_audio("c1", b"\x01" * 4096))
        await until(lambda: service.metrics.prelaunch_dropped_bytes >= 4096, what="a pre-launch audio drop")
        gate.set()
        await h.finish()

    assert service.metrics.prelaunch_dropped_bytes >= 4096
    assert len(records_at(caplog_loguru, logging.WARNING, "pre-launch")) == 1
    assert service.metrics.summary()["prelaunch_dropped_bytes"] == service.metrics.prelaunch_dropped_bytes


@pytest.mark.unit
async def test_overflow_drops_the_oldest_audio_and_keeps_every_utterance_end() -> None:
    """240, 240 and 480 bytes against the 480-byte cap: the two older chunks go, oldest first, and all
    three utterance ends survive in order, including the one between the dropped chunks."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, prelaunch_max_seconds=CAP_SECONDS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _utterance(h, "c1", _pcm(1, 240))
        await _utterance(h, "c2", _pcm(2, 240))
        await _utterance(h, "c3", _pcm(3, CAP_BYTES))
        await processed(h, "after the pre-launch audio")
        assert service.metrics.prelaunch_dropped_bytes == 480

        gate.set()
        await rec.wait("on_avatar_connected")
        await until(lambda: room.session.end_utterance_count >= 3, what="the queued utterance ends delivered")
        await h.finish()

    assert _audio_work(room.session)[:4] == [
        ("end_utterance", None),
        ("end_utterance", None),
        ("send_audio", 3),
        ("end_utterance", None),
    ]
    assert _sent(room.session) == [_pcm(3, CAP_BYTES)]
    assert service.metrics.prelaunch_dropped_bytes == 480
    assert service.metrics.summary()["prelaunch_dropped_bytes"] == 480


@pytest.mark.unit
async def test_audio_exactly_filling_the_cap_is_kept_whole() -> None:
    """A third 240-byte chunk past the full cap then drops only the oldest one."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, prelaunch_max_seconds=CAP_SECONDS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", _pcm(1, 240)))
        await h.queue(tts_audio("c1", _pcm(2, 240)))
        await processed(h, "at the cap")
        assert service.metrics.prelaunch_dropped_bytes == 0

        await h.queue(tts_audio("c1", _pcm(3, 240)))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await processed(h, "past the cap")
        assert service.metrics.prelaunch_dropped_bytes == 240

        gate.set()
        await rec.wait("on_avatar_connected")
        await until(lambda: room.session.end_utterance_count >= 1, what="the utterance delivered")
        await h.finish()

    assert [c[1][0] for c in room.session.send_audio_calls] == [2, 3]
    assert service.metrics.prelaunch_dropped_bytes == 240


@pytest.mark.unit
async def test_a_chunk_larger_than_the_cap_is_dropped_and_later_audio_still_flows() -> None:
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, prelaunch_max_seconds=CAP_SECONDS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", _pcm(1, 2 * CAP_BYTES)))
        await processed(h, "after the oversized chunk")
        assert service.metrics.prelaunch_dropped_bytes == 2 * CAP_BYTES

        gate.set()
        await rec.wait("on_avatar_connected")
        await _utterance(h, "c2", _pcm(2, 240))
        await until(lambda: room.session.end_utterance_count >= 1, what="the next utterance delivered")
        await h.finish()

    assert [c[1][0] for c in room.session.send_audio_calls] == [2]
    assert service.metrics.prelaunch_dropped_bytes == 2 * CAP_BYTES


@pytest.mark.unit
@pytest.mark.parametrize(
    ("output_rate", "kept"),
    [(24000, [AUDIO_2, AUDIO_3]), (16000, [AUDIO_3])],
    ids=["24kHz-keeps-two", "16kHz-keeps-one"],
)
async def test_prelaunch_audio_is_capped_by_the_pipeline_output_rate(
    output_rate: int, kept: list[bytes]
) -> None:
    """The cap is `prelaunch_max_seconds` of 16-bit mono at the pipeline's output rate: 0.1 s is 4800 bytes
    at 24 kHz, two of the three 1920-byte chunks, and 3200 bytes at 16 kHz, one. The oldest is dropped."""
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, prelaunch_max_seconds=0.1, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    params = PipelineParams(audio_out_sample_rate=output_rate)

    async with bidirectional_pipeline(service, pipeline_params=params) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
            await h.queue(tts_audio("c1", chunk))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.frame(TTSStoppedFrame)  # every chunk ahead of it is in the send queue
        launch.set()
        await rec.wait("on_avatar_connected")
        await until(lambda: room.session.end_utterance_count >= 1, what="the pre-launch utterance to be sent")
        await h.finish()

    sent = [c[1] for c in room.session.send_audio_calls]
    for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
        assert (chunk in sent) is (chunk in kept), f"chunk {chunk[:1]!r}: sent={chunk in sent}"
    assert service.metrics.prelaunch_dropped_bytes == (3 - len(kept)) * len(AUDIO_1)


# --- interruption ----------------------------------------------------------------------------------


@pytest.mark.unit
async def test_interruption_session_active() -> None:
    """An `InterruptionFrame` in SESSION_ACTIVE: queued audio is cleared, `interrupt` reaches the session
    before any new `send_audio`, a late stale-context chunk never reaches the session, the new context
    flows, and a speaking avatar gets an immediate `BotStoppedSpeakingFrame`. A `UserStartedSpeakingFrame`
    adds NO interrupt: only `InterruptionFrame` interrupts.

    `hold_send` parks the first chunk inside the fake so the interruption finds audio still queued.
    `stop_drain_timeout=60` is longer than the 30 s test timeout: if the interruption path failed to
    release the playback-idle gate, `stop()` would hang and the test timeout would be the failure.
    """
    hold = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=60.0)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        for _ in range(3):
            await h.queue(tts_audio("c1", AUDIO_1))
        await until(
            lambda: len(session.send_audio_calls) == 1, what="the first c1 chunk (held) at the session"
        )
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)

        await h.queue(UserStartedSpeakingFrame())
        await h.frame(UserStartedSpeakingFrame)
        assert session.interrupt_count == 0, "UserStartedSpeakingFrame must not interrupt"

        await h.queue(InterruptionFrame())
        # no playback_finished was ever emitted, so the only source of a Stopped is the interruption path
        await h.frame(BotStoppedSpeakingFrame)
        await until(lambda: session.interrupt_count == 1, what="interrupt at the session")
        hold.set()  # the held send was cancelled with its task; later sends pass straight through

        await h.queue(tts_audio("c1", AUDIO_1))  # late chunk of the interrupted context: stale, dropped
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await until(
            lambda: any(c[1] == AUDIO_2 for c in session.send_audio_calls), what="the c2 chunk at the session"
        )

        names = [c[0] for c in session.calls]
        first_interrupt = names.index("interrupt")
        after = [c for c in session.calls[first_interrupt + 1 :] if c[0] == "send_audio"]
        assert [c[1] for c in after] == [AUDIO_2], "only the new context may be sent after the interrupt"
        assert [c[1] for c in session.send_audio_calls] == [AUDIO_1, AUDIO_2]
        assert session.interrupt_count == 1

        await h.queue(InterruptionFrame())  # clears c2 and releases the idle gate; stop() below is prompt
        await until(lambda: session.interrupt_count == 2, what="second interrupt")
        await h.finish()

    assert count(h.down, BotStartedSpeakingFrame) == 1
    assert count(h.down, BotStoppedSpeakingFrame) == 1
    assert_absent(h.down, TTSAudioRawFrame)
    assert_absent(h.up, ErrorFrame)


def _audio_work_after_interrupt(session: FakeAvatarSession, n: int) -> list[tuple[str, int | None]]:
    """`_audio_work` for the calls recorded after the session's `n`-th completed `interrupt`."""
    interrupts = [i for i, c in enumerate(session.calls) if c[0] == "interrupt"]
    if len(interrupts) < n:
        return []
    after = session.calls[interrupts[n - 1] + 1 :]
    return [
        (c[0], c[1][0] if c[0] == "send_audio" else None)
        for c in after
        if c[0] in ("send_audio", "end_utterance")
    ]


@pytest.mark.unit
async def test_a_late_stop_of_a_stale_context_does_not_end_the_next_utterance() -> None:
    """A stale context's late `TTSStoppedFrame` arriving in the middle of the next utterance queues no
    utterance end: the new context goes out as one byte stream, ended only by its own stop. The late stop
    is still forwarded (*frame contract*)."""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: AUDIO_1 in _sent(session), what="the c1 chunk at the session")
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 1, what="interrupt at the session")

        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await h.queue(tts_audio("c1", AUDIO_1))  # late chunk of the interrupted context
        await h.queue(TTSStoppedFrame(context_id="c1"))  # and its late end, mid-c2
        await h.queue(tts_audio("c2", AUDIO_3))
        await h.queue(TTSStoppedFrame(context_id="c2"))
        await until(
            lambda: (
                _audio_work_after_interrupt(session, 1)[-1:] == [("end_utterance", None)]
                and AUDIO_3 in _sent(session)
            ),
            what="c2 sent and ended",
        )
        assert _audio_work_after_interrupt(session, 1) == [
            ("send_audio", AUDIO_2[0]),
            ("send_audio", AUDIO_3[0]),
            ("end_utterance", None),
        ], "c2 must go out as one byte stream, with no end between its chunks"
        await h.finish()

    stops = [f.context_id for f in frames_of(h.down, TTSStoppedFrame)]
    assert "c1" in stops, "the late stop is still forwarded"
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_a_late_start_of_a_stale_context_keeps_the_current_context_interruptible() -> None:
    """A stale context's late `TTSStartedFrame` arriving while the next context streams leaves that
    context current, so the next interruption marks it stale too: its audio after that interruption never
    reaches the avatar."""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="a"))
        await h.queue(tts_audio("a", AUDIO_1))
        await until(lambda: AUDIO_1 in _sent(session), what="the a chunk at the session")
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 1, what="the first interrupt")

        await h.queue(TTSStartedFrame(context_id="b"))
        await h.queue(tts_audio("b", AUDIO_2))
        await until(lambda: AUDIO_2 in _sent(session), what="the first b chunk at the session")
        await h.queue(TTSStartedFrame(context_id="a"))  # late start of the interrupted context
        # InterruptionFrame is a system frame and would overtake the late start still in the queue
        await processed(h, "late start handled")
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 2, what="the second interrupt")

        await h.queue(tts_audio("b", AUDIO_3))  # b was interrupted: stale, dropped
        await h.queue(TTSStartedFrame(context_id="c"))
        await h.queue(tts_audio("c", AUDIO_1))
        # FIFO: once c's chunk is at the session, a b chunk queued before it would have been sent first
        await until(
            lambda: ("send_audio", AUDIO_1[0]) in _audio_work_after_interrupt(session, 2),
            what="the c chunk after the second interrupt",
        )
        assert ("send_audio", AUDIO_3[0]) not in _audio_work(session), (
            "b's audio after the second interruption reached the avatar"
        )
        await h.finish()

    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_interruption_launching() -> None:
    """An `InterruptionFrame` while LAUNCHING: no `interrupt` call (there is no session yet), the
    pre-launch audio already queued is kept (*pre-launch interruption*: nothing has played), and the send
    loop drains it, then what came after, once SESSION_ACTIVE. The cases around it are in
    `test_a_greeting_queued_before_the_session_is_active.py`."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, stop_drain_timeout=0.2)

    async with bidirectional_pipeline(service) as h:
        assert service.state is ServiceState.LAUNCHING
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.frame(TTSStoppedFrame)
        await h.queue(InterruptionFrame())
        await h.frame(InterruptionFrame)
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await h.queue(TTSStoppedFrame(context_id="c2"))
        await until(lambda: count(h.down, TTSStoppedFrame) == 2, what="both utterances processed")
        assert room.session.calls == []

        gate.set()
        await until(lambda: room.session.end_utterance_count >= 2, what="both utterances sent")
        await h.finish()

    assert [c[1] for c in room.session.send_audio_calls] == [AUDIO_1, AUDIO_2]
    assert room.session.interrupt_count == 0


@pytest.mark.unit
async def test_double_interruption_cancels_previous_interrupt_task() -> None:
    """Two `InterruptionFrame` s back to back: the first in-flight `interrupt()` is cancelled by the
    second (`hold_interrupt` keeps it in flight), exactly one interrupt completes, nothing raises, and
    audio flows afterwards."""
    hold = asyncio.Event()
    session = FakeAvatarSession(hold_interrupt=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(InterruptionFrame())
        await h.queue(InterruptionFrame())
        await until(lambda: count(h.down, InterruptionFrame) == 2, what="both interruptions forwarded")
        await until(lambda: session.interrupt_entered >= 1, what="an interrupt in flight")
        assert session.interrupt_count == 0, "the held interrupt cannot have completed"

        hold.set()
        await until(lambda: session.interrupt_count == 1, what="exactly one interrupt completing")
        await h.queue(TTSStartedFrame(context_id="c3"))
        await h.queue(tts_audio("c3", AUDIO_3))
        await until(lambda: any(c[1] == AUDIO_3 for c in session.send_audio_calls), what="audio afterwards")
        assert session.interrupt_count == 1
        await h.finish()

    assert_absent(h.up, ErrorFrame)


class _HoldBotStopped(FrameProcessor):
    """Parks the first downstream `BotStoppedSpeakingFrame`. The service pushes it from inside its
    interruption handling, so while it is parked the interruption has advanced its generation but not yet
    started its own `interrupt()`."""

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True, name="hold-bot-stopped")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, BotStoppedSpeakingFrame)
            and direction == FrameDirection.DOWNSTREAM
            and not self.entered.is_set()
        ):
            self.entered.set()
            await self.release.wait()
        await self.push_frame(frame, direction)


@pytest.mark.unit
async def test_overlapping_interruptions_both_reach_the_avatar_and_leave_one_working_send_loop() -> None:
    """The first `interrupt()` completes only once a second interruption is under way: both reach the avatar
    worker, the next audio is sent exactly once, and nothing is left running.

    It does not isolate `_interrupt_and_resume`'s own *generation* check, which is defence in depth: a stale
    send loop that check let start would exit at `_send_loop`'s own generation check before taking an item."""
    hold = asyncio.Event()
    session = FakeAvatarSession(hold_interrupt=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    parker = _HoldBotStopped()

    async with bidirectional_pipeline(Pipeline([service, parker])) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_entered == 1, what="the first interrupt in flight")

        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        await h.queue(InterruptionFrame())
        await until(parker.entered.is_set, what="the second interruption parked mid-way")

        hold.set()
        await until(lambda: session.interrupt_count == 1, what="the first interrupt completing")
        parker.release.set()
        await until(lambda: session.interrupt_count == 2, what="the second interrupt completing")

        await h.queue(TTSStartedFrame(context_id="c3"))
        await h.queue(tts_audio("c3", AUDIO_3))
        await until(lambda: AUDIO_3 in _sent(session), what="audio after both interruptions")
        await h.finish()

    assert session.interrupt_entered == 2
    assert _sent(session) == [AUDIO_3]
    assert_absent(h.up, ErrorFrame)


# --- terminal states -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("terminal", ["ended", "degraded"])
async def test_terminal_states_pass_through_stop_and_interruption(terminal: str) -> None:
    """In ENDED (after the avatar left) and DEGRADED: TTS started/audio/stopped and the interruption all
    pass through downstream, the session gets no new work, and nothing is left running (the observable
    form of "no new send loop"; *frame contract*)."""
    if terminal == "ended":
        room = FakeAvatarRoom(JoinMode.DIRECT)
        service = make_service(room)
    else:
        room = FakeAvatarRoom(
            JoinMode.DIRECT, fail_with=SynthesiaError("bad key", type=ErrorType.AUTH, status=401)
        )
        service = make_service(room, required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        if terminal == "ended":
            await rec.wait("on_avatar_connected")
            room.session.emit_avatar_left("avatar_left_room", humans_present=True)
            await rec.wait("on_session_ended")
            assert service.state is ServiceState.ENDED
        else:
            await rec.wait("on_session_ended")
            assert service.state is ServiceState.DEGRADED
        work_before = _work(room.session)

        await h.queue(TTSStartedFrame(context_id="c9"))
        await h.queue(tts_audio("c9", AUDIO_3))
        await h.queue(TTSStoppedFrame(context_id="c9"))
        await h.frame(TTSStoppedFrame)  # the three are processed in order before the interruption
        await h.queue(InterruptionFrame())
        await h.frame(InterruptionFrame)
        await h.finish()

    assert [f.audio for f in frames_of(h.down, TTSAudioRawFrame) if f.audio == AUDIO_3] == [AUDIO_3]
    assert_ordered(h.down, [TTSStartedFrame, TTSStoppedFrame, InterruptionFrame])
    assert _work(room.session) == work_before


# --- graceful stop -------------------------------------------------------------------------------


# The drain tests: `stop_drain_timeout=20` is four times `until()`'s 5 s, so every close they observe was
# caused by the avatar going idle, never by the bound; it stays under the 30 s test timeout, so a stop
# that never sees the avatar idle still ends the pipeline and fails cleanly.
DRAIN_BOUND = 20.0
HYSTERESIS = 0.05


@pytest.mark.unit
async def test_a_graceful_stop_waits_for_audio_the_avatar_has_not_started_playing() -> None:
    """Audio sent counts as busy even before `playback_started`: after the drain the session stays open,
    and closes once `playback_finished` ends the speaking state."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, stop_drain_timeout=DRAIN_BOUND, bot_stopped_hysteresis_secs=HYSTERESIS)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await until(lambda: session.end_utterance_count == 1, what="the utterance sent")

        await h.queue(EndFrame())
        await until(lambda: session.end_utterance_count == 2, what="the send queue drained by the stop")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        assert session.aclose_count == 0, "the stop must still be waiting for the avatar"

        session.emit_playback_finished(1.0, False)
        await until(lambda: session.aclose_count == 1, what="the session closed once the avatar is idle")
        await h.finish(send_end_frame=False)

    assert count(h.down, BotStartedSpeakingFrame) == 1
    assert count(h.down, BotStoppedSpeakingFrame) == 1
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"


@pytest.mark.unit
async def test_a_graceful_stop_waits_while_the_avatar_is_still_speaking() -> None:
    """Speaking again after one playback ended, with nothing left to send: after the drain the session
    stays open while `BotSpeakingFrame`s keep coming, and closes at the last `playback_finished`."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(
        room,
        stop_drain_timeout=DRAIN_BOUND,
        bot_stopped_hysteresis_secs=HYSTERESIS,
        bot_speaking_interval_secs=0.02,
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await until(lambda: session.end_utterance_count == 1, what="the utterance sent")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        session.emit_playback_finished(1.0, False)
        await h.frame(BotStoppedSpeakingFrame)

        session.emit_playback_started()
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 2, what="speaking again")
        await h.queue(EndFrame())
        await until(lambda: session.end_utterance_count == 2, what="the send queue drained by the stop")
        speaking = count(h.down, BotSpeakingFrame)
        await until(
            lambda: count(h.down, BotSpeakingFrame) > speaking,
            what="the avatar still speaking after the drain",
        )
        assert session.aclose_count == 0, "the stop must still be waiting for the avatar"

        session.emit_playback_finished(2.0, False)
        await until(lambda: session.aclose_count == 1, what="the session closed once the avatar is idle")
        await h.finish(send_end_frame=False)

    assert count(h.down, BotStoppedSpeakingFrame) == 2


@pytest.mark.unit
async def test_a_graceful_stop_drains_every_queued_utterance() -> None:
    """Two utterances queued behind a held chunk when the pipeline ends are both written and ended before
    the close, even though the avatar goes idle after the first.

    `hold_send` holds the first chunk until the stop begins (a synchronous `on_before_process_frame` handler
    sees the `EndFrame`); a second hold, installed at the first utterance end, holds the second utterance's
    chunk until the avatar is idle."""
    first_hold = asyncio.Event()
    second_hold = asyncio.Event()
    session = FakeAvatarSession(hold_send=first_hold)

    def on_call(call: tuple) -> None:
        if call[0] == "end_utterance" and session.end_utterance_count == 1:
            session.hold_send = second_hold

    session.on_call = on_call
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=DRAIN_BOUND, bot_stopped_hysteresis_secs=HYSTERESIS)
    rec = EventRecorder(service)

    def release_first_chunk_as_the_stop_begins(_processor: FrameProcessor, frame: Frame) -> None:
        if isinstance(frame, EndFrame):
            first_hold.set()

    service.add_event_handler("on_before_process_frame", release_first_chunk_as_the_stop_begins)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await _utterance(h, "c1", AUDIO_1)
        await _utterance(h, "c2", AUDIO_2)
        await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk held")
        await until(lambda: count(h.down, TTSStoppedFrame) == 2, what="both utterances queued")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)

        await h.queue(EndFrame())
        await until(lambda: len(session.send_audio_calls) == 2, what="the second chunk held")
        session.emit_playback_finished(1.0, False)
        await h.frame(BotStoppedSpeakingFrame)  # the avatar is idle; the second utterance is not done
        second_hold.set()
        await until(lambda: session.aclose_count == 1, what="the session closed after the drain")
        await h.finish(send_end_frame=False)

    names = [c[0] for c in session.calls if c[0] in ("send_audio", "end_utterance", "aclose")]
    assert names.count("end_utterance") == 3, names  # two utterances and the stop's own end
    assert names[-1] == "aclose", names
    assert _sent(session) == [AUDIO_1, AUDIO_2]


@pytest.mark.unit
async def test_an_interruption_during_the_stop_hysteresis_leaves_no_pending_stop() -> None:
    """An interruption inside the stop hysteresis cancels the pending stop, so it cannot later release the
    stop's wait: after the old hysteresis would have elapsed (twice over, `h.sleep`; 0.5 s is about 10x
    the steps it races) the stop still waits for the new utterance to play."""
    hysteresis = 0.5
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, stop_drain_timeout=DRAIN_BOUND, bot_stopped_hysteresis_secs=hysteresis)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await _utterance(h, "c1", AUDIO_1)
        await until(lambda: session.end_utterance_count == 1, what="the first utterance sent")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)

        finished = session.emit_playback_finished(1.0, False)
        assert finished is not None
        await finished  # the stop hysteresis is now running
        await h.queue(InterruptionFrame())
        await h.frame(BotStoppedSpeakingFrame)  # at once, from the interruption
        await until(lambda: session.interrupt_count == 1, what="interrupt at the session")
        await _utterance(h, "c2", AUDIO_2)
        await until(lambda: session.end_utterance_count == 2, what="the new utterance sent")
        await h.queue(EndFrame())
        await until(lambda: session.end_utterance_count == 3, what="the send queue drained by the stop")

        await h.sleep(2 * hysteresis)
        assert session.aclose_count == 0, "the stop must still be waiting for the new utterance to play"
        session.emit_playback_started()
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 2, what="the new utterance playing")
        session.emit_playback_finished(2.0, False)
        await until(lambda: session.aclose_count == 1, what="the session closed once the avatar is idle")
        await h.finish(send_end_frame=False)

    assert count(h.down, BotStoppedSpeakingFrame) == 2


@pytest.mark.unit
async def test_stop_drain_times_out_without_playback() -> None:
    """`stop_drain_timeout=0.2` with a worker that never reports playback: `stop()` still completes and
    the session is closed once (not timed: completing is the outcome)."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await until(lambda: room.session.send_audio_calls, what="audio at the session")
        await h.finish()

    assert room.session.aclose_count == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"
    assert count(h.down, EndFrame) == 1


@pytest.mark.unit
async def test_graceful_stop_sends_audio_still_in_the_send_queue() -> None:
    """Audio still queued when the `EndFrame` arrives reaches the avatar worker, with its utterance end,
    before the session is closed, although nothing is playing.

    `hold_interrupt` keeps the avatar worker from acknowledging an interruption, and the send loop restarts
    only once it has, so the next utterance is still queued when the pipeline ends. `h.sleep(0.3)` is an
    ordering margin, not a synchronisation: nothing observable marks `stop()` having begun its drain (the
    `EndFrame` is forwarded only once `stop()` returns), so the acknowledgement is released 0.3 s after the
    `EndFrame` is queued, when a stop that skipped the drain would already have closed the session.
    `stop_drain_timeout` keeps its 5 s default, over 10x that margin.
    """
    acknowledge = asyncio.Event()
    calls: list[str] = []
    session = FakeAvatarSession(hold_interrupt=acknowledge)

    def on_call(call: tuple) -> None:
        calls.append(call[0])
        if call[0] == "end_utterance" and session.end_utterance_count == 1:
            session.emit_playback_started()
            session.emit_playback_finished(1.0, False)

    session.on_call = on_call
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, bot_stopped_hysteresis_secs=0.01)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(InterruptionFrame())
        await until(
            lambda: session.interrupt_entered == 1, what="the interruption to reach the avatar worker"
        )
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await h.queue(TTSStoppedFrame(context_id="c2"))
        await h.frame(TTSStoppedFrame)  # the chunk ahead of it is in the send queue
        assert session.send_audio_calls == []

        await h.queue(EndFrame())
        await h.sleep(0.3)  # the ordering margin: see the docstring
        acknowledge.set()
        await h.finish(send_end_frame=False)

    assert AUDIO_2 in [c[1] for c in session.send_audio_calls]
    closed_at = calls.index("aclose")
    assert "send_audio" in calls[:closed_at], f"the queued chunk must reach the avatar worker first: {calls}"
    sent_at = calls.index("send_audio")
    assert "end_utterance" in calls[sent_at:closed_at], f"and so must the end of its utterance: {calls}"
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"


@pytest.mark.unit
async def test_graceful_stop_is_bounded_when_the_avatar_worker_never_takes_the_audio() -> None:
    """`stop_drain_timeout` bounds the drain of a chunk the avatar worker never accepts, and nothing queued
    behind that chunk is written once the stop gives up.

    The wait ends at the close or at a second write, whichever comes first. A `CancelFrame` does not cut a
    graceful stop short, so the `finally` releases the chunk: a stop that ignored its bound then fails the
    wait instead of hanging the run."""
    hold = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _utterance(h, "c1", AUDIO_1, AUDIO_2, AUDIO_3)
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk stuck")
            await h.frame(TTSStoppedFrame)
            await h.queue(EndFrame())
            await until(
                lambda: session.aclose_count == 1 or len(session.send_audio_calls) > 1,
                what="the session closed after the stop (or a second write)",
            )
            assert _sent(session) == [AUDIO_1], "audio was written after the pipeline began ending"
            await rec.wait("on_session_ended")
            await h.finish(send_end_frame=False)
        finally:
            hold.set()

    assert _sent(session) == [AUDIO_1]
    assert session.end_utterance_count == 0
    assert session.aclose_count == 1
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"


@pytest.mark.unit
async def test_graceful_stop_with_nothing_sent_tears_down_at_once() -> None:
    """With nothing ever sent there is no playback to wait for. `stop_drain_timeout=60` is far past
    `until()`'s 5 s, and the stop runs outside a pipeline so the `finally` can cancel a stop that waited."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=60.0)
    rec = EventRecorder(service)
    setup, task_manager = make_setup()
    await service.setup(setup)
    await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the session to be active")

    stop = asyncio.create_task(service.stop(EndFrame()))
    try:
        await until(stop.done, what="stop() to return without waiting for a playback")
    finally:
        stop.cancel()  # a no-op once it has returned
        with contextlib.suppress(asyncio.CancelledError):
            await stop
        await service.cleanup()

    assert rec.payload("on_session_ended")[0] == "pipeline_ended"
    assert room.session.aclose_count == 1
    assert_no_dangling(task_manager)


# --- interruption edge cases ---------------------------------------------------------------------


@pytest.mark.unit
async def test_s2s_frames_without_context_id() -> None:
    """Speech-to-speech services emit TTS frames with no `context_id`: `None` is never treated as stale,
    so audio after an interruption still reaches the session."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame())
        await h.queue(tts_audio(None, AUDIO_1))
        await until(lambda: len(room.session.send_audio_calls) == 1, what="first S2S chunk")
        await h.queue(InterruptionFrame())
        await until(lambda: room.session.interrupt_count == 1, what="interrupt")
        await h.queue(TTSStartedFrame())
        await h.queue(tts_audio(None, AUDIO_2))
        await until(
            lambda: any(c[1] == AUDIO_2 for c in room.session.send_audio_calls), what="second S2S chunk"
        )
        await h.queue(TTSStoppedFrame())
        await h.frame(TTSStoppedFrame)
        await h.finish()

    assert_absent(h.up, ErrorFrame)
    assert_absent(h.down, TTSAudioRawFrame)


@pytest.mark.unit
async def test_unexpected_send_failure_keeps_the_send_loop_and_warns_once(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A session whose `send_audio` raises (`send_error`, which the real session never does) must not kill
    the send loop: once it stops raising, the next utterance still reaches the session. Three failures of
    one exception type log one WARNING, and no ErrorFrame is pushed."""
    session = FakeAvatarSession(send_error=RuntimeError("bug in the send path"))
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        for _ in range(3):
            await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 3, what="three failing sends")
        await h.queue(TTSStoppedFrame())

        session.send_error = None
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await until(
            lambda: any(c[1] == AUDIO_2 for c in session.send_audio_calls),
            what="the next utterance, after the failures",
        )
        await h.finish()

    assert_absent(h.up, ErrorFrame)
    assert len(records_at(caplog_loguru, logging.WARNING, "audio send failed unexpectedly")) == 1
