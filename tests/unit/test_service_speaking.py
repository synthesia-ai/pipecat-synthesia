"""The speaking state (*speaking-state ownership*).

The service is the only emitter of `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` / `BotSpeakingFrame`,
derived from the worker's playback events with a stop hysteresis. Both sides of the hysteresis boundary are
tested with a 10x margin: a restart with no `await` in between (same tick) versus a wait of 10x the window.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TTSStartedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, tts_audio

UP = FrameDirection.UPSTREAM


@pytest.mark.unit
async def test_bot_speaking_synthesis() -> None:
    """Started once per utterance in both directions; finished (twice) then started with no yield in between
    produces no Stopped; finished then 10x the hysteresis produces exactly one Stopped; `BotSpeakingFrame`
    is periodic while speaking (>= 1, never a count) and stops after Stopped.

    `bot_stopped_hysteresis_secs=0.05` and `bot_speaking_interval_secs=0.02` are the config seams; the
    0.5 s wait is 10x the hysteresis.
    """
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, bot_stopped_hysteresis_secs=0.05, bot_speaking_interval_secs=0.02)
    rec = EventRecorder(service)
    session = room.session

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")

        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        await h.frame(BotStartedSpeakingFrame, direction=UP)
        assert count(h.down, BotStartedSpeakingFrame) == 1
        assert count(h.up, BotStartedSpeakingFrame) == 1
        await until(lambda: count(h.down, BotSpeakingFrame) >= 1, what="a periodic BotSpeakingFrame")

        # same tick: the next utterance starts before the hysteresis can fire -> no Stopped (unbounded margin)
        session.emit_playback_finished(1.0, False)
        session.emit_playback_finished(1.0, False)
        session.emit_playback_started()
        await session.wait_emitted()
        await h.sleep(0.5)
        assert count(h.down, BotStoppedSpeakingFrame) == 0
        assert count(h.down, BotStartedSpeakingFrame) == 1, "an already-speaking avatar restarts silently"

        session.emit_playback_finished(2.0, False)
        await h.frame(BotStoppedSpeakingFrame)
        await h.frame(BotStoppedSpeakingFrame, direction=UP)
        speaking_at_stop = count(h.down, BotSpeakingFrame)
        await h.sleep(0.2)  # 10x the BotSpeaking interval: a live timer would have added ~10 more
        assert count(h.down, BotSpeakingFrame) == speaking_at_stop
        await h.finish()

    assert count(h.down, BotStartedSpeakingFrame) == 1
    assert count(h.down, BotStoppedSpeakingFrame) == 1
    assert count(h.up, BotStartedSpeakingFrame) == 1
    assert count(h.up, BotStoppedSpeakingFrame) == 1
    assert service.metrics.playback_finished_count == 3


@pytest.mark.unit
async def test_ending_the_speaking_state_stops_its_periodic_timer() -> None:
    """The first speaking period's `BotSpeakingFrame` timer must stop at the interruption. With
    `bot_speaking_interval_secs=30` a timer that was not stopped is still asleep when the pipeline has
    ended, and the harness finds it running."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, bot_speaking_interval_secs=30.0, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    session = room.session

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        await h.queue(InterruptionFrame())
        await h.frame(BotStoppedSpeakingFrame)
        await until(lambda: session.interrupt_count == 1, what="interrupt at the session")

        session.emit_playback_started()
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 2, what="speaking again")
        await h.finish()

    assert count(h.down, BotStoppedSpeakingFrame) == 2


class _BlockOnStarted(FrameProcessor):
    """Direct-mode downstream processor that parks the first `BotStartedSpeakingFrame` until released."""

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True, name="blocker")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self.entered.set()
            await self.release.wait()
        await self.push_frame(frame, direction)


@pytest.mark.unit
async def test_speaking_state_atomic_under_interruption() -> None:
    """The speaking flag flips before the broadcast awaits (*state-before-await*): with a downstream
    processor holding the `BotStartedSpeakingFrame` push, an `InterruptionFrame` still produces exactly one
    Stopped, and the held Started completes later."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    blocker = _BlockOnStarted()

    async with bidirectional_pipeline(Pipeline([service, blocker])) as h:
        await rec.wait("on_avatar_connected")
        room.session.emit_playback_started()
        await asyncio.wait_for(blocker.entered.wait(), 5.0)
        assert count(h.down, BotStartedSpeakingFrame) == 0  # held inside the blocker

        await h.queue(InterruptionFrame())
        await h.frame(BotStoppedSpeakingFrame)
        await h.frame(BotStoppedSpeakingFrame, direction=UP)

        blocker.release.set()
        await h.frame(BotStartedSpeakingFrame)
        await h.frame(BotStartedSpeakingFrame, direction=UP)
        await until(lambda: room.session.interrupt_count == 1, what="interrupt")
        await h.finish()

    assert count(h.down, BotStartedSpeakingFrame) == 1
    assert count(h.down, BotStoppedSpeakingFrame) == 1
    assert count(h.up, BotStartedSpeakingFrame) == 1
    assert count(h.up, BotStoppedSpeakingFrame) == 1


@pytest.mark.unit
async def test_playback_cut_short_by_the_avatar_worker_ends_the_speaking_state() -> None:
    """A `playback_finished(interrupted=True)` the bot never asked for still ends the speaking state after
    the stop hysteresis."""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, bot_stopped_hysteresis_secs=0.01, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the chunk at the avatar worker")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        session.emit_playback_finished(0.4, interrupted=True)
        await h.frame(BotStoppedSpeakingFrame)
        await session.wait_emitted()
        await h.finish()

    assert service.metrics.playback_finished_count == 1
    assert service.metrics.interruption_latency_seconds == []
