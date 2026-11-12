"""The speaking state when the interrupted utterance's `playback_finished` arrives late.

An interruption ends the speaking state at once, with no stop hysteresis, and sends clear buffer. The
avatar worker then reports the interrupted utterance as `playback_finished(interrupted=True)`. The next
utterance's speaking state must not depend on whether that report arrives before or after the next
utterance's `playback_started`:

* finish, then start: the order the avatar worker usually sends them in;
* start, then finish: livekit-agents' `AvatarRunner` sends the interrupted finish only when
  its video generator's `clear_buffer()` returns, so a next utterance whose audio comes out of the generator
  first has its `playback_started` sent first. The late finish is about the interrupted utterance; the next
  one is still playing, so no `BotStoppedSpeakingFrame` may follow it.

Each pair is emitted back to back, as the session delivers two playback events that arrive together: two
callback tasks, in arrival order. `bot_stopped_hysteresis_secs=0.05` is the config seam; 0.5 s, ten times
it, is the "and nothing happened" wait.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, delivered, tts_audio

UP = FrameDirection.UPSTREAM
HYSTERESIS = 0.05
MARGIN = 10 * HYSTERESIS


@pytest.mark.unit
@pytest.mark.parametrize("order", ["finish_then_start", "start_then_finish"])
async def test_the_next_utterance_stays_speaking_whenever_the_interrupted_finish_arrives(order: str) -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the c1 chunk at the avatar worker")
        await delivered(session.emit_playback_started())
        await h.frame(BotStartedSpeakingFrame)

        await h.queue(InterruptionFrame())
        await h.frame(BotStoppedSpeakingFrame)
        await until(lambda: session.interrupt_count == 1, what="clear buffer at the avatar worker")

        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await until(lambda: len(session.send_audio_calls) == 2, what="the c2 chunk at the avatar worker")

        if order == "finish_then_start":
            session.emit_playback_finished(0.4, interrupted=True)
            session.emit_playback_started()
        else:
            session.emit_playback_started()
            session.emit_playback_finished(0.4, interrupted=True)
        await session.wait_emitted()
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 2, what="the c2 utterance playing")
        await h.sleep(MARGIN)  # a stop timer the late finish armed has long fired
        stopped_down = count(h.down, BotStoppedSpeakingFrame)
        stopped_up = count(h.up, BotStoppedSpeakingFrame)

        session.emit_playback_finished(1.0, interrupted=False)  # c2 ends
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) >= 2, what="the bot stopped after c2")
        await h.finish()

    assert stopped_down == 1, "BotStoppedSpeakingFrame downstream while the c2 utterance was still playing"
    assert stopped_up == 1, "BotStoppedSpeakingFrame upstream while the c2 utterance was still playing"
    assert count(h.down, BotStoppedSpeakingFrame) == 2
