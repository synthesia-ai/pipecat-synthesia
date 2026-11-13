"""The speaking state when the interrupted utterance's `playback_finished` outlives a whole later utterance.

An interruption ends the speaking state at once and sends clear buffer. livekit-agents' `AvatarRunner`
sends the interrupted utterance's `playback_finished(interrupted=True)` only when its video generator's
`clear_buffer()` returns. If that is slower than the whole next utterance, the avatar worker reports, in
order: the next utterance's `playback_started` and `playback_finished(interrupted=False)`, a third
utterance's `playback_started`, and only then the late interrupted finish. That finish is about the first
utterance; the third is still playing, so no `BotStoppedSpeakingFrame` may follow it.

`bot_stopped_hysteresis_secs=0.05` is the config seam; 0.5 s, ten times it, is the "and nothing happened"
wait.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSStartedFrame,
)

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, delivered, tts_audio

HYSTERESIS = 0.05
MARGIN = 10 * HYSTERESIS


@pytest.mark.unit
async def test_a_late_interrupted_finish_after_a_later_utterance_does_not_stop_the_playing_one() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        # c1 plays, and the bot interrupts it
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the c1 chunk at the avatar worker")
        await delivered(session.emit_playback_started())
        await h.frame(BotStartedSpeakingFrame)
        await h.queue(InterruptionFrame())
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) == 1, what="the interruption's stop")
        await until(lambda: session.interrupt_count == 1, what="clear buffer at the avatar worker")

        # c2 plays and finishes normally, before the avatar worker answers the clear buffer
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await until(lambda: len(session.send_audio_calls) == 2, what="the c2 chunk at the avatar worker")
        await delivered(session.emit_playback_started())
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 2, what="the c2 utterance playing")
        await delivered(session.emit_playback_finished(1.0, interrupted=False))
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) == 2, what="the bot stopped after c2")

        # c3 starts, then the interrupted c1 finish finally arrives
        await h.queue(TTSStartedFrame(context_id="c3"))
        await h.queue(tts_audio("c3", AUDIO_3))
        await until(lambda: len(session.send_audio_calls) == 3, what="the c3 chunk at the avatar worker")
        await delivered(session.emit_playback_started())
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 3, what="the c3 utterance playing")
        await delivered(session.emit_playback_finished(0.4, interrupted=True))
        await h.sleep(MARGIN)  # a stop timer the late finish armed has long fired
        stopped_down = count(h.down, BotStoppedSpeakingFrame)
        stopped_up = count(h.up, BotStoppedSpeakingFrame)

        await delivered(session.emit_playback_finished(1.0, interrupted=False))  # c3 ends
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) >= 3, what="the bot stopped after c3")
        await h.finish()

    assert stopped_down == 2, "BotStoppedSpeakingFrame downstream while the c3 utterance was still playing"
    assert stopped_up == 2, "BotStoppedSpeakingFrame upstream while the c3 utterance was still playing"
    assert count(h.down, BotStoppedSpeakingFrame) == 3
