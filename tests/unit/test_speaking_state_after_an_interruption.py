"""The speaking state never swallows a `playback_started`, and an interruption never leaves the bot speaking.

An interruption ends the speaking state at once and sends clear buffer; the avatar worker answers with the
interrupted utterance's `playback_finished(interrupted=True)`. That finish arms no stop timer only when a
chunk was sent since the interruption, because the next utterance may already be playing. With nothing
sent since, any `playback_started` since was the interrupted utterance's own, arriving late, and the
finish ends it like any other. What this file pins, on frames:

* that late start, while the clear buffer is still out and then accepted: as many Stopped as Started;
* the same with a second interruption while the first's clear buffer is still out, and with the
  interrupted finish arriving while the clear buffer is still out, which holds only because the
  interruption resets "sent since" before its first await;
* a clear buffer the worker did not apply: its start is real, and its normal finish stops it;
* the next utterance's start before the interrupted finish: the next utterance keeps speaking.

A known limit, until playback events name their byte stream (the ask in docs/protocol-compatibility.md):
if the interrupted utterance's late start arrives after the next utterance's first chunk was sent, and
that next utterance never plays or the next utterance's send failed ("sent since" is set before the send's
await), the interrupted finish is taken as answering the interruption, and the bot is left speaking.

`hold_interrupt` keeps clear buffer out; `bot_stopped_hysteresis_secs=0.05` is the config seam, and ten
of it is the "and nothing happened" wait.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, delivered, tts_audio

HYSTERESIS = 0.05
MARGIN = 10 * HYSTERESIS


def speaking_frames(h: Harness) -> tuple[int, int]:
    return count(h.down, BotStartedSpeakingFrame), count(h.down, BotStoppedSpeakingFrame)


async def interrupt_before_c1_starts(
    *, interruptions: int = 1, finished_interrupted: bool = True, finish_while_out: bool = False
) -> tuple[int, int]:
    """c1 is written; the interruptions' clear buffer is out when c1's own late start arrives; then the
    worker answers, and reports c1's finish (or reports it first, with `finish_while_out`). Returns
    (Started, Stopped) ten hystereses later."""
    session = FakeAvatarSession(hold_interrupt=asyncio.Event())
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the c1 chunk at the avatar worker")
        for n in range(1, interruptions + 1):
            await h.queue(InterruptionFrame())
            await until(lambda n=n: session.interrupt_entered == n, what=f"clear buffer {n} out")
        await delivered(session.emit_playback_started())
        await until(lambda: speaking_frames(h)[0] == 1, what="the late start speaking")
        if finish_while_out:
            await delivered(session.emit_playback_finished(0.1, interrupted=finished_interrupted))
        assert session.hold_interrupt is not None
        session.hold_interrupt.set()
        await until(lambda: session.interrupt_count == 1, what="clear buffer answered")
        if not finish_while_out:
            await delivered(session.emit_playback_finished(0.1, interrupted=finished_interrupted))
        await h.sleep(MARGIN)
        frames = speaking_frames(h)
        await h.finish()
    return frames


@pytest.mark.unit
@pytest.mark.parametrize(
    "interruptions, finish_while_out",
    [(1, False), (2, False), (1, True)],
    ids=["one-interruption", "a-second-while-pending", "finished-while-clear-buffer-out"],
)
async def test_a_late_start_while_clear_buffer_is_out_is_stopped_by_its_interrupted_finish(
    interruptions: int, finish_while_out: bool
) -> None:
    started, stopped = await interrupt_before_c1_starts(
        interruptions=interruptions, finish_while_out=finish_while_out
    )
    assert (started, stopped) == (1, 1), "the bot was left speaking"


@pytest.mark.unit
async def test_a_start_after_a_clear_buffer_the_worker_did_not_apply_is_real_and_its_finish_stops_it() -> (
    None
):
    assert await interrupt_before_c1_starts(interruptions=1, finished_interrupted=False) == (1, 1)


@pytest.mark.unit
async def test_the_next_utterance_starting_before_the_interrupted_finish_keeps_speaking() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the c1 chunk at the avatar worker")
        await delivered(session.emit_playback_started())
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 1, what="clear buffer answered")
        await h.queue(tts_audio("c2", AUDIO_2))
        await until(lambda: len(session.send_audio_calls) == 2, what="the c2 chunk at the avatar worker")
        await delivered(session.emit_playback_started())  # c2's
        await delivered(session.emit_playback_finished(0.1, interrupted=True))  # c1's, late
        await h.sleep(MARGIN)
        frames = speaking_frames(h)
        await h.finish()
    assert frames == (2, 1), "c2 was stopped by c1's interrupted finish"


@pytest.mark.unit
@pytest.mark.parametrize("stray", [False, True], ids=["no-stray-finish", "a-stray-interrupted-finish"])
async def test_an_interrupted_finish_for_discarded_early_audio_changes_nothing(stray: bool) -> None:
    """The avatar worker answers audio it discarded before binding the bot with `playback_finished(0.0,
    interrupted=True)`, before any of the bot's audio has played. With no interruption pending it ends no
    speaking state, and the utterance that plays next starts and stops exactly as it would without it."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        if stray:
            await delivered(session.emit_playback_finished(0.0, interrupted=True))
            await h.sleep(MARGIN)
        assert speaking_frames(h) == (0, 0)
        assert not service._speaking
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the c1 chunk at the avatar worker")
        if stray:  # the worker's own order: the stray finish at the bind, then the held audio's start
            await delivered(session.emit_playback_finished(0.0, interrupted=True))
        await delivered(session.emit_playback_started())
        await h.sleep(MARGIN)
        assert speaking_frames(h) == (1, 0), "the stray finish stopped the utterance that followed it"
        await delivered(session.emit_playback_finished(0.4, interrupted=False))
        await until(lambda: speaking_frames(h) == (1, 1), what="c1's finish stopping it")
        await h.sleep(MARGIN)
        frames = speaking_frames(h)
        await h.finish()
    assert frames == (1, 1)


@pytest.mark.unit
async def test_an_interruption_before_the_worker_binds_the_bot_is_answered_at_the_bind() -> None:
    """The session is active once the avatar is in the room, a little before the avatar worker binds the
    bot. A clear buffer in that window gets no finish of its own: the worker answers it at the bind, with
    the discarded audio's `playback_finished(0.0, interrupted=True)`. That answer clears the pending
    interruption and stops nothing, and the next utterance then starts and stops normally."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the c1 chunk at the avatar worker")
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 1, what="clear buffer answered")
        await delivered(session.emit_playback_finished(0.0, interrupted=True))  # at the bind
        await h.sleep(MARGIN)
        assert service._interrupt_started is None
        assert speaking_frames(h) == (0, 0)

        await h.queue(tts_audio("c2", AUDIO_2))
        await until(lambda: len(session.send_audio_calls) == 2, what="the c2 chunk at the avatar worker")
        await delivered(session.emit_playback_started())
        await until(lambda: speaking_frames(h) == (1, 0), what="c2 speaking")
        await delivered(session.emit_playback_finished(0.4, interrupted=False))
        await until(lambda: speaking_frames(h) == (1, 1), what="c2's finish stopping it")
        await h.finish()
