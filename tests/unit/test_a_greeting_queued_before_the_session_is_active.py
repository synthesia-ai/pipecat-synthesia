"""A greeting queued while the avatar launches survives the user speaking before the session is active.

Pipecat sends an `InterruptionFrame` at every user turn start, even when the bot is not speaking. The
avatar takes seconds to launch, so a user who speaks early sends one while the greeting's audio is still
pre-launch audio. Nothing of the greeting has played, so the first audio held when the user speaks is kept
(*pre-launch interruption*); only the rest of an utterance still being generated is dropped, as stale, and a
later turn drops what was queued since.

`FakeAvatarRoom(hold_launch=)` keeps the service launching until the test lets it go. The answer to the
user's question, queued after the interruption, is the marker that the queue has drained: the send queue is
a FIFO, so once the answer's utterance end is at the avatar worker, everything queued ahead of it has been
sent or dropped. Each test asserts the worker's exact work order, which also pins that the utterance end the
interruption adds is added only when an utterance is open: an extra one would show as a second end.

The join mode is parametrized, as both reach the same code path; so is the context id: a cascade TTS service
sets one per utterance, and OpenAI Realtime sets none. Without context ids nothing is stale, so the rest of an
interrupted utterance is not dropped; only a later user turn drops it, with the answer it superseded.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from pipecat.frames.frames import InterruptionFrame, TTSStartedFrame, TTSStoppedFrame

from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, records_at, tts_audio

GREETING = AUDIO_1
ANSWER = AUDIO_2
END = "end_utterance"


async def _utterance(h: Harness, context_id: str | None, audio: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    await h.queue(tts_audio(context_id, audio))
    await h.queue(TTSStoppedFrame(context_id=context_id))


def _work(room: FakeAvatarRoom) -> list[bytes | str]:
    """The avatar worker's audio work in order: each chunk sent, and each utterance end."""
    return [c[1] if c[0] == "send_audio" else END for c in room.session.calls if c[0] in ("send_audio", END)]


def _answer_ended(room: FakeAvatarRoom) -> bool:
    work = _work(room)
    return ANSWER in work and END in work[work.index(ANSWER) :]


async def _launch_and_answer(
    h: Harness, room: FakeAvatarRoom, launch: asyncio.Event, rec: EventRecorder
) -> None:
    """Let the launch finish, then queue the answer and wait until the worker has it, ended."""
    launch.set()
    await rec.wait("on_avatar_connected")
    await _utterance(h, "answer", ANSWER)
    await until(lambda: _answer_ended(room), what="the answer and its utterance end at the avatar worker")


@pytest.mark.unit
@pytest.mark.parametrize("join_mode", [JoinMode.DIRECT, JoinMode.RELAYED], ids=["direct", "relayed"])
@pytest.mark.parametrize(
    "greeting_context", ["greeting", None], ids=["cascade-context-id", "realtime-no-context-id"]
)
async def test_the_greeting_reaches_the_avatar_worker_after_a_user_turn_starts_during_the_launch(
    join_mode: JoinMode, greeting_context: str | None, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    launch = asyncio.Event()
    room = FakeAvatarRoom(join_mode, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        assert service.state is ServiceState.LAUNCHING
        await _utterance(h, greeting_context, GREETING)
        await processed(h, "the greeting queued")
        await h.queue(InterruptionFrame())  # the user's turn starts
        await h.frame(InterruptionFrame)
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:4] == [GREETING, END, ANSWER, END], "the greeting must play once, before the answer"
    assert room.session.interrupt_count == 0, "nothing had played, so there was nothing to clear"
    assert service.metrics.prelaunch_dropped_bytes == 0
    assert len(records_at(caplog_loguru, logging.INFO, "pre-launch audio")) == 1


@pytest.mark.unit
async def test_the_rest_of_an_utterance_still_generating_is_dropped_and_the_queued_part_ends_alone() -> None:
    """The user's turn starts mid-greeting, while the TTS service is still producing it: the audio queued
    before the interruption is kept and ends as its own utterance; the greeting's later chunk and its late
    stop are the interrupted context's (*stale context*), so neither reaches the avatar worker as audio nor
    ends the answer early."""
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="greeting"))
        await h.queue(tts_audio("greeting", GREETING))
        await processed(h, "the greeting's first chunk queued")
        await h.queue(InterruptionFrame())
        await h.frame(InterruptionFrame)
        await h.queue(tts_audio("greeting", AUDIO_3))  # the rest of the greeting, late
        await h.queue(TTSStoppedFrame(context_id="greeting"))
        await processed(h, "the greeting's late audio and stop")
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:4] == [GREETING, END, ANSWER, END], (
        "only the greeting's first chunk is sent, ended before the answer"
    )


@pytest.mark.unit
async def test_a_user_turn_during_the_launch_with_nothing_queued_adds_no_utterance_end(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(InterruptionFrame())
        await h.frame(InterruptionFrame)
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:2] == [ANSWER, END], "a stray utterance end reached the avatar worker"
    assert records_at(caplog_loguru, logging.INFO, "pre-launch audio") == [], "nothing was kept"


@pytest.mark.unit
async def test_an_answer_a_later_user_turn_superseded_during_the_launch_is_dropped(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A slow launch, and a user who says "hello?" twice: the greeting, queued before the user first spoke,
    is kept; the answer to the first turn, which the second turn superseded, is dropped as an interruption
    drops unplayed audio once the session is active; the answer to the second turn follows the greeting."""
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _utterance(h, "greeting", GREETING)
        await processed(h, "the greeting queued")
        await h.queue(InterruptionFrame())  # the user's first turn
        await h.frame(InterruptionFrame)
        await _utterance(h, "first answer", AUDIO_3)
        await processed(h, "the first answer queued")
        await h.queue(InterruptionFrame())  # the user's second turn
        await processed(h, "the second turn")
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:4] == [GREETING, END, ANSWER, END], "the superseded answer must not play"
    assert len(records_at(caplog_loguru, logging.INFO, "pre-launch audio")) == 1, (
        "INFO at the first turn only"
    )
    assert len(records_at(caplog_loguru, logging.DEBUG, "superseded")) == 1


@pytest.mark.unit
async def test_a_greeting_queued_after_a_turn_that_found_nothing_queued_is_kept_through_the_next() -> None:
    """The user's first turn finds nothing queued, so it keeps nothing and sets no boundary. The greeting
    queued after it is the first audio held when the user speaks again, so that turn keeps it, and the answer
    follows."""
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(InterruptionFrame())  # the user's first turn, with nothing queued
        await h.frame(InterruptionFrame)
        await _utterance(h, "greeting", GREETING)
        await processed(h, "the greeting queued")
        await h.queue(InterruptionFrame())  # the user's second turn
        await processed(h, "the second turn")
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:4] == [GREETING, END, ANSWER, END], "a greeting queued after the first turn must play"


@pytest.mark.unit
async def test_a_turn_that_finds_only_an_empty_utterance_sets_no_boundary() -> None:
    """A TTS request that times out produces a `TTSStartedFrame` and a `TTSStoppedFrame` with no audio
    between them, which queues an utterance end and nothing else. A turn that finds only that has found no
    audio, so it keeps nothing and sets no boundary, and the greeting queued after it is kept by the next
    turn. The empty utterance's own end reaches the worker first, and closes nothing there."""
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="timed out"))
        await h.queue(TTSStoppedFrame(context_id="timed out"))
        await processed(h, "an utterance with no audio")
        await h.queue(InterruptionFrame())  # the user's first turn
        await h.frame(InterruptionFrame)
        await _utterance(h, "greeting", GREETING)
        await processed(h, "the greeting queued")
        await h.queue(InterruptionFrame())  # the user's second turn
        await processed(h, "the second turn")
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:5] == [END, GREETING, END, ANSWER, END], "the greeting must play before the answer"


@pytest.mark.unit
async def test_without_a_greeting_the_first_reply_held_is_kept_as_if_it_were_the_opening() -> None:
    """The accepted cost of the rule. The plugin cannot tell a greeting from a reply, so it keeps the first
    audio held when the user speaks. For a bot with no greeting, spoken over twice during the launch, that is
    the reply to the first turn: it plays before the reply to the latest. A bot with a greeting pays the same
    way when the user's first turn comes before the greeting: the reply to it is kept with the greeting. Here
    the third turn still drops what was queued after the kept reply; replies that arrive together, before any
    later turn, would all be kept, which Pipecat normally prevents by cancelling a reply still being produced
    when the user speaks again."""
    launch = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=launch)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(InterruptionFrame())  # the user's first turn, with nothing queued
        await h.frame(InterruptionFrame)
        await _utterance(h, "first answer", AUDIO_3)
        await processed(h, "the first answer queued")
        await h.queue(InterruptionFrame())  # the user's second turn
        await processed(h, "the second turn")
        await _utterance(h, "superseded", AUDIO_1)
        await processed(h, "a reply the third turn supersedes")
        await h.queue(InterruptionFrame())  # the user's third turn
        await processed(h, "the third turn")
        await _launch_and_answer(h, room, launch, rec)
        await h.finish()

    assert _work(room)[:4] == [AUDIO_3, END, ANSWER, END]
