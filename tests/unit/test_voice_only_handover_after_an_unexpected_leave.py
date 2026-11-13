"""The voice-only handover after an unexpected leave, when the service's leave handling yields part-way.

When the avatar leaves while humans remain, the bot continues voice-only: the audio still in the send
queue is pushed to the user transport, and later TTS audio passes through. These tests pin that the
handover keeps the bot's speech whole and in order even if the leave handling yields to the event loop
before the handover is done:

* audio that was queued at the leave reaches the transport before any audio that arrived after it;
* audio that was queued at the leave is handed over, not written to the session that is closing.

`_on_avatar_left` sets ENDED and then awaits `_stop_speaking_now()` before `_end_session()` takes the
send queue, and `_end_session()` awaits `push_error()` before it pushes the queued audio. Each of those
awaits yields only if something on its path does. The tests make each one yield with a Pipecat event
handler on the service, the way a developer's handler would: a sync `on_before_push_frame` handler that
awaits when the leave's `BotStoppedSpeakingFrame` is pushed, or an `on_error` handler that awaits when the
leave's error is pushed. `FakeAvatarSession(hold_send=)` keeps the first chunk in flight at the avatar
worker, so the chunks after it are still in the send queue when the avatar leaves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio

#: A chunk of the next response, which starts after the avatar has left.
AUDIO_4 = b"\x04\x00" * 960

Handler = Callable[[object, Frame], Awaitable[None]]


def _hold_the_first(frame_type: type[Frame], entered: asyncio.Event, release: asyncio.Event) -> Handler:
    """A handler that, the first time it sees a `frame_type`, waits for `release`; otherwise returns."""
    held = False

    async def hold(_processor: object, frame: Frame) -> None:
        nonlocal held
        if held or not isinstance(frame, frame_type):
            return
        held = True
        entered.set()
        await release.wait()

    return hold


async def _queue_one_utterance_with_its_first_chunk_in_flight(h: Harness, session: FakeAvatarSession) -> None:
    await h.queue(TTSStartedFrame(context_id="c1"))
    for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
        await h.queue(tts_audio("c1", chunk))
    await processed(h, "the first utterance queued")
    await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")


_NAMES = {AUDIO_1: "AUDIO_1", AUDIO_2: "AUDIO_2", AUDIO_3: "AUDIO_3", AUDIO_4: "AUDIO_4"}


def _named(chunks: list[bytes]) -> list[str]:
    """Chunks by name, so a failure shows their order rather than 2 KB of bytes each."""
    return [_NAMES.get(chunk, f"<{len(chunk)} other bytes>") for chunk in chunks]


def _audio_downstream(h: Harness) -> list[str]:
    return _named([f.audio for f in frames_of(h.down, TTSAudioRawFrame) if isinstance(f, TTSAudioRawFrame)])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event", "held_frame"),
    [
        pytest.param("on_before_push_frame", BotStoppedSpeakingFrame, id="bot-stopped-speaking-push-yields"),
        pytest.param("on_error", ErrorFrame, id="error-handler-yields"),
    ],
)
async def test_audio_queued_at_an_unexpected_leave_reaches_the_transport_before_later_audio(
    event: str, held_frame: type[Frame]
) -> None:
    """The order starts at the chunk that was in flight: once the avatar has left, a chunk in flight at the
    avatar worker is not one anyone heard, so the handover hands it over first."""
    in_flight = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    session = FakeAvatarSession(hold_send=in_flight)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    service.add_event_handler(event, _hold_the_first(held_frame, entered, release))

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_one_utterance_with_its_first_chunk_in_flight(h, session)
            session.emit_playback_started()
            await h.frame(BotStartedSpeakingFrame)

            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(entered.is_set, what="the leave handling to yield")
            # the next response starts while the leave is still being handled
            await h.queue(TTSStartedFrame(context_id="c2"))
            await h.queue(tts_audio("c2", AUDIO_4))
            await processed(h, "the next response's first chunk")
            release.set()
            await rec.wait("on_session_ended")
        finally:
            release.set()
            in_flight.set()
        await h.finish()

    assert _audio_downstream(h) == ["AUDIO_1", "AUDIO_2", "AUDIO_3", "AUDIO_4"], (
        "the bot's speech reached the user out of order"
    )
    assert rec.payload("on_session_ended")[0] == "avatar_left"


@pytest.mark.unit
async def test_audio_queued_at_an_unexpected_leave_is_handed_over_not_sent_to_the_closing_session() -> None:
    """The send held at the leave is released while the leave handling has yielded (at the
    `BotStoppedSpeakingFrame` push). The send loop must not then take the queued chunks and write them to
    the session that is closing, where they are lost: they belong to the handover. The handover starts at the
    chunk that was in flight: once the avatar has left, a chunk in flight at the avatar worker is not one
    anyone heard."""
    in_flight = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    session = FakeAvatarSession(hold_send=in_flight)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    service.add_event_handler(
        "on_before_push_frame", _hold_the_first(BotStoppedSpeakingFrame, entered, release)
    )

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_one_utterance_with_its_first_chunk_in_flight(h, session)
            session.emit_playback_started()
            await h.frame(BotStartedSpeakingFrame)

            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(entered.is_set, what="the leave handling to yield")
            in_flight.set()  # the held send is released; the retired loop must send nothing more
            await processed(h, "the send loop has had its turn")
            release.set()
            await rec.wait("on_session_ended")
        finally:
            release.set()
            in_flight.set()
        await h.finish()

    sent = _named([call[1] for call in session.send_audio_calls])
    assert sent == ["AUDIO_1"], "queued audio was written to the session after the avatar left"
    assert _audio_downstream(h) == ["AUDIO_1", "AUDIO_2", "AUDIO_3"], "the queued audio was not handed over"
    assert rec.payload("on_session_ended")[0] == "avatar_left"
