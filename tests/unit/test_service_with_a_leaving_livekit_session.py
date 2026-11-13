"""The service while the real LiveKit session closes its byte stream after the avatar left.

When the avatar leaves, the session stops being alive at once, but it closes the byte stream it has open
before it delivers `on_avatar_left` (*one-open-stream rule*). During that close the service is still
SESSION_ACTIVE and knows only that the session is no longer alive. This file pins what the service does
in that window:
- the audio queued at an unexpected leave survives it: it reaches the user transport through the
  voice-only handover, rather than being drained by the send loop into the session, which drops it;
- so does the chunk whose send was in flight at the leave, whether its byte stream was still opening or
  already open: the session finishes that send before it reports the leave, but the avatar has gone;
- a graceful stop in it does not wait out `stop_drain_timeout` for a drain that can no longer happen.

`test_voice_only_handover_after_an_unexpected_leave.py` pins the service's half of the handover with a fake
session. This file needs the real `LiveKitAvatarSession`, because the window is its own. It runs on the
session rig (`_session_rig.py`), behind its `SessionRigRoom`: a direct-join `AvatarRoom` that launches it
with the service's own launch context, its API swapped for the rig's fake. `GatedOpenParticipant` holds
the first chunk's byte stream open, so that chunk is in flight at the avatar worker and the chunks after
it are still in the send queue when the avatar leaves; `HeldWriteParticipant` does the same for a chunk
written on a stream that is already open.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit import rtc as lk_rtc
from pipecat.frames.frames import EndFrame, TTSAudioRawFrame, TTSStartedFrame

from pipecat_synthesia.livekit import _types
from tests.fakes.fake_rtc import FakeRemoteParticipant
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio
from tests.unit._session_rig import STANDARD, Rig, SessionRigRoom
from tests.unit._slow_byte_streams import (
    GatedOpenParticipant,
    HeldWriteParticipant,
    hold_after_the_close_request,
)

CLIENT_INITIATED = lk_rtc.DisconnectReason.CLIENT_INITIATED
_NAMES = {AUDIO_1: "AUDIO_1", AUDIO_2: "AUDIO_2", AUDIO_3: "AUDIO_3"}
#: Far longer than `until()`'s 5 s: an input, so that waiting it out shows as a timeout, never as a pass.
LONG = 60.0


@pytest.fixture(autouse=True)
def _short_no_humans_left_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The human stays in every test here, so each leave waits out the *no-humans-left window* before it is
    unexpected; shortened, so that it does not cost each test its full 2 s."""
    monkeypatch.setattr(_types, "NO_HUMANS_LEFT_WINDOW_SECS", 0.05)


def _handed_over(h: Harness) -> list[str]:
    """The voice-only audio by chunk name, so a failure shows the order rather than 2 KB of bytes each."""
    return [
        _NAMES.get(f.audio, f"<{len(f.audio)} other bytes>")
        for f in frames_of(h.down, TTSAudioRawFrame)
        if isinstance(f, TTSAudioRawFrame)
    ]


async def test_audio_queued_at_an_unexpected_leave_is_handed_over_while_the_session_closes_its_stream() -> (
    None
):
    """The handover starts at the chunk whose stream was still opening at the leave: the session lets that
    send finish before it reports the leave, but the avatar has gone, so no one heard it."""
    participant = GatedOpenParticipant()
    rig = Rig(participant=participant)
    rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))  # a human remains
    service = make_service(SessionRigRoom(rig))
    rec = EventRecorder(service)
    gate = participant.open_gate = asyncio.Event()

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(participant.streams) == 1, what="the first chunk's byte stream opening")

            assert rig.avatar is not None
            rig.leave(rig.avatar, reason=CLIENT_INITIATED)
            gate.set()  # the chunk in flight completes while the session is closing its stream
            await rec.wait("on_session_ended")
        finally:
            gate.set()
        await h.finish()

    assert _handed_over(h) == ["AUDIO_1", "AUDIO_2", "AUDIO_3"], (
        "the audio in flight or queued at the leave did not reach the user, in order"
    )
    assert rec.payload("on_session_ended")[0] == "avatar_left"


async def test_the_chunk_written_at_an_unexpected_leave_starts_the_handover() -> None:
    """The first chunk has gone out on an open stream, and the second chunk's write is in flight at the leave.
    The session finishes that write before it reports the leave, but the avatar has gone: the handover
    starts with the second chunk, and not with the first, which the avatar had."""
    participant = HeldWriteParticipant(AUDIO_2)
    rig = Rig(participant=participant)
    rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))  # a human remains
    service = make_service(SessionRigRoom(rig))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(participant.entered.is_set, what="the second chunk's write in flight")

            assert rig.avatar is not None
            rig.leave(rig.avatar, reason=CLIENT_INITIATED)
            participant.gate.set()  # the write completes; the leave is reported only after it
            await rec.wait("on_session_ended")
        finally:
            participant.gate.set()
        await h.finish()

    assert participant.streams[0].writes == [AUDIO_1, AUDIO_2]
    assert _handed_over(h) == ["AUDIO_2", "AUDIO_3"], (
        "the audio in flight or queued at the leave did not reach the user, in order"
    )
    assert rec.payload("on_session_ended")[0] == "avatar_left"


async def test_a_graceful_stop_while_the_session_closes_its_stream_does_not_wait_for_a_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The send loop has already met the session no longer alive and stopped sending when the stop
    arrives, so nothing will ever drain. The session's close of the byte stream is held, which keeps the
    service in the window; it is released once the stop is under way."""
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", LONG)
    participant = GatedOpenParticipant()
    rig = Rig(participant=participant)
    rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=LONG)
    rec = EventRecorder(service)
    gate = participant.open_gate = asyncio.Event()
    release_close: asyncio.Event | None = None
    stopping: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(participant.streams) == 1, what="the first chunk's byte stream opening")
            release_close = hold_after_the_close_request(participant.streams[0])

            assert rig.avatar is not None
            rig.leave(rig.avatar, reason=CLIENT_INITIATED)
            gate.set()
            await until(lambda: participant.streams[0].closed, what="the session to start closing its stream")
            assert rec.count("on_avatar_disconnected") == 0, "the leave was delivered; the window is gone"

            stopping = asyncio.ensure_future(service.stop(EndFrame()))
            release_close.set()
            await until(stopping.done, what="the graceful stop to finish")
            await stopping
        finally:
            gate.set()
            if release_close is not None:
                release_close.set()
            if stopping is not None and not stopping.done():
                stopping.cancel()
                await asyncio.wait({stopping})
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "pipeline_ended"
