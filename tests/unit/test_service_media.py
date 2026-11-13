"""The media path.

A relayed join re-emits the avatar's media as Pipecat frames, but only after the transport said it is
ready (`OutputTransportReadyFrame`, travelling upstream; the media gate), and never as
`SpeechOutputAudioRawFrame` (*speaking-state ownership*). A direct join emits no media at all.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import (
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import tts_audio

UP = FrameDirection.UPSTREAM
PCM = b"\x00\x10" * 480


def _plain_audio(frames) -> list[OutputAudioRawFrame]:
    """Exactly `OutputAudioRawFrame`, not a subclass (TTSAudioRawFrame is one)."""
    return [f for f in frames if type(f) is OutputAudioRawFrame]


@pytest.mark.unit
async def test_relayed_join_media_gating() -> None:
    """Relayed join: media emitted before `OutputTransportReadyFrame` is dropped (never queued); after
    it, video arrives as `OutputImageRawFrame(size=(512, 512), format="RGB")` and audio as a PLAIN
    `OutputAudioRawFrame` (*speaking-state ownership*)."""
    room = FakeAvatarRoom(JoinMode.RELAYED)
    service = make_service(room)
    rec = EventRecorder(service)
    session = room.session

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")

        video_task = session.emit_video(512, 512)
        audio_task = session.emit_audio(PCM, 24000, 1)
        assert video_task is not None and audio_task is not None
        await video_task
        await audio_task
        assert_absent(h.down, OutputImageRawFrame, where="before OutputTransportReadyFrame")
        assert _plain_audio(h.down) == []

        await h.queue(OutputTransportReadyFrame(), UP)
        await h.frame(OutputTransportReadyFrame, direction=UP)  # forwarded upstream, not consumed
        assert service.metrics.transport_ready is not None

        await session.emit_video(512, 512)  # type: ignore[misc]
        await session.emit_audio(PCM, 24000, 1)  # type: ignore[misc]
        image = await h.frame(OutputImageRawFrame)
        assert image.size == (512, 512)
        assert image.format == "RGB"
        assert len(image.image) == 512 * 512 * 3
        await until(lambda: _plain_audio(h.down), what="a plain OutputAudioRawFrame downstream")
        audio = _plain_audio(h.down)[0]
        assert type(audio) is OutputAudioRawFrame
        assert (audio.audio, audio.sample_rate, audio.num_channels) == (PCM, 24000, 1)
        await h.finish()

    assert len(frames_of(h.down, OutputImageRawFrame)) == 1
    assert len(_plain_audio(h.down)) == 1
    assert service.metrics.first_video_frame is not None


@pytest.mark.unit
async def test_direct_join_emits_no_media() -> None:
    """Direct join: the room gets no media callbacks, and no media frame goes downstream even after the
    transport is ready."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(OutputTransportReadyFrame(), UP)
        await h.frame(OutputTransportReadyFrame, direction=UP)
        assert room.captured_ctx is not None
        assert room.captured_ctx.callbacks.on_video_frame is None
        assert room.captured_ctx.callbacks.on_audio_frame is None
        assert room.session.emit_video(512, 512) is None
        assert room.session.emit_audio(PCM, 24000, 1) is None
        await h.finish()

    assert_absent(h.down, OutputImageRawFrame)
    assert _plain_audio(h.down) == []


@pytest.mark.unit
async def test_relayed_join_without_transport_ready_sends_audio_but_no_media() -> None:
    """Relayed join with no `OutputTransportReadyFrame` ever: the outbound audio path still writes to
    the session while the inbound media path emits nothing. The media gate holds back only the media."""
    room = FakeAvatarRoom(JoinMode.RELAYED)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await until(lambda: room.session.send_audio_calls, what="TTS audio at the session")
        for _ in range(3):
            await room.session.emit_video(512, 512)  # type: ignore[misc]
            await room.session.emit_audio(PCM, 24000, 1)  # type: ignore[misc]
        assert_absent(h.down, OutputImageRawFrame, where="with the transport never ready")
        assert _plain_audio(h.down) == []
        await h.finish()

    assert room.session.send_audio_calls
    assert service.metrics.transport_ready is None
    assert service.metrics.first_video_frame is None
