"""The service records no time to first byte in this release.

The avatar worker's playback events don't say which byte stream they belong to, so a `playback_started`
can't always be matched to the utterance it starts. This file pins, with Pipecat's metrics enabled, that
the service pushes no TTFB `MetricsFrame` and `summary()` has no `utterance_ttfb_seconds`: after one
played utterance, and after an interruption of a sent chunk, where Pipecat would report any TTFB
measurement still open. And with Pipecat's defaults, which send every metrics-capable processor an initial
empty time to first byte and processing time, nothing in any `MetricsFrame` names the service at all.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import BotStartedSpeakingFrame, InterruptionFrame, MetricsFrame, TTSStartedFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.worker import PipelineParams

from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import delivered, tts_audio

METRICS = PipelineParams(enable_metrics=True)


def ttfb_reported(h: Harness, service: SynthesiaVideoService) -> list[TTFBMetricsData]:
    return [
        data
        for frame in frames_of(h.down, MetricsFrame)
        for data in getattr(frame, "data", [])
        if isinstance(data, TTFBMetricsData) and data.processor == service.name
    ]


@pytest.mark.unit
async def test_one_played_utterance_records_no_time_to_first_byte() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service, pipeline_params=METRICS) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await until(lambda: bool(room.session.send_audio_calls), what="the chunk at the session")
        await delivered(room.session.emit_playback_started())
        await h.frame(BotStartedSpeakingFrame)
        await delivered(room.session.emit_playback_finished(1.0, interrupted=False))
        await h.finish()

    assert ttfb_reported(h, service) == []
    assert "utterance_ttfb_seconds" not in service.metrics.summary()


@pytest.mark.unit
async def test_an_interruption_of_a_sent_chunk_reports_no_time_to_first_byte() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service, pipeline_params=METRICS) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await until(lambda: bool(room.session.send_audio_calls), what="the chunk at the session")
        await h.queue(InterruptionFrame())
        await until(lambda: room.session.interrupt_count == 1, what="clear buffer answered")
        await h.finish()

    assert ttfb_reported(h, service) == []


@pytest.mark.unit
async def test_pipecat_sends_no_initial_empty_metrics_for_the_service() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service, pipeline_params=METRICS) as h:
        await rec.wait("on_avatar_connected")
        await h.finish()

    metrics_frames = frames_of(h.down, MetricsFrame)
    assert metrics_frames, "Pipecat's initial metrics frame never reached the end of the pipeline"
    named = [data for frame in metrics_frames for data in frame.data if data.processor == service.name]
    assert named == []
