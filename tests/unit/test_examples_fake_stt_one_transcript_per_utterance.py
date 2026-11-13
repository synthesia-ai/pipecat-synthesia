"""The examples' fake STT gives one transcript per spoken utterance, however many audio chunks it hears.

`--fake-brain` with a live microphone sends the fake STT a ~20 ms audio chunk at a time, for as long
as the call lasts. A transcript per chunk floods the pipeline with user turns. One per utterance is
what a real STT gives, and the user aggregator's voice activity detection is what delimits an
utterance: it broadcasts `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`, and the
fake STT, upstream of it in the examples' pipeline, sees them travelling upstream.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection

from tests.harness import bidirectional_pipeline
from tests.support.frames import count, frames_of
from tests.support.waiting import until

FAKE_BRAIN = Path(__file__).resolve().parents[2] / "examples" / "_fake_brain.py"

SAMPLE_RATE = 16000
#: 20 ms of 16-bit mono audio, the size of chunk a live microphone delivers.
CHUNK = bytes(SAMPLE_RATE // 50 * 2)
CHUNKS_PER_UTTERANCE = 50


def _load_fake_brain() -> ModuleType:
    """`examples/_fake_brain.py`, loaded from its file: it imports nothing from `examples/`."""
    spec = importlib.util.spec_from_file_location("_fake_brain", FAKE_BRAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("utterances", [1, 2])
async def test_continuous_audio_gives_one_transcript_per_utterance(utterances: int) -> None:
    fake_brain = _load_fake_brain()
    stt = fake_brain.FakeSTT()
    params = PipelineParams(audio_in_sample_rate=SAMPLE_RATE, enable_metrics=True)

    async with bidirectional_pipeline(stt, pipeline_params=params) as h:
        for utterance in range(1, utterances + 1):
            await h.queue(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await until(
                lambda n=utterance: count(h.up, VADUserStartedSpeakingFrame) == n,
                what="the fake STT to pass on the start of speech",
            )
            for _ in range(CHUNKS_PER_UTTERANCE):
                await h.queue(InputAudioRawFrame(audio=CHUNK, sample_rate=SAMPLE_RATE, num_channels=1))
            await until(
                lambda n=utterance: count(h.down, InputAudioRawFrame) == n * CHUNKS_PER_UTTERANCE,
                what="the fake STT to take every audio chunk",
            )
            assert count(h.down, TranscriptionFrame) == utterance - 1, (
                "a transcript came before the end of speech"
            )
            await h.queue(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            await until(
                lambda n=utterance: count(h.down, TranscriptionFrame) >= n,
                what="the utterance's transcript after the end of speech",
            )

    transcripts = frames_of(h.down, TranscriptionFrame)
    assert len(transcripts) == utterances, (
        f"{utterances} utterance(s) of {CHUNKS_PER_UTTERANCE} chunks each gave {len(transcripts)} transcripts"
    )
    texts = {t.text for t in transcripts if isinstance(t, TranscriptionFrame)}
    assert texts == {fake_brain.DEFAULT_TRANSCRIPT}, texts


@pytest.mark.unit
async def test_the_fake_stt_starts_without_an_error_or_warning(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """An ERROR or WARNING at startup in a scripted stand-in reads like a fault in the example it runs in.

    An STT service that gives no `ttfs_p99_latency` logs a WARNING at start that it is using
    Pipecat's default. Metrics are on here because the example bots turn them on.
    """
    stt = _load_fake_brain().FakeSTT()
    params = PipelineParams(audio_in_sample_rate=SAMPLE_RATE, enable_metrics=True)
    async with bidirectional_pipeline(stt, pipeline_params=params):
        pass

    problems = [r.getMessage() for r in caplog_loguru.records if r.levelno >= logging.WARNING]
    assert not problems, problems
