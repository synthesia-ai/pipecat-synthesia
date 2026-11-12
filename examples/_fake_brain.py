"""Scripted STT/LLM/TTS stand-ins: the `fake` brain, for running without STT, LLM or TTS keys.

Used by `_brain.build_brain("fake")`, which backs the examples' `--fake-brain` flag. The examples
still talk to the real Synthesia API and LiveKit; the example smoke test
(`tests/integration/test_examples_smoke.py`) also swaps in a fake avatar worker, which is what
makes it free and hermetic instead of billable and network-flaky.

What is fake and what is real matters here. **VAD is real** -- the user aggregator still runs
Silero, so turn-taking and barge-in behave exactly as they do in production, and an interruption
test against this brain is meaningful. Only the *words* are canned and the *voice* is a tone.

The tone is deliberate rather than silence: the avatar's mouth moves, so a relayed join produces
real video motion and an interruption is audible as the tone cutting out mid-word.
"""

from __future__ import annotations

import math
import struct
from collections.abc import AsyncGenerator
from typing import Any

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService, LLMSettings
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService, TTSSettings
from pipecat.utils.time import time_now_iso8601

__all__ = ["FakeLLM", "FakeSTT", "FakeTTS", "build_fake_brain"]

DEFAULT_TRANSCRIPT = "Tell me something interesting."
DEFAULT_REPLY = (
    "Here is a thought for you. The avatar you are looking at is rendered remotely and streamed "
    "back in real time, which is why you can interrupt it mid-sentence."
)


class FakeSTT(SegmentedSTTService):
    """Emits a fixed transcript once per utterance, so the pipeline advances with no STT vendor.

    The user aggregator's VAD decides *when* the user is speaking: `SegmentedSTTService` buffers the
    audio between its start and stop frames and calls `run_stt` once per utterance, not once per
    audio chunk. This only supplies the words.
    """

    def __init__(self, transcript: str = DEFAULT_TRANSCRIPT, **kwargs: Any) -> None:
        # See the note in FakeLLM.__init__ about NOT_GIVEN settings fields.
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        # The transcript comes the moment speech stops, so the latency from end of speech to final
        # transcript is 0. Left unset, Pipecat logs a WARNING at start that it is using a default.
        kwargs.setdefault("ttfs_p99_latency", 0.0)
        super().__init__(**kwargs)
        self._transcript = transcript

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield TranscriptionFrame(self._transcript, "", time_now_iso8601())


class FakeLLM(LLMService):
    """Replies with a canned sentence, triggered by whichever frame reaches it first.

    In the shipped wiring, ``_brain.build_brain("fake")`` puts the user aggregator *between* STT
    and the LLM, so the live path is the ``LLMContextFrame`` one.

    The bare-``TranscriptionFrame`` branch is a safety net, not the normal route: it only fires if
    someone reorders the chain so the LLM sits upstream of the aggregator. It cannot fire in the
    shipped order, because ``LLMUserAggregator`` handles a ``TranscriptionFrame`` without pushing
    it downstream. It is kept because a stand-in that goes silently mute after a reorder turns a
    smoke test from failing into passing-but-vacuous, which is the worse failure.
    """

    def __init__(self, reply: str = DEFAULT_REPLY, **kwargs: Any) -> None:
        # Every settings field must be initialised or Pipecat logs an ERROR per field at startup
        # ("the following fields are NOT_GIVEN"), which in a reference integration reads like a
        # fault. None is the documented value for "this service does not support it" (Pipecat 1.10).
        kwargs.setdefault(
            "settings",
            LLMSettings(
                model=None,
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
            ),
        )
        super().__init__(**kwargs)
        self._reply = reply

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame | TranscriptionFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(self._reply))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class FakeTTS(TTSService):
    """Speaks a 220 Hz tone whose duration tracks the text length."""

    TONE_HZ = 220.0
    SECONDS_PER_CHAR = 0.055
    MAX_SECONDS = 12.0
    CHUNK_SECONDS = 0.02
    AMPLITUDE = 12000

    def __init__(self, *, sample_rate: int = 24000, **kwargs: Any) -> None:
        # See the note in FakeLLM.__init__ about NOT_GIVEN settings fields.
        kwargs.setdefault("settings", TTSSettings(model=None, voice=None, language=None))
        super().__init__(sample_rate=sample_rate, **kwargs)
        # `self.sample_rate` stays 0 until the pipeline's StartFrame negotiates one, so keep the
        # configured value for callers that drive run_tts directly (tests).
        self._fallback_sample_rate = sample_rate

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        rate = self.sample_rate or self._fallback_sample_rate
        total_samples = int(rate * min(len(text) * self.SECONDS_PER_CHAR, self.MAX_SECONDS))
        chunk_samples = int(rate * self.CHUNK_SECONDS)

        yield TTSStartedFrame()
        await self.start_ttfb_metrics()
        sample = 0
        while sample < total_samples:
            count = min(chunk_samples, total_samples - sample)
            pcm = b"".join(
                struct.pack(
                    "<h", int(self.AMPLITUDE * math.sin(2 * math.pi * self.TONE_HZ * (sample + i) / rate))
                )
                for i in range(count)
            )
            sample += count
            await self.stop_ttfb_metrics()
            yield TTSAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1)
        yield TTSStoppedFrame()


def build_fake_brain() -> list[FrameProcessor]:
    """The scripted chain, in pipeline order: STT, then LLM, then TTS."""
    return [FakeSTT(), FakeLLM(), FakeTTS()]
