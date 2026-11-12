"""The STT/LLM/TTS half of the example pipelines, kept out of the bot files.

Both examples build their conversational half the same way, so it lives here once. The
avatar-specific code stays in the bot files, where a reader looking for "what does the Synthesia
plugin need?" can see it without wading through vendor setup.

Three brains:

``cascade``  Cartesia STT -> OpenAI LLM -> Cartesia TTS. The default. One vendor key covers both
             ends of the pipeline.
``s2s``      OpenAI Realtime. Speech in, speech out, one WebSocket, no STT or TTS vendor at all --
             it emits the same TTSStartedFrame / TTSAudioRawFrame / TTSStoppedFrame sequence the
             avatar consumes, so it drops into the same pipeline slot.
``fake``     Scripted stand-ins. No vendor keys and no network, so the example smoke test can run
             in CI against the fake worker.

Why not Deepgram, which a reader of other Pipecat examples would expect: Cartesia covers STT as well as TTS
(``CartesiaSTTService``, model ``ink-whisper``), which keeps the example to a single vendor key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor

BrainKind = Literal["cascade", "s2s", "fake"]

SYSTEM_PROMPT = (
    "You are a friendly video avatar in a live call. Keep replies to one or two short "
    "sentences -- your words are spoken aloud, so avoid lists, markdown and emoji. "
    "If the user interrupts you, stop and listen."
)


@dataclass
class Brain:
    """The processors that turn user audio into spoken replies.

    ``pre_avatar`` is the complete ordered chain that sits between ``transport.input()`` and the
    avatar, user aggregator included. It has to be one list rather than "processors plus an
    aggregator": the aggregator's position differs per brain -- the cascade needs it *between*
    STT and the LLM, while the speech-to-speech brain has no separate STT for it to follow.

    ``assistant_aggregator`` goes last in the pipeline, after ``transport.output()``.
    """

    pre_avatar: list[FrameProcessor]
    assistant_aggregator: FrameProcessor
    context: LLMContext


def _require(name: str, *, needed_for: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set, and {needed_for} needs it.\n"
            f"Run scripts/check_credentials.py to see what is configured."
        )
    return value


def build_brain(kind: BrainKind = "cascade") -> Brain:
    """Assemble the conversational half of the pipeline.

    VAD lives in the user aggregator, not in the transport: ``TransportParams``, ``LiveKitParams``
    and ``DailyParams`` have no ``vad_analyzer`` field and, being pydantic models with
    ``extra='ignore'``, silently drop one that is passed. A ``vad_analyzer=`` on a transport param
    object is a no-op that looks like it works (Pipecat 1.10).
    """
    # Empty context: the system prompt goes on the LLM service as `system_instruction`, not in as
    # an initial "system" message. Passing it as a message is deprecated since Pipecat 1.9.0 and
    # emits a DeprecationWarning on every run.
    context = LLMContext([])
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    if kind == "fake":
        # Sibling import: the examples run as scripts, so `examples/` is on sys.path, not the
        # repo root.
        from _fake_brain import build_fake_brain

        # Same ordering as the cascade below, deliberately: the aggregator goes BETWEEN the STT
        # and the LLM. An earlier version led with the aggregator (copied from the s2s branch,
        # which has no STT for it to follow), which put it upstream of FakeSTT where it never saw
        # a transcription. The point of the fake brain is to exercise the real pipeline, so it has
        # to be wired in the real order.
        fake_stt, fake_llm, fake_tts = build_fake_brain()
        pre_avatar = [fake_stt, pair.user(), fake_llm, fake_tts]
    elif kind == "s2s":
        from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

        # One service for the whole loop. It pushes TTSStartedFrame / TTSAudioRawFrame /
        # TTSStoppedFrame exactly as a TTS service would, which is what the avatar consumes.
        # No STT stage, so the user aggregator leads rather than following one.
        pre_avatar = [
            pair.user(),
            OpenAIRealtimeLLMService(
                api_key=_require("OPENAI_API_KEY", needed_for="--s2s"),
                settings=OpenAIRealtimeLLMService.Settings(system_instruction=SYSTEM_PROMPT),
            ),
        ]
    else:
        from pipecat.services.cartesia.stt import CartesiaSTTService
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.services.openai.llm import OpenAILLMService

        cartesia_key = _require("CARTESIA_API_KEY", needed_for="the default pipeline")
        # The user aggregator sits between STT and the LLM: it turns transcription frames into
        # context messages, so it has to see STT's output before the LLM runs.
        pre_avatar = [
            CartesiaSTTService(api_key=cartesia_key),
            pair.user(),
            OpenAILLMService(
                api_key=_require("OPENAI_API_KEY", needed_for="the default pipeline"),
                settings=OpenAILLMService.Settings(system_instruction=SYSTEM_PROMPT),
            ),
            CartesiaTTSService(
                api_key=cartesia_key,
                voice_id=os.environ.get("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
            ),
        ]

    return Brain(
        pre_avatar=pre_avatar,
        assistant_aggregator=pair.assistant(),
        context=context,
    )
