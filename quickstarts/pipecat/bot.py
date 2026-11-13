"""Synthesia Interactive Avatar in a Pipecat voice bot, in one file plus its shutdown helper.

    uv run python bot.py -t webrtc

Then open <http://localhost:7860> and allow microphone access.

This mirrors `examples/bot_relayed_join.py` in the pipecat-synthesia repository, flattened into
one file: a quickstart a reader can copy whole. Its one sibling, `_shutdown.py`, ends calls in order
on Ctrl-C or SIGTERM.

The avatar joins a private LiveKit room the plugin creates and owns, and its video and audio are
relayed into the pipeline as ordinary Pipecat frames — so SmallWebRTC (or Daily) delivers it and
your transport never touches LiveKit. If your users are already in a LiveKit room, use
`LiveKitTransportRoom` instead; see the plugin's README.

Two things that are easy to get wrong, both marked below:

* **VAD goes on the user aggregator**, never on the transport params. `TransportParams` has no
  `vad_analyzer` field and silently drops one, giving a bot that never hears you.
* **The five `video_out_*` settings are not decoration.** The avatar renders 512x512 at 25 fps;
  leave Pipecat's defaults and every frame is resized — wrong aspect ratio, a pinned CPU core, and
  latency that grows over a call.
"""

from __future__ import annotations

import os

from _shutdown import run_under_signals
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams

from pipecat_synthesia import SynthesiaVideoService
from pipecat_synthesia.livekit import PrivateLiveKitRoom

load_dotenv(override=True)

SYSTEM_PROMPT = (
    "You are a friendly video avatar in a live call. Keep replies to one or two short "
    "sentences -- your words are spoken aloud, so avoid lists, markdown and emoji. "
    "If the user interrupts you, stop and listen."
)

# The geometry of the avatar's video. See the module docstring.
VIDEO = dict(
    video_out_enabled=True,
    video_out_is_live=True,
    video_out_width=512,
    video_out_height=512,
    video_out_framerate=25,
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True, **VIDEO),
}


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


async def bot(runner_args) -> None:
    """Entry point discovered by `pipecat.runner.run.main`."""
    transport = await create_transport(runner_args, transport_params)

    context = LLMContext([])
    pair = LLMContextAggregatorPair(
        context,
        # THE one correct home for vad_analyzer. Not on TransportParams -- see the docstring.
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    cartesia_key = require("CARTESIA_API_KEY")
    stt = CartesiaSTTService(api_key=cartesia_key)
    llm = OpenAILLMService(
        api_key=require("OPENAI_API_KEY"),
        # Both of these go in `settings`, and it matters. `OpenAILLMService.__init__` takes
        # `**kwargs`, so a top-level `system_instruction=` is swallowed and the bot silently
        # ignores its prompt; a top-level `model=` still works but is deprecated. This is the
        # same class of bug as putting `vad_analyzer` on TransportParams -- see below.
        settings=OpenAILLMService.Settings(model="gpt-4o-mini", system_instruction=SYSTEM_PROMPT),
    )
    tts = CartesiaTTSService(
        api_key=cartesia_key,
        voice_id=os.environ.get("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
    )

    # --- everything Synthesia-specific is these lines --------------------------------------
    avatar = SynthesiaVideoService(
        avatar_ids=[require("SYNTHESIA_AVATAR_ID")],
        room=PrivateLiveKitRoom(),  # reads LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
    )

    @avatar.event_handler("on_avatar_connected")
    async def _connected(service, info):
        logger.info(f"avatar connected: session={info.session_id} id={info.active_avatar_id}")

    @avatar.event_handler("on_avatar_disconnected")
    async def _disconnected(service, reason):
        logger.warning(f"avatar disconnected: {reason}")

    @avatar.event_handler("on_session_ended")
    async def _ended(service, reason):
        logger.info(f"avatar session ended: {reason}")

    @avatar.event_handler("on_metrics")
    async def _metrics(service, metrics):
        logger.info(f"avatar metrics: {metrics.summary()}")

    # ----------------------------------------------------------------------------------------

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            pair.user(),
            llm,
            tts,
            avatar,  # consumes the TTS audio and renders the avatar speaking it
            transport.output(),
            pair.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def _greet(transport, client):
        # Greet when the human connects, not at pipeline start, or the greeting plays before
        # anyone is there to hear it.
        logger.info("client connected; greeting")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _end(transport, client):
        logger.info("client disconnected; ending")
        # Not `worker.end()`: at Pipecat 1.10 the runner cancels a worker that is still draining,
        # so the call would end as a cancel (`pipeline_cancelled`) and cut off whatever the avatar
        # is still saying. An EndFrame is a normal end.
        await worker.stop_when_done()

    # Ends the call in order on Ctrl-C or SIGTERM, however the development runner runs the bot: on its
    # own, or as one call in its web server. Without that, the process can die before the avatar session
    # is torn down, and the session keeps billing until the avatar worker ends it, a few seconds after the
    # bot's participant leaves the private room. See _shutdown.py.
    await run_under_signals(worker, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
