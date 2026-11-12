"""Synthesia avatar over a non-LiveKit Pipecat transport (relayed join): SmallWebRTC or Daily.

The avatar joins a private LiveKit room of its own, and its video and audio are relayed into your
pipeline as frames, so whichever transport you are already using -- SmallWebRTC, Daily -- carries
it to the user. Your own transport needs no LiveKit involvement at all.

    # Talk to it over WebRTC in your browser, no STT/TTS vendor needed:
    uv run python examples/bot_relayed_join.py -t webrtc --s2s

    # The default pipeline (Cartesia STT + OpenAI LLM + Cartesia TTS):
    uv run python examples/bot_relayed_join.py -t webrtc

    # A scripted brain instead of STT, LLM and TTS, so no vendor keys (the Synthesia and LiveKit
    # credentials below are still needed):
    uv run python examples/bot_relayed_join.py -t webrtc --fake-brain

Then open http://localhost:7860 and allow microphone access.

Environment: see `.env.example`. Always needed are SYNTHESIA_API_KEY, SYNTHESIA_AVATAR_ID and
LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET -- the LiveKit credentials are for the
avatar's *private* room, not for your transport.

Two things about the relayed join that are easy to get wrong:

`-t livekit` is untested here: Pipecat's LiveKit transport gained video output in 1.11, after this
plugin's design, and nothing here has been run against it. If your users are in a LiveKit room, use
`bot_direct_join.py` instead, which puts the avatar in that same room.

The five `video_out_*` settings are not decoration. The avatar renders 512x512 at 25 fps; Pipecat's
defaults are a different size and framerate, so every frame gets PIL-resized -- which distorts the
aspect ratio, burns a core, and adds latency that grows over a call. `video_out_is_live=True` marks
the stream as continuous rather than a fixed-length clip.

Over Daily, a call can have several people in it: the bot greets the first and ends the call when
the last one leaves, not the first. SmallWebRTC is one-to-one, so there it greets the one person
who connects and ends when they disconnect. Over Daily it ends only once nobody has been in the call
for 6 s, so that someone rejoining does not end it; over SmallWebRTC a reconnect is a new call, so it
ends at once. Either way the session ends with `pipeline_ended`: the avatar worker cannot see the
people in the call.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

# Sibling imports, not `examples._brain`: these are run as scripts
# (`python examples/bot_relayed_join.py`), so `examples/` is what lands on sys.path, not the
# repo root.
from _brain import build_brain
from _presence import NO_HUMANS_GRACE_SECS, Action, Presence
from _shutdown import run_under_signals
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams

from pipecat_synthesia import SynthesiaVideoService
from pipecat_synthesia.livekit import PrivateLiveKitRoom

# The avatar renders at this geometry; matching it avoids a per-frame resize. See the module
# docstring.
VIDEO = dict(
    video_out_enabled=True,
    video_out_is_live=True,
    video_out_width=512,
    video_out_height=512,
    video_out_framerate=25,
)


def _daily_params():
    # Imported here, not at module scope: pipecat.transports.daily raises ImportError unless the
    # Daily SDK is installed, and `-t webrtc` should not require it. create_transport only calls
    # the factory for the transport actually selected.
    from pipecat.transports.daily.transport import DailyParams

    return DailyParams(audio_in_enabled=True, audio_out_enabled=True, **VIDEO)


transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True, **VIDEO),
    "daily": _daily_params,
}


def _client_id(client) -> str:
    """The id of whoever on_client_connected / on_client_disconnected is reporting.

    Daily passes its participant dict; SmallWebRTC passes its one peer connection.
    """
    return client["id"] if isinstance(client, Mapping) else client.pc_id


def _add_example_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register this example's own flags.

    They go on the *runner's* parser, not a private one: `pipecat.runner.run.main` adds its
    arguments to whatever parser it is given and then parses strictly, so a flag it has never
    heard of is an error. Passing the parser in is the supported way to extend it.
    """
    parser.add_argument("--s2s", action="store_true", help="OpenAI Realtime; no STT/TTS vendor")
    parser.add_argument("--fake-brain", action="store_true", help="scripted STT/LLM/TTS; no vendor keys")
    return parser


# `bot()` runs inside the runner and is not handed the parsed args, so read them here too. A
# throwaway parser with parse_known_args ignores the runner's own flags.
OPTS, _ = _add_example_args(argparse.ArgumentParser(add_help=False)).parse_known_args()


def no_humans_grace_secs(runner_args) -> float:
    """How long nobody must be in the call before it ends, for the transport the runner hands the bot.

    Over SmallWebRTC, Pipecat reports a disconnect only when the peer connection closes, and a reconnect
    arrives as a new `bot()` call, so nothing in this call could cancel a grace: it would only add
    billing. Over Daily a participant can leave and rejoin the same room, so the call waits.
    """
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        return 0.0
    return NO_HUMANS_GRACE_SECS


async def bot(runner_args):
    """Entry point discovered by `pipecat.runner.run.main`."""
    transport = await create_transport(runner_args, transport_params)

    kind = "fake" if OPTS.fake_brain else "s2s" if OPTS.s2s else "cascade"
    logger.info(f"brain: {kind}")
    brain = build_brain(kind)

    # --- everything Synthesia-specific is these lines ---------------------------------
    avatar = SynthesiaVideoService(
        avatar_ids=[os.environ["SYNTHESIA_AVATAR_ID"]],
        room=PrivateLiveKitRoom(),
    )

    @avatar.event_handler("on_avatar_connected")
    async def _connected(service, info):
        logger.info(f"avatar connected: session={info.session_id} id={info.active_avatar_id}")

    @avatar.event_handler("on_avatar_disconnected")
    async def _disconnected(service, reason):
        logger.info(f"avatar disconnected: {reason}")

    @avatar.event_handler("on_session_ended")
    async def _session_ended(service, reason):
        logger.info(f"avatar session ended: {reason}")

    @avatar.event_handler("on_metrics")
    async def _metrics(service, metrics):
        logger.info(f"avatar metrics: {metrics.summary()}")

    # ----------------------------------------------------------------------------------

    pipeline = Pipeline(
        [
            transport.input(),
            *brain.pre_avatar,
            avatar,
            transport.output(),
            brain.assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    # The people in the call. The avatar is never among them: it is in the private room.
    presence = Presence(on_end=lambda: act(Action.END), grace_secs=no_humans_grace_secs(runner_args))

    async def act(action: Action) -> None:
        if action is Action.GREET:
            # Greet when the first *human* is there, not at pipeline start, or the greeting plays
            # before anyone is there to hear it.
            logger.info("first human present; greeting")
            await worker.queue_frames([LLMRunFrame()])
        elif action is Action.END:
            logger.info("last human left; ending")
            # Not `worker.end()`: at Pipecat 1.10 the runner cancels a worker that is still
            # draining, so the call would end as a cancel (`pipeline_cancelled`) and cut off
            # whatever the avatar is still saying. An EndFrame is a normal end.
            await worker.stop_when_done()

    if isinstance(runner_args, DailyRunnerArguments):
        # Daily can say who is already in the room. SmallWebRTC has no such call, and nothing to
        # report: its one peer arrives through on_client_connected.
        @transport.event_handler("on_connected")
        async def _seed(transport, data):
            # participants() includes the bot itself, under "local". Counting it as a human would
            # mean the call never ends, and keeps billing.
            await act(presence.seed(pid for pid in transport.participants() if pid != "local"))

    @transport.event_handler("on_client_connected")
    async def _joined(transport, client):
        logger.info(f"client connected: {_client_id(client)}")
        await act(presence.joined(_client_id(client)))

    @transport.event_handler("on_client_disconnected")
    async def _left(transport, client):
        logger.info(f"client disconnected: {_client_id(client)}")
        presence.left(_client_id(client))

    @worker.event_handler("on_pipeline_error")
    async def _error(worker, frame):
        logger.error(f"pipeline error: {frame.error} (usable={frame.processor.is_usable})")

    # Ends the call in order on Ctrl-C or SIGTERM, however the development runner runs the bot: on its
    # own, or as one call in its web server. Without that, the process can die before the avatar session
    # is torn down, and the session keeps billing until the avatar worker ends it, a few seconds after the
    # bot's participant leaves the private room. See _shutdown.py and docs/deployment.md.
    try:
        await run_under_signals(worker, runner_args)
    finally:
        presence.close()


if __name__ == "__main__":
    from pipecat.runner.run import main

    if OPTS.s2s and OPTS.fake_brain:
        sys.exit("--s2s and --fake-brain are mutually exclusive")
    main(_add_example_args(argparse.ArgumentParser(description=__doc__.splitlines()[0])))
