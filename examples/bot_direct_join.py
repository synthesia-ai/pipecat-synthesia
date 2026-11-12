"""Synthesia avatar in your own LiveKit room (direct join: the avatar joins the room you are in).

Your users are already in a LiveKit room. The avatar joins *that* room as an ordinary participant
and publishes its own lip-synced video track, so your client renders it the same way it renders
any other participant -- no video plumbing in your pipeline at all.

    uv run python examples/mint_tokens.py --room my-room   # prints LIVEKIT_ROOM and BOT_TOKEN
    uv run python examples/bot_direct_join.py --s2s

Then join the same room in a browser with the HUMAN_TOKEN that command printed (for example at
<https://meet.livekit.io> -> Custom).

Environment: SYNTHESIA_API_KEY, SYNTHESIA_AVATAR_ID, LIVEKIT_URL, LIVEKIT_ROOM, BOT_TOKEN, plus
LIVEKIT_API_KEY / LIVEKIT_API_SECRET so the plugin can mint the avatar's own token.

How this differs from `bot_relayed_join.py`, and why each difference matters
----------------------------------------------------------------------------
**No `video_out_*` settings.** The avatar publishes video into the room itself. Nothing flows
through your pipeline, so there is no geometry to match and no frames to resize.

**`room.input_filter()` is not optional.** The avatar's audio arrives on the bot's own input, so
without the filter the bot hears the avatar speak, transcribes it, and replies to itself in a
loop. The filter drops audio frames from the avatar's identity. The service WARNS once at startup if
it is missing -- it cannot raise, because it never sees transport.input()'s frames, so nothing at
runtime will stop you shipping without it.

**The greeting is filtered by participant identity.** `on_first_participant_joined` is wrong here:
the avatar is itself a participant and usually the first to arrive, so the greeting would fire at
the avatar and play to an empty room. We greet once, for the first participant that is not the
avatar, including one who was already in the room when the bot connected.

**The call ends when the last human leaves.** Ending the pipeline takes the bot, and the avatar
with it, out of the room, so ending on any leave would take the avatar away from everyone still in
the call. The bot counts the humans in the room, including any who were there before it connected,
and ends once none have been there for 6 s, so that a reconnect does not end the call. The avatar
leaving is not a human leaving. The avatar worker stops a second sooner, once its own 5 s default
has passed with no humans, so the session normally ends with `no_humans_left`, and the bot then
ends the call. Someone who rejoins before the plugin has recorded that end (within about 2 s of the
avatar leaving) gets the voice-only bot instead: with a human present, the plugin reports the
avatar's leave as `avatar_left` and carries on without it, as after an avatar crash. A rejoin after
the plugin has recorded the end, such as a page reload whose leave is client-initiated (see
`_presence.py`), is too late: the end is `no_humans_left`, and the bot ends the call although that
person is back.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Sibling imports, not `examples._brain`: these run as scripts, so `examples/` is on sys.path.
from _brain import build_brain
from _presence import Action, Presence
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.workers.runner import WorkerRunner

from pipecat_synthesia import SynthesiaVideoService
from pipecat_synthesia.livekit import LiveKitTransportRoom


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"Run `uv run python examples/mint_tokens.py --room <name>` and put its output in .env."
        )
    return value


async def end_with_the_session(reason: str, presence: Presence, worker: PipelineWorker) -> None:
    """End the call when the avatar worker ended the session because no humans were left.

    This is how the call normally ends once the last human has left: the worker stops about a
    second before this example's own grace runs out. Every other reason leaves the call running,
    and the plugin continues voice-only: after a session cap, an avatar crash with a human still in
    the room, or someone rejoining before the plugin has recorded the end, which the plugin, seeing a
    human present, reports as `avatar_left`. A rejoin after the plugin has recorded the end, such as
    a page reload, comes too late: the end is `no_humans_left`, so the call ends.
    """
    if reason != "no_humans_left":
        return
    logger.info("the avatar worker ended the session with no humans left; ending")
    presence.close()
    await worker.stop_when_done()


async def run(kind: str) -> None:
    transport = LiveKitTransport(
        _require("LIVEKIT_URL"),
        _require("BOT_TOKEN"),
        _require("LIVEKIT_ROOM"),
        LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    logger.info(f"brain: {kind}")
    brain = build_brain(kind)

    # --- everything Synthesia-specific is these lines ---------------------------------
    room = LiveKitTransportRoom(transport)
    avatar = SynthesiaVideoService(
        avatar_ids=[os.environ["SYNTHESIA_AVATAR_ID"]],
        room=room,
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
        await end_with_the_session(reason, presence, worker)

    @avatar.event_handler("on_metrics")
    async def _metrics(service, metrics):
        logger.info(f"avatar metrics: {metrics.summary()}")

    # ----------------------------------------------------------------------------------

    pipeline = Pipeline(
        [
            transport.input(),
            # Must sit directly after transport.input(): see the module docstring.
            room.input_filter(),
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

    # The humans in the room. The avatar is a participant too, and normally arrives before the
    # human: greeting it would play to an empty room, and its leaving is not the call ending.
    presence = Presence(ignore={avatar.avatar_identity}, on_end=lambda: act(Action.END))

    async def act(action: Action) -> None:
        if action is Action.GREET:
            logger.info("first human present; greeting")
            await worker.queue_frames([LLMRunFrame()])
        elif action is Action.END:
            logger.info("last human left; ending")
            # Not `worker.end()`: at Pipecat 1.10 the runner cancels a worker that is still
            # draining, so the call would end as a cancel (`pipeline_cancelled`) and cut off
            # whatever the avatar is still saying. An EndFrame is a normal end.
            await worker.stop_when_done()

    @transport.event_handler("on_connected")
    async def _seed(transport):
        # on_participant_connected fires only for later joins. Anyone already in the room when
        # the bot connects is reported here, and get_participants() never includes the bot.
        await act(presence.seed(transport.get_participants()))

    @transport.event_handler("on_participant_connected")
    async def _joined(transport, participant_id: str):
        # participant_id is an identity string.
        logger.info(f"participant joined: {participant_id}")
        await act(presence.joined(participant_id))

    @transport.event_handler("on_participant_disconnected")
    async def _left(transport, participant_id: str):
        logger.info(f"participant left: {participant_id}")
        presence.left(participant_id)

    @worker.event_handler("on_pipeline_error")
    async def _error(worker, frame):
        logger.error(f"pipeline error: {frame.error} (usable={frame.processor.is_usable})")

    # handle_sigterm: a container stop sends SIGTERM, not SIGINT. Without this the process dies
    # before the avatar session is torn down, and the session keeps billing until the avatar worker
    # ends it, a few seconds after the bot's participant leaves the transport room. SIGINT (Ctrl-C)
    # is on by default; SIGTERM is not.
    # add_workers() then run(): passing the worker straight to run() is deprecated.
    runner = WorkerRunner(handle_sigterm=True)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        presence.close()


def main() -> int:
    # Load .env ourselves. `bot_relayed_join.py` gets this for free because
    # `pipecat.runner.run.main` calls `load_dotenv(override=True)` at import; this example does not
    # use the dev runner (it connects to a room you already have), so without this every variable
    # in run() looks unset and the script exits telling you to set what you already set. Here
    # rather than at module scope, so that importing this file loads nothing.
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s2s", action="store_true", help="OpenAI Realtime; no STT/TTS vendor")
    parser.add_argument("--fake-brain", action="store_true", help="scripted STT/LLM/TTS; no vendor keys")
    args = parser.parse_args()
    if args.s2s and args.fake_brain:
        return int(bool(sys.stderr.write("--s2s and --fake-brain are mutually exclusive\n")))

    kind = "fake" if args.fake_brain else "s2s" if args.s2s else "cascade"
    asyncio.run(run(kind))
    return 0


if __name__ == "__main__":
    sys.exit(main())
