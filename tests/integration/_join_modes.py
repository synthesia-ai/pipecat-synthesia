"""Builders and constants shared by the direct-join and relayed-join suites. Not a test module.

Everything here is the *product's* public constructor surface plus the two things a test needs on top
of it: a way to say "the bot speaks one utterance" in one line, and a probe that records the frames
passing a point in the pipeline. Nothing here decides anything: a helper that quietly corrected a
constructor argument would make the join-mode tests agree with the helper rather than with the product.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.protocol import models
from pipecat.frames.frames import Frame, StartFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from pipecat_synthesia.livekit._types import LiveKitCredentials
from pipecat_synthesia.livekit.private_room import PrivateLiveKitRoom
from pipecat_synthesia.livekit.token import DEFAULT_BOT_NAME
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from tests.support.livekit_dev import mint_dev_token

if TYPE_CHECKING:
    from tests.harness import Harness
    from tests.support.livekit_dev import LiveKitDev

__all__ = [
    "AGENT",
    "AVATAR_IDS",
    "CHUNK_A",
    "CHUNK_B",
    "SAMPLE_RATE",
    "DIRECT_JOIN_BOT",
    "STANDARD",
    "WAIT",
    "FrameProbe",
    "livekit_transport",
    "private_room",
    "direct_join_bot_token",
    "speak",
    "transport_room",
]

#: Two avatars, so readiness is a map and a swap has somewhere to go. Prefixed, as everything public is.
AVATAR_IDS = ("av_a", "av_b")

#: Participant kinds as the *server* reports them in `ListParticipants`: the ground truth for the
#: *agent-kind requirement*, and not the same thing as the `video.agent` grant.
AGENT = models.ParticipantInfo.Kind.AGENT
STANDARD = models.ParticipantInfo.Kind.STANDARD

#: The worker's rate; also the harness's `PipelineParams.audio_out_sample_rate` default, so the stream
#: attributes a test asserts on (`sample_rate="24000"`) are the ones a real pipeline would send.
SAMPLE_RATE = 24000

#: 40 ms of 24 kHz mono s16le each, distinguishable by their first byte so a test can say *which*
#: utterance a recorded stream carried instead of only how many there were.
CHUNK_A = b"\x01\x00" * 960
CHUNK_B = b"\x02\x00" * 960

#: Every network-observable wait in this tier. A backstop, not a budget: durations are inputs, never
#: assertions, so the assertion is always "eventually", never "within".
WAIT = 10.0

#: The direct-join bot's LiveKit identity. Not `pipecat-bot` (the relayed join's default) on purpose: in
#: a direct join the identity comes from the developer's own token, and a test that reused the relayed
#: join's default could pass while the adapter read the wrong one.
DIRECT_JOIN_BOT = "pcs-direct-join-bot"


def private_room(dev: LiveKitDev, *, room_name: str, **kwargs) -> PrivateLiveKitRoom:
    """A relayed join against the dev server: a private room this adapter creates and deletes.

    The dev server's ``ws://`` URL is accepted because this suite's conftest sets the room adapters'
    test seam for it.
    """
    return PrivateLiveKitRoom(
        url=dev.url,
        api_key=dev.api_key,
        api_secret=dev.api_secret,
        room_name=room_name,
        **kwargs,
    )


def direct_join_bot_token(dev: LiveKitDev, *, room_name: str, identity: str = DIRECT_JOIN_BOT) -> str:
    """The bot token for a direct join, as the docs tell developers to mint it: the agent kind and the
    agent grant (*agent-kind requirement*).

    Not the product's `mint_bot_token`, which mints the private room's bot, a standard participant.
    """
    return mint_dev_token(
        dev, identity=identity, room=room_name, kind="agent", agent_grant=True, name=DEFAULT_BOT_NAME
    )


def livekit_transport(dev: LiveKitDev, *, room_name: str, token: str) -> LiveKitTransport:
    """A real ``LiveKitTransport`` for the bot's own room: the room a direct join borrows.

    A fresh one per test: its client caches the room object and reference-counts connect/disconnect, so
    a transport reused across tests would be connected to a room the previous test deleted.
    """
    return LiveKitTransport(
        dev.url,
        token,
        room_name,
        LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )


def transport_room(dev: LiveKitDev, transport: LiveKitTransport, **kwargs) -> LiveKitTransportRoom:
    """A direct join: the avatar joins the room ``transport`` is already connected to."""
    return LiveKitTransportRoom(
        transport,
        credentials=LiveKitCredentials(api_key=dev.api_key, api_secret=dev.api_secret),
        **kwargs,
    )


async def speak(h: Harness, *, context_id: str | None, audio: bytes = CHUNK_A, end: bool = True) -> None:
    """One utterance the way a TTS service produces it: started, one chunk, stopped.

    ``end=False`` omits the ``TTSStoppedFrame`` — a dropped utterance boundary, which is what the 1 s
    idle close in ``utterance_streamer.py`` exists to survive.
    """
    await h.queue(TTSStartedFrame(context_id=context_id))
    await h.queue(
        TTSAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1, context_id=context_id)
    )
    if end:
        await h.queue(TTSStoppedFrame(context_id=context_id))


class FrameProbe(FrameProcessor):
    """Records every frame passing one point in the pipeline and forwards all of them unchanged.

    Direct mode, like the harness's own capture processors: the probe must not reorder anything it
    observes, and a processor with its own queue would make "the filter dropped this before the probe
    saw it" indistinguishable from "the probe had not got to it yet".
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("enable_direct_mode", True)
        super().__init__(**kwargs)
        #: Frames seen travelling DOWNSTREAM, in arrival order; ``StartFrame`` excluded, as in the harness.
        self.seen: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and not isinstance(frame, StartFrame):
            self.seen.append(frame)
        await self.push_frame(frame, direction)
