"""The speaking state when the avatar worker is replaced while it plays.

A same-identity replacement swaps the avatar worker under the session: the replaced worker's utterance
stops playing, and the worker that replaces it never sends that utterance's `playback_finished`. The
*replacement grace* keeps the session alive across it, so the service never sees a leave either. With an
*early start* this happens on a path of its own: the session request is retried after the avatar has
started speaking, and the retry's worker replaces it (seen live: a 502 after 3.9 s with the worker
publishing, then a second worker 2 s later).

What this file pins:
* the bot is not left speaking after the replacement: it stops once the stop hysteresis has run out;
* a replacement that starts playing within the hysteresis keeps the bot speaking, with no stop between.

Runs the real `LiveKitAvatarSession` on the session rig behind the service; `bot_stopped_hysteresis_secs`
is the config seam, and twice it is the "and nothing happened" wait.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

from pipecat_synthesia.protocol import RPC_PLAYBACK_STARTED
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, tts_audio
from tests.unit._session_rig import AVATAR, Rig, SessionRigRoom, make_avatar


async def _speaking_then_replaced(h: Harness, rig: Rig, rec: EventRecorder) -> None:
    await rec.wait("on_avatar_connected")
    await h.queue(tts_audio("c1", AUDIO_1))
    await until(lambda: rig.participant.streams, what="c1 at the avatar")
    await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)
    await until(lambda: count(h.down, BotStartedSpeakingFrame) == 1, what="the bot speaking")

    # the replaced worker leaves as livekit-server reports it (no reason), and its replacement joins
    assert rig.avatar is not None
    rig.depart(rig.avatar)
    rig.join(make_avatar())


@pytest.mark.unit
async def test_a_replacement_while_the_avatar_speaks_ends_the_speaking_state() -> None:
    rig = Rig()
    service = make_service(SessionRigRoom(rig), bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _speaking_then_replaced(h, rig, rec)
        await until(
            lambda: count(h.down, BotStoppedSpeakingFrame) == 1,
            what="the bot no longer speaking after the replacement",
        )
        assert rec.count("on_avatar_disconnected") == 0, "a replacement is not a leave"


@pytest.mark.unit
async def test_a_replacement_that_starts_playing_within_the_hysteresis_keeps_the_bot_speaking() -> None:
    hysteresis = 0.5
    rig = Rig()
    service = make_service(
        SessionRigRoom(rig), bot_stopped_hysteresis_secs=hysteresis, stop_drain_timeout=0.2
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _speaking_then_replaced(h, rig, rec)
        await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)  # the replacement plays
        await asyncio.sleep(2 * hysteresis)
        assert count(h.down, BotStoppedSpeakingFrame) == 0, "the bot stopped and started again"
        assert count(h.down, BotStartedSpeakingFrame) == 1
