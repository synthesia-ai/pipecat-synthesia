"""Sending audio while the avatar worker is replaced by one with the same identity.

The replaced worker is gone for a moment before the new one joins (the *replacement grace* covers that
gap). A byte stream opened in that moment can fail. With an *early start*, a session request retried after
the avatar started speaking can cause exactly this replacement, and nothing holds the bot's audio for it.

What this file pins: a send that fails in the gap costs that utterance only; the send loop goes on, the
session stays active, and the next utterance reaches the new worker. Runs the real `LiveKitAvatarSession`
on the session rig behind the service.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import TTSStartedFrame, TTSStoppedFrame

from pipecat_synthesia.service import ServiceState
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio
from tests.unit._session_rig import Rig, SessionRigRoom, make_avatar


async def _utterance(h: Harness, context_id: str, audio: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    await h.queue(tts_audio(context_id, audio))
    await h.queue(TTSStoppedFrame(context_id=context_id))


@pytest.mark.unit
async def test_a_send_that_fails_while_the_avatar_is_replaced_costs_only_that_utterance() -> None:
    rig = Rig()
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await _utterance(h, "c1", AUDIO_1)
        await until(lambda: len(rig.participant.streams) == 1, what="c1 at the first worker")

        # the first worker leaves as livekit-server reports a replacement; opening a stream in the gap fails
        assert rig.avatar is not None
        rig.depart(rig.avatar)
        rig.participant.fail_next_open = RuntimeError("no participant with that identity")
        await _utterance(h, "c2", AUDIO_2)
        await processed(h, "c2 sent into the gap")

        rig.join(make_avatar())  # the new worker
        await _utterance(h, "c3", AUDIO_3)
        await until(
            lambda: any(stream.data == AUDIO_3 for stream in rig.participant.streams),
            what="c3 at the new worker",
        )
        assert not any(AUDIO_2 in stream.data for stream in rig.participant.streams), "c2's send failed"
        assert service.state is ServiceState.SESSION_ACTIVE
        assert rec.count("on_avatar_disconnected") == 0, "a replacement is not a leave"
