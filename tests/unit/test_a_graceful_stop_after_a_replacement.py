"""A graceful stop after the avatar worker was replaced, when the bot's audio since reached no worker.

After a same-identity replacement, the next audio can reach no worker: a byte stream opened in the gap
before the new worker joins can fail (the test double fails it; if it opened instead, no worker would read
it either), or the rest of an utterance goes on the byte stream opened to the replaced worker, whose header
the new worker never saw. Nothing will ever play it. When that audio went out within the stop hysteresis
of the replacement, as here, a later graceful stop with nothing queued must still end at once, not wait
out its drain timeout for a playback that never comes. Sent later than that, the stop does wait its full
drain timeout: a documented limit (README, "The avatar worker replaced mid-call"), not fixed, so not
pinned here. (A stop that might also wait for audio the worker that stays has not yet
started was considered and rejected: it made every such stop wait. The cost, a goodbye after a retry cut
short in a window of the stop hysteresis, is documented in the README's known limits.)

Runs the real `LiveKitAvatarSession` on the session rig behind the service. The drain timeout is longer
than `until()`'s wait, so waiting it out shows as a failure.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSStartedFrame, TTSStoppedFrame

from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.protocol import (
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_STARTED,
    TOKEN_ATTR_LAUNCH,
)
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, processed, tts_audio
from tests.unit._session_rig import AVATAR, Rig, SessionRigRoom, make_avatar

LAUNCH = "0123456789abcdef"
DRAIN = 10.0


async def _utterance(h: Harness, context_id: str, audio: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    await h.queue(tts_audio(context_id, audio))
    await h.queue(TTSStoppedFrame(context_id=context_id))


async def _the_stop_ends_at_once(h: Harness) -> None:
    finishing = asyncio.ensure_future(h.finish())  # a graceful stop, nothing queued
    try:
        await until(finishing.done, what="the stop ending with nothing left that will play")
    finally:
        await finishing


@pytest.mark.unit
@pytest.mark.parametrize("send_in_the_gap", [False, True], ids=["nothing-sent", "a-send-in-the-gap"])
async def test_a_stop_after_a_replacement_does_not_wait_for_audio_that_reached_no_worker(
    send_in_the_gap: bool,
) -> None:
    rig = Rig()
    service = make_service(SessionRigRoom(rig), bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=DRAIN)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await _utterance(h, "c1", AUDIO_1)
        await until(lambda: len(rig.participant.streams) == 1, what="c1 at the first worker")
        await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)
        await until(lambda: rig.participant.streams[0].closed, what="c1's end")

        assert rig.avatar is not None
        rig.depart(rig.avatar)  # the first worker is replaced while it plays c1
        if send_in_the_gap:
            rig.participant.fail_next_open = RuntimeError("no participant with that identity")
            await _utterance(h, "c2", AUDIO_2)
            await processed(h, "c2 sent into the gap")
            assert rig.participant.fail_next_open is None, "c2's stream was never opened"
        rig.join(make_avatar())  # the worker that stays; it never got c2
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) == 1, what="the stop timer firing")
        assert not any(AUDIO_2 in s.data for s in rig.participant.streams), "c2 reached no worker"

        await _the_stop_ends_at_once(h)


@pytest.mark.unit
async def test_a_stop_after_a_mid_utterance_replacement_does_not_wait_for_the_rest_on_the_old_stream() -> (
    None
):
    rig = Rig()
    service = make_service(SessionRigRoom(rig), bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=DRAIN)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(rig.participant.streams) == 1, what="c1 at the first worker")
        await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)

        assert rig.avatar is not None
        rig.depart(rig.avatar)  # replaced mid-utterance
        rig.join(make_avatar())
        await h.queue(tts_audio("c1", AUDIO_2))  # the rest of c1, on the stream opened to the old worker
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await until(lambda: rig.participant.streams[0].closed, what="c1's end")
        assert len(rig.participant.streams) == 1
        assert AUDIO_2 in rig.participant.streams[0].data, "the rest of c1 did not go on the old stream"
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) == 1, what="the stop timer firing")

        await _the_stop_ends_at_once(h)


@pytest.mark.unit
async def test_a_stop_after_a_retry_whose_held_rest_went_on_the_replaced_workers_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early start; c1's stream open at the first worker, which plays it; the retry holds the rest of c1;
    the retry's worker replaces the first and the 201 lands at once, so the held rest goes on the old
    stream, and nothing plays it."""
    monkeypatch.setattr(session_module, "secrets", SimpleNamespace(token_hex=lambda _nbytes: LAUNCH))
    created, retry = asyncio.Event(), asyncio.Event()
    rig = Rig(
        api=FakeSessionApiClient(gate=created, retries=1, retry_gate=retry),
        attributes={TOKEN_ATTR_LAUNCH: LAUNCH},
    )
    rig.participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    service = make_service(SessionRigRoom(rig), bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=DRAIN)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(rig.participant.streams) == 1, what="c1's stream open at the first worker")
        await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")
        await h.queue(tts_audio("c1", AUDIO_2))  # held
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await processed(h, "the rest of c1 held")

        assert rig.avatar is not None
        rig.depart(rig.avatar)
        rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: LAUNCH}))
        created.set()  # the 201, for the worker that stays
        await until(lambda: rig.participant.streams[0].closed, what="the rest of c1 sent, and c1 ended")
        assert len(rig.participant.streams) == 1
        assert AUDIO_2 in rig.participant.streams[0].data, "the held rest of c1 did not go on the old stream"
        await until(lambda: count(h.down, BotStoppedSpeakingFrame) == 1, what="the stop timer firing")

        await _the_stop_ends_at_once(h)
