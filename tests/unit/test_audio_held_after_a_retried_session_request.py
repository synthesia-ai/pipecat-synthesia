"""The bot's audio while a session request retried after an *early start* is on its way.

`SessionApiClient` retries a failed session request with the same token, and the failed attempt may already
have launched the avatar worker in the room: the retry's worker then joins as the same identity and
replaces it. Seen live: a 502 after 3.9 s with the worker publishing, a second worker 2 s
later, and a reply sent in between never heard. So after an early start, a retry holds the bot's audio
until the request settles: its 201 follows the worker that stays. The *launch timeout* bounds the hold.

The hold sits in the service's send loop: a held item stays queued. What this file pins:
* a reply queued during the hold reaches the replacement worker whole, and the first worker gets nothing
  after the retry;
* when the request then fails or runs out of time, every chunk held reaches the user voice-only, none the
  avatar;
* a held chunk does not count as sent: the bot is not reported speaking over the hold;
* during a graceful stop the hold still applies, until the request settles: the goodbye reaches the worker
  that stays after a replacement, goes to the avatar in the room when the request fails (even after the
  launch timeout ran out during the stop), and is lost if the request is still out when the drain times out.

Runs the real `LiveKitAvatarSession` on the session rig behind the service; the launch id is pinned so the
avatar in the room is this launch's.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.protocol import (
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    TOKEN_ATTR_LAUNCH,
)
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count, frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio
from tests.unit._session_rig import AVATAR, Rig, SessionRigRoom, make_avatar

LAUNCH = "0123456789abcdef"
FAILURE = SynthesiaError("the API returned 502 twice", type=ErrorType.CONNECTION)
#: How long "and nothing was sent" is watched for; an input, not an outcome.
NOTHING_HAPPENS = 0.2


@pytest.fixture(autouse=True)
def _launch_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_module, "secrets", SimpleNamespace(token_hex=lambda _nbytes: LAUNCH))


def _retried_rig(*, error: BaseException | None = None) -> tuple[Rig, asyncio.Event, asyncio.Event]:
    """This launch's avatar already publishing (an early start); the 201 and the retry each held."""
    created, retry = asyncio.Event(), asyncio.Event()
    api = FakeSessionApiClient(gate=created, retries=1, retry_gate=retry, error=error)
    rig = Rig(api=api, attributes={TOKEN_ATTR_LAUNCH: LAUNCH})
    rig.participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    return rig, created, retry


async def _utterance(h: Harness, context_id: str, *chunks: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    for chunk in chunks:
        await h.queue(tts_audio(context_id, chunk))
    await h.queue(TTSStoppedFrame(context_id=context_id))


def _bytes_at_the_avatar(rig: Rig) -> int:
    return sum(len(stream.data) for stream in rig.participant.streams)


@pytest.mark.unit
async def test_a_reply_during_the_hold_reaches_the_replacement_worker_whole() -> None:
    rig, created, retry = _retried_rig()
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        await _utterance(h, "c1", AUDIO_1)  # the greeting
        await until(
            lambda: _bytes_at_the_avatar(rig) == len(AUDIO_1), what="the greeting at the first worker"
        )
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")

        await _utterance(h, "c2", AUDIO_2, AUDIO_3)  # the reply, during the hold
        await processed(h, "the reply queued")
        await asyncio.sleep(NOTHING_HAPPENS)
        assert len(rig.participant.streams) == 1, "the first worker got audio after the retry"

        assert rig.avatar is not None
        rig.depart(rig.avatar)  # the retry's worker replaces the first
        rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: LAUNCH}))
        created.set()  # then the 201, for the worker that stays
        await rec.wait("on_avatar_connected")
        await until(lambda: len(rig.participant.streams) == 2, what="the reply at the new worker")
        await until(lambda: rig.participant.streams[1].closed, what="the reply's end")

    assert rig.participant.streams[1].data == AUDIO_2 + AUDIO_3


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ["the-request-fails", "the-launch-timeout"])
async def test_audio_held_when_the_request_fails_reaches_the_user_voice_only(outcome: str) -> None:
    """Every held chunk reaches the user voice-only, in order. The request's failure ends the hold as it ends
    the launch, so the send loop may wake just before the failure path retires it: at most one chunk then
    reaches the abandoned avatar, which is sent clear buffer, and the chunk whose send was cut short goes back
    for the handover. The launch timeout closes the session first, so nothing reaches the avatar then."""
    rig, created, retry = _retried_rig(error=FAILURE)
    service = make_service(SessionRigRoom(rig), required=False, launch_timeout=1.0, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")
        await _utterance(h, "c1", AUDIO_1, AUDIO_2, AUDIO_3)
        await processed(h, "c1 queued")

        if outcome == "the-request-fails":
            created.set()
        (reason,) = await rec.wait("on_session_ended")
        await processed(h, "after the handover")

    assert reason == "degraded"
    assert _bytes_at_the_avatar(rig) <= (len(AUDIO_1) if outcome == "the-request-fails" else 0)
    assert [f.audio for f in frames_of(h.down, TTSAudioRawFrame)] == [AUDIO_1, AUDIO_2, AUDIO_3]


@pytest.mark.unit
async def test_a_held_chunk_does_not_count_as_sent_after_an_interruption() -> None:
    """The interrupted utterance's `playback_started` arrives late, after the next utterance's chunk was
    queued and held; then the worker answers the clear buffer with its interrupted finish. Nothing is
    playing and nothing was sent since the interruption, so the bot stops speaking."""
    rig, created, retry = _retried_rig()
    service = make_service(SessionRigRoom(rig), bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=0.2)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(rig.participant.streams) == 1, what="c1 at the avatar")
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")

        await h.queue(InterruptionFrame())
        await until(
            lambda: any(c["method"] == RPC_CLEAR_BUFFER for c in rig.participant.perform_rpc_calls),
            what="the clear buffer",
        )
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))  # held
        await processed(h, "c2 queued")
        await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)  # c1's, late
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 1, what="the late start")
        finished = json.dumps({"playback_position": 0.1, "interrupted": True})
        await rig.participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=finished)

        await until(lambda: count(h.down, BotStoppedSpeakingFrame) == 1, what="the bot no longer speaking")
        assert len(rig.participant.streams) == 1, "c2 is still held"
        created.set()


async def _the_avatar_plays_it(rig: Rig) -> None:
    await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)
    finished = json.dumps({"playback_position": 1.0, "interrupted": False})
    await rig.participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=finished)


@pytest.mark.unit
async def test_a_goodbye_held_at_a_graceful_stop_reaches_the_worker_that_stays() -> None:
    """The app stops while the bot's last words are held; within the stop's drain the retry's worker replaces
    the first and the 201 lands for the one that stays. The hold still applies during the stop, so the
    goodbye reaches the worker that stays, and none of it the first."""
    rig, created, retry = _retried_rig()
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=10.0, bot_stopped_hysteresis_secs=0.05)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")
        await _utterance(h, "bye", AUDIO_1, AUDIO_2)  # the goodbye, held
        await processed(h, "the goodbye queued")

        finishing = asyncio.ensure_future(h.finish())  # a graceful stop
        try:
            await asyncio.sleep(NOTHING_HAPPENS)  # the stop is under way
            assert not rig.participant.streams, "the goodbye went to the worker about to be replaced"
            assert rig.avatar is not None
            rig.depart(rig.avatar)  # the retry's worker replaces the first
            rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: LAUNCH}))
            created.set()  # then the 201, for the worker that stays
            await until(
                lambda: rig.participant.streams and rig.participant.streams[0].closed, what="the goodbye"
            )
            await _the_avatar_plays_it(rig)
            await until(finishing.done, what="the stop finishing without waiting out its drain")
        finally:
            created.set()
            await finishing

    assert [s.data for s in rig.participant.streams] == [AUDIO_1 + AUDIO_2]


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ["the-request-fails", "the-launch-timeout-then-the-request-fails"])
async def test_a_graceful_stop_waits_for_the_request_to_settle(outcome: str) -> None:
    """The app stops while the goodbye is held, and the retried request then fails inside the drain: the
    hold ends, the goodbye goes to the avatar in the room, and the drain finishes with it. In the second case
    the launch timeout runs out first, during the stop, so the service has stopped waiting for the request;
    its failure still ends the hold. The drain timeout is longer than `until()`'s wait, so waiting it out
    shows as a failure; the launch timeout is an input placed inside the drain."""
    rig, created, retry = _retried_rig(error=FAILURE)
    launch_timeout = 1.0 if outcome == "the-launch-timeout-then-the-request-fails" else 30.0
    service = make_service(
        SessionRigRoom(rig),
        launch_timeout=launch_timeout,
        stop_drain_timeout=10.0,
        bot_stopped_hysteresis_secs=0.05,
    )

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")
        await _utterance(h, "bye", AUDIO_1, AUDIO_2)  # the goodbye, held
        await processed(h, "the goodbye queued")

        finishing = asyncio.ensure_future(h.finish())  # a graceful stop
        try:
            if outcome == "the-launch-timeout-then-the-request-fails":
                await asyncio.sleep(launch_timeout)  # the launch timeout runs out during the stop
            else:
                await asyncio.sleep(NOTHING_HAPPENS)
            assert not rig.participant.streams, "the goodbye was sent before the request settled"
            created.set()  # the retried request fails
            await until(lambda: _bytes_at_the_avatar(rig) == len(AUDIO_1) + len(AUDIO_2), what="the goodbye")
            await _the_avatar_plays_it(rig)
            await until(finishing.done, what="the stop finishing without waiting out its drain")
        finally:
            created.set()
            await finishing

    assert [s.data for s in rig.participant.streams] == [AUDIO_1 + AUDIO_2]


@pytest.mark.unit
async def test_a_request_still_out_when_the_drain_runs_out_loses_the_goodbye() -> None:
    """The documented cost: during a graceful stop the hold waits for the request, bounded by the drain
    timeout; a request still out at that bound loses the goodbye, and the stop still ends at the bound."""
    rig, created, retry = _retried_rig()
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=0.5)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")
        await _utterance(h, "bye", AUDIO_1, AUDIO_2)  # the goodbye, held; the request never settles
        await processed(h, "the goodbye queued")
        finishing = asyncio.ensure_future(h.finish())
        try:
            await until(finishing.done, what="the stop ending at its drain timeout")
        finally:
            created.set()
            await finishing

    assert not rig.participant.streams


@pytest.mark.unit
@pytest.mark.parametrize("retried", [False, True], ids=["no-retry", "retried"])
async def test_a_graceful_stop_with_nothing_to_say_does_not_wait_for_the_request(retried: bool) -> None:
    """Only audio is held: a stop with nothing queued ends at once, as without a retry, rather than waiting
    for the request to settle or the drain to time out. The drain timeout is longer than `until()`'s wait."""
    created, retry = asyncio.Event(), asyncio.Event()
    rig = Rig(
        api=FakeSessionApiClient(gate=created, retries=1 if retried else 0, retry_gate=retry),
        attributes={TOKEN_ATTR_LAUNCH: LAUNCH},
    )
    rig.participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=10.0)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        if retried:
            retry.set()
            await until(lambda: rig.api.retries_reported == 1, what="the retry")
        finishing = asyncio.ensure_future(h.finish())  # nothing queued, the request still out
        try:
            await until(finishing.done, what="the stop, with nothing to drain")
        finally:
            created.set()
            await finishing


@pytest.mark.unit
async def test_a_failed_request_sends_at_most_one_chunk_on_a_stream_already_open() -> None:
    """The utterance's stream to the first worker was opened before the retry, so a held chunk woken by the
    request's failure needs no new stream: at most one chunk still reaches the abandoned avatar before the
    failure path retires the send loop, and the rest of the utterance reaches the user voice-only, in
    order. Needs the fake's writer to yield as rtc's does: without that, the woken loop would write every
    held chunk in one step."""
    rig, created, retry = _retried_rig(error=FAILURE)
    service = make_service(SessionRigRoom(rig), required=False, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: _bytes_at_the_avatar(rig) == len(AUDIO_1), what="c1's stream open at the avatar")
        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry")
        await h.queue(tts_audio("c1", AUDIO_2))
        await h.queue(tts_audio("c1", AUDIO_3))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await processed(h, "the rest of c1 held")

        created.set()  # the retried request fails
        await rec.wait("on_session_ended")
        await processed(h, "after the handover")

    assert len(rig.participant.streams) == 1, "the held chunks needed no new stream"
    assert len(rig.participant.streams[0].writes) <= 2, "more than one chunk reached the abandoned avatar"
    assert [f.audio for f in frames_of(h.down, TTSAudioRawFrame)] == [AUDIO_2, AUDIO_3]
