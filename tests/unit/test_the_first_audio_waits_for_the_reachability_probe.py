"""The bot's first audio waits until the avatar worker has answered the reachability probe.

When the user is in the room before the bot starts, the bot asks for a greeting at once, and the greeting
waits in the send queue while the avatar launches. Once the avatar worker has published its video the session
goes active and, before this fix, the greeting went out straight away. But on LiveKit Cloud the worker's own
connection can take up to about a second more to learn that the bot exists, and a byte stream from a sender it
doesn't know yet never reaches it: the greeting was silently lost. A request from that sender, though, is only
delayed until the worker learns of it.

So as the launch ends, the session sends the worker one request, the *reachability probe*, on a method no
worker registers. LiveKit answers it with `UNSUPPORTED_METHOD`, and only once the worker's connection knows
the bot: that answer shows the next stream will arrive. The send loop holds the bot's audio until the probe
has finished, however it finishes: answered, failed, out of time, or cancelled by the close. A worker that
never answers costs the bound, and then the audio goes anyway, with one warning.

`ack_when(event, answer)` scripts a worker that acknowledges the probe only once `event` is set: the moment
its connection learns of the bot. These run the real `LiveKitAvatarSession` on the session rig, alone or
behind the service, whose send loop is where the audio is held. Each test names the change to the product
that must turn it red.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from livekit import rtc as lk_rtc
from pipecat.frames.frames import InterruptionFrame, TTSStartedFrame, TTSStoppedFrame

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.protocol import RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE, RPC_REACHABILITY_PROBE
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_rtc import ack_when, never_returns
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, processed, records_at, tts_audio
from tests.unit._session_rig import AVATAR, GRACE_MARGIN, Rig, SessionRigRoom, launched

GREETING = AUDIO_1
ANSWER = AUDIO_2
CODES = lk_rtc.RpcError.ErrorCode
#: The probe task's name, which is how test 5 finds it among the loop's tasks.
PROBE_TASK = "synthesia-reachability-probe"
#: How long "and nothing was sent" is watched for; an input, not an outcome.
NOTHING_HAPPENS = GRACE_MARGIN
#: A bound that runs out inside a test.
SHORT_BOUND = 0.1
#: Far longer than `until()`'s 5 s: a hold released inside `until()` did not wait for this bound.
LONG_BOUND = 60.0


def unsupported_method() -> lk_rtc.RpcError:
    """What a worker that doesn't implement the probe answers: today's worker. Fresh for every test."""
    return lk_rtc.RpcError(CODES.UNSUPPORTED_METHOD, "Method not supported at destination")


def probe_calls(rig: Rig) -> list[dict[str, Any]]:
    return [call for call in rig.participant.perform_rpc_calls if call["method"] == RPC_REACHABILITY_PROBE]


def probe_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return records_at(caplog, logging.WARNING, "reachability probe")


async def _utterance(h: Harness, context_id: str, audio: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    await h.queue(tts_audio(context_id, audio))
    await h.queue(TTSStoppedFrame(context_id=context_id))


# ------------------------------------------------------------------------------------- behind the service


@pytest.mark.unit
async def test_the_greeting_waits_for_the_probes_answer_then_goes_whole() -> None:
    """The greeting is queued while the avatar launches. The launch ends, and the worker's connection has not
    learned of the bot yet: nothing goes out. Once it has, and answers the probe, the greeting goes out once,
    whole, on one stream.

    Sabotage: `sending` ignores the probe's task. The greeting then goes out before the worker answers."""
    created, learned = asyncio.Event(), asyncio.Event()
    rig = Rig(api=FakeSessionApiClient(gate=created), probe_answer=ack_when(learned, unsupported_method()))
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _utterance(h, "greeting", GREETING)  # queued while the avatar launches
        await processed(h, "the greeting queued")
        created.set()
        await rec.wait("on_avatar_connected")
        await until(lambda: bool(probe_calls(rig)), what="the reachability probe")
        await asyncio.sleep(NOTHING_HAPPENS)
        assert rig.participant.streams == [], "the greeting went out before the worker could know the bot"

        learned.set()
        await until(
            lambda: bool(rig.participant.streams) and rig.participant.streams[0].closed,
            what="the whole greeting",
        )
        assert [stream.data for stream in rig.participant.streams] == [GREETING]


@pytest.mark.unit
async def test_an_interruption_while_the_probe_waits_does_not_block_the_next_reply(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """The user speaks while the greeting is held behind the probe: the greeting is dropped, as any queued
    audio is at an interruption, and the send loop starts again. The answer queued after it still waits for
    the probe, then goes out; nothing deadlocks, and nothing is logged as an error.

    Sabotage: `wait_sending` awaits the probe's task directly. Cancelling the send loop at the interruption
    then cancels the probe, and the answer goes out before the worker has answered."""
    learned = asyncio.Event()
    rig = Rig(probe_answer=ack_when(learned, unsupported_method()))
    rig.participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    service = make_service(SessionRigRoom(rig), stop_drain_timeout=0.2)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the session active")
        await until(lambda: bool(probe_calls(rig)), what="the reachability probe")
        await _utterance(h, "greeting", GREETING)
        await processed(h, "the greeting queued")
        await h.queue(InterruptionFrame())
        await _utterance(h, "answer", ANSWER)
        await processed(h, "the answer queued")
        await asyncio.sleep(NOTHING_HAPPENS)
        assert rig.participant.streams == [], "audio went out before the worker answered the probe"

        learned.set()
        await until(
            lambda: bool(rig.participant.streams) and rig.participant.streams[-1].closed, what="the answer"
        )
        assert [stream.data for stream in rig.participant.streams] == [ANSWER]

    errors = [record for record in caplog_loguru.records if record.levelno >= logging.ERROR]
    assert errors == [], caplog_loguru.text


# ------------------------------------------------------------------------------------------- the session


@pytest.mark.unit
async def test_the_probe_goes_out_once_on_its_own_method_with_the_whole_bound_to_be_acknowledged() -> None:
    """One request, on a method of its own, with an empty payload, to the avatar. Its acknowledgement
    allowance is the whole bound, because a request sent before the worker knows the bot is acknowledged late
    rather than never.

    Sabotage: send `lk.clear_buffer` as the probe, or give it the interruption's timeouts, or retry it in
    short attempts."""
    learned = asyncio.Event()  # never set: the worker never learns of the bot during the test
    async with launched(probe_answer=ack_when(learned, unsupported_method())) as (rig, session):
        await until(lambda: bool(probe_calls(rig)), what="the reachability probe")
        await asyncio.sleep(NOTHING_HAPPENS)

        (call,) = probe_calls(rig)
        assert call["destination_identity"] == AVATAR
        assert call["payload"] == ""
        assert call["max_round_trip_latency"] == _types.REACHABILITY_PROBE_SECS
        assert call["response_timeout"] >= _types.REACHABILITY_PROBE_SECS
        assert [c for c in rig.participant.perform_rpc_calls if c["method"] == RPC_CLEAR_BUFFER] == []
        assert not session.sending


@pytest.mark.unit
async def test_no_answer_within_the_bound_sends_anyway_warns_once_and_never_retries(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """A worker that never acknowledges: the hold ends at the bound, with exactly one warning, and the probe
    is never sent again. Here the 2 s timeout is the assertion: at the default 3 s bound it fails.

    Sabotage: read the bound when the module is imported, not when the probe starts (the test then waits 3 s
    and fails its 2 s timeout); or retry the probe."""
    monkeypatch.setattr(_types, "REACHABILITY_PROBE_SECS", SHORT_BOUND)
    async with launched(probe_answer=ack_when(asyncio.Event(), unsupported_method())) as (rig, session):
        await asyncio.wait_for(session.wait_sending(), 2.0)
        assert session.sending
        assert len(probe_warnings(caplog_loguru)) == 1, caplog_loguru.text

        await asyncio.sleep(10 * SHORT_BOUND)
        assert len(probe_calls(rig)) == 1, "the probe was sent again"
        assert len(probe_warnings(caplog_loguru)) == 1, caplog_loguru.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer, warns",
    [("a-string", False), ("unsupported-method", False), ("application-error", True)],
)
async def test_an_answer_releases_the_hold_at_once_and_only_a_failure_warns(
    answer: str, warns: bool, monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """Any string, and LiveKit's `UNSUPPORTED_METHOD` (what today's worker answers), are answers: the hold
    ends at once, with no warning. Any other error ends it at once too, rather than at the bound, with one
    warning. The bound is far longer than `until()`'s wait, so a release that waited for it fails here.

    Sabotage: treat `UNSUPPORTED_METHOD` as a failure; or hold until the bound after an error."""
    monkeypatch.setattr(_types, "REACHABILITY_PROBE_SECS", LONG_BOUND)
    scripted: object = {
        "a-string": lambda: "ok",
        "unsupported-method": unsupported_method,
        "application-error": lambda: lk_rtc.RpcError(CODES.APPLICATION_ERROR, "Application error"),
    }[answer]()
    async with launched(probe_answer=scripted) as (_rig, session):
        await until(lambda: session.sending, what="the hold released without waiting for the bound")
        assert len(probe_warnings(caplog_loguru)) == (1 if warns else 0), caplog_loguru.text


@pytest.mark.unit
async def test_the_close_releases_the_hold_and_stops_the_probe(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The session closes while the probe waits: whatever waits on the hold returns at once, the probe's task
    is gone, and nothing is logged about the probe. Here the 1 s timeout is the assertion.

    Sabotage: drop the probe's cancel from the close (`_close_everything`)."""
    monkeypatch.setattr(_types, "REACHABILITY_PROBE_SECS", LONG_BOUND)
    async with launched(probe_answer=never_returns) as (rig, session):
        await until(lambda: bool(probe_calls(rig)), what="the reachability probe")
        waiting = asyncio.ensure_future(session.wait_sending())
        try:
            await asyncio.sleep(0)
            assert not waiting.done(), "nothing held the audio"
            await session.aclose()
            await asyncio.wait_for(waiting, 1.0)
        finally:
            if not waiting.done():
                waiting.cancel()
            await asyncio.wait({waiting})

        left = [task for task in asyncio.all_tasks() if task.get_name() == PROBE_TASK and not task.done()]
        assert left == [], "the probe outlived the close"
        assert probe_warnings(caplog_loguru) == [], caplog_loguru.text


@pytest.mark.unit
async def test_a_leave_while_the_probe_waits_logs_no_probe_failure(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The avatar leaves (a declared leave, not a replacement) while the probe waits, and the probe then fails
    because the recipient is gone. The leave is what gets reported; a probe failure after it means nothing,
    and is not logged.

    Sabotage: warn whether or not the session is still alive (drop the `_alive` check)."""
    monkeypatch.setattr(_types, "REACHABILITY_PROBE_SECS", LONG_BOUND)
    learned = asyncio.Event()
    gone = lk_rtc.RpcError(CODES.RECIPIENT_DISCONNECTED, "Recipient disconnected")
    async with launched(probe_answer=ack_when(learned, gone)) as (rig, session):
        await until(lambda: bool(probe_calls(rig)), what="the reachability probe")
        assert rig.avatar is not None
        rig.depart(rig.avatar, reason=lk_rtc.DisconnectReason.CLIENT_INITIATED)  # a leave, not a replacement
        await until(lambda: bool(rig.rec.left), what="the leave reported")
        learned.set()  # the request reaches the worker's side, which is gone
        await until(lambda: session.sending, what="the probe finished")
        assert probe_warnings(caplog_loguru) == [], caplog_loguru.text


@pytest.mark.unit
async def test_the_probe_starts_as_the_launch_ends_not_at_the_first_audio() -> None:
    """No audio is ever sent, and the probe still goes out: it starts as the launch ends, so its round trip
    overlaps the bot's first reply instead of delaying it.

    Sabotage: start the probe lazily, from `send_audio` or `wait_sending`."""
    async with launched() as (rig, _session):
        await until(lambda: bool(probe_calls(rig)), what="the reachability probe, with no audio sent")
