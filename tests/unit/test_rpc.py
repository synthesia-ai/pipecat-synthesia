"""`PlaybackRpcHandlers`, `perform_clear_buffer`, `perform_swap_avatar`, `error_from_rpc`.

Both playback handlers are registered, answer with a `str`, never raise, and reject any caller but the
avatar. `clear_buffer` is bounded and never raises. Every `RpcError` code maps to its own error type
rather than all to CONNECTION.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import pytest
from livekit import rtc as lk_rtc
from loguru import logger

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._types import CLEAR_BUFFER_SECS, RPC_ACK_SECS
from pipecat_synthesia.livekit.rpc import (
    PlaybackRpcHandlers,
    error_from_rpc,
    perform_clear_buffer,
    perform_swap_avatar,
)
from pipecat_synthesia.protocol import (
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_REJECT,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    RPC_SWAP_AVATAR,
    SWAP_DEFAULT,
)
from tests.fakes.fake_rtc import FakeLocalParticipant, never_returns, reply_after
from tests.support.waiting import until

AVATAR = "synthesia-avatar-agent"
LOG = logger.bind(component="test")
CODES = lk_rtc.RpcError.ErrorCode


class Recorder:
    """The session's playback callbacks: `on_started()` and `on_finished(position, interrupted)`."""

    def __init__(self, *, fail: bool = False) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []
        self._fail = fail

    async def on_started(self) -> None:
        self.started += 1
        if self._fail:
            raise RuntimeError("callback blew up")

    async def on_finished(self, position: float, interrupted: bool) -> None:
        self.finished.append((position, interrupted))
        if self._fail:
            raise RuntimeError("callback blew up")


def make_handlers(
    participant: FakeLocalParticipant, recorder: Recorder, *, expected_caller: str = AVATAR
) -> PlaybackRpcHandlers:
    return PlaybackRpcHandlers(
        participant,
        expected_caller=expected_caller,
        on_started=recorder.on_started,
        on_finished=recorder.on_finished,
        log=LOG,
    )


# --- PlaybackRpcHandlers -------------------------------------------------------------------------


async def test_register_records_both_methods_in_order_and_is_idempotent() -> None:
    """Both playback handlers are registered, once each: without `lk.playback_finished`, the worker ends
    its session after the first utterance."""
    participant = FakeLocalParticipant()
    handlers = make_handlers(participant, Recorder())

    handlers.register()
    handlers.register()

    assert participant.registration_order == [RPC_PLAYBACK_STARTED, RPC_PLAYBACK_FINISHED]
    assert set(participant.rpc_methods) == {RPC_PLAYBACK_STARTED, RPC_PLAYBACK_FINISHED}


async def test_handlers_schedule_the_callbacks_with_parsed_values() -> None:
    participant = FakeLocalParticipant()
    recorder = Recorder()
    make_handlers(participant, recorder).register()

    assert await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR) == "ok"
    payload = json.dumps({"playback_position": 1.25, "interrupted": True, "synchronized_transcript": None})
    assert await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=payload) == "ok"

    await until(lambda: recorder.started == 1, what="on_started")
    await until(lambda: recorder.finished == [(1.25, True)], what="on_finished")


async def test_missing_keys_become_defaults() -> None:
    participant = FakeLocalParticipant()
    recorder = Recorder()
    make_handlers(participant, recorder).register()

    await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload="{}")

    await until(lambda: recorder.finished == [(0.0, False)], what="the default payload")


async def test_a_foreign_caller_is_rejected_and_no_callback_runs() -> None:
    """The bot must not act on another participant's playback events: they are rejected, and no
    callback runs."""
    participant = FakeLocalParticipant()
    recorder = Recorder()
    make_handlers(participant, recorder).register()

    assert await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity="someone-else") == "reject"
    assert (
        await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity="someone-else", payload="{}")
        == "reject"
    )

    await asyncio.sleep(0)  # give a (wrongly) scheduled callback task a chance to run
    assert recorder.started == 0
    assert recorder.finished == []


async def test_a_raising_callback_does_not_propagate(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """A raising callback is logged and the worker still gets "ok": a raise reaching the worker arrives as
    `APPLICATION_ERROR`, which is fatal after three retries."""
    participant = FakeLocalParticipant()
    recorder = Recorder(fail=True)
    make_handlers(participant, recorder).register()

    assert await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR) == "ok"
    assert await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload="{}") == "ok"

    await until(lambda: len(recorder.finished) == 1, what="the failing callback to run")
    await until(
        lambda: any(record.levelno == logging.ERROR for record in caplog_loguru.records),
        what="the guard to log the failure",
    )


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "null",
        "[]",
        '{"playback_position":"x"}',
        '{"interrupted":"yes"}',
        '{"playback_position":true}',
        "not json at all",
        "x" * 1_000_000,
    ],
)
async def test_handler_return_is_str_and_never_raises_on_any_payload(payload: str) -> None:
    """Whatever arrives, the handler answers with a `str` and the callback gets usable defaults.

    `interrupted` is deliberately truthiness-based (`"yes"` -> True): the worker only ever sends a
    JSON bool, and guessing at a string would be a silent behaviour change if it ever sent one.
    """
    participant = FakeLocalParticipant()
    recorder = Recorder()
    make_handlers(participant, recorder).register()

    result = await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=payload)

    assert isinstance(result, str)
    assert result == RPC_HANDLER_RESPONSE
    await until(lambda: len(recorder.finished) == 1, what="the callback")
    position, interrupted = recorder.finished[0]
    assert position == 0.0
    assert isinstance(interrupted, bool)
    assert interrupted is (payload == '{"interrupted":"yes"}'), (
        "a non-bool `interrupted` is taken for its truthiness; the worker only ever sends a JSON bool"
    )


async def test_unregister_twice_is_safe() -> None:
    participant = FakeLocalParticipant()
    handlers = make_handlers(participant, Recorder())
    handlers.register()

    handlers.unregister()
    handlers.unregister()

    assert participant.rpc_methods == {}
    assert participant.unregistered == [RPC_PLAYBACK_STARTED, RPC_PLAYBACK_FINISHED]


# --- perform_clear_buffer ------------------------------------------------------------------------


async def test_clear_buffer_ok() -> None:
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, log=LOG) is True

    call = participant.perform_rpc_calls[0]
    assert call["destination_identity"] == AVATAR
    assert call["method"] == RPC_CLEAR_BUFFER
    assert call["payload"] == ""
    # Both, by value. The ACK wait is `max_round_trip_latency`, which defaults to 7 s (livekit rtc 1.1.19);
    # ours is half the timeout, at most RPC_ACK_SECS. Comparing the two kwargs with each other would pass if
    # both were dropped (`None == None`), the regression this catches.
    assert call["response_timeout"] == CLEAR_BUFFER_SECS
    assert call["max_round_trip_latency"] == min(RPC_ACK_SECS, CLEAR_BUFFER_SECS / 2)


async def test_clear_buffer_reject_returns_false_with_one_warning(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A "reject" answer means the worker is bound to a different bot identity: False, and one warning."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_REJECT)

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, log=LOG) is False

    warnings = [record for record in caplog_loguru.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog_loguru.text
    assert "reject" in warnings[0].getMessage()


async def test_clear_buffer_rpc_error_returns_false() -> None:
    participant = FakeLocalParticipant()
    participant.script_rpc(
        RPC_CLEAR_BUFFER, lk_rtc.RpcError(CODES.RECIPIENT_NOT_FOUND, "Recipient not found")
    )

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, log=LOG) is False


async def test_clear_buffer_that_never_answers_returns_false() -> None:
    """A clear_buffer the worker never answers returns False. The result is the assertion, never the
    elapsed time."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_CLEAR_BUFFER, never_returns)

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, timeout=0.05, log=LOG) is False


class SlowToAnswerParticipant(FakeLocalParticipant):
    """Answers each RPC after `delay` seconds: an input that models a worker that is slow but in time."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    async def perform_rpc(self, **kwargs: Any) -> str:
        await asyncio.sleep(self.delay)
        return await super().perform_rpc(**kwargs)


async def test_a_short_clear_buffer_timeout_still_hears_a_worker_that_answers() -> None:
    """The overall bound is the RPC's own timeout plus a margin, never less: an answer at 80% of the timeout
    is heard, so a bound shrunk below 0.4 s fails here."""
    participant = SlowToAnswerParticipant(delay=0.4)
    participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, timeout=0.5, log=LOG) is True
    assert len(participant.perform_rpc_calls) == 1


async def test_a_clear_buffer_answered_in_0_2_s_at_its_default_timeout_is_heard() -> None:
    """At its default 2 s timeout, the worker's deadline for clear buffer is rtc's 1 s floor, which a worker
    answering in 0.2 s meets (live round trips are about 0.15 s)."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_CLEAR_BUFFER, reply_after(0.2, RPC_HANDLER_RESPONSE))

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, log=LOG) is True


async def test_a_clear_buffer_answered_well_inside_its_timeout_is_heard() -> None:
    """A worker that answers 1.25 s into a 3 s clear buffer is heard.

    rtc sends the worker's handler a deadline of `response_timeout - max_round_trip_latency`, at least 1 s;
    here 1.5 s. Passing the timeout as both values made it 1 s, so this answer was a `RESPONSE_TIMEOUT` and
    the call returned False.
    """
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_CLEAR_BUFFER, reply_after(1.25, RPC_HANDLER_RESPONSE))

    assert await perform_clear_buffer(participant, avatar_identity=AVATAR, timeout=3.0, log=LOG) is True


# --- perform_swap_avatar -------------------------------------------------------------------------


async def test_swap_strips_the_av_prefix_on_the_wire() -> None:
    """The swap payload carries the unprefixed avatar id, and the result is prefixed again: the worker keys
    its avatars by the unprefixed id and strips nothing (*unprefixed ids on the wire*)."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, json.dumps({"status": "ok", "avatar_id": "abc123"}))

    assert (
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_abc123", timeout=5.0)
        == "av_abc123"
    )

    call = participant.perform_rpc_calls[0]
    assert call["method"] == RPC_SWAP_AVATAR
    assert json.loads(call["payload"]) == {"avatar_id": "abc123"}
    assert call["response_timeout"] == 5.0


async def test_swap_default_is_passed_through() -> None:
    """`SWAP_DEFAULT` goes to the worker as-is: the worker resolves it to the first entry of `avatarIds`."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, json.dumps({"status": "ok", "avatar_id": "abc123"}))

    await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id=SWAP_DEFAULT, timeout=5.0)

    assert json.loads(participant.perform_rpc_calls[0]["payload"]) == {"avatar_id": SWAP_DEFAULT}


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ok", "avatar_id": "abc123"},
        {"avatar_id": "abc123"},
        {"avatar_id": "av_abc123"},
        {"status": "ok", "avatar_id": "abc123", "some_future_key": 1},
    ],
)
async def test_swap_success_variants_return_a_prefixed_id(response: dict[str, object]) -> None:
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, json.dumps(response))

    assert (
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_abc123", timeout=5.0)
        == "av_abc123"
    )


async def test_swap_error_body_carries_the_worker_code() -> None:
    participant = FakeLocalParticipant()
    participant.script_rpc(
        RPC_SWAP_AVATAR,
        json.dumps({"error": "avatar abc123 still preparing", "code": "AVATAR_PREPARING"}),
    )

    with pytest.raises(SynthesiaError) as excinfo:
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_abc123", timeout=5.0)

    assert excinfo.value.type is ErrorType.AVATAR_SWAP_FAILED
    assert excinfo.value.code == "AVATAR_PREPARING"
    assert "still preparing" in str(excinfo.value)


@pytest.mark.parametrize(
    "confirmed,expected",
    [
        ("abc.123", "av_abc.123"),  # a character the avatar-id pattern does not allow
        ("a" * 80, "av_" + "a" * 80),  # longer than the pattern allows
        ("av_abc/123", "av_abc/123"),  # already prefixed: kept as it is
    ],
)
async def test_swap_confirmed_id_outside_the_pattern_is_still_a_success(
    confirmed: str, expected: str
) -> None:
    """The worker has already swapped when it answers, so a confirmed id the avatar-id pattern would reject
    is returned, prefixed, rather than reported as a failed swap (*tolerant reader*)."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, json.dumps({"status": "ok", "avatar_id": confirmed}))

    result = await perform_swap_avatar(
        participant, avatar_identity=AVATAR, avatar_id="av_abc123", timeout=5.0
    )

    assert result == expected


@pytest.mark.parametrize(
    "response",
    ["not json", '"a string"', "[]", "{}", '{"avatar_id": 7}', '{"avatar_id": ""}', '{"avatar_id": "  "}'],
)
async def test_swap_malformed_response(response: str) -> None:
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, response)

    with pytest.raises(SynthesiaError) as excinfo:
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_abc123", timeout=5.0)

    assert excinfo.value.type is ErrorType.AVATAR_SWAP_FAILED
    assert str(excinfo.value) == "avatar swap returned a malformed response"


@pytest.mark.slow
async def test_a_swap_answered_after_5_s_succeeds_within_the_default_15_s_timeout() -> None:
    """A worker that takes 5 s to swap, inside the default `swap_timeout` of 15 s, is heard.

    Found in a live run: with the timeout passed to rtc as both `response_timeout` and
    `max_round_trip_latency`, the deadline rtc sent the worker's handler was 1 s, so every swap that took
    longer failed with `TIMEOUT` after about a second, whatever the timeout. It is now 13 s.
    """
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, reply_after(5.0, json.dumps({"status": "ok", "avatar_id": "b"})))

    assert (
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_b", timeout=15.0)
        == "av_b"
    )


@pytest.mark.parametrize(
    ("timeout", "answer_secs"),
    [(4.0, 1.5), (3.0, 1.25)],
    ids=["4s-timeout", "3s-timeout"],
)
async def test_the_worker_is_given_at_least_half_the_swap_timeout(timeout: float, answer_secs: float) -> None:
    """A swap the worker answers inside half its timeout succeeds: the avatar worker is given the timeout
    minus two seconds, and at least half of it. The 3 s case is where the half matters: a plain 2 s
    acknowledgement allowance would leave the worker 1 s."""
    participant = FakeLocalParticipant()
    participant.script_rpc(
        RPC_SWAP_AVATAR, reply_after(answer_secs, json.dumps({"status": "ok", "avatar_id": "b"}))
    )

    assert (
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_b", timeout=timeout)
        == "av_b"
    )


async def test_a_swap_answered_past_its_deadline_at_the_worker_is_a_response_timeout() -> None:
    """At a 4 s timeout the worker's deadline is 2 s: a swap it answers after 3 s is a `TIMEOUT` carrying
    LiveKit's `RESPONSE_TIMEOUT` code, sent by the worker's rtc when the deadline passes."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, reply_after(3.0, json.dumps({"status": "ok", "avatar_id": "b"})))

    with pytest.raises(SynthesiaError) as excinfo:
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_b", timeout=4.0)

    assert excinfo.value.type is ErrorType.TIMEOUT
    assert excinfo.value.code == str(int(CODES.RESPONSE_TIMEOUT))


@pytest.mark.parametrize("answer", [reply_after(30.0, "{}"), never_returns], ids=["answers-late", "never"])
async def test_a_swap_the_worker_does_not_answer_in_time_fails_within_its_timeout_and_a_margin(
    answer: object,
) -> None:
    """A swap the worker answers too late, or never, fails with `TIMEOUT` no later than its timeout plus
    the 0.5 s margin around the RPC. The upper bound is the assertion; how much sooner it fails is rtc's."""
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, answer)
    timeout = 1.0

    started = time.monotonic()
    with pytest.raises(SynthesiaError) as excinfo:
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_b", timeout=timeout)
    elapsed = time.monotonic() - started

    assert excinfo.value.type is ErrorType.TIMEOUT
    # 0.5 s of slack for timer lateness on a loaded CI machine: the bound being checked is timeout + 0.5.
    assert elapsed < timeout + 0.5 + 0.5, f"failed after {elapsed:.2f}s"


async def test_swap_rpc_error_is_mapped() -> None:
    participant = FakeLocalParticipant()
    participant.script_rpc(RPC_SWAP_AVATAR, lk_rtc.RpcError(CODES.RESPONSE_TIMEOUT, "Response timeout"))

    with pytest.raises(SynthesiaError) as excinfo:
        await perform_swap_avatar(participant, avatar_identity=AVATAR, avatar_id="av_abc123", timeout=0.1)

    assert excinfo.value.type is ErrorType.TIMEOUT
    assert excinfo.value.code == str(int(CODES.RESPONSE_TIMEOUT))


# --- error_from_rpc ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (CODES.UNSUPPORTED_METHOD, ErrorType.CONNECTION),
        (CODES.RECIPIENT_NOT_FOUND, ErrorType.CONNECTION),
        (CODES.RECIPIENT_DISCONNECTED, ErrorType.CONNECTION),
        (CODES.RESPONSE_TIMEOUT, ErrorType.TIMEOUT),
        (CODES.CONNECTION_TIMEOUT, ErrorType.TIMEOUT),
        (CODES.APPLICATION_ERROR, ErrorType.AVATAR_SWAP_FAILED),
        (CODES.SEND_FAILED, ErrorType.CONNECTION),
        (CODES.REQUEST_PAYLOAD_TOO_LARGE, ErrorType.CONNECTION),
        (CODES.UNSUPPORTED_SERVER, ErrorType.CONNECTION),
        (CODES.UNSUPPORTED_VERSION, ErrorType.CONNECTION),
        (CODES.RESPONSE_PAYLOAD_TOO_LARGE, ErrorType.CONNECTION),
    ],
)
def test_error_from_rpc_maps_every_code(code: int, expected: ErrorType) -> None:
    """Every `RpcError` code maps to its own error type: folding them all into CONNECTION would lose the
    retryability distinction."""
    err = error_from_rpc(lk_rtc.RpcError(code, "boom"), method=RPC_SWAP_AVATAR)

    assert err.type is expected
    assert err.code == str(int(code))
    if expected is not ErrorType.AVATAR_SWAP_FAILED:
        # APPLICATION_ERROR is excluded on purpose: it carries the remote's message verbatim, which
        # `test_error_from_rpc_keeps_an_application_error_message_verbatim` owns. An `or` here would
        # make that row assert nothing at all.
        assert RPC_SWAP_AVATAR in str(err)


def test_error_from_rpc_keeps_an_application_error_message_verbatim() -> None:
    err = error_from_rpc(
        lk_rtc.RpcError(CODES.APPLICATION_ERROR, "avatar abc123 unavailable"),
        method=RPC_SWAP_AVATAR,
    )

    assert str(err) == "avatar abc123 unavailable"


def test_error_from_rpc_names_an_unknown_code() -> None:
    err = error_from_rpc(lk_rtc.RpcError(4242, "something new"), method=RPC_SWAP_AVATAR)

    assert err.type is ErrorType.CONNECTION
    assert "4242" in str(err)
    assert err.code == "4242"
