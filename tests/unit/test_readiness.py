"""`AvatarReadiness`: the worker's two readiness attributes, read tolerantly.

The worker keys `avatar_status` by the unprefixed avatar id, and only ever sets `avatar_ready="true"`: the
key is absent otherwise, never `"false"`. Everything this class exposes carries the `av_` prefix
(*unprefixed ids on the wire*), so the interesting parts are the translation and the *tolerant reader*.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from loguru import logger

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit.readiness import AvatarReadiness
from pipecat_synthesia.protocol import (
    ATTR_AVATAR_READY,
    ATTR_AVATAR_STATUS,
    AVATAR_STATE_FAILED,
    AVATAR_STATE_PREPARING,
    AVATAR_STATE_READY,
)

LOG = logger.bind(component="test")
IDS = ("av_one", "av_two")


def status_attr(**states: object) -> dict[str, str]:
    """Build the worker's `avatar_status` attribute from unprefixed id -> state (or a full status dict)."""
    payload = {raw: state if isinstance(state, dict) else {"state": state} for raw, state in states.items()}
    return {ATTR_AVATAR_STATUS: json.dumps(payload, separators=(",", ":"), sort_keys=True)}


def make_readiness() -> AvatarReadiness:
    return AvatarReadiness(IDS, LOG)


def test_status_is_keyed_by_the_prefixed_id() -> None:
    readiness = make_readiness()

    assert readiness.update(status_attr(one=AVATAR_STATE_READY, two=AVATAR_STATE_PREPARING)) is True

    assert readiness.status == {"av_one": AVATAR_STATE_READY, "av_two": AVATAR_STATE_PREPARING}
    assert readiness.state_of("av_one") == AVATAR_STATE_READY
    assert readiness.all_ready is False


def test_an_unchanged_update_reports_no_change() -> None:
    readiness = make_readiness()
    attributes = status_attr(one=AVATAR_STATE_READY, two=AVATAR_STATE_READY)

    assert readiness.update(attributes) is True
    assert readiness.update(attributes) is False
    assert readiness.all_ready is True


def test_avatar_ready_marks_everything_ready_except_failures() -> None:
    """The worker sets `avatar_ready` only once every avatar is ready, but a FAILED avatar in the
    status map must not be overwritten by it."""
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_FAILED))

    assert readiness.update({ATTR_AVATAR_READY: "true"}) is True

    assert readiness.status == {"av_one": AVATAR_STATE_FAILED, "av_two": AVATAR_STATE_READY}
    assert readiness.all_ready is False


def test_avatar_ready_with_no_status_map_marks_every_known_id() -> None:
    readiness = make_readiness()

    assert readiness.update({ATTR_AVATAR_READY: "true"}) is True

    assert readiness.all_ready is True


def test_an_unknown_state_is_kept_and_is_not_ready() -> None:
    """A worker that gains a state must not turn into a spurious failure here."""
    readiness = make_readiness()

    readiness.update(status_attr(one="rehearsing", two=AVATAR_STATE_READY))

    assert readiness.state_of("av_one") == "rehearsing"
    assert readiness.all_ready is False


def test_a_status_entry_for_an_unlisted_avatar_is_kept_but_does_not_gate() -> None:
    readiness = make_readiness()

    readiness.update(
        status_attr(one=AVATAR_STATE_READY, two=AVATAR_STATE_READY, three=AVATAR_STATE_PREPARING)
    )

    assert readiness.state_of("av_three") == AVATAR_STATE_PREPARING
    assert readiness.all_ready is True, "only the session's own avatars gate readiness"


def test_unparseable_keys_are_skipped() -> None:
    readiness = make_readiness()

    readiness.update({ATTR_AVATAR_STATUS: json.dumps({"": "ready", "no spaces!": "ready"})})

    assert readiness.status == {}


@pytest.mark.parametrize(
    "state",
    [pytest.param(7, id="a-number"), pytest.param("", id="empty"), pytest.param(["ready"], id="a-list")],
)
def test_a_status_entry_whose_state_is_not_a_non_empty_string_is_ignored(state: object) -> None:
    """The rest of the same map is still read (*tolerant reader*)."""
    readiness = make_readiness()

    readiness.update(status_attr(one={"state": state}, two=AVATAR_STATE_READY))

    assert "av_one" not in readiness.status
    assert readiness.state_of("av_two") == AVATAR_STATE_READY
    assert readiness.all_ready is False


@pytest.mark.parametrize("raw", ["not json", "[]", '"a string"', "null"])
def test_malformed_status_is_ignored_with_one_warning(
    raw: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_READY))

    assert readiness.update({ATTR_AVATAR_STATUS: raw}) is False
    assert readiness.update({ATTR_AVATAR_STATUS: raw}) is False

    assert readiness.status == {"av_one": AVATAR_STATE_READY}, "nothing was lost"
    warnings = [record for record in caplog_loguru.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog_loguru.text


async def test_wait_ready_returns_when_an_update_arrives() -> None:
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_PREPARING))
    waiter = asyncio.ensure_future(readiness.wait_ready("av_one", timeout=5))

    await asyncio.sleep(0)
    assert not waiter.done()

    readiness.update(status_attr(one=AVATAR_STATE_READY))

    await waiter


async def test_wait_ready_returns_immediately_when_already_ready() -> None:
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_READY))

    await readiness.wait_ready("av_one", timeout=5)


async def test_wait_ready_raises_on_failed() -> None:
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_FAILED))

    with pytest.raises(SynthesiaError) as excinfo:
        await readiness.wait_ready("av_one", timeout=5)

    assert excinfo.value.type is ErrorType.AVATAR_SWAP_FAILED
    assert excinfo.value.code == "AVATAR_FAILED"


async def test_wait_ready_raises_on_an_unknown_id() -> None:
    readiness = make_readiness()

    with pytest.raises(SynthesiaError) as excinfo:
        await readiness.wait_ready("av_nope", timeout=5)

    assert excinfo.value.type is ErrorType.UNKNOWN_AVATAR


async def test_wait_ready_times_out_while_preparing() -> None:
    """A wait on an avatar that stays preparing raises TIMEOUT. The timeout is an input; the duration is
    never measured."""
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_PREPARING))

    with pytest.raises(SynthesiaError) as excinfo:
        await readiness.wait_ready("av_one", timeout=0.05)

    assert excinfo.value.type is ErrorType.TIMEOUT
    assert "still preparing" in str(excinfo.value)


async def test_wait_ready_keeps_waiting_through_an_unrelated_update() -> None:
    readiness = make_readiness()
    readiness.update(status_attr(one=AVATAR_STATE_PREPARING, two=AVATAR_STATE_PREPARING))
    waiter = asyncio.ensure_future(readiness.wait_ready("av_one", timeout=5))
    await asyncio.sleep(0)

    readiness.update(status_attr(two=AVATAR_STATE_READY))
    await asyncio.sleep(0)
    assert not waiter.done(), "another avatar going ready must not resolve this wait"

    readiness.update(status_attr(one=AVATAR_STATE_READY))
    await waiter
