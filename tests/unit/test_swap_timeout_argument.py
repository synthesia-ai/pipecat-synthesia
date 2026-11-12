"""`swap_avatar(timeout=...)`: a positive bound, or `None` for the service's default.

What this pins: `0`, a negative number and NaN are a `ValueError`, raised before anything is sent and
never pushed as an `ErrorFrame`, because a bad argument is a mistake in the calling code, not a failed swap
(*misuse vs session errors*); before this, `0` silently meant the default. `None` still means the default,
and a positive timeout reaches the session unchanged. The timeouts here are inputs only: the fake session
records what it was given, and no test waits on one.
"""

from __future__ import annotations

import math

import pytest
from pipecat.frames.frames import ErrorFrame

from pipecat_synthesia import JoinMode, ServiceState
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent

pytestmark = pytest.mark.unit

#: A default no test would pick by accident, set through the config seam.
DEFAULT_SWAP_TIMEOUT = 7.25


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, math.nan], ids=["zero", "zero-float", "negative", "nan"])
async def test_a_timeout_that_is_not_positive_is_a_value_error_and_nothing_is_sent(bad: float) -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"), swap_timeout=DEFAULT_SWAP_TIMEOUT)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        with pytest.raises(ValueError, match="timeout must be positive"):
            await h.call(lambda: service.swap_avatar("av_b", timeout=bad))
        assert service.state is ServiceState.SESSION_ACTIVE
        await h.finish()

    assert room.session.swap_calls == []
    assert service.active_avatar_id == "av_a"
    assert_absent(h.up, ErrorFrame)


async def test_none_means_the_default_and_a_positive_timeout_is_passed_on() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"), swap_timeout=DEFAULT_SWAP_TIMEOUT)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        assert await h.call(lambda: service.swap_avatar("av_b")) == "av_b"
        assert await h.call(lambda: service.swap_avatar("av_a", timeout=None)) == "av_a"
        assert await h.call(lambda: service.swap_avatar("av_b", timeout=2.5)) == "av_b"
        await h.finish()

    assert [(call[1], call[2]) for call in room.session.swap_calls] == [
        ("av_b", DEFAULT_SWAP_TIMEOUT),
        ("av_a", DEFAULT_SWAP_TIMEOUT),
        ("av_b", 2.5),
    ]
    assert_absent(h.up, ErrorFrame)
