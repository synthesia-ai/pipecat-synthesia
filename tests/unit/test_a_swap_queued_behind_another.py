"""A swap asked for while another is still running waits for its turn, within its own `timeout`.

Swaps run one at a time, in the order they are asked for
(`test_a_swap_requested_while_the_pending_swap_waits.py` pins why). `swap_avatar(timeout=...)` still bounds
the caller's whole wait, the wait for the earlier swap included:

* a swap still waiting for its turn when its `timeout` runs out fails with a `TIMEOUT` error, pushed once
  and raised, and never reaches the avatar worker; the earlier swap, and the next one, are unaffected;
* a swap that gets its turn is given what is left of its `timeout`, not all of it again.

The earlier swap is held by `HeldSwapSession` (`tests/support/held_swap.py`), a `FakeAvatarSession` whose
first swap is recorded and then waits for an event the test sets, as a swap to a preparing avatar waits
for it to be ready. Timeouts are
inputs: `SHORT` is the queued swap's bound, `LONG` one no test reaches, and `EARLIER_SWAP` how long the
earlier swap takes in the second test.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import ErrorFrame

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.held_swap import HeldSwapSession
from tests.support.waiting import until
from tests.unit._session_rig import settle

pytestmark = pytest.mark.unit

AVATAR_IDS = ("av_a", "av_b", "av_c")
SHORT = 0.1
LONG = 60.0
EARLIER_SWAP = 0.05


async def test_a_swap_still_waiting_for_its_turn_when_its_timeout_runs_out_fails_with_a_timeout() -> None:
    release = asyncio.Event()
    session = HeldSwapSession(release)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session), avatar_ids=AVATAR_IDS)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        earlier = asyncio.ensure_future(service.swap_avatar("av_b", timeout=LONG))
        queued: asyncio.Future[str] | None = None
        try:
            await until(lambda: len(session.swap_calls) == 1, what="the earlier swap at the avatar worker")
            queued = asyncio.ensure_future(service.swap_avatar("av_c", timeout=SHORT))
            await until(queued.done, what="the queued swap to give up at its own timeout")
            assert not earlier.done(), "the queued swap waited for the earlier one, past its own timeout"

            release.set()
            assert await earlier == "av_b"
            assert await service.swap_avatar("av_c", timeout=LONG) == "av_c"  # the next swap gets its turn
        finally:
            release.set()
            for task in (earlier, queued):
                if task is not None:
                    await settle(task)
        state = service.state
        await h.finish()

    assert queued is not None
    failure = queued.exception()
    assert isinstance(failure, SynthesiaError), f"the queued swap did not fail: {queued!r}"
    assert failure.type is ErrorType.TIMEOUT
    assert [call[1] for call in session.swap_calls].count("av_c") == 1, (
        "the timed-out swap reached the worker"
    )
    pushed = [frame for frame in frames_of(h.up, ErrorFrame) if isinstance(frame, ErrorFrame)]
    assert [frame.exception for frame in pushed] == [failure]
    assert state is ServiceState.SESSION_ACTIVE
    assert service.active_avatar_id == "av_c"


async def test_a_swap_that_waited_for_its_turn_has_what_is_left_of_its_timeout() -> None:
    release = asyncio.Event()
    session = HeldSwapSession(release)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session), avatar_ids=AVATAR_IDS)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        earlier = asyncio.ensure_future(service.swap_avatar("av_b", timeout=LONG))
        queued: asyncio.Future[str] | None = None
        try:
            await until(lambda: len(session.swap_calls) == 1, what="the earlier swap at the avatar worker")
            queued = asyncio.ensure_future(service.swap_avatar("av_c", timeout=LONG))
            await h.sleep(EARLIER_SWAP)  # the earlier swap takes this long
            release.set()
            assert await queued == "av_c"
        finally:
            release.set()
            for task in (earlier, queued):
                if task is not None:
                    await settle(task)
        await h.finish()

    (given,) = [call[2] for call in session.swap_calls if call[1] == "av_c"]
    assert 0 < given < LONG, f"the queued swap was given {given}s of its {LONG}s after waiting for its turn"
