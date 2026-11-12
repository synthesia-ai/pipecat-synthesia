"""A `cleanup()` that joined a teardown whose run is then cancelled.

Teardown is *single flight*: the first of `stop()`, `cancel()` and `cleanup()` runs it, and a later caller
joins the run in progress. A run that is cancelled part-way releases whoever joined it and clears the
slot, so that a later `cleanup()` runs the remaining steps. This file pins the joiner's side: a `cleanup()`
that had already joined the run when the run was cancelled returns only once the teardown has finished,
so the session end is recorded (`on_session_ended` and `on_metrics`, once each).

The run is `stop()`'s, which Pipecat runs inline on the processor's process task. The test runs `stop()`
in a task of its own and cancels that task, as Pipecat cancels the process task when a `CancelFrame`
arrives. Pipecat calls `cleanup()` only after the `CancelFrame` has passed the processor, so this order
needs a cancel from outside Pipecat's own lifecycle. The session's close is held (`hold_aclose`), which
keeps the run in its session close while `cleanup()` joins it, and is released as the run is cancelled.
The service is driven by hand, as in `test_service_lifecycle.py`, so that the order is exact; it gets an
HTTP session of the test's, which the test closes.
"""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from pipecat.frames.frames import EndFrame

from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import make_service, make_setup
from tests.support.events import EventRecorder
from tests.support.waiting import assert_no_dangling, until


@pytest.mark.unit
async def test_a_cleanup_that_joined_a_cancelled_teardown_returns_once_the_teardown_has_finished() -> None:
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    http = aiohttp.ClientSession()
    stopping: asyncio.Future[None] | None = None
    cleaning: asyncio.Future[None] | None = None
    try:
        service = make_service(room, session=http)
        rec = EventRecorder(service)
        setup, tm = make_setup()
        await service.setup(setup)
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")

        stopping = asyncio.ensure_future(service.stop(EndFrame()))
        await until(lambda: session.aclose_count == 1, what="stop()'s teardown closing the session")
        cleaning = asyncio.ensure_future(service.cleanup())  # joins the run on its first turn ...
        stopping.cancel()  # ... which comes before this cancel reaches the run
        release_close.set()
        await until(cleaning.done, what="cleanup() to return")
        await cleaning
        await asyncio.wait({stopping})
        assert stopping.cancelled(), "the run was not cancelled: the interleaving did not happen"

        assert rec.count("on_session_ended") == 1, f"cleanup() returned first; events: {rec.names()}"
        assert rec.count("on_metrics") == 1, f"cleanup() returned first; events: {rec.names()}"
        assert_no_dangling(tm)
    finally:
        release_close.set()
        for task in (stopping, cleaning):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait({task})
        await http.close()
