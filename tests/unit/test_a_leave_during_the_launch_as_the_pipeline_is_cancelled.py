"""A leave during the launch, whose session close finishes as the pipeline is cancelled.

When the avatar leaves during the launch, the launch closes the session `launch()` returned and reports a
failed launch: with `required=True`, FAILED, a permanent error and an `EndWorkerFrame` upstream; with
`required=False`, DEGRADED and an error. A `cancel()` during that close starts the teardown, which cancels
the launch. This file pins the case where the close's completion is delivered to the launch before the
`CancelFrame` is processed, so the launch resumes before the teardown can cancel it. The end is still the
pipeline's, as for any launch outcome that races the end (`test_cancel_as_the_launch_finishes.py`): the
state stays ENDED, the end reason is `pipeline_cancelled`, no error is pushed and no `EndWorkerFrame` goes
upstream.

The leave is delivered inside `launch()` (`during_launch`), so `launch()` returns a session that has left,
and `hold_aclose` holds the service's close of it. The test releases the close and counts event-loop
turns with `asyncio.sleep(0)`, which waits for no time: one turn for the close to finish, one for its
completion to reach the launch's wait. `cancel()` runs in that turn, before the launch's next one; one
turn earlier, the teardown cancels the launch before it resumes. `cancel()` is called directly, as Pipecat
calls it for a `CancelFrame`, so that the moment is exact.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import CancelFrame, EndWorkerFrame, ErrorFrame

from pipecat_synthesia.room import JoinMode, LaunchContext
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent
from tests.support.waiting import until
from tests.unit._service_helpers import delivered


async def _the_avatar_leaves(room: FakeAvatarRoom, _ctx: LaunchContext) -> None:
    await delivered(room.session.emit_avatar_left("avatar_left_room", humans_present=True))


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "not_required"])
async def test_a_leave_during_the_launch_is_not_reported_once_the_pipeline_is_cancelled(
    required: bool,
) -> None:
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, during_launch=_the_avatar_leaves)
    service = make_service(room, required=required)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: session.aclose_count == 1, what="the launch closing the session that left")
            assert service.state is ServiceState.LAUNCHING

            release_close.set()
            await asyncio.sleep(0)  # one loop turn: the close finishes
            assert session.aclose_completed
            await asyncio.sleep(0)  # one loop turn: its completion reaches the launch's wait
            # ... and the CancelFrame is processed in that turn, before the launch resumes
            await service.cancel(CancelFrame())

            await rec.wait("on_session_ended")
        finally:
            release_close.set()
        up = list(h.up)
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "pipeline_cancelled"
    assert service.state is ServiceState.ENDED
    assert rec.count("on_error") == 0, rec.names()
    assert_absent(up, ErrorFrame, where="upstream after the cancel")
    assert_absent(up, EndWorkerFrame, where="upstream after the cancel")
    assert rec.count("on_session_ended") == 1
