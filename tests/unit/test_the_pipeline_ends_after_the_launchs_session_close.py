"""A session the launch is closing as the pipeline ends: the pipeline finishes only once the close has.

The launch closes two sessions itself, because it never stores them where the teardown would close them:
one whose avatar left as the launch returned, and one that launched as the pipeline ended. The teardown's
first step cancels the launch, and the close it cancels carries on as a detached close (*cancel-proof
cleanup*). The teardown waits for it, bounded by what is left of the backstop it gives any session close,
so the pipeline does not finish while a session is still closing. In a relayed join that close deletes
the bot's private room, and a bot process that exits when its pipeline ends would cut the delete short.

Each test holds the session's close, releases it `CLOSE_SECS` after the pipeline starts ending, and records
whether the close had finished when the pipeline did. `CLOSE_SECS` is an input: long enough that a
pipeline that does not wait for the close finishes first, far below the backstop, which only bounds the
wait. `required=False` keeps the launch's own outcome from ending the pipeline, so the only end is the
test's.

- A leave during the launch, on a fake session (`FakeAvatarSession(hold_aclose=)`) and on the real
  `LiveKitAvatarSession` over an owned room, whose close disconnects and deletes the room
  (`FakeRoom.disconnect_event` holds the disconnect); each ended by an `EndFrame` and by a `CancelFrame`.
- A session that launched as the pipeline was cancelled. `cancel()` starts the teardown in a task, so the
  launch's return, released in the same loop turn (`FakeAvatarRoom(hold_launch=)`), reaches the launch
  first, and the launch closes the session; `stop()` cancels the launch before it can resume, so only a
  cancel reaches this case. `cancel()` is called directly, as Pipecat calls it for a `CancelFrame`, so the
  moment is exact. Below Python 3.12, Pipecat's `wait_for2` delivers the teardown's cancel to the launch
  together with its session, and the launch closes that session inline, before the cancel ends it, so the
  case does not arise there and the test is skipped. From 3.12 the close is detached, as in a leave.

The last test pins the bound: a close still running when its backstop runs out does not hold the end. The
backstop is shrunk to `BACKSTOP_SECS`, an input, and the close is held until the pipeline has finished.
"""

from __future__ import annotations

import asyncio
import sys

import pytest
from livekit import rtc as lk_rtc
from pipecat.frames.frames import CancelFrame, EndFrame, Frame

from pipecat_synthesia import service as service_module
from pipecat_synthesia.room import AvatarSession, JoinMode, LaunchContext
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service
from tests.support.waiting import until
from tests.unit._service_helpers import delivered
from tests.unit._session_rig import Rig, SessionRigRoom

CLOSE_SECS = 0.5
BACKSTOP_SECS = 0.2
ENDS = pytest.mark.parametrize("end", [EndFrame, CancelFrame], ids=["EndFrame", "CancelFrame"])


async def _the_avatar_leaves(room: FakeAvatarRoom, _ctx: LaunchContext) -> None:
    await delivered(room.session.emit_avatar_left("avatar_left_room"))


class _LeavesAsTheLaunchReturns(SessionRigRoom):
    """A relayed join's launch, on the private room it owns, whose avatar leaves as `launch()` returns."""

    join_mode = JoinMode.RELAYED

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        session = await super().launch(ctx)
        assert self.rig.avatar is not None
        self.rig.depart(self.rig.avatar, reason=lk_rtc.DisconnectReason.CLIENT_INITIATED)
        return session


@pytest.mark.unit
@ENDS
async def test_the_pipeline_finishes_after_the_close_of_a_fake_session_that_left_during_the_launch(
    end: type[Frame],
) -> None:
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, during_launch=_the_avatar_leaves)
    service = make_service(room, required=False)
    release: asyncio.TimerHandle | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: session.aclose_count == 1, what="the launch closing the session that left")
            release = asyncio.get_running_loop().call_later(CLOSE_SECS, release_close.set)
            await h.queue(end())
            await h.finish(send_end_frame=False)
            closed_when_the_pipeline_finished = session.aclose_completed
        finally:
            if release is not None:
                release.cancel()
            release_close.set()

    await until(lambda: session.aclose_completed, what="the close to finish, so nothing outlives the test")
    assert closed_when_the_pipeline_finished, "the pipeline finished while the session was still closing"


@pytest.mark.unit
@ENDS
async def test_the_pipeline_finishes_after_the_private_room_of_a_session_that_left_in_the_launch_is_deleted(
    end: type[Frame],
) -> None:
    rig = Rig(owns_room=True)
    release_disconnect = asyncio.Event()
    rig.room.disconnect_event = release_disconnect
    service = make_service(_LeavesAsTheLaunchReturns(rig), required=False)
    release: asyncio.TimerHandle | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await until(
                rig.room.disconnect_started.is_set, what="the launch's close disconnecting the private room"
            )
            release = asyncio.get_running_loop().call_later(CLOSE_SECS, release_disconnect.set)
            await h.queue(end())
            await h.finish(send_end_frame=False)
            deletes_when_the_pipeline_finished = rig.delete_calls
        finally:
            if release is not None:
                release.cancel()
            release_disconnect.set()

    await until(lambda: rig.delete_calls == 1, what="the private room's delete, so nothing outlives the test")
    assert deletes_when_the_pipeline_finished == 1, "the pipeline finished before the room was deleted"


@pytest.mark.unit
@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="below Python 3.12, Pipecat's wait_for2 hands the cancel to the launch with its session, and the "
    "launch closes it inline, so no close is detached and the test would pass without exercising the case",
)
async def test_the_pipeline_finishes_after_the_close_of_a_session_that_launched_as_it_was_cancelled() -> None:
    release_launch = asyncio.Event()
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, hold_launch=release_launch)
    service = make_service(room, required=False)
    release: asyncio.TimerHandle | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            release = asyncio.get_running_loop().call_later(CLOSE_SECS, release_close.set)
            release_launch.set()  # the avatar has joined: launch() returns on the next turn
            await service.cancel(CancelFrame())  # ... and the CancelFrame is processed first
            await until(lambda: session.aclose_count == 1, what="the launch closing the session it produced")
            await h.queue(CancelFrame())
            await h.finish(send_end_frame=False)
            closed_when_the_pipeline_finished = session.aclose_completed
        finally:
            if release is not None:
                release.cancel()
            release_close.set()

    assert not room.cancelled, "the cancel reached launch() before it returned: not the case under test"
    await until(lambda: session.aclose_completed, what="the close to finish, so nothing outlives the test")
    assert closed_when_the_pipeline_finished, "the pipeline finished while the session was still closing"


@pytest.mark.unit
async def test_a_close_that_outlasts_its_backstop_does_not_hold_the_end_of_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "SESSION_ACLOSE_BACKSTOP_SECS", BACKSTOP_SECS)
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, during_launch=_the_avatar_leaves)
    service = make_service(room, required=False)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: session.aclose_count == 1, what="the launch closing the session that left")
            await h.queue(EndFrame())
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish once the close's backstop ran out")
            await finishing
            closed_when_the_pipeline_finished = session.aclose_completed
        finally:
            release_close.set()
            if finishing is not None:
                await asyncio.wait({finishing})

    await until(lambda: session.aclose_completed, what="the close to finish, so nothing outlives the test")
    assert not closed_when_the_pipeline_finished, "the close was released before the pipeline finished"
