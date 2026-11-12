"""The avatar leaves just as a launch that started early returns.

After an *early start* the launch returns before the Synthesia API's 201, so the session has no session id
yet. If the avatar has left by the time the service takes the session, the launch has failed, and the
service reports it as a failed launch, with no session id on the error, rather than failing to read one.

Runs the real `LiveKitAvatarSession` on the session rig, because only the real session refuses to give a
session id before the 201. The room adapter makes the avatar leave between the session's return and the
service taking it; the 201 never comes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from livekit import rtc as lk_rtc
from pipecat.frames.frames import EndFrame

from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.protocol import TOKEN_ATTR_LAUNCH
from pipecat_synthesia.room import AvatarSession, JoinMode, LaunchContext
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.unit._service_helpers import delivered
from tests.unit._session_rig import Rig, SessionRigRoom

LAUNCH = "0123456789abcdef"


class LeavesAsTheLaunchReturns(SessionRigRoom):
    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        session = await super().launch(ctx)
        rig = self.rig
        assert rig.avatar is not None
        rig.depart(rig.avatar, reason=lk_rtc.DisconnectReason.CLIENT_INITIATED)  # a leave, not a replacement
        return session


@pytest.mark.unit
async def test_a_leave_as_an_early_launch_returns_ends_as_a_failed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "secrets", SimpleNamespace(token_hex=lambda _nbytes: LAUNCH))
    rig = Rig(api=FakeSessionApiClient(gate=asyncio.Event()), attributes={TOKEN_ATTR_LAUNCH: LAUNCH})
    service = make_service(LeavesAsTheLaunchReturns(rig), stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        (reason,) = await rec.wait("on_session_ended")
        await h.frame(EndFrame)  # the service's own end
        await h.finish(send_end_frame=False)

    assert reason == "launch_failed"
    assert service.state is ServiceState.FAILED
    assert rec.count("on_avatar_connected") == 0


@pytest.mark.unit
async def test_a_leave_during_an_early_launch_ends_as_a_failed_launch_with_the_fake_session() -> None:
    """The same on the fake session, which, like the real one, has no session id before the 201: the avatar
    leaves while the launch runs, and the launch returns before the 201."""

    async def leave(room: FakeAvatarRoom, _ctx: LaunchContext) -> None:
        await delivered(room.session.emit_avatar_left("avatar_left_room"))

    session = FakeAvatarSession(hold_created=asyncio.Event())  # the 201 never comes
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, during_launch=leave)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        (reason,) = await rec.wait("on_session_ended")
        await h.frame(EndFrame)  # the service's own end
        await h.finish(send_end_frame=False)

    assert reason == "launch_failed"
    assert session.aclose_count == 1
