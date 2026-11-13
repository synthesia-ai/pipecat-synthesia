"""Every line at INFO or above that the service and its session print after the 201 shows the session id.

Loguru's default format, which developers and the examples use, never prints bound fields, so a session
id that is only bound is invisible. What this file pins, for a relayed join under a sink with loguru's
default format: no line before the Synthesia API's 201 has the id; every line at INFO or above from the
service and its session after it does, including the lines of loggers derived before the 201 (the private
room's and the session's); the DEBUG lines after it do too; and the summary line prints its values, with its
fields still bound.

The relayed join is the real `PrivateLiveKitRoom` on its `room_factory` and `api_factory` seams, behind a
small `AvatarRoom` that swaps the launch context's API for a fake, because no public route takes a fake
Synthesia API client.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import loguru
import pytest
from loguru import logger
from loguru._defaults import LOGURU_FORMAT

from pipecat_synthesia.livekit.private_room import PrivateLiveKitRoom
from pipecat_synthesia.room import AvatarRoom, AvatarSession, JoinMode, LaunchContext
from tests.fakes.fake_lk_api import LiveKitApiFactory, RoomFactory
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.secrets import LK_API_SECRET
from tests.unit._session_rig import make_avatar

SESSION_ID = "ses_printed"
INFO = logger.level("INFO").no


class RelayedJoinOnAFakeApi(AvatarRoom):
    """The real `PrivateLiveKitRoom`; the avatar joins as the fake Synthesia API answers its request."""

    join_mode = JoinMode.RELAYED

    def __init__(self, on_201: Any) -> None:
        self.rooms = RoomFactory(name="private-room")
        self.adapter = PrivateLiveKitRoom(
            url="wss://livekit.example",
            api_key="APIlivekit",
            api_secret=LK_API_SECRET,
            room_name="private-room",
            room_factory=self.rooms,
            api_factory=LiveKitApiFactory(),
        )

        async def avatar_joins(*_args: Any) -> None:
            private = self.rooms.room
            private.emit("participant_connected", private.add_participant(make_avatar()))
            on_201()  # the 201 reaches the session as this returns

        self.api = FakeSessionApiClient(session_id=SESSION_ID, on_create=avatar_joins)

    @property
    def avatar_identity(self) -> str:
        return self.adapter.avatar_identity

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        return await self.adapter.launch(dataclasses.replace(ctx, api=self.api))


def from_the_plugin(messages: list[loguru.Message]) -> list[loguru.Message]:
    return [m for m in messages if m.record["extra"].get("component") == "pipecat_synthesia"]


@pytest.mark.unit
async def test_lines_after_the_201_print_the_session_id_and_lines_before_it_do_not() -> None:
    messages: list[loguru.Message] = []
    at_201: list[int] = []
    handler_id = logger.add(
        lambda m: messages.append(m), format=str(LOGURU_FORMAT), level="DEBUG", colorize=False
    )
    try:
        service = make_service(RelayedJoinOnAFakeApi(on_201=lambda: at_201.append(len(messages))))
        rec = EventRecorder(service)
        async with bidirectional_pipeline(service) as h:
            await rec.wait("on_avatar_connected")
            await h.finish()
    finally:
        logger.remove(handler_id)

    before = from_the_plugin(messages[: at_201[0]])
    after = [m for m in from_the_plugin(messages[at_201[0] :]) if m.record["level"].no >= INFO]
    assert before and [str(m) for m in before if SESSION_ID in str(m)] == []
    assert [str(m) for m in after if f"[{SESSION_ID}] " not in str(m)] == []
    debug = [m for m in from_the_plugin(messages[at_201[0] :]) if m.record["level"].name == "DEBUG"]
    assert debug and [str(m) for m in debug if f"[{SESSION_ID}] " not in str(m)] == []
    # the private room's and the session's loggers were derived before the 201, and never rebound
    modules = {m.record["name"] for m in after}
    assert {"pipecat_synthesia.livekit.private_room", "pipecat_synthesia.livekit.session"} <= modules
    summary = next(m for m in after if "synthesia session summary" in str(m))
    assert "end_reason" in str(summary) and summary.record["extra"]["end_reason"] == "pipeline_ended"
