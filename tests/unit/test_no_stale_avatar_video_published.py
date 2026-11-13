"""A launch that fails before the 201 reports no `avatar_video_published`.

Until the Synthesia API's 201, the avatar publishing in the room may be a stale one with the same
identity, so a failed launch with no session id clears the metric. What this file pins: such a launch,
with the avatar's video already published, reports no `avatar_video_published`; a launch that fails
after the 201, and a successful one, with the video published before the session is active, still
report it; and a pipeline that ends during the launch, after the video was published but before the 201,
reports no `avatar_video_published` either, although the launch's cancellation runs no failure handler.
"""

from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode, LaunchContext
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until


async def video_published_then_failed(_room: FakeAvatarRoom, ctx: LaunchContext) -> None:
    await ctx.callbacks.on_track_published("video")
    raise SynthesiaError("the session request failed", type=ErrorType.CONNECTION)


async def created_then_video_published(_room: FakeAvatarRoom, ctx: LaunchContext) -> None:
    ctx.callbacks.on_session_created("ses_video", "req_video", monotonic())
    await ctx.callbacks.on_track_published("video")


async def created_then_video_published_then_failed(room: FakeAvatarRoom, ctx: LaunchContext) -> None:
    await created_then_video_published(room, ctx)
    raise SynthesiaError("the avatar did not join", type=ErrorType.TIMEOUT)


@pytest.mark.unit
@pytest.mark.parametrize(
    "during_launch, reported",
    [
        (video_published_then_failed, False),
        (created_then_video_published_then_failed, True),
        (created_then_video_published, True),
    ],
    ids=["failed-before-the-201", "failed-after-the-201", "succeeded"],
)
async def test_avatar_video_published_only_for_a_launch_past_the_201(during_launch, reported: bool) -> None:
    service = make_service(FakeAvatarRoom(JoinMode.RELAYED, during_launch=during_launch), required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait(
            "on_avatar_connected" if during_launch is created_then_video_published else "on_metrics"
        )
        await h.finish()

    assert ("avatar_video_published" in rec.payload("on_metrics")[0].summary()) is reported


@pytest.mark.unit
async def test_a_pipeline_ended_during_the_launch_before_the_201_reports_no_video_published() -> None:
    published = asyncio.Event()

    async def video_published_then_waits(_room: FakeAvatarRoom, ctx: LaunchContext) -> None:
        await ctx.callbacks.on_track_published("video")
        published.set()
        await asyncio.Event().wait()  # never reaches the 201

    service = make_service(FakeAvatarRoom(JoinMode.RELAYED, during_launch=video_published_then_waits))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(published.is_set, what="the avatar's video published during the launch")
        await h.finish()

    summary = rec.payload("on_metrics")[0].summary()
    assert summary.get("end_reason") == "pipeline_ended", summary
    assert "avatar_video_published" not in summary, summary
