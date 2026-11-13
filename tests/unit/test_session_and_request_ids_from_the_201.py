"""The session and request ids reach the metrics and the errors at the Synthesia API's 201.

A session bills and can be looked up from its 201, whether or not the launch then succeeds, and support
needs both ids. What this file pins:

* the avatar session reports the 201 through `on_session_created(session_id, request_id, created_at)`
  once, on the clock of `LaunchTimings`;
* after a launch that fails after the 201, the summary, the pushed error and the `on_metrics` payload
  carry both ids, while the service's own `session_id` stays None;
* a failed swap's raised and pushed error, and the error a swap raises after the session ended, carry
  both ids, whether the swap was asked for after the end or was waiting for its turn when it came;
* an adapter that never reports its 201 still gets the session id into `summary()`, at session active;
* a line the plugin logs after the 201 but before the session is active carries the session id.

The service tests report the 201 from a `FakeAvatarRoom`'s `during_launch`, as a real room adapter's
session does. The first test drives the internal `LiveKitAvatarSession` on the session rig, because no
public room adapter can be given a fake Synthesia API client.
"""

from __future__ import annotations

import asyncio
import dataclasses
from time import monotonic

import pytest

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode, LaunchContext
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.held_swap import HeldSwapSession
from tests.support.waiting import until
from tests.unit._session_rig import Rig

SESSION_ID = "ses_ids"
REQUEST_ID = "req_ids"
IDS = (SESSION_ID, REQUEST_ID)


async def reports_201(_room: FakeAvatarRoom, ctx: LaunchContext) -> None:
    ctx.callbacks.on_session_created(SESSION_ID, REQUEST_ID, monotonic())


async def reports_201_then_fails(room: FakeAvatarRoom, ctx: LaunchContext) -> None:
    await reports_201(room, ctx)
    raise SynthesiaError("the avatar did not join", type=ErrorType.TIMEOUT)


def room_reporting_201(session: FakeAvatarSession | None = None) -> FakeAvatarRoom:
    session = session or FakeAvatarSession(session_id=SESSION_ID)
    return FakeAvatarRoom(JoinMode.DIRECT, during_launch=reports_201, session=session)


@pytest.mark.unit
async def test_the_session_reports_the_201_once_on_the_launch_timings_clock() -> None:
    rig = Rig(api=FakeSessionApiClient(session_id=SESSION_ID, request_id=REQUEST_ID))
    calls: list[tuple[str, str | None, float]] = []
    callbacks = dataclasses.replace(rig.ctx.callbacks, on_session_created=lambda *call: calls.append(call))
    rig.ctx = dataclasses.replace(rig.ctx, callbacks=callbacks)

    session = await rig.launch()
    try:
        assert calls == [(SESSION_ID, REQUEST_ID, session.timings.session_created)]
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_a_launch_that_fails_after_the_201_reports_both_ids_everywhere() -> None:
    service = make_service(FakeAvatarRoom(JoinMode.RELAYED, during_launch=reports_201_then_fails))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_metrics")
        await h.finish()

    assert service.session_id is None, "the session never became active"
    err = rec.payload("on_error")[0].exception
    assert (err.session_id, err.request_id) == IDS
    for summary in (service.metrics.summary(), rec.payload("on_metrics")[0].summary()):
        assert (summary.get("session_id"), summary.get("request_id")) == IDS, summary
        assert "session_created" in summary, summary


@pytest.mark.unit
async def test_a_failed_swap_raises_and_pushes_an_error_with_both_ids() -> None:
    refused = SynthesiaError("swap refused", type=ErrorType.AVATAR_SWAP_FAILED)
    room = room_reporting_201(FakeAvatarSession(session_id=SESSION_ID, swap_error=refused))
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        with pytest.raises(SynthesiaError) as caught:
            await service.swap_avatar("av_b")
        await h.finish()

    assert (caught.value.session_id, caught.value.request_id) == IDS
    pushed = rec.payload("on_error")[0].exception
    assert (pushed.session_id, pushed.request_id) == IDS


@pytest.mark.unit
async def test_a_swap_after_the_session_ended_raises_an_error_with_both_ids() -> None:
    room = room_reporting_201()
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        room.session.emit_avatar_left("avatar_left_room", humans_present=True)
        await rec.wait("on_session_ended")
        with pytest.raises(SynthesiaError) as caught:
            await service.swap_avatar("av_b")
        await h.finish()

    assert (caught.value.session_id, caught.value.request_id) == IDS


@pytest.mark.unit
async def test_a_swap_whose_session_ended_while_it_waited_its_turn_carries_both_ids() -> None:
    release = asyncio.Event()
    session = HeldSwapSession(release, session_id=SESSION_ID)
    service = make_service(room_reporting_201(session), avatar_ids=("av_a", "av_b", "av_c"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        earlier = asyncio.ensure_future(service.swap_avatar("av_b"))
        await until(lambda: len(session.swap_calls) == 1, what="the earlier swap at the avatar worker")
        queued = asyncio.ensure_future(service.swap_avatar("av_c"))
        # The leave's callback task is created after the queued swap's task, so the queued swap is already
        # waiting for its turn when the session ends.
        session.emit_avatar_left("avatar_left_room", humans_present=True)
        await rec.wait("on_session_ended")
        release.set()
        await earlier
        with pytest.raises(SynthesiaError) as caught:
            await queued
        await h.finish()

    assert (caught.value.session_id, caught.value.request_id) == IDS


@pytest.mark.unit
async def test_an_adapter_that_never_reports_the_201_still_gets_the_session_id_into_the_summary() -> None:
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=FakeAvatarSession(session_id=SESSION_ID)))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.finish()

    assert service.metrics.summary().get("session_id") == SESSION_ID


@pytest.mark.unit
async def test_a_line_logged_after_the_201_and_before_session_active_carries_the_session_id(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    # required=False, so the launch failing after the 201 logs a warning instead of ending the pipeline:
    # a line the service logs between the 201 and a session active that never comes.
    room = FakeAvatarRoom(JoinMode.RELAYED, during_launch=reports_201_then_fails)
    service = make_service(room, required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_metrics")
        await h.finish()

    assert service.session_id is None, "the session never became active"
    unavailable = [r for r in caplog_loguru.records if "Synthesia avatar unavailable" in r.getMessage()]
    assert len(unavailable) == 1, caplog_loguru.messages
    assert unavailable[0].extra.get("session_id") == SESSION_ID  # type: ignore[attr-defined]
