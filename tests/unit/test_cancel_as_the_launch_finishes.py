"""A pipeline end that arrives in the same event-loop turn as the launch's outcome.

`cancel()` ends the session at once: it marks the service ENDED and starts the teardown in a task, which
is what cancels the launch. `stop()` runs the same teardown inline. These tests pin that the launch's
outcome cannot undo the end, whichever way the launch ends:

* a launch that returns a session must not make the session active again: no `on_avatar_connected`, no
  pre-launch audio sent to the avatar worker, the state stays ENDED, a session that `launch()` returned
  is closed exactly once, and the session ends once, with the end's own reason;
* a launch that fails must not turn the cancel into a failed launch: no error pushed, no
  `EndWorkerFrame`, the state stays ENDED, and the end reason stays `pipeline_cancelled`.

The interleaving is built with `FakeAvatarRoom(hold_launch=)`: the test releases the launch and calls
`cancel()` (or `stop()`) without yielding in between. `cancel()` never yields while the service is
LAUNCHING, so the launch's completion is already scheduled when the teardown starts. That is the real
event "the avatar joined (or the launch failed) and the CancelFrame or EndFrame was processed in the same
loop turn". The methods are called directly, as Pipecat calls them for those frames, so the moment is
exact; the harness pipeline is there to carry the pre-launch audio and to record what the service pushes.

How the bug shows depends on the Python version, because Pipecat replaces `asyncio.wait_for` with
`wait_for2.wait_for` below Python 3.12. There, a cancel that races the launch's completion raises a
`CancelledError` that carries the session, and the service drops it unclosed. From 3.12, the launch
resumes before the teardown, so the session is reported connected, or the failure is reported, after
the end. `room.cancelled` tells the two outcomes apart: when the room saw the cancel, it rolled the
launch back and there is no session to close.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import CancelFrame, EndFrame, EndWorkerFrame, ErrorFrame, TTSStartedFrame

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState, SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent
from tests.unit._service_helpers import AUDIO_1, processed, tts_audio


async def _end(service: SynthesiaVideoService, how: str) -> None:
    if how == "cancel":
        await service.cancel(CancelFrame())
    else:
        await service.stop(EndFrame())


@pytest.mark.unit
@pytest.mark.parametrize(("how", "reason"), [("cancel", "pipeline_cancelled"), ("stop", "pipeline_ended")])
async def test_a_launch_that_returns_as_the_pipeline_ends_never_makes_the_session_active(
    how: str, reason: str
) -> None:
    """`stop_drain_timeout=0.2` is an input: it only shortens the final `EndFrame`'s drain wait if the
    service wrongly believes a session is still active when the harness ends the pipeline."""
    release = asyncio.Event()
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=release, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await processed(h, "pre-launch audio queued")
        assert service.state is ServiceState.LAUNCHING

        release.set()  # the avatar has joined: launch() returns on the next turn
        await _end(service, how)  # ... and the end is processed first
        assert service.state is ServiceState.ENDED

        await rec.wait("on_session_ended")
        await h.finish()

    assert rec.count("on_avatar_connected") == 0, rec.names()
    assert session.send_audio_calls == [], "pre-launch audio reached the avatar worker after the end"
    assert service.state is ServiceState.ENDED
    if not room.cancelled:
        assert session.aclose_count == 1, "launch() returned a session and nothing closed it"
    assert session.aclose_count <= 1
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == reason


@pytest.mark.unit
async def test_a_launch_that_fails_as_the_pipeline_is_cancelled_is_not_reported_as_a_failed_launch() -> None:
    """`required=True` (the default), so a launch failure on its own would push a permanent error and an
    `EndWorkerFrame`. After the cancel, neither may happen."""
    release = asyncio.Event()
    refused = SynthesiaError("bad key", type=ErrorType.AUTH, status=401)
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=release, fail_with=refused)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        assert service.state is ServiceState.LAUNCHING

        release.set()  # the launch fails on the next turn
        await service.cancel(CancelFrame())  # ... and the CancelFrame is processed first
        assert service.state is ServiceState.ENDED

        await rec.wait("on_session_ended")
        up = list(h.up)
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "pipeline_cancelled"
    assert service.state is ServiceState.ENDED
    assert rec.count("on_error") == 0, rec.names()
    assert_absent(up, ErrorFrame, where="upstream after the cancel")
    assert_absent(up, EndWorkerFrame, where="upstream after the cancel")
    assert rec.count("on_session_ended") == 1
