"""The pipeline does not finish while the session whose request failed after an *early start* is closing.

The failed session is closed in a task of its own, so the voice-only bot need not wait for it. The teardown
still waits for that close, bounded, as it waits for any session's: in a relayed join the close deletes the
private room, and a bot process that exits once its pipeline ends would cut that short. An unexpected leave,
whose close also runs in a task, is the control.

The close takes `CLOSE_SECS` here, an input: a relayed join's room disconnect and delete can take seconds.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.worker import ProcessorUnusablePolicy

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.protocol import RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE, TOKEN_ATTR_LAUNCH
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, delivered, tts_audio
from tests.unit._session_rig import Rig, SessionRigRoom

FAILURE = SynthesiaError("the API dropped the connection", type=ErrorType.CONNECTION)
CLOSE_SECS = 0.5
LAUNCH = "0123456789abcdef"


@pytest.mark.unit
@pytest.mark.parametrize(
    "case", ["required-failed-201", "not-required-failed-201-then-end", "not-required-leave-then-end"]
)
async def test_the_pipeline_does_not_finish_before_the_session_is_closed(case: str) -> None:
    release_close, created = asyncio.Event(), asyncio.Event()
    session = FakeAvatarSession(hold_created=created, hold_aclose=release_close, created_error=FAILURE)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, required=case.startswith("required"), stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    closed_when_the_pipeline_finished: bool | None = None

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        try:
            await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
            await h.queue(tts_audio("c1", AUDIO_1))
            await until(lambda: session.send_audio_calls, what="audio at the avatar before the 201")

            if case.endswith("leave-then-end"):
                await delivered(session.emit_avatar_left("avatar_left_room"))
            else:
                created.set()
            await until(lambda: session.aclose_count >= 1, what="the close started")
            asyncio.get_running_loop().call_later(CLOSE_SECS, release_close.set)
            await rec.wait("on_session_ended")

            if not case.startswith("required"):
                await h.queue(EndFrame())  # the user hangs up
            await h.finish(send_end_frame=False)
            closed_when_the_pipeline_finished = session.aclose_completed
        finally:
            release_close.set()

    assert rec.count("on_session_ended") == 1
    assert closed_when_the_pipeline_finished, "the pipeline finished while the session was still closing"


@pytest.mark.unit
async def test_the_real_session_releases_its_owned_room_before_the_pipeline_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real session, with an owned room as a relayed join's private room is, whose disconnect takes
    `CLOSE_SECS`. required=True: the request failing after the early start ends the pipeline."""
    monkeypatch.setattr(session_module, "secrets", SimpleNamespace(token_hex=lambda _nbytes: LAUNCH))
    gate, release_disconnect = asyncio.Event(), asyncio.Event()
    rig = Rig(
        owns_room=True,
        api=FakeSessionApiClient(gate=gate, error=FAILURE),
        attributes={TOKEN_ATTR_LAUNCH: LAUNCH},
    )
    rig.participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    rig.room.disconnect_event = release_disconnect
    service = make_service(SessionRigRoom(rig), required=True, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    deleted_when_the_pipeline_finished: int | None = None

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        try:
            await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
            gate.set()  # the request fails
            await rig.room.disconnect_started.wait()
            asyncio.get_running_loop().call_later(CLOSE_SECS, release_disconnect.set)
            await rec.wait("on_session_ended")
            await h.finish(send_end_frame=False)  # the service ends the pipeline itself
            deleted_when_the_pipeline_finished = rig.delete_calls
        finally:
            release_disconnect.set()

    assert deleted_when_the_pipeline_finished == 1, "the pipeline finished before the owned room was deleted"
