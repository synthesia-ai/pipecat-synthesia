"""The service after an *early start*: the session is active, and audio flows, before the 201.

`FakeAvatarSession(hold_created=...)` is a session whose launch returned before the Synthesia API's 201,
which `wait_session_created()` then waits for (see `test_an_early_start_on_this_launchs_video.py` for the
real session). In that window the service is SESSION_ACTIVE: queued audio goes to the avatar, an
interruption interrupts it, a leave ends it. Only what needs the session id waits for the 201:
`on_avatar_connected`, then `on_avatar_ready`, and `session_id`.

A 201 that fails after an early start ends the service as that failed launch would have without it:
`required` decides between FAILED and DEGRADED, and the session is closed.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import (
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.worker import ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, delivered, processed, tts_audio

SESSION_ID = "ses_fake"  # FakeAvatarSession's default


async def _utterance(h: Harness, context_id: str, audio: bytes) -> None:
    await h.queue(TTSStartedFrame(context_id=context_id))
    await h.queue(tts_audio(context_id, audio))
    await h.queue(TTSStoppedFrame(context_id=context_id))


def _early(
    join_mode: JoinMode = JoinMode.DIRECT, **kwargs
) -> tuple[FakeAvatarRoom, FakeAvatarSession, asyncio.Event]:
    created = asyncio.Event()
    session = FakeAvatarSession(hold_created=created, **kwargs)
    return FakeAvatarRoom(join_mode, session=session), session, created


@pytest.mark.unit
async def test_audio_reaches_the_avatar_before_on_avatar_connected() -> None:
    room, session, created = _early()
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _utterance(h, "c1", AUDIO_1)
        await until(
            lambda: session.end_utterance_count == 1, what="the utterance at the avatar before the 201"
        )
        assert service.state is ServiceState.SESSION_ACTIVE
        assert rec.count("on_avatar_connected") == 0
        assert service.session_id is None

        created.set()
        (connected,) = await rec.wait("on_avatar_connected")
        assert connected.session_id == SESSION_ID
        assert service.session_id == SESSION_ID
        await h.finish()

    assert [c[1] for c in session.send_audio_calls] == [AUDIO_1]


@pytest.mark.unit
async def test_on_avatar_ready_never_comes_before_on_avatar_connected() -> None:
    """The avatar reports every avatar ready while the 201 is on its way: `on_avatar_ready` waits for
    `on_avatar_connected`, and fires once."""
    room, session, created = _early(all_ready=True, avatar_status={"av_a": "ready"})
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")
        await delivered(session.emit_status({"av_a": "ready"}, True))
        assert rec.count("on_avatar_ready") == 0

        created.set()
        await rec.wait("on_avatar_ready")
        await h.finish()

    names = [n for n in rec.names() if n in ("on_avatar_connected", "on_avatar_ready")]
    assert names == ["on_avatar_connected", "on_avatar_ready"]


@pytest.mark.unit
async def test_an_interruption_before_the_201_interrupts_the_avatar() -> None:
    """Audio has reached the avatar, so an interruption in the window is a real one: `interrupt` reaches
    the session, as after `on_avatar_connected`. A guard: a service that kept the window short of
    SESSION_ACTIVE would leave the avatar talking over the user until the 201."""
    room, session, created = _early()
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: session.send_audio_calls, what="audio at the avatar before the 201")
        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 1, what="interrupt at the avatar before the 201")

        created.set()
        await rec.wait("on_avatar_connected")
        await h.finish()


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "not-required"])
async def test_a_session_request_failing_after_an_early_start_ends_as_a_failed_launch(required: bool) -> None:
    """The 201 fails after audio has gone to the avatar: the session is closed, no `on_avatar_connected`,
    and the end is the failed launch's. `required=True`: FAILED, and the service ends the pipeline.
    `required=False`: DEGRADED, and later TTS audio passes through voice-only."""
    room, session, created = _early(
        created_error=SynthesiaError("the API dropped the connection", type=ErrorType.CONNECTION)
    )
    service = make_service(room, required=required, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        await _utterance(h, "c1", AUDIO_1)
        await until(
            lambda: session.end_utterance_count == 1, what="the utterance at the avatar before the 201"
        )

        created.set()
        await rec.wait("on_session_ended")
        assert service.state is (ServiceState.FAILED if required else ServiceState.DEGRADED)
        assert session.aclose_count == 1
        if required:
            await h.frame(EndFrame)  # the service's own end
            await h.finish(send_end_frame=False)
        else:
            await _utterance(h, "c2", AUDIO_2)
            await until(lambda: count(h.down, TTSAudioRawFrame) == 1, what="later audio passed through")
            await h.finish()

    assert rec.count("on_avatar_connected") == 0
    assert rec.payload("on_session_ended")[0] == ("launch_failed" if required else "degraded")
    assert [c[1] for c in session.send_audio_calls] == [AUDIO_1]


@pytest.mark.unit
async def test_the_avatar_leaving_before_the_201_ends_the_session_without_on_avatar_connected() -> None:
    room, session, created = _early()
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")
        await delivered(session.emit_avatar_left("avatar_left_room"))
        await rec.wait("on_session_ended")
        created.set()  # the 201 lands after the end
        await processed(h, "after the 201")
        await h.finish()

    assert rec.count("on_avatar_connected") == 0
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert service.state is ServiceState.ENDED


@pytest.mark.unit
async def test_a_201_that_lands_while_the_pipeline_stops_fires_no_on_avatar_connected() -> None:
    """The pipeline is ending (a graceful stop waits for the avatar to finish speaking) when the 201 lands:
    no `on_avatar_connected`, as a launch never made active once the pipeline is ending. Without this
    rule it could even follow `on_avatar_disconnected`."""
    room, session, created = _early()
    service = make_service(room, bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=5.0)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _utterance(h, "c1", AUDIO_1)
        await until(lambda: session.end_utterance_count == 1, what="c1 at the avatar before the 201")
        await delivered(session.emit_playback_started())

        await h.queue(EndFrame())
        await until(lambda: session.end_utterance_count == 2, what="the graceful stop's final end")
        created.set()  # the stop now waits for the avatar to stop speaking
        await delivered(session.emit_playback_finished())
        await h.finish(send_end_frame=False)

    assert rec.count("on_avatar_connected") == 0
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"


@pytest.mark.unit
async def test_a_relayed_join_passes_the_avatars_media_before_the_201() -> None:
    """In a relayed join the avatar's video and its speech reach the user through the service. Once the
    session is active the avatar may be speaking, so its media passes from then, not from the 201, or the
    user would miss the start of what it says."""
    room, session, created = _early(JoinMode.RELAYED)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(OutputTransportReadyFrame(), FrameDirection.UPSTREAM)
        await h.frame(OutputTransportReadyFrame, direction=FrameDirection.UPSTREAM)
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")
        await delivered(session.emit_video(512, 512))
        await delivered(session.emit_audio(b"\x00\x10" * 480, 24000, 1))
        assert count(h.down, OutputImageRawFrame) == 1
        assert count(h.down, OutputAudioRawFrame) == 1
        assert rec.count("on_avatar_connected") == 0

        created.set()
        await rec.wait("on_avatar_connected")
        await h.finish()


@pytest.mark.unit
@pytest.mark.parametrize("end", ["leave-before-201", "201-during-stop"])
async def test_a_session_that_was_active_reports_its_join_whenever_it_ends(end: str) -> None:
    """The join timings need no session id: a session that went active reports them in `on_metrics`, even
    when it ends before the service takes the 201."""
    room, session, created = _early()
    service = make_service(room, stop_drain_timeout=5.0)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await _utterance(h, "c1", AUDIO_1)
        await until(lambda: session.end_utterance_count == 1, what="c1 at the avatar before the 201")
        if end == "leave-before-201":
            await delivered(session.emit_avatar_left("avatar_left_room"))
            await rec.wait("on_session_ended")
            created.set()
            await h.finish()
        else:
            await delivered(session.emit_playback_started())
            await h.queue(EndFrame())
            await until(lambda: session.end_utterance_count == 2, what="the graceful stop's final end")
            created.set()  # the 201 lands while the stop waits for the avatar to finish
            await delivered(session.emit_playback_finished())
            await h.finish(send_end_frame=False)

    (metrics,) = rec.payload("on_metrics")
    assert metrics.avatar_joined is not None, metrics.summary()
    assert metrics.room_connected is not None
    assert metrics.session_requested is not None
