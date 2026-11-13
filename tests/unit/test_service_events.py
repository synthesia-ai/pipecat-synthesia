"""Events, avatar swaps, the avatar leaving, metrics, configuration warnings and secrets."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    InterruptionFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.errors import ErrorCategory

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.events import REGISTERED_EVENTS, SESSION_END_REASONS, AvatarConnected, AvatarReady
from pipecat_synthesia.frames import SynthesiaVideoSettings, SynthesiaVideoUpdateSettingsFrame
from pipecat_synthesia.metrics import SessionMetrics
from pipecat_synthesia.room import JoinMode, LaunchContext
from pipecat_synthesia.service import ServiceState, SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, count, first, frames_of
from tests.support.secrets import assert_no_secrets
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, delivered, processed, records_at, tts_audio

UP = FrameDirection.UPSTREAM


def _swap_delta(
    avatar_id: str | None, *, service: FrameProcessor | None = None
) -> SynthesiaVideoUpdateSettingsFrame:
    return SynthesiaVideoUpdateSettingsFrame(
        delta=SynthesiaVideoSettings(avatar_id=avatar_id), service=service
    )


def _swapped_to(session: FakeAvatarSession) -> list[str]:
    return [call[1] for call in session.swap_calls]


async def _end_session_without_error(service: SynthesiaVideoService, room: FakeAvatarRoom) -> None:
    """The no-humans-left end: the session ends normally, with no ErrorFrame."""
    rec = EventRecorder(service)
    room.session.emit_avatar_left("avatar_left_room", humans_present=False)
    await rec.wait("on_session_ended")
    assert service.state is ServiceState.ENDED


# --- settings frames and avatar swap ------------------------------------------------------------------


@pytest.mark.unit
async def test_settings_frame_for_another_service_passes_through_untouched() -> None:
    """Forwarded as the same object with its delta intact, asking this service for nothing; the same change
    aimed at this service swaps and is consumed."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)
    other = FrameProcessor(name="another-service")

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        for_other = _swap_delta("av_b", service=other)
        await h.queue(for_other)
        await processed(h, "after the frame for another service")
        assert room.session.swap_calls == []
        assert service.active_avatar_id == "av_a"

        await h.queue(_swap_delta("av_b", service=service))
        await until(lambda: service.active_avatar_id == "av_b", what="the swap asked of this service")
        await processed(h, "after the frame for this service")
        await h.finish()

    forwarded = first(h.down, SynthesiaVideoUpdateSettingsFrame)
    assert forwarded is for_other
    assert forwarded.delta is not None and forwarded.delta.avatar_id == "av_b"
    assert count(h.down, SynthesiaVideoUpdateSettingsFrame) == 1, "the frame for this service is consumed"
    assert _swapped_to(room.session) == ["av_b"]
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_deprecated_settings_dict_is_applied_like_a_delta() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(SynthesiaVideoUpdateSettingsFrame(settings={"avatar_id": "av_b"}))
        await until(lambda: service.active_avatar_id == "av_b", what="the swap from the settings dict")
        await h.finish()

    assert _swapped_to(room.session) == ["av_b"]
    assert_absent(h.down, SynthesiaVideoUpdateSettingsFrame)
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_avatar_id_of_none_asks_for_no_swap() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(_swap_delta(None))
        await processed(h, "after the None change")
        assert room.session.swap_calls == []
        assert service.active_avatar_id == "av_a"

        await h.queue(_swap_delta("av_b"))
        await until(lambda: service.active_avatar_id == "av_b", what="the later real swap")
        await h.finish()

    assert _swapped_to(room.session) == ["av_b"]
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_avatar_id_change_during_the_launch_is_a_pending_swap() -> None:
    """Applied once the session is active, as one swap."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        assert service.state is ServiceState.LAUNCHING
        await h.queue(_swap_delta("av_b"))
        await processed(h, "after the change during the launch")
        assert room.session.swap_calls == []
        assert service.active_avatar_id == "av_a"

        gate.set()
        await rec.wait("on_avatar_connected")
        await until(lambda: service.active_avatar_id == "av_b", what="the pending swap to apply")
        await h.finish()

    assert _swapped_to(room.session) == ["av_b"]
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_avatar_id_change_after_the_session_ended_is_ignored() -> None:
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await _end_session_without_error(service, room)
        await h.queue(_swap_delta("av_b"))
        await processed(h, "after the change in ENDED")
        await h.finish()

    assert room.session.swap_calls == []
    assert service.active_avatar_id == "av_a"
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_successful_swap_moves_the_active_avatar_and_the_settings() -> None:
    """The settings follow the swap: a change to the avatar now active asks for nothing, and a change back
    swaps back."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        assert await h.call(lambda: service.swap_avatar("av_b")) == "av_b"
        assert service.active_avatar_id == "av_b"

        await h.queue(_swap_delta("av_b"))
        await processed(h, "after the change to the active avatar")
        assert _swapped_to(room.session) == ["av_b"], "the settings already name av_b: no second swap"

        await h.queue(_swap_delta("av_a"))
        await until(lambda: service.active_avatar_id == "av_a", what="the swap back to av_a")
        await h.finish()

    assert _swapped_to(room.session) == ["av_b", "av_a"]
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
@pytest.mark.parametrize("when", ["launching", "ended"])
async def test_swap_with_no_active_session_raises_without_pushing(when: str) -> None:
    """A caller-state error: raised as CONNECTION, never pushed, nothing sent."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=gate)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        if when == "ended":
            gate.set()
            await rec.wait("on_avatar_connected")
            await _end_session_without_error(service, room)
        with pytest.raises(SynthesiaError) as excinfo:
            await h.call(lambda: service.swap_avatar("av_b"))
        assert excinfo.value.type is ErrorType.CONNECTION
        gate.set()
        await h.finish()

    assert room.session.swap_calls == []
    assert service.active_avatar_id == "av_a"
    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_swap_failure_keeps_usable() -> None:
    """A swap failure (`UNKNOWN_AVATAR`, truly INVALID_REQUEST) is reported once, non-permanently
    (*permanent-error rule*): (a) via a settings frame -> exactly one ErrorFrame with category APPLICATION,
    `is_usable` True, the active avatar unchanged and the settings restored (proved by a second identical
    delta triggering a second swap attempt), no "Error processing frame" frame; (b) via the method ->
    raises with the session id and pushes one more ErrorFrame."""
    session = FakeAvatarSession(swap_error=SynthesiaError("not prepared", type=ErrorType.UNKNOWN_AVATAR))
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")

        await h.queue(_swap_delta("av_b"))
        error = await h.frame(ErrorFrame, direction=UP)
        assert error.category is ErrorCategory.APPLICATION
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.type is ErrorType.UNKNOWN_AVATAR
        assert error.exception.category is ErrorCategory.INVALID_REQUEST
        assert error.exception.session_id == session.session_id
        await processed(h, "after the rejected change")
        assert service.is_usable is True
        assert service.active_avatar_id == "av_a"
        assert len(session.swap_calls) == 1
        assert count(h.up, ErrorFrame) == 1

        # the settings were restored to av_a: the same delta is a change again and triggers another attempt
        await h.queue(_swap_delta("av_b"))
        await until(lambda: len(session.swap_calls) == 2, what="a second swap attempt")
        await until(lambda: count(h.up, ErrorFrame) == 2, what="the second ErrorFrame")
        assert service.is_usable is True

        with pytest.raises(SynthesiaError) as excinfo:
            await h.call(lambda: service.swap_avatar("av_b"))
        assert excinfo.value.session_id == session.session_id
        assert excinfo.value.type is ErrorType.UNKNOWN_AVATAR
        await until(lambda: count(h.up, ErrorFrame) == 3, what="the method's ErrorFrame")
        assert service.is_usable is True
        assert service.active_avatar_id == "av_a"
        await h.finish()

    for frame in frames_of(h.up, ErrorFrame):
        assert isinstance(frame, ErrorFrame)
        assert not frame.error.startswith("Error processing frame"), frame.error
        assert frame.category is ErrorCategory.APPLICATION
    assert rec.count("on_error") == 3


# --- the avatar leaves ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False])
async def test_avatar_left_humans_present(required: bool) -> None:
    """An unexpected leave (the avatar leaves while humans remain): ENDED, one NON-permanent ErrorFrame,
    `on_avatar_disconnected` and `on_session_ended("avatar_left")` once each, TTS passes through, the
    pipeline keeps running, the session is closed. The same for both values of `required`, which governs
    the launch only."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, required=required)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        room.session.emit_avatar_left("avatar_left_room", humans_present=True)
        error = await h.frame(ErrorFrame, direction=UP)
        assert service.state is ServiceState.ENDED
        assert error.category is not None and error.category.is_permanent is False
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.type is ErrorType.CONNECTION
        assert error.exception.session_id == room.session.session_id
        assert service.is_usable is True
        await h.queue(tts_audio("c1"))
        await h.frame(TTSAudioRawFrame)
        await until(lambda: room.session.aclose_count >= 1, what="session.aclose")
        assert h.worker.has_finished() is False
        assert_absent(h.down, EndFrame, where="before we sent one")
        await h.finish()

    assert rec.count("on_avatar_disconnected") == 1
    assert rec.payload("on_avatar_disconnected")[0] == "avatar_left_room"
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert count(h.up, ErrorFrame) == 1
    assert room.session.send_audio_calls == []


@pytest.mark.unit
async def test_avatar_left_no_humans(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The no-humans-left end is a normal end: INFO, no ErrorFrame, events once. Nobody is left to hear the
    queued audio, so it is neither sent nor handed on, including the chunk in flight (held by `hold_send`),
    which goes back to the queue and is dropped with it."""
    in_flight = asyncio.Event()
    session = FakeAvatarSession(hold_send=in_flight)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await h.queue(tts_audio("c1", AUDIO_2))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.frame(TTSStoppedFrame)  # both chunks are in the send queue
        await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")

        session.emit_avatar_left("avatar_left_room", humans_present=False)
        await rec.wait("on_session_ended")
        assert service.state is ServiceState.ENDED
        in_flight.set()
        await h.finish()

    sent = [c[1] for c in session.send_audio_calls]
    assert AUDIO_1 in sent
    assert AUDIO_2 not in sent, "queued audio must not reach the avatar worker after the session ended"
    assert_absent(h.down, TTSAudioRawFrame, where="downstream after a no-humans-left end")
    assert_absent(h.up, ErrorFrame)
    assert rec.count("on_avatar_disconnected") == 1
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "no_humans_left"
    assert records_at(caplog_loguru, logging.INFO, "no human")
    assert room.session.aclose_count >= 1


@pytest.mark.unit
async def test_avatar_left_during_stop() -> None:
    """Two leave signals while `stop()` is draining (video stops, then the avatar leaves the room): no
    ErrorFrame, `on_avatar_disconnected` once with the first reason, and the session ends with the
    teardown's own reason. The service must not rely on the session delivering a leave only once."""
    session = FakeAvatarSession()

    def on_call(call: tuple) -> None:
        # the final UtteranceEnd of stop() is the second end_utterance the fake sees
        if call[0] == "end_utterance" and session.end_utterance_count == 2:
            session.emit_avatar_left("avatar_video_stopped", humans_present=True)
            session.emit_avatar_left("avatar_left_room", humans_present=True)

    session.on_call = on_call
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=2.0)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.queue(EndFrame())
        await h.finish(send_end_frame=False)
        await session.wait_emitted()

    assert len(session.emitted) == 2, "both leave signals were delivered"
    assert_absent(h.up, ErrorFrame)
    assert rec.count("on_avatar_disconnected") == 1
    assert rec.payload("on_avatar_disconnected")[0] == "avatar_video_stopped"
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"
    assert session.aclose_count >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("humans_present", "end_reason"),
    [(True, "avatar_left"), (False, "no_humans_left")],
    ids=["unexpected-leave", "no-humans-left"],
)
async def test_leave_mid_utterance_ends_the_speaking_state(humans_present: bool, end_reason: str) -> None:
    """At the leave, not only when the pipeline later ends."""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await until(lambda: len(session.send_audio_calls) == 1, what="the utterance at the avatar worker")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)

        session.emit_avatar_left("avatar_left_room", humans_present=humans_present)
        await h.frame(BotStoppedSpeakingFrame)
        await rec.wait("on_session_ended")
        await h.finish()

    assert count(h.down, BotStoppedSpeakingFrame) == 1
    assert rec.payload("on_session_ended")[0] == end_reason


# --- readiness and connected-handler races -------------------------------------------------------


@pytest.mark.unit
async def test_ready_emitted_once_under_race() -> None:
    """A status callback fires inside `launch()` (before SESSION_ACTIVE) and `all_ready` is already True
    when it returns: `on_avatar_ready` fires exactly once, after `on_avatar_connected`; a later status
    repeats nothing."""

    async def status_during_launch(room: FakeAvatarRoom, ctx: LaunchContext) -> None:
        room.session.set_status({"av_a": "ready"}, all_ready=True)
        await ctx.callbacks.on_avatar_status({"av_a": "ready"}, True)  # delivered before launch() returns

    room = FakeAvatarRoom(JoinMode.DIRECT, during_launch=status_during_launch)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        ready = await rec.wait("on_avatar_ready")
        assert isinstance(ready[0], AvatarReady)
        assert ready[0].status == {"av_a": "ready"}
        names = rec.names()
        assert names.index("on_avatar_connected") < names.index("on_avatar_ready")
        task = room.session.emit_status({"av_a": "ready"}, all_ready=True)
        assert task is not None
        await task
        await h.finish()

    assert rec.count("on_avatar_ready") == 1
    assert service.metrics.avatar_ready is not None


@pytest.mark.unit
async def test_left_during_connected_handler() -> None:
    """An `on_avatar_connected` handler that makes the avatar leave at once: no exception, ENDED,
    `on_session_ended("avatar_left")` and `on_avatar_disconnected` once, `on_avatar_ready` at most once and
    never before connected."""
    room = FakeAvatarRoom(JoinMode.DIRECT, session=FakeAvatarSession(all_ready=True))
    service = make_service(room)
    rec = EventRecorder(service)

    @service.event_handler("on_avatar_connected")
    async def _leave(_service, _info: AvatarConnected) -> None:
        task = room.session.emit_avatar_left("avatar_left_room", humans_present=True)
        assert task is not None
        await task

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_session_ended")
        assert service.state is ServiceState.ENDED
        await h.queue(tts_audio("c1"))
        await h.frame(TTSAudioRawFrame)
        await h.finish()

    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert rec.count("on_avatar_disconnected") == 1
    assert rec.count("on_avatar_ready") <= 1
    names = rec.names()
    if "on_avatar_ready" in names:
        assert names.index("on_avatar_connected") < names.index("on_avatar_ready")


# --- events --------------------------------------------------------------------------------------


@pytest.mark.unit
async def test_events_registered_and_fire(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """Every `REGISTERED_EVENTS` name accepts a handler (no "not registered" WARNING) and fires with the
    documented payload type; `on_error` carries an ErrorFrame whose exception is our SynthesiaError. The
    event set is checked by containment, so a new event does not break it."""
    session = FakeAvatarSession(swap_error=SynthesiaError("nope", type=ErrorType.UNKNOWN_AVATAR))
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    assert records_at(caplog_loguru, logging.WARNING, "not registered") == []

    async with bidirectional_pipeline(service) as h:
        connected = await rec.wait("on_avatar_connected")
        assert isinstance(connected[0], AvatarConnected)
        assert connected[0].session_id == session.session_id

        task = session.emit_status({"av_a": "ready"}, all_ready=True)
        assert task is not None
        await task
        ready = await rec.wait("on_avatar_ready")
        assert isinstance(ready[0], AvatarReady)

        with pytest.raises(SynthesiaError):
            await h.call(lambda: service.swap_avatar("av_b"))
        error = await rec.wait("on_error")
        assert isinstance(error[0], ErrorFrame)
        assert isinstance(error[0].exception, SynthesiaError)

        session.emit_avatar_left("avatar_left_room", humans_present=True)
        disconnected = await rec.wait("on_avatar_disconnected")
        assert isinstance(disconnected[0], str)
        ended = await rec.wait("on_session_ended")
        assert ended[0] in SESSION_END_REASONS
        metrics = await rec.wait("on_metrics")
        assert isinstance(metrics[0], SessionMetrics)
        assert metrics[0].end_reason == ended[0]
        await h.finish()

    assert set(REGISTERED_EVENTS) <= set(rec.names())
    assert rec.count("on_session_ended") == 1 and rec.count("on_metrics") == 1
    assert rec.names().index("on_session_ended") < rec.names().index("on_metrics")


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        AvatarConnected(session_id="ses", avatar_identity="a", active_avatar_id="av_a", room_name=None),
        AvatarReady(session_id="ses", status={}),
    ],
    ids=["avatar-connected", "avatar-ready"],
)
def test_event_payloads_are_immutable(payload: Any) -> None:
    """Every handler gets the same instance, so one handler's assignment would change what the rest see."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        payload.session_id = "replaced"


# --- metrics -------------------------------------------------------------------------------------


@pytest.mark.unit
async def test_metrics_summary_fields_and_timings() -> None:
    """Launch timings are copied from `session.timings` (the fake stamps them all at once, so
    `session_created <= avatar_joined` here); `summary()` contains the documented keys (containment, values
    never asserted)."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        timings = room.session.timings
        assert service.metrics.room_connected == timings.room_connected
        assert service.metrics.session_requested == timings.session_requested
        assert service.metrics.session_created == timings.session_created
        assert service.metrics.avatar_joined == timings.avatar_joined
        assert service.metrics.session_created is not None and service.metrics.avatar_joined is not None
        assert service.metrics.session_created <= service.metrics.avatar_joined
        assert service.metrics.launch_started is not None
        await h.finish()

    summary = service.metrics.summary()
    assert {
        "room_connected",
        "session_requested",
        "session_created",
        "avatar_joined",
        "ended",
        "end_reason",
        "playback_finished_count",
        "prelaunch_dropped_bytes",
    } <= set(summary)
    assert summary["end_reason"] == "pipeline_ended"


@pytest.mark.unit
async def test_avatar_video_published_is_stamped_once_by_the_first_video_track() -> None:
    """Not by an audio track, and not moved by a second video track."""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await delivered(session.emit_track_published("audio"))
        assert service.metrics.avatar_video_published is None
        await delivered(session.emit_track_published("video"))
        first = service.metrics.avatar_video_published
        assert first is not None
        await delivered(session.emit_track_published("video"))
        assert service.metrics.avatar_video_published == first
        await h.finish()

    assert "avatar_video_published" in service.metrics.summary()


@pytest.mark.unit
async def test_first_utterance_without_tts_started_is_sent_and_speaks() -> None:
    """The session's first chunk, with no `TTSStartedFrame` ahead of it, still reaches the avatar worker, and
    its playback starts the speaking state."""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(tts_audio(None, AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk at the avatar worker")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)
        await h.finish()

    assert [call[1] for call in session.send_audio_calls] == [AUDIO_1]


@pytest.mark.unit
def test_summary_reports_launch_timestamps_as_millisecond_deltas_from_the_launch_start() -> None:
    metrics = SessionMetrics(
        launch_started=100.0,
        room_connected=100.1236,
        session_created=100.25,
        avatar_joined=100.75,
        ended=102.0,
    )
    summary = metrics.summary()

    expected = {"room_connected": 0.1236, "session_created": 0.25, "avatar_joined": 0.75, "ended": 2.0}
    assert set(expected) <= set(summary)
    for name, delta in expected.items():
        assert abs(summary[name] - delta) <= 0.0005, (name, summary[name])
        assert summary[name] == round(summary[name], 3), (name, summary[name])


@pytest.mark.unit
def test_summary_lists_the_per_utterance_and_per_interruption_timings() -> None:
    metrics = SessionMetrics(
        launch_started=0.0, utterance_ttfb_seconds=[0.12345, 0.5], interruption_latency_seconds=[0.2468]
    )
    summary = metrics.summary()

    assert summary["utterance_ttfb_seconds"] == pytest.approx([0.123, 0.5], abs=1e-9)
    assert summary["interruption_latency_seconds"] == pytest.approx([0.247], abs=1e-9)


@pytest.mark.unit
async def test_interruption_latency_is_not_recorded() -> None:
    """Not recorded in this release: a `playback_finished(interrupted=True)` does not say which clear buffer
    it answers, so no rule can tell which interruption it measures. The finishes still count."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    session = room.session
    service = make_service(room, bot_stopped_hysteresis_secs=0.05, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: len(session.send_audio_calls) == 1, what="the chunk sent")
        session.emit_playback_started()
        await h.frame(BotStartedSpeakingFrame)

        await h.queue(InterruptionFrame())
        await until(lambda: session.interrupt_count == 1, what="interrupt at the session")
        await delivered(session.emit_playback_finished(0.5, True))
        await h.finish()

    assert service.metrics.interruption_latency_seconds == []
    assert "interruption_latency_seconds" not in service.metrics.summary()
    assert service.metrics.playback_finished_count == 1


# --- configuration warnings ----------------------------------------------------------------------


class _DirectJoinRoomWithoutFilterFlag(FakeAvatarRoom):
    """A direct-join room that, like an `AvatarRoom` from elsewhere, has no `filter_installed` attribute."""

    def __init__(self) -> None:
        super().__init__(JoinMode.DIRECT)
        del self.filter_installed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("make_room", "warns"),
    [
        (lambda: FakeAvatarRoom(JoinMode.DIRECT, filter_installed=False), True),
        (lambda: FakeAvatarRoom(JoinMode.DIRECT, filter_installed=True), False),
        (_DirectJoinRoomWithoutFilterFlag, False),
        (lambda: FakeAvatarRoom(JoinMode.RELAYED, filter_installed=False), False),
    ],
    ids=["direct-join-filter-missing", "direct-join-filter-installed", "direct-join-no-flag", "relayed-join"],
)
async def test_direct_join_filter_warning(
    caplog_loguru: pytest.LogCaptureFixture, make_room: Callable[[], FakeAvatarRoom], warns: bool
) -> None:
    """One WARNING naming `input_filter`, only for a direct join whose room reports the avatar audio filter
    missing: a room with no such flag gets the benefit of the doubt, and a relayed join has no filter."""
    async with bidirectional_pipeline(make_service(make_room())) as h:
        await h.finish()

    assert len(records_at(caplog_loguru, logging.WARNING, "input_filter")) == (1 if warns else 0)


class _MinimalOutputTransport(BaseOutputTransport):
    """A real `BaseOutputTransport` subclass with no media; only its `_params` matter to the service."""

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("params", "expect_warning"),
    [
        (TransportParams(), True),
        (
            TransportParams(
                video_out_enabled=True,
                video_out_is_live=True,
                video_out_width=512,
                video_out_height=512,
                video_out_framerate=25,
            ),
            False,
        ),
    ],
    ids=["defaults", "configured-for-synthesia"],
)
async def test_relayed_join_video_params_warning(
    caplog_loguru: pytest.LogCaptureFixture, params: TransportParams, expect_warning: bool
) -> None:
    """Relayed join with a downstream output transport: default `TransportParams` produce one WARNING
    naming `video_out`; 512x512 @ 25 fps live produces none."""
    room = FakeAvatarRoom(JoinMode.RELAYED)
    service = make_service(room)
    transport = _MinimalOutputTransport(params)
    async with bidirectional_pipeline(Pipeline([service, transport])) as h:
        await h.finish()
    hits = records_at(caplog_loguru, logging.WARNING, "video_out")
    assert len(hits) == (1 if expect_warning else 0), [r.getMessage() for r in hits]


# --- secrets -------------------------------------------------------------------------------------


@pytest.mark.unit
async def test_no_secrets_in_logs(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """A full launch, a swap failure, an avatar-left error and the teardown log nothing that contains the
    shared secret sentinels; nor do the ErrorFrames' messages and exceptions (*redaction rule*)."""
    session = FakeAvatarSession(swap_error=SynthesiaError("nope", type=ErrorType.UNKNOWN_AVATAR))
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        with pytest.raises(SynthesiaError):
            await h.call(lambda: service.swap_avatar("av_b"))
        session.emit_avatar_left("avatar_left_room", humans_present=True)
        await rec.wait("on_session_ended")
        await until(lambda: count(h.up, ErrorFrame) == 2, what="both ErrorFrames")
        await h.finish()

    assert_no_secrets(caplog_loguru.text)
    for frame in frames_of(h.up, ErrorFrame):
        assert isinstance(frame, ErrorFrame)
        assert_no_secrets(frame.error, str(frame.exception), repr(frame.exception))
    assert_no_secrets(repr(service.metrics), str(service.metrics.summary()))


# --- usability after a permanent push ----------------------------------------------------------------


@pytest.mark.unit
async def test_settings_frame_in_failed_does_not_revive_usability() -> None:
    """After the `required=True` launch failure made the processor unusable, a settings frame must not
    revive it: `is_usable` stays False and `on_usable_changed` fired exactly once. Nothing calls
    `set_usable(True)` after a permanent push (*permanent-error rule*).

    The service ends the pipeline after that failure, so the settings frame is queued from `on_error`,
    which Pipecat awaits before it pushes the ErrorFrame: it enters the pipeline ahead of the EndFrame.
    `hold_launch` keeps the launch in progress until the handler is registered."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(
        JoinMode.DIRECT,
        fail_with=SynthesiaError("bad key", type=ErrorType.AUTH, status=401),
        hold_launch=gate,
    )
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:

        @service.event_handler("on_error")
        async def _settings_after_failure(_service, _error) -> None:
            await h.queue(_swap_delta("av_b"))
            await h.queue(TextFrame("marker"))

        gate.set()
        await h.frame(TextFrame)  # the settings frame ahead of it has been processed
        assert service.state is ServiceState.FAILED
        assert service.is_usable is False
        assert service.active_avatar_id == "av_a"
        await h.finish()

    assert rec.count("on_usable_changed") == 1
    assert rec.payload("on_usable_changed")[0] is False
    assert room.session.swap_calls == []
    assert count(h.up, ErrorFrame) == 1, "the settings change in FAILED must report nothing more"


@pytest.mark.unit
async def test_constructor_settings_avatar_id_is_applied_when_session_active() -> None:
    """`settings=SynthesiaVideoSettings(avatar_id=...)` on the constructor naming a different avatar is a
    swap request, applied once SESSION_ACTIVE exactly like a settings frame that arrived while LAUNCHING."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(
        room, avatar_ids=("av_a", "av_b"), settings=SynthesiaVideoSettings(avatar_id="av_b")
    )
    rec = EventRecorder(service)
    assert service.active_avatar_id == "av_a"

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await until(lambda: service.active_avatar_id == "av_b", what="the constructor swap to apply")
        assert [c[1] for c in room.session.swap_calls] == ["av_b"]
        await h.finish()

    assert_absent(h.up, ErrorFrame)


@pytest.mark.unit
async def test_refused_pending_swap_from_the_constructor_is_tried_again_on_a_later_change() -> None:
    """A refused constructor swap puts the settings back to the active avatar, so the same change later is
    a change again and is tried as a swap."""
    session = FakeAvatarSession(swap_error=SynthesiaError("not prepared", type=ErrorType.UNKNOWN_AVATAR))
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(
        room, avatar_ids=("av_a", "av_b"), settings=SynthesiaVideoSettings(avatar_id="av_b")
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        error = await h.frame(ErrorFrame, direction=UP)
        assert error.category is ErrorCategory.APPLICATION
        assert service.is_usable is True
        assert service.active_avatar_id == "av_a"
        assert [c[1] for c in session.swap_calls] == ["av_b"]

        await h.queue(_swap_delta("av_b"))
        await until(lambda: len(session.swap_calls) == 2, what="the later change to be tried as a swap")
        await until(lambda: count(h.up, ErrorFrame) == 2, what="the second refusal reported")
        await h.finish()

    assert service.active_avatar_id == "av_a"
    assert service.is_usable is True
