"""An unexpected leave that resumes after the pipeline started ending: the end is the pipeline's.

`_on_avatar_left` reads whether the pipeline is ending once, on entry, then awaits before it ends the
session: it broadcasts `BotStoppedSpeakingFrame` when the bot was speaking, and an app's sync
`on_before_push_frame` handler can make that push wait. If the pipeline starts ending meanwhile, the
teardown's one check for an end in progress has already found none, and the leave, resuming, used to end
the session as `avatar_left` with an error, behind the teardown's back. So the leave reads it again right
before it would end the session, with no await between that read and the session being marked finished,
and takes its "avatar left during shutdown" path instead: one INFO line, no error, and the end is the
pipeline's own, `on_session_ended` with the pipeline's reason, then `on_metrics`.

That relies on `stop()`, `cancel()` and the teardown each marking the pipeline as ending before their
first await. The first test ends the pipeline each of those ways while the leave waits: an `EndFrame`
(`stop()`), a `CancelFrame` (`cancel()`), and `cleanup()` called directly, as Pipecat calls it for a
processor it cleans up without ending, which reaches the teardown without either. (These pin that each
marks it before the teardown closes the session, where the leave resumes; that it is their first act is
read from the code, not pinned.)

Each test holds the leave at its `BotStoppedSpeakingFrame` (`_hold_the_leave`; the bot was speaking, through
`session.emit_playback_started()`), ends the pipeline, and releases the leave:
- while the teardown is held in its session close (`FakeAvatarSession(hold_aclose=)`), with an app
  `on_error` handler that takes `APP_HANDLER_SECS` and the `EventRecorder` registered after it, so that
  the recorder sees what any handler sees;
- the same, with audio queued for the voice-only handover: `FakeAvatarSession(hold_send=)` keeps the
  first chunk in flight, so the next two are still queued when the avatar leaves. The leave leaves that
  audio to the teardown's end, which drops it, as any end without an error drops leftover audio: none of
  it reaches the user transport, and none stays held;
- once the teardown has finished, so the leave resumes with the session already ended.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)

from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState, SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service, make_setup
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.waiting import assert_no_dangling, until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, delivered, processed, records_at, tts_audio

END_EVENTS = ("on_error", "on_session_ended", "on_metrics")
#: How long the app's `on_error` handler takes, were the leave to push an error.
APP_HANDLER_SECS = 0.1
SHUTDOWN_LINE = "avatar left during shutdown"


def _hold_the_leave(service: SynthesiaVideoService) -> tuple[asyncio.Event, asyncio.Event]:
    """An app `on_before_push_frame` handler that waits, at the first `BotStoppedSpeakingFrame`, for the
    returned release event; it sets the returned `held` event first."""
    held = asyncio.Event()
    release = asyncio.Event()

    @service.event_handler("on_before_push_frame")
    async def _hold(_processor: object, frame: Frame) -> None:
        if isinstance(frame, BotStoppedSpeakingFrame) and not held.is_set():
            held.set()
            await release.wait()

    return held, release


def _app_on_error_then_recorder(service: SynthesiaVideoService) -> EventRecorder:
    @service.event_handler("on_error")
    async def _app_on_error(_processor: object, _error: object) -> None:
        await asyncio.sleep(APP_HANDLER_SECS)

    return EventRecorder(service)


def _end_names(rec: EventRecorder) -> list[str]:
    return [name for name in rec.names() if name in END_EVENTS]


async def _speaking_then_leaving(session: FakeAvatarSession, held: asyncio.Event) -> None:
    """The bot is speaking; the avatar leaves with humans present; the leave waits at its broadcast."""
    await delivered(session.emit_playback_started())
    session.emit_avatar_left("avatar_left_room", humans_present=True)
    await until(held.is_set, what="the leave's BotStoppedSpeakingFrame, held by the app's handler")


def _assert_the_pipelines_end(rec: EventRecorder, reason: str, caplog: pytest.LogCaptureFixture) -> None:
    assert _end_names(rec) == ["on_session_ended", "on_metrics"], rec.names()
    assert rec.payload("on_session_ended")[0] == reason
    assert len(records_at(caplog, logging.INFO, SHUTDOWN_LINE)) == 1, "no trace of the leave during shutdown"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("end", "reason"),
    [(EndFrame, "pipeline_ended"), (CancelFrame, "pipeline_cancelled")],
    ids=["EndFrame", "CancelFrame"],
)
async def test_a_leave_resuming_while_the_teardown_closes_the_session_leaves_the_end_to_the_pipeline(
    end: type[Frame], reason: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    held, release_leave = _hold_the_leave(service)
    rec = _app_on_error_then_recorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _speaking_then_leaving(session, held)
            await h.queue(end())  # the user hangs up
            await until(lambda: session.aclose_count == 1, what="the teardown closing the session")
            release_leave.set()
            await session.wait_emitted()  # the leave runs to its end
            release_close.set()
            await h.finish(send_end_frame=False)
        finally:
            release_leave.set()
            release_close.set()

    _assert_the_pipelines_end(rec, reason, caplog_loguru)


@pytest.mark.unit
async def test_a_leave_resuming_while_a_cleanup_closes_the_session_leaves_the_end_to_the_cleanup(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """`cleanup()` without a stop or a cancel first reaches the teardown directly. Driven outside a pipeline
    (`make_setup`), since a processor Pipecat has cleaned up processes no more frames, so a harness
    pipeline could not end after it."""
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    held, release_leave = _hold_the_leave(service)
    rec = _app_on_error_then_recorder(service)
    setup, task_manager = make_setup()
    await service.setup(setup)
    cleaning: asyncio.Future[None] | None = None
    try:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the session active")
        await _speaking_then_leaving(session, held)
        cleaning = asyncio.ensure_future(service.cleanup())
        await until(lambda: session.aclose_count == 1, what="the teardown closing the session")
        release_leave.set()
        await session.wait_emitted()
        release_close.set()
        await cleaning
    finally:
        release_leave.set()
        release_close.set()
        if cleaning is None:
            cleaning = asyncio.ensure_future(service.cleanup())
        await asyncio.wait({cleaning})

    _assert_the_pipelines_end(rec, "pipeline_cleanup", caplog_loguru)
    assert_no_dangling(task_manager)


@pytest.mark.unit
async def test_audio_queued_for_the_handover_of_a_leave_resuming_in_the_teardown_is_dropped(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    release_close = asyncio.Event()
    hold_send = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=release_close, hold_send=hold_send)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    held, release_leave = _hold_the_leave(service)
    rec = _app_on_error_then_recorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            await _speaking_then_leaving(session, held)
            await h.queue(EndFrame())  # the user hangs up
            await until(lambda: session.aclose_count == 1, what="the teardown closing the session")
            release_leave.set()
            await session.wait_emitted()
            release_close.set()
            await h.finish(send_end_frame=False)
        finally:
            release_leave.set()
            release_close.set()
            hold_send.set()

    _assert_the_pipelines_end(rec, "pipeline_ended", caplog_loguru)
    assert frames_of(h.down, TTSAudioRawFrame) == [], "the queued audio was handed over after the end"
    # No public seam shows the hold once the pipeline has ended: audio left in it would hold back, rather
    # than pass through, any TTS audio after the end.
    assert service._held_for_handover is None, "the teardown's end left the handover's audio held"


@pytest.mark.unit
async def test_a_leave_resuming_after_the_teardown_finished_leaves_the_end_to_the_pipeline(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    session = FakeAvatarSession()
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    held, release_leave = _hold_the_leave(service)
    rec = _app_on_error_then_recorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _speaking_then_leaving(session, held)
            await h.queue(EndFrame())  # the user hangs up
            await rec.wait("on_metrics")  # the teardown has ended the session
            release_leave.set()
            await session.wait_emitted()
            await h.finish(send_end_frame=False)
        finally:
            release_leave.set()
            with contextlib.suppress(Exception):
                await session.wait_emitted()

    _assert_the_pipelines_end(rec, "pipeline_ended", caplog_loguru)
