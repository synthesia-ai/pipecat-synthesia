"""A pipeline that ends while an app's `on_error` handler is still running: the end events still fire, once.

Pipecat runs `on_error` handlers inline, one after another, inside the task that pushed the error. The
service records the end of the session (`_finished`) before it pushes the error that ends it, then fires
`on_session_ended` and `on_metrics` once the push returns. So an `on_error` handler that awaits something
holds the end events back, and the handlers registered after it, and the pipeline can end meanwhile:
- after a failed launch, the error is pushed from the launch task, which the teardown cancels;
- after an unexpected leave, the error is pushed from the leave's callback task, which nothing cancels.
The teardown first waits, up to `ON_ERROR_WAIT_SECS`, for the error to reach every handler, then fires
whichever end events have not fired. Within the wait, every listener sees `on_error` before
`on_session_ended`. Past it, the end events fire anyway, and the pipeline ends: handlers still to run may
run after them, or never, where the end cancels the task running them. Either way each end event fires
once, with the end reason of the end that came first, and the error is not pushed again. An app that bills
or closes its own records from those events would otherwise lose the session, or see it end late: after
the pipeline had finished, which a bot process that exits with its pipeline never reaches.

Two kinds of test:
- The app's handler is registered first, as an app does, and the `EventRecorder` after it, so the recorder
  sees what any later handler sees. The app's handler takes `APP_HANDLER_SECS`, within the wait, or
  `SLOW_HANDLER_SECS`, far past it; the real `ON_ERROR_WAIT_SECS` applies.
- The recorder is registered first, and the app's handler (`_hold_on_error`) awaits an event the test sets
  only once the pipeline has finished, so it always outlasts the wait. These tests shrink the wait to
  `SHORT_WAIT_SECS` (`short_on_error_wait`): an input, so that each does not take a second.
The pipeline ends as when the user hangs up: an `EndFrame` or a `CancelFrame` queued at its source. The
pipeline must finish while a held handler still waits, so the wait for it is bounded (`until`): a teardown
that ran the handler again, or waited for it without a bound, fails rather than hangs.

Some cases hold the end earlier, in the voice-only handover, before its error is pushed: a sync
`on_before_push_frame` handler waits when the first handed-over chunk is pushed, the way
`test_voice_only_handover_after_an_unexpected_leave.py` makes the handover yield, and
`FakeAvatarSession(hold_send=)` keeps the first chunk in flight so that later chunks are still queued at
the leave. Past the wait the teardown fires the end events, and the error, pushed after them, would follow
`on_session_ended`; so it is dropped instead, and never reaches `on_error`.

The teardown closes the session whatever the handler holds up:
- a failed launch has no session to close;
- the leave starts the session's close itself before it pushes its error, and the teardown closes it too;
- a failed pending swap pushes its error from the launch task with the session active, its close not yet
  started when the teardown's cancel interrupts the handler: the teardown closes it, and the end is the
  pipeline's own, since nothing had ended the session before. A swap's error ends nothing, so the teardown
  does not wait for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, TTSAudioRawFrame, TTSStartedFrame

from pipecat_synthesia import service as service_module
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.frames import SynthesiaVideoSettings
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio

FAILURE = SynthesiaError("bad key", type=ErrorType.AUTH, status=401)
END_EVENTS = ("on_error", "on_session_ended", "on_metrics")
ENDS = pytest.mark.parametrize("end", [EndFrame, CancelFrame], ids=["EndFrame", "CancelFrame"])
REASON_OF = {EndFrame: "pipeline_ended", CancelFrame: "pipeline_cancelled"}
FINISHED = "the pipeline to finish while the app's on_error handler waits"
#: How long the app's `on_error` handler takes: well within the teardown's wait for it, or far past it.
APP_HANDLER_SECS = 0.1
SLOW_HANDLER_SECS = 5.0
#: How long a short voice-only handover takes: well within the teardown's wait for it.
HANDOVER_SECS = 0.1
SHORT_WAIT_SECS = 0.05


@pytest.fixture
def short_on_error_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """The teardown's wait for the end's error, shrunk for tests whose handler outlasts any wait."""
    monkeypatch.setattr(service_module, "ON_ERROR_WAIT_SECS", SHORT_WAIT_SECS)


def _hold_on_error(service: SynthesiaVideoService) -> tuple[asyncio.Event, asyncio.Event]:
    """Add an app `on_error` handler that waits for the returned release event; it sets `entered` first."""
    entered = asyncio.Event()
    release = asyncio.Event()

    @service.event_handler("on_error")
    async def _on_error(_processor: object, _error: object) -> None:
        entered.set()
        await release.wait()

    return entered, release


def _end_error_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The INFO lines about the end's error at the teardown: dropped ("not pushed"), or pushed while
    handlers were still running past the wait ("not dispatched")."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
        and ("not pushed" in record.getMessage() or "not dispatched" in record.getMessage())
    ]


def _assert_end_events_once_in_order(rec: EventRecorder) -> None:
    for name in END_EVENTS:
        assert rec.count(name) == 1, f"{name} fired {rec.count(name)} times; recorded: {rec.names()}"
    assert [name for name in rec.names() if name in END_EVENTS] == list(END_EVENTS), rec.names()


@pytest.mark.unit
@pytest.mark.usefixtures("short_on_error_wait")
@pytest.mark.parametrize("required", [True, False], ids=["required", "not_required"])
@ENDS
async def test_the_end_events_fire_once_when_the_pipeline_ends_in_the_on_error_handler_of_a_failed_launch(
    required: bool, end: type[Frame]
) -> None:
    launch_gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=FAILURE, hold_launch=launch_gate)
    service = make_service(room, required=required)
    rec = EventRecorder(service)
    handler_entered, release_handler = _hold_on_error(service)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            launch_gate.set()  # the launch fails
            await until(handler_entered.is_set, what="the app's on_error handler for the failed launch")
            await h.queue(end())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what=FINISHED)
            await finishing
        finally:
            release_handler.set()
            if finishing is not None:
                await asyncio.wait({finishing})

    _assert_end_events_once_in_order(rec)
    assert rec.payload("on_session_ended")[0] == ("launch_failed" if required else "degraded")
    assert rec.payload("on_metrics")[0].end_reason == rec.payload("on_session_ended")[0]


@pytest.mark.unit
@pytest.mark.usefixtures("short_on_error_wait")
@ENDS
async def test_end_events_have_fired_when_the_pipeline_ends_in_the_on_error_handler_of_an_unexpected_leave(
    end: type[Frame],
) -> None:
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    handler_entered, release_handler = _hold_on_error(service)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(handler_entered.is_set, what="the app's on_error handler for the leave")
            await h.queue(end())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what=FINISHED)
            await finishing
            fired_when_the_pipeline_finished = {name: rec.count(name) for name in END_EVENTS}
        finally:
            release_handler.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()  # the leave's callback, released, runs to its end

    assert fired_when_the_pipeline_finished == dict.fromkeys(END_EVENTS, 1), rec.names()
    _assert_end_events_once_in_order(rec)
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert session.aclose_completed


@pytest.mark.unit
@ENDS
async def test_the_session_is_closed_when_the_pipeline_ends_in_the_on_error_handler_of_a_pending_swap(
    end: type[Frame],
) -> None:
    session = FakeAvatarSession(swap_error=SynthesiaError("not prepared", type=ErrorType.UNKNOWN_AVATAR))
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    # a pending swap to av_b, applied once the session is active
    settings = SynthesiaVideoSettings(avatar_id="av_b")
    service = make_service(room, avatar_ids=("av_a", "av_b"), settings=settings)
    rec = EventRecorder(service)
    handler_entered, release_handler = _hold_on_error(service)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await until(handler_entered.is_set, what="the app's on_error handler for the failed swap")
            assert session.aclose_count == 0, "the session's close started before the pipeline ended"
            await h.queue(end())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what=FINISHED)
            await finishing
        finally:
            release_handler.set()
            if finishing is not None:
                await asyncio.wait({finishing})

    assert session.aclose_completed, "the teardown left the session open"
    _assert_end_events_once_in_order(rec)
    assert rec.payload("on_session_ended")[0] == REASON_OF[end]


@pytest.mark.unit
@pytest.mark.usefixtures("short_on_error_wait")
@ENDS
async def test_the_error_of_a_leave_still_handing_over_audio_never_follows_the_end_events(
    end: type[Frame], caplog_loguru: pytest.LogCaptureFixture
) -> None:
    hold_send = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold_send)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    handover_held = asyncio.Event()
    release_handover = asyncio.Event()

    @service.event_handler("on_before_push_frame")
    async def _hold_the_handover(_processor: object, frame: Frame) -> None:
        if isinstance(frame, TTSAudioRawFrame) and not handover_held.is_set():
            handover_held.set()
            await release_handover.wait()

    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(handover_held.is_set, what="the leave's voice-only handover")
            await h.queue(end())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish while the handover waits")
            await finishing
            fired_when_the_pipeline_finished = {name: rec.count(name) for name in END_EVENTS}
        finally:
            release_handover.set()
            hold_send.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()  # the leave's callback, released, runs to its end

    assert fired_when_the_pipeline_finished == {"on_error": 0, "on_session_ended": 1, "on_metrics": 1}
    assert [name for name in rec.names() if name in END_EVENTS] == ["on_session_ended", "on_metrics"]
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    # a support reader can see that an error was owed and why it did not arrive, in one line
    (line,) = _end_error_lines(caplog_loguru)
    assert "not pushed" in line and "avatar_left" in line, line


@pytest.mark.unit
@pytest.mark.usefixtures("short_on_error_wait")
async def test_the_error_of_a_degraded_launch_cut_short_in_its_handover_is_dropped_and_logged(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A failed launch with `required=False` hands its pre-launch audio over before it pushes its error, as a
    leave does; the pipeline ends meanwhile (an `on_before_push_frame` handler holds the handover). The end
    stands as `degraded`, the end events fire once, its error is not pushed, and one INFO line says so."""
    launch_gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=FAILURE, hold_launch=launch_gate)
    service = make_service(room, required=False)
    rec = EventRecorder(service)
    handover_held = asyncio.Event()
    release_handover = asyncio.Event()

    @service.event_handler("on_before_push_frame")
    async def _hold_the_handover(_processor: object, frame: Frame) -> None:
        if isinstance(frame, TTSAudioRawFrame) and not handover_held.is_set():
            handover_held.set()
            await release_handover.wait()

    finishing: asyncio.Future[None] | None = None
    async with bidirectional_pipeline(service) as h:
        try:
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2):
                await h.queue(tts_audio("c1", chunk))  # pre-launch audio, queued
            await processed(h, "pre-launch audio queued")
            launch_gate.set()  # the launch fails: degraded, its handover first
            await until(handover_held.is_set, what="the degraded launch's voice-only handover")
            await h.queue(EndFrame())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish while the handover waits")
            await finishing
        finally:
            release_handover.set()
            if finishing is not None:
                await asyncio.wait({finishing})

    assert rec.payload("on_session_ended")[0] == "degraded"
    assert rec.count("on_session_ended") == 1
    assert rec.count("on_metrics") == 1
    assert rec.count("on_error") == 0
    (line,) = _end_error_lines(caplog_loguru)
    assert "not pushed" in line and "degraded" in line, line


# --- on_error handlers after the app's: the order every listener sees ---------------------------------------


def _app_on_error_then_recorder(
    service: SynthesiaVideoService, *, secs: float
) -> tuple[asyncio.Event, asyncio.Event, EventRecorder]:
    """The app's `on_error` handler first, as an app registers it before anything else; it sleeps `secs`
    unless released earlier. Then an `EventRecorder`, whose `on_error` handler Pipecat runs after the app's.
    """
    entered = asyncio.Event()
    release = asyncio.Event()

    @service.event_handler("on_error")
    async def _app_on_error(_processor: object, _error: object) -> None:
        entered.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(release.wait(), secs)

    return entered, release, EventRecorder(service)


def _end_names(rec: EventRecorder) -> list[str]:
    return [name for name in rec.names() if name in END_EVENTS]


@pytest.mark.unit
async def test_a_listener_after_the_app_sees_a_leave_error_before_the_end_events() -> None:
    """The app's handler takes `APP_HANDLER_SECS`, well within the teardown's wait for it; an `EndFrame`
    arrives while it runs."""
    session = FakeAvatarSession()
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    entered, release, rec = _app_on_error_then_recorder(service, secs=APP_HANDLER_SECS)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(entered.is_set, what="the app's on_error handler for the leave")
            await h.queue(EndFrame())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish")
            await finishing
        finally:
            release.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()

    assert _end_names(rec) == list(END_EVENTS)
    assert rec.payload("on_session_ended")[0] == "avatar_left"


@pytest.mark.unit
async def test_a_listener_after_the_app_sees_a_failed_launchs_error_before_the_end_events() -> None:
    """`required=True`, and a `CancelFrame` arrives while the app's handler takes `APP_HANDLER_SECS`. The
    teardown cancels the launch, which is pushing the error, only once the error has reached every
    handler."""
    launch_gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=FAILURE, hold_launch=launch_gate)
    service = make_service(room, required=True)
    entered, release, rec = _app_on_error_then_recorder(service, secs=APP_HANDLER_SECS)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            launch_gate.set()  # the launch fails
            await until(entered.is_set, what="the app's on_error handler for the failed launch")
            await h.queue(CancelFrame())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish")
            await finishing
        finally:
            release.set()
            if finishing is not None:
                await asyncio.wait({finishing})

    assert rec.count("on_error") == 1, rec.names()
    assert _end_names(rec) == list(END_EVENTS)
    assert rec.payload("on_session_ended")[0] == "launch_failed"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case", "end", "after_the_end"),
    [
        pytest.param("leave", EndFrame, ["on_session_ended", "on_metrics", "on_error"], id="leave-EndFrame"),
        pytest.param(
            "failed_launch", CancelFrame, ["on_session_ended", "on_metrics"], id="failed-launch-CancelFrame"
        ),
    ],
)
async def test_an_app_handler_that_outlasts_the_wait_does_not_hold_the_end(
    case: str, end: type[Frame], after_the_end: list[str], caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The app's handler would take `SLOW_HANDLER_SECS`, far past `ON_ERROR_WAIT_SECS`, the teardown's
    wait for it; the test releases it once the pipeline has finished, so as not to wait it out. The end
    events fire while it still runs, and the pipeline ends. The recorder, behind the app's handler, gets
    the error after them where the task pushing it runs on (a leave's callback task), and never where the
    end cancels that task (the launch, after a failed launch with `required=True`). One INFO line says the
    handlers had not finished; the error was pushed, so none says it was dropped."""
    session = FakeAvatarSession()
    if case == "leave":
        room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
        service = make_service(room)
        launch_gate = None
    else:
        launch_gate = asyncio.Event()
        room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=FAILURE, hold_launch=launch_gate)
        service = make_service(room, required=True)
    entered, release, rec = _app_on_error_then_recorder(service, secs=SLOW_HANDLER_SECS)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            if launch_gate is None:
                await rec.wait("on_avatar_connected")
                session.emit_avatar_left("avatar_left_room", humans_present=True)
            else:
                launch_gate.set()  # the launch fails
            await until(entered.is_set, what="the app's on_error handler")
            await h.queue(end())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish while the app's handler runs")
            await finishing
            when_the_pipeline_finished = _end_names(rec)
        finally:
            release.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()

    assert when_the_pipeline_finished == ["on_session_ended", "on_metrics"]
    assert _end_names(rec) == after_the_end
    (line,) = _end_error_lines(caplog_loguru)  # pushed, so not dropped: only the handlers are late
    assert "not dispatched" in line, line


@pytest.mark.unit
@pytest.mark.parametrize("case", ["leave", "degraded_launch"])
async def test_the_error_of_a_short_handover_reaches_a_later_listener_before_the_end_events(
    case: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The other side of the wait: an end still handing audio over when the pipeline ends, whose handover
    takes `HANDOVER_SECS`, well within the real `ON_ERROR_WAIT_SECS`. Its error is pushed, and the
    recorder, registered after the app's `on_error` handler (which takes `APP_HANDLER_SECS`), gets it
    before the end events; nothing is dropped. The handover is an unexpected leave's, or a failed launch's
    with `required=False` (degraded), whose launch the teardown would otherwise cancel mid-handover."""
    hold_send = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold_send)
    launch_gate: asyncio.Event | None = None
    if case == "leave":
        service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    else:
        launch_gate = asyncio.Event()
        room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=FAILURE, hold_launch=launch_gate)
        service = make_service(room, required=False)
    handover_held = asyncio.Event()

    @service.event_handler("on_before_push_frame")
    async def _slow_handover(_processor: object, frame: Frame) -> None:
        if isinstance(frame, TTSAudioRawFrame) and not handover_held.is_set():
            handover_held.set()
            await asyncio.sleep(HANDOVER_SECS)

    _entered, release, rec = _app_on_error_then_recorder(service, secs=APP_HANDLER_SECS)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            if launch_gate is None:
                await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            if launch_gate is None:
                await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
                session.emit_avatar_left("avatar_left_room", humans_present=True)
            else:
                launch_gate.set()  # the launch fails: degraded, its handover first
            await until(handover_held.is_set, what="the voice-only handover")
            await h.queue(EndFrame())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish")
            await finishing
        finally:
            release.set()
            hold_send.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()

    assert _end_names(rec) == list(END_EVENTS)
    assert rec.payload("on_session_ended")[0] == ("avatar_left" if case == "leave" else "degraded")
    assert _end_error_lines(caplog_loguru) == []


@pytest.mark.unit
@pytest.mark.usefixtures("short_on_error_wait")
@pytest.mark.parametrize("app_handler", [True, False], ids=["app-handler-first", "recorder-only"])
async def test_a_leave_error_not_pushed_within_the_wait_is_not_pushed_later_behind_the_end_events(
    app_handler: bool, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """An unexpected leave's handover outlasts the teardown's wait for its error (shrunk to
    `SHORT_WAIT_SECS`), then finishes while the teardown is still closing the session (held by
    `FakeAvatarSession(hold_aclose=)`). The leave's callback task is never cancelled, so its error must
    already have been dropped when the wait ran out: pushed now, it would reach `on_error` behind the end
    events, or, with no handler holding it, after the pipeline had already started ending, unlogged. With
    `app_handler`, an app `on_error` handler that awaits comes first and the recorder after it; it must
    never run. Exactly one INFO line says the error was not pushed."""
    hold_send = asyncio.Event()
    release_close = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold_send, hold_aclose=release_close)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    handover_held = asyncio.Event()
    release_handover = asyncio.Event()

    @service.event_handler("on_before_push_frame")
    async def _hold_the_handover(_processor: object, frame: Frame) -> None:
        if isinstance(frame, TTSAudioRawFrame) and not handover_held.is_set():
            handover_held.set()
            await release_handover.wait()

    app_entered = asyncio.Event()
    release_app = asyncio.Event()
    if app_handler:

        @service.event_handler("on_error")
        async def _app_on_error(_processor: object, _error: object) -> None:
            app_entered.set()
            await release_app.wait()

    rec = EventRecorder(service)  # after the app's handler, as any later listener
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(handover_held.is_set, what="the leave's voice-only handover")
            await h.queue(EndFrame())  # the user hangs up
            # the leave's own close, then the teardown's: the teardown is past its wait, closing the session
            await until(lambda: session.aclose_count == 2, what="the teardown in its session close")
            release_handover.set()
            await until(
                lambda: app_entered.is_set() or all(task.done() for task in session.emitted),
                what="the leave to finish its handover, and push its error or not",
            )
            release_close.set()
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish")
            await finishing
        finally:
            release_handover.set()
            release_close.set()
            hold_send.set()
            release_app.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()

    assert _end_names(rec) == ["on_session_ended", "on_metrics"], rec.names()
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert not app_entered.is_set(), "the app's on_error handler got the error after the pipeline ended"
    (line,) = _end_error_lines(caplog_loguru)
    assert "not pushed" in line and "avatar_left" in line, line


@pytest.mark.unit
async def test_an_error_dispatched_as_the_wait_runs_out_logs_no_line(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """An unexpected leave's handover finishes in the same event-loop turn as the teardown's wait for its
    error runs out (the wait shrunk to 0, and the handover released as the EndFrame arrives, so the two
    coincide every time). The leave pushes its error, every handler returns at once, and on_error comes
    before the end events: nothing was held back, so no INFO line says the error was not dispatched."""
    monkeypatch.setattr(service_module, "ON_ERROR_WAIT_SECS", 0)
    hold_send = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold_send)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    handover_held = asyncio.Event()
    release_handover = asyncio.Event()

    @service.event_handler("on_before_push_frame")
    async def _hold_the_handover(_processor: object, frame: Frame) -> None:
        if isinstance(frame, TTSAudioRawFrame) and not handover_held.is_set():
            handover_held.set()
            await release_handover.wait()

    @service.event_handler("on_before_process_frame")
    async def _release_the_handover_at_the_end(_processor: object, frame: Frame) -> None:
        if isinstance(frame, EndFrame):
            release_handover.set()

    rec = EventRecorder(service)
    finishing: asyncio.Future[None] | None = None

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(handover_held.is_set, what="the leave's voice-only handover")
            await h.queue(EndFrame())  # the user hangs up
            finishing = asyncio.ensure_future(h.finish(send_end_frame=False))
            await until(finishing.done, what="the pipeline to finish")
            await finishing
        finally:
            release_handover.set()
            hold_send.set()
            if finishing is not None:
                await asyncio.wait({finishing})
            await session.wait_emitted()

    assert _end_names(rec) == list(END_EVENTS), rec.names()
    assert _end_error_lines(caplog_loguru) == []
