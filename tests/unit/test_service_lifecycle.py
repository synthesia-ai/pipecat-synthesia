"""The service lifecycle: setup, frame pass-through, launch failure and timeout, teardown.

Conventions: state only via `service.state`; events via `EventRecorder`; frames via the
`tests.support.frames` filters; asynchronous outcomes via `until()` / `h.frame()`; durations are inputs
named in each docstring; no private attributes.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import logging
import warnings
from time import monotonic

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InterruptionFrame,
    OutputTransportReadyFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.worker import ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.errors import ErrorCategory
from yarl import URL

from pipecat_synthesia import service as service_module
from pipecat_synthesia._async import await_cancel_proof
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode, LaunchContext
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import assert_strict_frames, bidirectional_pipeline, make_service, make_setup
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, count, frames_of
from tests.support.secrets import API_KEY, assert_no_secrets
from tests.support.waiting import assert_no_dangling, until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, delivered, processed, records_at, tts_audio

UP = FrameDirection.UPSTREAM


def _auth_401() -> SynthesiaError:
    return SynthesiaError("bad key", type=ErrorType.AUTH, status=401)


def _errors_logged(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Pipecat's task wrapper logs an unexpected exception in the launch task at ERROR, and only there."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


# --- setup ---------------------------------------------------------------------------------------


@pytest.mark.unit
async def test_setup_returns_fast() -> None:
    """`setup()` returns in ≪ the room's injected 5 s launch (< 1.0 s, never a tight bound), because
    Pipecat gives all processors' setup a shared 20 s.

    `launch_delay=5` is the injected slow operation. A bound far below an injected slow duration is the
    one kind of wall-clock assertion allowed, because it cannot flake.
    """
    room = FakeAvatarRoom(JoinMode.DIRECT, launch_delay=5.0)
    service = make_service(room)
    setup, tm = make_setup()

    started = monotonic()
    await service.setup(setup)
    elapsed = monotonic() - started

    assert elapsed < 1.0, f"setup() took {elapsed:.2f}s: it must not wait on the room"
    assert service.state is ServiceState.LAUNCHING
    await service.cleanup()
    # cleanup() ran before the launch task got a turn, so launch() was cancelled before it started
    # (nothing to observe) or while it slept (the room records the cancel); either way no session.
    assert room.cancelled or room.launch_calls == 0
    assert room.session.aclose_count == 0
    assert service.state is ServiceState.ENDED
    assert_no_dangling(tm)


# --- the launch succeeds -------------------------------------------------------------------------


@pytest.mark.unit
async def test_session_active_path(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """`on_avatar_ready` fires once, when the status first reports every session avatar ready; nothing is
    logged at ERROR. (The events' payload types, and that `on_session_ended` and `on_metrics` fire once
    each at a pipeline end, are pinned in test_service_events.py and test_teardown_idempotent_all_orders.)"""
    session = FakeAvatarSession()
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)
    assert service.session_id is None

    async with bidirectional_pipeline(service) as h:
        connected = await rec.wait("on_avatar_connected")
        assert connected[0].active_avatar_id == "av_a"
        assert service.session_id == session.session_id
        assert service.state is ServiceState.SESSION_ACTIVE

        await delivered(session.emit_status({"av_a": "ready", "av_b": "preparing"}, all_ready=False))
        await delivered(session.emit_status({"av_a": "ready", "av_b": "ready"}, all_ready=True))
        ready = await rec.wait("on_avatar_ready")
        assert ready[0].session_id == session.session_id
        assert ready[0].status == {"av_a": "ready", "av_b": "ready"}
        await delivered(session.emit_status({"av_a": "ready", "av_b": "ready"}, all_ready=True))
        await h.finish()

    names = rec.names()
    assert rec.count("on_avatar_connected") == 1
    assert rec.count("on_avatar_ready") == 1
    assert names.index("on_avatar_connected") < names.index("on_avatar_ready")
    assert names.index("on_session_ended") < names.index("on_metrics")
    assert rec.payload("on_metrics")[0].end_reason == "pipeline_ended"
    assert _errors_logged(caplog_loguru) == []


@pytest.mark.unit
async def test_session_says_which_avatar_is_active() -> None:
    """The service mirrors the session's active avatar rather than assuming the first, and never swaps to
    "correct" it."""
    session = FakeAvatarSession(active_avatar_id="av_b")
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room, avatar_ids=("av_a", "av_b"))
    rec = EventRecorder(service)
    assert service.active_avatar_id == "av_a"

    async with bidirectional_pipeline(service) as h:
        connected = await rec.wait("on_avatar_connected")
        assert connected[0].active_avatar_id == "av_b"
        assert service.active_avatar_id == "av_b"
        await h.finish()

    assert session.swap_calls == []


# --- pass-through --------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("end_with", ["EndFrame", "CancelFrame"])
async def test_passthrough_all_other_frames(end_with: str) -> None:
    """The ONLY strict-sequence test: with `launch_delay=60` no task can emit during it (state stays
    LAUNCHING, no drain, no ErrorFrame), so every frame downstream/upstream is one we forwarded
    (*frame contract*).

    Frames are waited for one group at a time (`h.frame`) rather than fired in a burst because the
    `InterruptionFrame` is a SystemFrame: it overtakes queued data frames and resets the process queue.
    """
    room = FakeAvatarRoom(JoinMode.DIRECT, launch_delay=60.0)
    service = make_service(room)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TextFrame("hello"))
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1"))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.frame(TTSStoppedFrame)
        await h.queue(InterruptionFrame())
        await h.frame(InterruptionFrame)
        await h.queue(OutputTransportReadyFrame(), UP)
        await h.queue(TextFrame("up"), UP)
        await h.frame(TextFrame, direction=UP)
        assert service.state is ServiceState.LAUNCHING
        if end_with == "EndFrame":
            await h.finish()
        else:
            await h.queue(CancelFrame())
            await h.finish(send_end_frame=False)

    tail = EndFrame if end_with == "EndFrame" else CancelFrame
    assert_strict_frames(h.down, [TextFrame, TTSStartedFrame, TTSStoppedFrame, InterruptionFrame, tail])
    assert_strict_frames(h.up, [OutputTransportReadyFrame, TextFrame])
    assert room.cancelled is True


# --- launch failure -------------------------------------------------------------------------------


@pytest.mark.unit
async def test_required_true_failure_ends_the_pipeline() -> None:
    """A 401 with `required=True`, under Pipecat's default `ProcessorUnusablePolicy.CONTINUE`: FAILED, one
    permanent upstream ErrorFrame with the true category, `is_usable` False, and the service ends the
    pipeline itself, so an EndFrame reaches the sink without the test sending one.
    `on_session_ended("launch_failed")` and `on_metrics` fire once (*permanent-error rule*)."""
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401())
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        error = await h.frame(ErrorFrame, direction=UP)
        assert service.state is ServiceState.FAILED
        assert error.category is ErrorCategory.AUTHENTICATION
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.status == 401 and error.exception.type is ErrorType.AUTH
        assert service.is_usable is False

        await h.frame(EndFrame)  # the service's own end: nothing else sends one under CONTINUE
        await h.finish(send_end_frame=False)

    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "launch_failed"
    assert service.metrics.end_reason == "launch_failed"
    assert rec.count("on_metrics") == 1
    assert count(h.up, ErrorFrame) == 1
    assert room.session.send_audio_calls == []


@pytest.mark.unit
async def test_required_true_failure_drops_prelaunch_audio() -> None:
    """Pre-launch audio queued before a `required=True` launch failure is dropped, not handed to the
    transport: the pipeline is ending, so the bot must not greet the user and then hang up.
    `hold_launch` keeps the launch in progress until the audio is queued."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401(), hold_launch=gate)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.frame(TTSStoppedFrame)  # the audio before it was processed (and queued as pre-launch audio)
        assert service.state is ServiceState.LAUNCHING
        gate.set()
        await rec.wait("on_session_ended")
        assert service.state is ServiceState.FAILED
        await h.finish()

    assert_absent(h.down, TTSAudioRawFrame, where="after a required=True launch failure")
    assert room.session.send_audio_calls == []


@pytest.mark.unit
async def test_required_true_failure_drops_tts_audio_in_flight() -> None:
    """TTS audio that reaches the service after a `required=True` launch failure, but ahead of the EndFrame
    that ends the pipeline, is dropped (*frame contract*).

    Pipecat awaits `on_error` handlers before it pushes the ErrorFrame, so frames this handler queues enter
    the pipeline source ahead of that EndFrame, as a TTS service still streaming its greeting would. The
    `TextFrame` behind the audio proves the audio reached the service."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401(), hold_launch=gate)
    service = make_service(room)

    async with bidirectional_pipeline(service) as h:

        @service.event_handler("on_error")
        async def _greeting_in_flight(_service, _error) -> None:
            await h.queue(TTSStartedFrame(context_id="c1"))
            await h.queue(tts_audio("c1", AUDIO_2))
            await h.queue(TTSStoppedFrame(context_id="c1"))
            await h.queue(TextFrame("after the audio"))

        gate.set()
        await h.frame(TextFrame)
        assert service.state is ServiceState.FAILED
        await h.finish()

    assert_absent(h.down, TTSAudioRawFrame, where="after a required=True launch failure")
    assert room.session.send_audio_calls == []


@pytest.mark.unit
async def test_required_false_failure(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The same 401 with `required=False`: DEGRADED, `is_usable` stays True, the frame carries
    APPLICATION (non-permanent) while `exception.category` is the true AUTHENTICATION, queued pre-launch
    audio is handed to the transport (the voice-only handover) and later audio passes through, the
    pipeline keeps running, one WARNING, `on_session_ended("degraded")`; `is_usable` re-checked after
    later audio (*permanent-error rule*)."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401(), hold_launch=gate)
    service = make_service(room, required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await h.queue(tts_audio("c1", AUDIO_3))
        await h.queue(TTSStoppedFrame(context_id="c1"))
        await h.frame(TTSStoppedFrame)  # the audio before it was processed (and queued as pre-launch audio)
        assert service.state is ServiceState.LAUNCHING
        assert_absent(h.down, TTSAudioRawFrame, where="downstream while the launch runs")
        gate.set()

        error = await h.frame(ErrorFrame, direction=UP)
        assert service.state is ServiceState.DEGRADED
        assert service.is_usable is True
        assert error.category is ErrorCategory.APPLICATION
        assert error.category.is_permanent is False
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.category is ErrorCategory.AUTHENTICATION

        await until(lambda: count(h.down, TTSAudioRawFrame) == 2, what="the pre-launch audio handed over")
        handed_over = [
            f.audio for f in frames_of(h.down, TTSAudioRawFrame) if isinstance(f, TTSAudioRawFrame)
        ]
        assert handed_over == [AUDIO_1, AUDIO_3], "once each, in order"

        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await h.queue(TTSStoppedFrame(context_id="c2"))
        await until(
            lambda: any(f.audio == AUDIO_2 for f in frames_of(h.down, TTSAudioRawFrame)),
            what="later TTS audio passing through in DEGRADED",
        )
        assert service.is_usable is True, "nothing after the launch failure may flip usability"
        assert h.worker.has_finished() is False
        assert_absent(h.down, EndFrame, where="before we sent one")
        await h.finish()

    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "degraded"
    assert rec.count("on_metrics") == 1
    assert len(records_at(caplog_loguru, logging.WARNING, "voice-only")) == 1
    assert room.session.send_audio_calls == []


# What a failed launch leaves behind, by `required`: the terminal state, the end reason, and usability.
_FAILED_LAUNCH = {
    True: (ServiceState.FAILED, "launch_failed", False),
    False: (ServiceState.DEGRADED, "degraded", True),
}


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "not-required"])
@pytest.mark.parametrize("session_dies", [False, True], ids=["leave-report-only", "leave-and-dead"])
async def test_avatar_leaving_during_the_launch_fails_the_launch(required: bool, session_dies: bool) -> None:
    """The leave report alone fails the launch, and is a launch failure, not a disconnect. Delivering the
    callback directly leaves the session claiming to be alive; `emit_avatar_left` also marks it dead."""

    async def leave_during_launch(room: FakeAvatarRoom, ctx: LaunchContext) -> None:
        if session_dies:
            await delivered(room.session.emit_avatar_left("avatar_left_room", humans_present=True))
        else:
            await ctx.callbacks.on_avatar_left("avatar_left_room", True)

    room = FakeAvatarRoom(JoinMode.DIRECT, during_launch=leave_during_launch)
    await _assert_launch_failed_after_the_room_returned(room, required)


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "not-required"])
async def test_session_dead_at_the_end_of_the_launch_fails_the_launch(required: bool) -> None:
    """A leave emitted before the launch installed the callbacks marks the fake session dead and delivers
    nothing."""
    session = FakeAvatarSession()
    assert session.emit_avatar_left("room_connection_lost") is None
    assert session.is_alive is False
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    await _assert_launch_failed_after_the_room_returned(room, required)


async def _assert_launch_failed_after_the_room_returned(room: FakeAvatarRoom, required: bool) -> None:
    state, reason, usable = _FAILED_LAUNCH[required]
    service = make_service(room, required=required)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        error = await h.frame(ErrorFrame, direction=UP)
        assert service.state is state
        assert service.is_usable is usable
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.type is ErrorType.CONNECTION
        assert error.exception.category is ErrorCategory.CONNECTIVITY
        assert error.exception.session_id == room.session.session_id
        await until(lambda: room.session.aclose_count >= 1, what="the session handed back to be closed")
        await rec.wait("on_session_ended")
        await h.finish()

    assert rec.payload("on_session_ended")[0] == reason
    assert rec.count("on_avatar_connected") == 0
    assert rec.count("on_avatar_disconnected") == 0
    assert room.session.send_audio_calls == []


@pytest.mark.unit
async def test_unexpected_exception_from_the_room_fails_the_launch(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """An aiohttp error whose `repr` carries the Authorization header: an UNKNOWN launch failure naming the
    exception type, and the API key appears nowhere (*redaction rule*)."""
    request = aiohttp.RequestInfo(
        URL("https://api.example.invalid/v2/sessions"),
        "POST",
        CIMultiDictProxy(CIMultiDict({"Authorization": f"Bearer {API_KEY}"})),
        URL("https://api.example.invalid/v2/sessions"),
    )

    async def fail(_room: FakeAvatarRoom, _ctx: LaunchContext) -> None:
        raise aiohttp.ClientResponseError(request, (), status=502, message="Bad Gateway")

    room = FakeAvatarRoom(JoinMode.DIRECT, during_launch=fail)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        error = await h.frame(ErrorFrame, direction=UP)
        assert service.state is ServiceState.FAILED
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.type is ErrorType.UNKNOWN
        assert "ClientResponseError" in error.exception.message
        await rec.wait("on_session_ended")
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "launch_failed"
    assert_no_secrets(caplog_loguru.text, error.error, str(error.exception), repr(error.exception))


# --- launch timeout ------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(("required", "reason"), [(True, "launch_timeout"), (False, "degraded")])
async def test_launch_timeout_cancels_the_launch(required: bool, reason: str) -> None:
    """`launch_timeout=0.3` (the config seam) against a 5 s launch: the one `wait_for` cancels `launch()`
    (*launch timeout*), the room observes the cancel, and the session ends with the documented reason."""
    room = FakeAvatarRoom(JoinMode.DIRECT, launch_delay=5.0)
    service = make_service(room, launch_timeout=0.3, required=required)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        ended = await rec.wait("on_session_ended")
        assert ended[0] == reason
        assert room.cancelled is True
        assert service.state is (ServiceState.FAILED if required else ServiceState.DEGRADED)
        error = await h.frame(ErrorFrame, direction=UP)
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.type is ErrorType.TIMEOUT
        await h.finish()

    assert service.metrics.end_reason == reason
    assert rec.count("on_session_ended") == 1


@pytest.mark.unit
async def test_launch_failure_reason_is_launch_failed() -> None:
    """A non-TIMEOUT launch failure with `required=True` ends with reason "launch_failed"."""
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=SynthesiaError("refused", type=ErrorType.CONNECTION))
    service = make_service(room)
    rec = EventRecorder(service)
    async with bidirectional_pipeline(service) as h:
        assert (await rec.wait("on_session_ended"))[0] == "launch_failed"
        assert service.state is ServiceState.FAILED
        await h.finish()
    assert service.metrics.end_reason == "launch_failed"


# --- teardown ------------------------------------------------------------------------------------


async def _session_active_by_hand(room: FakeAvatarRoom, **overrides):
    """Set a service up outside a pipeline and wait for SESSION_ACTIVE; returns (service, recorder,
    task manager)."""
    service = make_service(room, **overrides)
    rec = EventRecorder(service)
    setup, tm = make_setup()
    await service.setup(setup)
    await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")
    return service, rec, tm


async def _run_step(service, step: str) -> None:
    if step == "stop":
        await service.stop(EndFrame())
    elif step == "cancel":
        await service.cancel(CancelFrame())
    else:
        await service.cleanup()


_FIRST_REASON = {"stop": "pipeline_ended", "cancel": "pipeline_cancelled", "cleanup": "pipeline_cleanup"}


@pytest.mark.unit
@pytest.mark.parametrize("order", list(itertools.permutations(("stop", "cancel", "cleanup"))))
async def test_teardown_idempotent_all_orders(order: tuple[str, str, str]) -> None:
    """Every order of stop/cancel/cleanup on a started service: `aclose` once, `on_session_ended` once
    with the first caller's reason, nothing left running (*single flight*)."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service, rec, tm = await _session_active_by_hand(room)

    for step in order:
        await _run_step(service, step)
    if "cleanup" != order[-1]:
        await service.cleanup()  # a real pipeline always ends with cleanup(); it must still be a no-op

    assert room.session.aclose_count == 1
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == _FIRST_REASON[order[0]]
    assert service.metrics.end_reason == _FIRST_REASON[order[0]]
    assert rec.count("on_metrics") == 1
    assert service.state is ServiceState.ENDED
    assert_no_dangling(tm)


@pytest.mark.unit
async def test_injected_aiohttp_session_is_not_closed() -> None:
    """A `session=` passed in belongs to the caller: teardown leaves it open."""
    http = aiohttp.ClientSession()
    try:
        room = FakeAvatarRoom(JoinMode.DIRECT)
        service, _rec, tm = await _session_active_by_hand(room, session=http)
        await service.stop(EndFrame())
        await service.cleanup()
        assert http.closed is False
        assert_no_dangling(tm)
    finally:
        await http.close()


@pytest.mark.unit
async def test_own_aiohttp_session_is_closed() -> None:
    """Without `session=`, the service owns its aiohttp session and closes it at teardown.

    Observed without touching a private: an unclosed `aiohttp.ClientSession` emits a `ResourceWarning`
    when it is garbage-collected, so the scenario runs in its own frame, everything it created is
    collected under `catch_warnings(record=True)`, and no such warning may have been recorded.
    """

    async def scenario() -> None:
        room = FakeAvatarRoom(JoinMode.DIRECT)
        service, _rec, tm = await _session_active_by_hand(room)
        await service.stop(EndFrame())
        await service.cleanup()
        assert_no_dangling(tm)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await scenario()
        gc.collect()

    leaks = [
        w for w in caught if issubclass(w.category, ResourceWarning) and "client session" in str(w.message)
    ]
    assert not leaks, f"the service's own aiohttp session was not closed: {[str(w.message) for w in leaks]}"


@pytest.mark.unit
async def test_cleanup_on_never_set_up_service() -> None:
    """`cleanup()` on a service whose `setup()` never ran returns, ends the state machine and reports
    `on_session_ended("pipeline_cleanup")` once. Pipecat may call `cleanup()` on a half-built processor;
    even then a `session=` passed in belongs to the caller and stays open."""
    http = aiohttp.ClientSession()
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service = make_service(room, session=http)
    rec = EventRecorder(service)

    try:
        await service.cleanup()
        await service.cleanup()  # idempotent
        assert http.closed is False
    finally:
        await http.close()

    assert service.state is ServiceState.ENDED
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_cleanup"
    assert rec.count("on_metrics") == 1
    assert room.session.aclose_count == 0


@pytest.mark.unit
async def test_cancel_interrupts_the_avatar_and_is_not_held_by_the_teardown() -> None:
    """Mid-utterance, a `CancelFrame` travels on at once, while the interrupt it starts is still in flight at
    the avatar worker (`hold_interrupt`). The teardown lets that interrupt complete before it closes the
    session, and the close (`hold_aclose`) holds nothing up either; the session then ends once, with
    `pipeline_cancelled`."""
    acknowledge = asyncio.Event()
    closing = asyncio.Event()
    session = FakeAvatarSession(hold_interrupt=acknowledge, hold_aclose=closing)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            await h.queue(tts_audio("c1"))
            await until(lambda: len(session.send_audio_calls) == 1, what="the utterance under way")

            await h.queue(CancelFrame())
            await h.frame(CancelFrame)  # forwarded, although the interrupt and the session close are held
            await until(lambda: session.interrupt_entered == 1, what="the interrupt in flight")

            acknowledge.set()
            await until(lambda: session.aclose_count == 1, what="the teardown reaching the session close")
            names = [call[0] for call in session.calls]
            assert "interrupt" in names, "the teardown cut the interrupt short"
            assert names.index("interrupt") < names.index("aclose"), names
            assert session.aclose_completed is False
            assert rec.count("on_session_ended") == 0, "the session has not ended while aclose() is pending"
        finally:
            acknowledge.set()
            closing.set()
        await rec.wait("on_session_ended")
        await h.finish(send_end_frame=False)

    assert session.interrupt_count == 1
    assert session.aclose_completed is True
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_cancelled"
    assert service.metrics.end_reason == "pipeline_cancelled"
    assert service.state is ServiceState.ENDED


@pytest.mark.unit
async def test_a_hung_interrupt_holds_the_teardown_only_up_to_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt the avatar worker never acknowledges (`hold_interrupt`, never set) does not hold the
    teardown: past the bound (an input, shrunk to 0.05 s) the teardown stops waiting, closes the session
    and ends it, and nothing is left running. What is asserted is the outcome, never a duration."""
    monkeypatch.setattr(service_module, "INTERRUPTION_WAIT_SECS", 0.05)
    never = asyncio.Event()
    session = FakeAvatarSession(hold_interrupt=never)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service, rec, tm = await _session_active_by_hand(room)

    try:
        await service.cancel(CancelFrame())
        await until(lambda: session.interrupt_entered == 1, what="the interrupt in flight")
        await rec.wait("on_session_ended")
        await service.cleanup()
    finally:
        never.set()

    assert session.aclose_count == 1
    assert session.interrupt_count == 0, "the interrupt was still held when the teardown gave up on it"
    assert rec.payload("on_session_ended")[0] == "pipeline_cancelled"
    assert_no_dangling(tm)


@pytest.mark.unit
async def test_session_is_over_as_soon_as_cancel_returns() -> None:
    """The teardown finishes in a task, but a swap is already refused without reaching the avatar worker."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service, rec, tm = await _session_active_by_hand(room)

    await service.cancel(CancelFrame())
    assert service.state is ServiceState.ENDED
    with pytest.raises(SynthesiaError) as excinfo:
        await service.swap_avatar("av_b")
    assert excinfo.value.type is ErrorType.CONNECTION

    await service.cleanup()
    assert room.session.swap_calls == []
    assert rec.payload("on_session_ended")[0] == "pipeline_cancelled"
    assert_no_dangling(tm)


@pytest.mark.unit
async def test_avatar_leave_during_a_cancel_is_part_of_the_shutdown() -> None:
    """A leave already on its way when the cancel starts pushes no error and does not change the end reason.

    Emitted before `cancel()`, its callback task runs when the cancel first yields, so today it lands during
    the cancel and fires `on_avatar_disconnected`. That timing is the scheduler's, not a contract: after the
    cancel a leave is ignored, so the disconnect is asserted only as at most one."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    service, rec, tm = await _session_active_by_hand(room)

    leave = room.session.emit_avatar_left("room_connection_lost", humans_present=True)
    assert leave is not None
    await service.cancel(CancelFrame())
    await leave
    await service.cleanup()

    assert rec.count("on_avatar_disconnected") <= 1
    assert rec.count("on_error") == 0
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_cancelled"
    assert_no_dangling(tm)


@pytest.mark.unit
async def test_teardown_cancelled_midway_is_finished_by_cleanup() -> None:
    """A `stop()` cancelled while its teardown is blocked in `session.aclose()` must not count as a
    completed teardown: the later `cleanup()` runs the remaining steps, so the session is closed and
    `on_session_ended` fires exactly once, with cleanup's reason."""
    hold = asyncio.Event()
    session = FakeAvatarSession(hold_aclose=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service, rec, tm = await _session_active_by_hand(room)

    stop_task = asyncio.create_task(service.stop(EndFrame()))
    await until(lambda: session.aclose_count == 1, what="stop() to reach aclose()")
    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task
    assert rec.count("on_session_ended") == 0

    hold.set()
    await service.cleanup()

    assert session.aclose_completed is True
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "pipeline_cleanup"
    assert service.state is ServiceState.ENDED
    assert_no_dangling(tm)


# --- the session-close backstop ------------------------------------------------------------------


class CancelProofCloseSession(FakeAvatarSession):
    """A fake session whose `aclose()` is *cancel-proof cleanup*, as `LiveKitAvatarSession.aclose()` is: a
    cancel is absorbed until the close has finished, and only then re-raised. That is what makes
    `asyncio.wait_for` unable to bound it."""

    async def aclose(self) -> None:
        if await await_cancel_proof(asyncio.ensure_future(super().aclose())):
            raise asyncio.CancelledError


def _pending_closes() -> list[asyncio.Task]:
    """Session closes still running anywhere in the loop, whoever started them."""
    return [
        task
        for task in asyncio.all_tasks()
        if not task.done() and "aclose" in getattr(task.get_coro(), "__qualname__", "")
    ]


@pytest.mark.unit
async def test_teardown_waits_for_a_slow_close_only_up_to_the_backstop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session close that outlasts the backstop does not hold teardown hostage: `cleanup()` returns
    while the close is still running, the close then finishes on its own, and nothing is left running.

    The backstop is an input, shrunk to 0.05 s; the close is held by an event, so what is asserted is the
    order (teardown finished before the close did), never a wall-clock bound."""
    monkeypatch.setattr(service_module, "SESSION_ACLOSE_BACKSTOP_SECS", 0.05)
    hold = asyncio.Event()
    session = CancelProofCloseSession(hold_aclose=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service, rec, tm = await _session_active_by_hand(room)

    cleanup = asyncio.ensure_future(service.cleanup())
    try:
        await until(cleanup.done, what="cleanup() to return past the backstop")
        cleanup.result()
        assert session.aclose_completed is False, "teardown returned before the close finished"
        assert rec.count("on_session_ended") == 1
        assert service.state is ServiceState.ENDED
    finally:
        hold.set()

    await until(lambda: session.aclose_completed, what="the detached close to finish")
    await until(lambda: not _pending_closes(), what="no session close left running")
    assert session.aclose_count == 1
    assert_no_dangling(tm)


@pytest.mark.unit
async def test_session_that_left_during_launch_is_closed_within_the_backstop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other bounded close: the avatar leaves inside `launch()`, and the service closes the session it
    was handed. A close that outlasts the (injected, 0.05 s) backstop does not stop the launch failure
    from being reported, and the close still finishes afterwards."""
    monkeypatch.setattr(service_module, "SESSION_ACLOSE_BACKSTOP_SECS", 0.05)
    hold = asyncio.Event()

    async def leave(room: FakeAvatarRoom, _ctx) -> None:
        task = room.session.emit_avatar_left("avatar_left_room", humans_present=True)
        assert task is not None
        await task

    session = CancelProofCloseSession(hold_aclose=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, during_launch=leave)
    service = make_service(room)
    rec = EventRecorder(service)

    try:
        async with bidirectional_pipeline(service) as h:
            error = await h.frame(ErrorFrame, direction=UP)
            assert isinstance(error.exception, SynthesiaError)
            assert error.exception.type is ErrorType.CONNECTION
            assert session.aclose_completed is False, "the failure was reported before the close finished"
            hold.set()
            await until(lambda: session.aclose_completed, what="the detached close to finish")
            await h.finish()
    finally:
        hold.set()

    await until(lambda: not _pending_closes(), what="no session close left running")
    assert rec.payload("on_session_ended")[0] == "launch_failed"


# --- teardown during the launch ------------------------------------------------------------------


@pytest.mark.unit
async def test_teardown_while_launching_waits_for_launch_cleanup() -> None:
    """`cleanup()` during LAUNCHING waits for the room's cancellation cleanup to COMPLETE (a flag on the
    fake, not a duration) and never closes a session that does not exist.

    `cancel_cleanup_delay=1.5` exceeds Pipecat's default 1 s `cancel_task` bound, so this test fails if
    `_launch` were cancelled with that default instead of `LAUNCH_CANCEL_SECS`.

    On Pipecat 1.13+ this is also the path a cancel takes when it lands before the worker's start task has
    returned: `cleanup()` alone, no `cancel()`.
    """
    room = FakeAvatarRoom(JoinMode.DIRECT, launch_delay=5.0, cancel_cleanup_delay=1.5)
    service = make_service(room)
    rec = EventRecorder(service)
    setup, tm = make_setup()
    await service.setup(setup)
    assert service.state is ServiceState.LAUNCHING
    await until(lambda: room.launch_calls == 1, what="launch() to be in progress")

    await service.cleanup()

    assert room.cancelled is True
    assert room.cancel_cleanup_completed is True
    assert room.session.aclose_count == 0
    assert service.state is ServiceState.ENDED
    assert rec.payload("on_session_ended")[0] == "pipeline_cleanup"
    assert_no_dangling(tm)


@pytest.mark.unit
async def test_teardown_during_the_launch_is_quiet(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The pipeline ends while the room is still launching, with pre-launch audio queued: the launch is
    cancelled without draining, the audio is dropped (neither sent nor handed on), no session is closed,
    and nothing is logged at ERROR."""
    room = FakeAvatarRoom(JoinMode.DIRECT, launch_delay=5.0)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: room.launch_calls == 1, what="the launch to be under way")
        assert service.state is ServiceState.LAUNCHING
        await h.queue(tts_audio("c1"))
        await h.finish()

    assert room.cancelled is True
    assert_absent(h.down, TTSAudioRawFrame, where="after the pipeline ended during the launch")
    assert room.session.send_audio_calls == []
    assert room.session.aclose_count == 0
    assert service.state is ServiceState.ENDED
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"
    assert _errors_logged(caplog_loguru) == []


# --- processor_unusable_policy -------------------------------------------------------------------


@pytest.mark.unit
async def test_required_false_401_keeps_pipeline_under_END_policy() -> None:
    """Under `ProcessorUnusablePolicy.END` a 401 with `required=False` must NOT end the pipeline: the
    upstream ErrorFrame is APPLICATION with the 401 on `.exception`, audio pushed afterwards reaches the
    sink, the `PipelineWorker` is still running, and the EndFrame that ends the run is the one WE send
    (*permanent-error rule*)."""
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401())
    service = make_service(room, required=False)

    async with bidirectional_pipeline(service, processor_unusable_policy=ProcessorUnusablePolicy.END) as h:
        error = await h.frame(ErrorFrame, direction=UP)
        assert error.category is ErrorCategory.APPLICATION
        assert isinstance(error.exception, SynthesiaError) and error.exception.status == 401

        await h.queue(tts_audio("c1"))
        await h.frame(TTSAudioRawFrame)  # primary, positive check: voice still flows after the error
        assert h.worker.has_finished() is False
        assert_absent(h.down, EndFrame, where="before we sent one")  # secondary check
        await h.finish()

    assert count(h.down, EndFrame) == 1


@pytest.mark.unit
async def test_required_true_failure_under_END_policy_ends_once() -> None:
    """Under `ProcessorUnusablePolicy.END` the PipelineWorker ends the pipeline for the permanent error,
    and the service asks for the same end: exactly one EndFrame reaches the sink. This pins how Pipecat
    1.14 to 1.16 handle the two requests, so an upgrade that delivers a second EndFrame is caught."""
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401())
    service = make_service(room)

    async with bidirectional_pipeline(service, processor_unusable_policy=ProcessorUnusablePolicy.END) as h:
        error = await h.frame(ErrorFrame, direction=UP)
        assert error.category is ErrorCategory.AUTHENTICATION
        await h.frame(EndFrame)
        await h.finish(send_end_frame=False)

    assert count(h.down, EndFrame) == 1
    assert_absent(h.down, CancelFrame)
    assert service.is_usable is False
    assert service.state is ServiceState.FAILED


@pytest.mark.unit
async def test_required_true_failure_under_CANCEL_policy_cancels() -> None:
    """Under `ProcessorUnusablePolicy.CANCEL` a permanent error in a running pipeline cancels it, and the
    service's own end request does not turn that into a graceful end: a CancelFrame reaches the sink, and no
    EndFrame does. The error stays permanent so that a developer's CANCEL is honoured.

    `hold_launch` keeps the launch in progress until the marker has proved the worker is pushing frames:
    on Pipecat 1.13+ the harness can yield before the worker's start task has returned, and a cancel in that
    window is resolved by cleanup alone, without a CancelFrame (the next test)."""
    gate = asyncio.Event()
    room = FakeAvatarRoom(JoinMode.DIRECT, fail_with=_auth_401(), hold_launch=gate)
    service = make_service(room)

    async with bidirectional_pipeline(service, processor_unusable_policy=ProcessorUnusablePolicy.CANCEL) as h:
        await processed(h, "pipeline running")
        gate.set()
        await h.frame(ErrorFrame, direction=UP)
        await h.frame(CancelFrame)
        await h.finish(send_end_frame=False)

    assert_absent(h.down, EndFrame, where="under CANCEL")
    assert service.is_usable is False
    assert service.state is ServiceState.FAILED


@pytest.mark.unit
@pytest.mark.parametrize("join_mode", [JoinMode.DIRECT, JoinMode.RELAYED], ids=["direct", "relayed"])
async def test_required_true_failure_during_startup_under_CANCEL_policy(join_mode: JoinMode) -> None:
    """The launch fails at once, inside `setup()`'s launch task, while the pipeline is still starting.

    Pipecat 1.13+ pushes the StartFrame while processors are still being set up, and a cancel before the
    worker's start task has returned is resolved by cleanup alone: no CancelFrame is pushed and
    `on_pipeline_finished` does not fire, so the service sees `cleanup()` and not `cancel()`. The
    StartFrame may or may not have reached the sink by then (it has on Python 3.11, not on 3.12, where the
    launch fails a loop turn sooner), hence `may_never_start=True`; both sub-paths are cleanup only.
    Either way the contract holds: no EndFrame, FAILED,
    unusable, `on_session_ended("launch_failed")` and `on_metrics` once, the worker finishes, nothing
    left running (the harness checks that on exit). Whether a CancelFrame reaches the sink is Pipecat's
    business, so it is not asserted."""
    room = FakeAvatarRoom(join_mode, fail_with=_auth_401())
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CANCEL, may_never_start=True
    ) as h:
        await h.frame(ErrorFrame, direction=UP)
        await h.finish(send_end_frame=False)

    assert_absent(h.down, EndFrame, where="under CANCEL, failure during startup")
    assert service.is_usable is False
    assert service.state is ServiceState.FAILED
    assert rec.count("on_session_ended") == 1
    assert rec.payload("on_session_ended")[0] == "launch_failed"
    assert rec.count("on_metrics") == 1
    assert h.worker.has_finished()


# --- StartFrame ----------------------------------------------------------------------------------


@pytest.mark.unit
async def test_start_frame_forwarded_while_room_slow() -> None:
    """With a 5 s room launch the pipeline still reaches `on_pipeline_started`: the StartFrame is
    forwarded before the avatar exists (a swallowed StartFrame would trip the harness's start
    timeout; *frame contract*)."""
    room = FakeAvatarRoom(JoinMode.DIRECT, launch_delay=5.0)
    service = make_service(room)
    async with bidirectional_pipeline(service) as h:
        assert service.state is ServiceState.LAUNCHING
        await h.queue(TextFrame("through"))
        await h.frame(TextFrame)
        await h.finish()
    assert room.cancelled is True
