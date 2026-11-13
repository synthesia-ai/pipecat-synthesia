"""A session request that fails, or never answers, after an *early start*.

After an early start the service is SESSION_ACTIVE before the Synthesia API's 201: audio has gone to the
avatar, the avatar may be speaking, and the user may interrupt. If the request then fails, the launch has
failed after all, and the service ends as a failed launch does (`required` decides). What this file pins:

* the speaking state ends with it: a bot left speaking would keep Pipecat's turn-taking believing the bot
  is still talking;
* an interruption while the avatar is being stopped drops the interrupted audio, as it would once the
  session is active, instead of handing it to the user transport voice-only;
* the avatar is told to stop (clear buffer) before the failed session closes: in a direct join it stays in
  the room, and would go on playing what it was sent;
* the *launch timeout* still bounds the session request, as it did when the launch waited for the 201:
  a request still out when it runs out ends the service as a launch timeout, through the same path;
* the voice-only bot does not wait for the failed session's close, as after an unexpected leave;
* a graceful end (EndFrame) still plays what was queued, voice-only ahead of the EndFrame, whether or not the
  failure path's clear buffer has answered when it arrives: a graceful end plays what was queued; a cancel
  does not.

`FakeAvatarSession(hold_created=...)` is a session whose launch returned before the 201;
`created_error` fails it; `hold_interrupt` holds the clear buffer, `hold_aclose` the close.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.worker import ProcessorUnusablePolicy

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.protocol import TOKEN_ATTR_LAUNCH
from pipecat_synthesia.room import AvatarSession, JoinMode, LaunchContext
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import count, frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, delivered, processed, tts_audio
from tests.unit._session_rig import Rig

FAILURE = SynthesiaError("the API dropped the connection", type=ErrorType.CONNECTION)
LAUNCH = "0123456789abcdef"
HYSTERESIS = 0.05


def _early(**kwargs) -> tuple[FakeAvatarRoom, FakeAvatarSession, asyncio.Event]:
    created = asyncio.Event()
    session = FakeAvatarSession(hold_created=created, **kwargs)
    return FakeAvatarRoom(JoinMode.DIRECT, session=session), session, created


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "not-required"])
async def test_a_failure_while_the_avatar_speaks_ends_the_speaking_state(required: bool) -> None:
    room, session, created = _early(created_error=FAILURE)
    service = make_service(
        room, required=required, bot_stopped_hysteresis_secs=HYSTERESIS, stop_drain_timeout=0.2
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: session.send_audio_calls, what="audio at the avatar before the 201")
        await delivered(session.emit_playback_started())
        await until(lambda: count(h.down, BotStartedSpeakingFrame) == 1, what="the bot speaking")

        created.set()
        await rec.wait("on_session_ended")
        await until(
            lambda: count(h.down, BotStoppedSpeakingFrame) == 1,
            what="the bot no longer speaking once the launch failed",
        )
        if not required:
            await h.finish()


@pytest.mark.unit
async def test_an_interruption_while_the_avatar_is_being_stopped_drops_the_interrupted_audio() -> None:
    """`required=False`: the service goes on voice-only. The user interrupts c2 while the failed session's
    avatar is still being stopped (clear buffer held); c2's audio must not reach the user transport after
    the interruption, as no interrupted audio would once the session is active."""
    release_interrupt = asyncio.Event()
    room, session, created = _early(created_error=FAILURE, hold_interrupt=release_interrupt)
    service = make_service(room, required=False, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: session.send_audio_calls, what="audio at the avatar before the 201")

        created.set()
        await until(lambda: session.interrupt_entered == 1, what="the avatar being stopped")
        await h.queue(TTSStartedFrame(context_id="c2"))
        await h.queue(tts_audio("c2", AUDIO_2))
        await processed(h, "c2 queued")
        await h.queue(InterruptionFrame())
        await processed(h, "interrupted")

        release_interrupt.set()
        await rec.wait("on_session_ended")
        await processed(h, "after the end")

    down = h.down
    interruption = next(i for i, f in enumerate(down) if isinstance(f, InterruptionFrame))
    after = [f.audio for f in frames_of(down[interruption:], TTSAudioRawFrame)]
    assert AUDIO_2 not in after, "the interrupted audio was handed over voice-only after the interruption"
    assert rec.payload("on_session_ended")[0] == "degraded"


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "not-required"])
async def test_the_avatar_is_stopped_before_the_failed_session_closes(required: bool) -> None:
    """In a direct join the avatar stays in the room after the failure, and would go on playing what it was
    sent, over the voice-only bot when the pipeline continues: clear buffer goes out before the close."""
    room, session, created = _early(created_error=FAILURE)
    service = make_service(room, required=required, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        await h.queue(tts_audio("c1", AUDIO_1))
        await until(lambda: session.send_audio_calls, what="audio at the avatar before the 201")

        created.set()
        await rec.wait("on_session_ended")
        if required:
            await h.finish(send_end_frame=False)

    names = [call[0] for call in session.calls]
    assert "interrupt" in names, "the avatar was never told to stop"
    assert names.index("interrupt") < names.index("aclose")


@pytest.mark.unit
async def test_the_launch_timeout_still_bounds_a_201_that_never_comes() -> None:
    """The launch timeout bounds the whole launch, the session request included. After an early start the
    service is active before the 201, but a request still out when the launch timeout runs out ends the
    service as a launch timeout, as it did when the launch waited for the 201."""
    room, session, _created = _early()  # the 201 never comes
    service = make_service(room, launch_timeout=0.3, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(
        service, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
    ) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="SESSION_ACTIVE")
        await rec.wait("on_session_ended")
        await h.finish(send_end_frame=False)

    assert rec.payload("on_session_ended")[0] == "launch_timeout"
    assert service.state is ServiceState.FAILED
    assert session.aclose_count == 1
    assert rec.count("on_avatar_connected") == 0


@pytest.mark.unit
@pytest.mark.parametrize("end", ["avatar-left", "failed-201"])
async def test_voice_only_audio_does_not_wait_for_the_failed_session_to_close(end: str) -> None:
    """`required=False`: once the avatar is stopped, the bot goes on voice-only at once, as after an
    unexpected leave, without waiting for the session's close, which can take seconds in a relayed join
    (the private room's disconnect and delete). The close is held open here."""
    release_close = asyncio.Event()
    room, session, created = _early(created_error=FAILURE, hold_aclose=release_close)
    service = make_service(room, required=False, stop_drain_timeout=0.2)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
            if end == "failed-201":
                created.set()
            else:
                await delivered(session.emit_avatar_left("avatar_left_room"))
            await until(lambda: session.aclose_count == 1, what="the session closing")

            await h.queue(TTSStartedFrame(context_id="c2"))
            await h.queue(tts_audio("c2", AUDIO_2))
            await until(
                lambda: count(h.down, TTSAudioRawFrame) == 1,
                what="later TTS audio voice-only while the session is still closing",
            )
        finally:
            release_close.set()
        await rec.wait("on_session_ended")


@pytest.mark.unit
@pytest.mark.parametrize("session_kind", ["real", "fake"])
async def test_a_failed_request_leaves_the_session_without_a_session_id(
    session_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There never was a 201, so neither the real session nor the fake one gives a session id or a 201
    time, before or after the failure: a service that read either there would be caught by both."""
    if session_kind == "real":
        monkeypatch.setattr(session_module, "secrets", SimpleNamespace(token_hex=lambda _nbytes: LAUNCH))
        gate = asyncio.Event()
        rig = Rig(api=FakeSessionApiClient(gate=gate, error=FAILURE), attributes={TOKEN_ATTR_LAUNCH: LAUNCH})
        session: AvatarSession = await asyncio.wait_for(rig.launch(), timeout=5)  # an early start
    else:
        gate = asyncio.Event()
        session = FakeAvatarSession(hold_created=gate, created_error=FAILURE)
    try:
        for _ in ("before", "after"):
            with pytest.raises(RuntimeError):
                _ = session.session_id
            assert session.timings.session_created is None
            if not gate.is_set():
                gate.set()
                with pytest.raises(SynthesiaError):
                    await session.wait_session_created()
    finally:
        await session.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("clear_buffer", ["answered-before-the-end", "still-out-at-the-end"])
async def test_a_goodbye_before_a_graceful_end_is_heard_voice_only(clear_buffer: str) -> None:
    """Early start, required=False: the session request fails, and the failure path tells the avatar to
    stop (clear buffer). The bot then says goodbye and ends the call gracefully, as an end-call tool does:
    TTS audio, then an EndFrame. The goodbye reaches the user transport voice-only, ahead of the EndFrame,
    whether the clear buffer had answered by then or not: a graceful end lets the last utterance play (see
    stop()). A cancel plays nothing (see
    test_a_cancel_while_the_failure_path_stops_the_avatar_plays_nothing)."""
    created = asyncio.Event()
    release_interrupt = asyncio.Event()
    session = FakeAvatarSession(hold_created=created, created_error=FAILURE, hold_interrupt=release_interrupt)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session), required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
            created.set()  # the session request fails after the early start
            await until(lambda: session.interrupt_entered == 1, what="the failure path stopping the avatar")
            if clear_buffer == "answered-before-the-end":
                release_interrupt.set()
                await rec.wait("on_session_ended")
            # the bot's goodbye, then a graceful end, as an end-call tool pushes them
            await h.queue(TTSStartedFrame(context_id="c2"))
            await h.queue(tts_audio("c2", AUDIO_2))
            await h.queue(TTSStoppedFrame(context_id="c2"))
            await h.queue(EndFrame())
            await until(lambda: any(isinstance(f, EndFrame) for f in h.down), what="the EndFrame downstream")
            await h.finish(send_end_frame=False)
        finally:
            release_interrupt.set()

    end_at = next(i for i, f in enumerate(h.down) if isinstance(f, EndFrame))
    heard = [f.audio for f in frames_of(h.down[:end_at], TTSAudioRawFrame)]
    assert AUDIO_2 in heard, (
        f"the goodbye never reached the user transport (end reason {rec.payload('on_session_ended')[0]!r})"
    )


@pytest.mark.unit
async def test_speech_queued_before_the_failure_is_handed_over_ahead_of_a_graceful_end() -> None:
    """Early start, required=False: the first chunk in flight, two more queued, when the session request
    fails; the bot ends the call (EndFrame) while the failure path's clear buffer is still out. The queued
    speech reaches the user transport voice-only, in order, ahead of the EndFrame: a graceful end plays what
    was queued."""
    created, release_interrupt, hold_send = asyncio.Event(), asyncio.Event(), asyncio.Event()
    session = FakeAvatarSession(
        hold_created=created, created_error=FAILURE, hold_interrupt=release_interrupt, hold_send=hold_send
    )
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session), required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the utterance queued")
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            created.set()  # the session request fails after the early start
            await until(lambda: session.interrupt_entered == 1, what="the failure path stopping the avatar")
            await h.queue(EndFrame())
            await until(lambda: any(isinstance(f, EndFrame) for f in h.down), what="the EndFrame downstream")
            await h.finish(send_end_frame=False)
        finally:
            release_interrupt.set()
            hold_send.set()

    end_at = next(i for i, f in enumerate(h.down) if isinstance(f, EndFrame))
    assert [f.audio for f in frames_of(h.down[:end_at], TTSAudioRawFrame)] == [AUDIO_1, AUDIO_2, AUDIO_3]
    assert frames_of(h.down[end_at:], TTSAudioRawFrame) == []
    assert rec.payload("on_session_ended")[0] == "pipeline_ended"
    assert rec.count("on_error") == 0


@pytest.mark.unit
@pytest.mark.parametrize("audio", ["queued-before-the-failure", "goodbye-after-the-failure"])
async def test_a_cancel_while_the_failure_path_stops_the_avatar_plays_nothing(audio: str) -> None:
    """The other half of the rule: a cancel plays nothing. Early start, required=False, the session request
    fails, and a CancelFrame arrives while the failure path's clear buffer is still out, with speech queued
    before the failure, or a goodbye held after it. No TTS audio reaches the user transport: the failure path
    still hands the speech over, but only after the CancelFrame, and Pipecat drops what the service pushes
    after it."""
    created, release_interrupt, hold_send = asyncio.Event(), asyncio.Event(), asyncio.Event()
    session = FakeAvatarSession(
        hold_created=created, created_error=FAILURE, hold_interrupt=release_interrupt, hold_send=hold_send
    )
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session), required=False)

    async with bidirectional_pipeline(service) as h:
        try:
            await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
            if audio == "queued-before-the-failure":
                await h.queue(TTSStartedFrame(context_id="c1"))
                for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                    await h.queue(tts_audio("c1", chunk))
                await processed(h, "the utterance queued")
                await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            created.set()
            await until(lambda: session.interrupt_entered == 1, what="the failure path stopping the avatar")
            if audio == "goodbye-after-the-failure":
                await h.queue(TTSStartedFrame(context_id="c2"))
                await h.queue(tts_audio("c2", AUDIO_2))
                await h.queue(TTSStoppedFrame(context_id="c2"))
                await processed(h, "the goodbye held behind the handover")
                # without the wait above, the CancelFrame overtakes the goodbye before the service sees it;
                # no public seam shows the hold, so check it here
                held = service._held_for_handover
                assert held is not None, "the goodbye was not held"
                assert [frame.audio for frame, _ in held] == [AUDIO_2], "the goodbye was not held"
            await h.queue(CancelFrame())
            await until(lambda: any(isinstance(f, CancelFrame) for f in h.down), what="the CancelFrame")
            await h.finish(send_end_frame=False)
        finally:
            release_interrupt.set()
            hold_send.set()

    assert frames_of(h.down, TTSAudioRawFrame) == []


@pytest.mark.unit
@pytest.mark.parametrize("end", ["request-fails", "avatar-leaves", "pipeline-ends", "request-succeeds"])
async def test_an_early_start_reports_the_avatars_video_only_once_the_201_lands(end: str) -> None:
    """The early start does not move the `avatar_video_published` boundary: no 201, no metric, even when the
    video that started the session was this launch's. The avatar publishes its video during the launch, the
    session goes active before the 201, and then, before any 201: the request fails, the avatar leaves (the
    call goes on voice-only), or the pipeline ends. With the request succeeding instead, it is reported."""
    created = asyncio.Event()

    async def video_published(_room: FakeAvatarRoom, ctx: LaunchContext) -> None:
        await ctx.callbacks.on_track_published("video")

    session = FakeAvatarSession(
        hold_created=created, created_error=FAILURE if end == "request-fails" else None
    )
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session, during_launch=video_published)
    service = make_service(room, required=False)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await until(lambda: service.state is ServiceState.SESSION_ACTIVE, what="the early start")
        if end == "request-fails":
            created.set()
            await rec.wait("on_metrics")
        elif end == "avatar-leaves":
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await rec.wait("on_metrics")
        elif end == "request-succeeds":
            created.set()
            await rec.wait("on_avatar_connected")
        await h.finish()  # "pipeline-ends": the pipeline ends with the 201 still out
        created.set()

    summary = rec.payload("on_metrics")[0].summary()
    assert ("avatar_video_published" in summary) is (end == "request-succeeds"), summary
