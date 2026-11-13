"""TTS audio that arrives while the voice-only handover is still being pushed waits behind it.

`test_voice_only_handover_after_an_unexpected_leave.py` pins the order after an unexpected leave. This file
pins the two other edges of the same hold:
- a failed launch with `required=False` hands over its pre-launch audio the same way, so audio that
  arrives while that handover is being pushed reaches the user after it, not in the middle of it;
- an interruption while audio is held drops it, as the interruption would have cleared it downstream had
  it passed straight through; and once the handover is over, later audio passes through again.

The yield inside the handover is a developer's sync `on_before_push_frame` handler that awaits, as in the
leave tests. `FakeAvatarRoom(hold_launch=, fail_with=)` makes the launch fail at a moment the test chooses,
and `FakeAvatarSession(hold_send=)` keeps the first chunk in flight at the avatar worker when it leaves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio

AUDIO_4 = b"\x04\x00" * 960
AUDIO_5 = b"\x05\x00" * 960
_NAMES = {AUDIO_1: "AUDIO_1", AUDIO_2: "AUDIO_2", AUDIO_3: "AUDIO_3", AUDIO_4: "AUDIO_4", AUDIO_5: "AUDIO_5"}

Handler = Callable[[object, Frame], Awaitable[None]]


def _hold_the_first(
    matches: Callable[[Frame], bool], entered: asyncio.Event, release: asyncio.Event
) -> Handler:
    """A handler that, the first time a pushed frame matches, waits for `release`; otherwise returns."""
    held = False

    async def hold(_processor: object, frame: Frame) -> None:
        nonlocal held
        if held or not matches(frame):
            return
        held = True
        entered.set()
        await release.wait()

    return hold


def _audio_downstream(h: Harness) -> list[str]:
    return [
        _NAMES.get(f.audio, f"<{len(f.audio)} other bytes>")
        for f in frames_of(h.down, TTSAudioRawFrame)
        if isinstance(f, TTSAudioRawFrame)
    ]


@pytest.mark.unit
async def test_audio_arriving_during_a_degraded_launchs_handover_reaches_the_user_after_it() -> None:
    release_launch = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    refused = SynthesiaError("bad key", type=ErrorType.AUTH, status=401)
    room = FakeAvatarRoom(JoinMode.DIRECT, hold_launch=release_launch, fail_with=refused)
    service = make_service(room, required=False)
    rec = EventRecorder(service)
    service.add_event_handler(
        "on_before_push_frame",
        _hold_the_first(lambda f: isinstance(f, TTSAudioRawFrame), entered, release),
    )

    async with bidirectional_pipeline(service) as h:
        try:
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the pre-launch audio queued")

            release_launch.set()  # the launch fails: voice-only, and the handover begins
            await until(entered.is_set, what="the handover's first push to yield")
            await h.queue(tts_audio("c1", AUDIO_3))  # arrives while the handover is being pushed
            await processed(h, "the later chunk")
            release.set()
            await rec.wait("on_session_ended")
            await until(lambda: len(_audio_downstream(h)) >= 3, what="all three chunks downstream")
        finally:
            release.set()
        await h.finish()

    assert _audio_downstream(h) == ["AUDIO_1", "AUDIO_2", "AUDIO_3"], (
        "the bot's speech reached the user out of order"
    )
    assert rec.payload("on_session_ended")[0] == "degraded"


@pytest.mark.unit
async def test_an_interruption_drops_the_audio_held_behind_the_handover() -> None:
    in_flight = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    session = FakeAvatarSession(hold_send=in_flight)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    service.add_event_handler(
        "on_before_push_frame",
        _hold_the_first(lambda f: isinstance(f, BotStoppedSpeakingFrame), entered, release),
    )

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await h.queue(TTSStartedFrame(context_id="c1"))
            for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
                await h.queue(tts_audio("c1", chunk))
            await processed(h, "the first utterance queued")
            await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")
            session.emit_playback_started()
            await h.frame(BotStartedSpeakingFrame)

            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(entered.is_set, what="the leave handling to yield")
            await h.queue(TTSStartedFrame(context_id="c2"))
            await h.queue(tts_audio("c2", AUDIO_4))  # held behind the handover
            await processed(h, "the next response's first chunk")
            await h.queue(InterruptionFrame())  # the user talks over the bot
            await h.frame(InterruptionFrame)
            release.set()
            await rec.wait("on_session_ended")

            await h.queue(TTSStartedFrame(context_id="c3"))
            await h.queue(tts_audio("c3", AUDIO_5))  # the next response, after the handover
            await processed(h, "the next response")
        finally:
            release.set()
            in_flight.set()
        await h.finish()

    assert _audio_downstream(h) == ["AUDIO_5"], "audio held at the interruption still reached the user"
    assert rec.payload("on_session_ended")[0] == "avatar_left"
