"""The chunk being sent to the avatar when it leaves mid-call: handed over, unless an interruption dropped it.

When the avatar leaves while humans remain, the bot continues voice-only, and the voice-only handover pushes
the send queue's audio to the user transport. The chunk the send loop has taken off the queue and is still
sending reaches no one at the avatar, which has left. This file pins, with the fake session, that:

* a leave delivered while a send is in flight hands that chunk over first, then the queued chunks, in
  order;
* a chunk in flight when an interruption arrives stays dropped, even if the avatar leaves while the
  interruption is still being handled: the interruption drops that audio on purpose. Both ways a leave can
  meet that chunk are covered: while its cancelled send is still unwinding, and after a send that outlived
  the cancel returns with the session no longer alive.

The `AvatarSession` interface lets a session deliver the leave while a send is in flight, and the fake does.
The LiveKit session finishes the send in flight before it reports the leave; that order is pinned in
`test_service_with_a_leaving_livekit_session.py`.

`FakeAvatarSession(hold_send=)` keeps the first chunk in flight at the avatar worker. For the interruption
cases the hold is a `_SlowToCancel`, a send that takes a while to give up once cancelled, so the leave can
be delivered after the interruption has cancelled the send and before the send loop has unwound from it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)

from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, tts_audio

_NAMES = {AUDIO_1: "AUDIO_1", AUDIO_2: "AUDIO_2", AUDIO_3: "AUDIO_3"}

Handler = Callable[[object, Frame], Awaitable[None]]


class _SlowToCancel(asyncio.Event):
    """A `hold_send` for a send that takes a while to give up once cancelled.

    Until cancelled it holds the send like a plain event. Once cancelled it sets `cancelled` and waits for
    `unwind`; then it re-raises the cancel or, with `absorb=True`, returns as if the send had finished
    (setting `absorbed`). `finished` is set when it stops waiting, whichever way; a second cancel while it
    waits for `unwind` ends it at once.
    """

    def __init__(self, *, absorb: bool = False) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()
        self.unwind = asyncio.Event()
        self.absorbed = False
        self.finished = False
        self._absorb = absorb

    async def wait(self) -> Literal[True]:
        try:
            return await super().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.unwind.wait()
            if self._absorb:
                self.absorbed = True
                return True
            raise
        finally:
            self.finished = True

    def release(self) -> None:
        """Let any send it holds finish, cancelled or not."""
        self.set()
        self.unwind.set()


def _hold_the_first(frame_type: type[Frame], entered: asyncio.Event, release: asyncio.Event) -> Handler:
    """A handler that, the first time it sees a `frame_type`, waits for `release`; otherwise returns."""
    held = False

    async def hold(_processor: object, frame: Frame) -> None:
        nonlocal held
        if held or not isinstance(frame, frame_type):
            return
        held = True
        entered.set()
        await release.wait()

    return hold


def _audio_downstream(h: Harness) -> list[str]:
    """The voice-only audio by chunk name, so a failure shows the order rather than 2 KB of bytes each."""
    return [
        _NAMES.get(f.audio, f"<{len(f.audio)} other bytes>")
        for f in frames_of(h.down, TTSAudioRawFrame)
        if isinstance(f, TTSAudioRawFrame)
    ]


async def _queue_one_utterance_with_its_first_chunk_in_flight(h: Harness, session: FakeAvatarSession) -> None:
    await h.queue(TTSStartedFrame(context_id="c1"))
    for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
        await h.queue(tts_audio("c1", chunk))
    await processed(h, "the utterance queued")
    await until(lambda: len(session.send_audio_calls) == 1, what="the first chunk in flight")


@pytest.mark.unit
async def test_a_leave_hands_over_the_chunk_in_flight_first() -> None:
    in_flight = asyncio.Event()
    session = FakeAvatarSession(hold_send=in_flight)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_one_utterance_with_its_first_chunk_in_flight(h, session)

            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await rec.wait("on_session_ended")
        finally:
            in_flight.set()
        await h.finish()

    assert _audio_downstream(h) == ["AUDIO_1", "AUDIO_2", "AUDIO_3"], (
        "the voice-only audio did not start with the chunk in flight at the leave"
    )
    assert rec.payload("on_session_ended")[0] == "avatar_left"


@pytest.mark.unit
async def test_an_interruption_drops_the_chunk_in_flight_even_if_the_avatar_leaves_as_it_is_handled() -> None:
    """The leave is delivered after the interruption has cancelled the send in flight, and before that
    send has unwound: the interruption is still waiting for it."""
    hold = _SlowToCancel()
    session = FakeAvatarSession(hold_send=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_one_utterance_with_its_first_chunk_in_flight(h, session)

            await h.queue(InterruptionFrame())  # the user talks over the bot
            await until(hold.cancelled.is_set, what="the interruption to cancel the send in flight")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await rec.wait("on_session_ended")
            assert not hold.finished, "the send stopped unwinding before the leave was handled"
            hold.unwind.set()
            await h.frame(InterruptionFrame)
        finally:
            hold.release()
        await h.finish()

    assert _audio_downstream(h) == [], "interrupted audio reached the user after the avatar left"
    assert rec.payload("on_session_ended")[0] == "avatar_left"


@pytest.mark.unit
async def test_a_send_that_outlives_an_interruption_does_not_bring_its_chunk_back_at_a_leave() -> None:
    """The interruption cancels the send in flight, but the send outlives the cancel and returns, with the
    session no longer alive, after the avatar has left and before the leave's handover takes the queue. The
    leave handling yields at its `BotStoppedSpeakingFrame` push, as a developer's sync handler can make it."""
    hold = _SlowToCancel(absorb=True)
    entered = asyncio.Event()
    release = asyncio.Event()
    session = FakeAvatarSession(hold_send=hold)
    room = FakeAvatarRoom(JoinMode.DIRECT, session=session)
    service = make_service(room)
    rec = EventRecorder(service)
    service.add_event_handler(
        "on_before_push_frame", _hold_the_first(BotStoppedSpeakingFrame, entered, release)
    )

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_one_utterance_with_its_first_chunk_in_flight(h, session)
            session.emit_playback_started()
            await h.frame(BotStartedSpeakingFrame)

            await h.queue(InterruptionFrame())  # the user talks over the bot
            await until(hold.cancelled.is_set, what="the interruption to cancel the send in flight")
            session.emit_avatar_left("avatar_left_room", humans_present=True)
            await until(entered.is_set, what="the leave handling to yield before its handover")
            hold.unwind.set()  # the send returns after all
            # the interruption waits for the send loop to end before it sends the interrupt
            await until(lambda: session.interrupt_entered == 1, what="the send loop to end")
            assert hold.absorbed, "the send did not outlive the cancel"
            release.set()
            await rec.wait("on_session_ended")
        finally:
            release.set()
            hold.release()
        await h.finish()

    assert _audio_downstream(h) == [], "interrupted audio reached the user after the avatar left"
    assert rec.payload("on_session_ended")[0] == "avatar_left"
