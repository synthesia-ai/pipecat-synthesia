"""A `send_audio` already in flight when the avatar leaves or the session closes leaves no byte stream open.

What this file pins (*one-open-stream rule*): a send that passed the session's liveness check before the
leave or the close may still open a byte stream, but that stream is closed before `on_avatar_left` is
delivered, or before `aclose()` returns, and no stream is opened after either. Two in-flight positions are
covered: a send whose open is still waiting for its writer, and a send queued behind the previous
utterance's close. Each runs against a leave and against a close.

`LiveKitAvatarSession` is internal, but it is the unit seam for the avatar session (see `_session_rig.py`).
`_types.STREAM_CLOSE_SECS` is raised far past `until()`'s 5 s, so a held close never runs out its budget
here: a close that overruns its budget is a different path, covered in
`test_the_next_byte_stream_after_a_slow_close.py`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from livekit import rtc as lk_rtc

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from tests.fakes.fake_rtc import FakeLocalParticipant
from tests.support.callbacks import CallbackRecorder
from tests.support.waiting import until
from tests.unit._session_rig import BOT, Rig, settle
from tests.unit._slow_byte_streams import GatedOpenParticipant, hold_after_the_close_request

CHUNK = b"\x01\x02" * 480  # 20 ms of 24 kHz mono s16le
CLIENT_INITIATED = lk_rtc.DisconnectReason.CLIENT_INITIATED
#: Far longer than `until()`'s 5 s: no close in this file may run out its budget.
LONG = 60.0


@dataclass(frozen=True)
class Streams:
    """What the bot had asked of the avatar worker at one moment: streams opened, and those still open."""

    opened: int
    unclosed: int


def streams_now(participant: FakeLocalParticipant) -> Streams:
    return Streams(opened=len(participant.streams), unclosed=len(participant.unclosed_streams()))


class LeaveSnapshotRecorder(CallbackRecorder):
    """Records the bot's byte streams at the moment `on_avatar_left` is delivered."""

    def __init__(self, participant: FakeLocalParticipant) -> None:
        super().__init__()
        self._participant = participant
        self.at_left: Streams | None = None

    async def on_avatar_left(self, reason: str, humans_present: bool | None) -> None:
        self.at_left = streams_now(self._participant)
        await super().on_avatar_left(reason, humans_present)


async def end_the_session(how: str, rig: Rig, session: LiveKitAvatarSession) -> asyncio.Future[None] | None:
    """Start the leave or the close; return once the session reports itself no longer alive."""
    if how == "leave":
        assert rig.avatar is not None
        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        ending = None
    else:
        ending = asyncio.ensure_future(session.aclose())
        await until(lambda: not session.is_alive, what="aclose() to have started")
    assert session.is_alive is False
    return ending


async def finish_ending(
    how: str,
    rec: LeaveSnapshotRecorder,
    participant: FakeLocalParticipant,
    ending: asyncio.Future[None] | None,
) -> Streams:
    """Wait for the leave to be delivered, or the close to return; the streams at that moment."""
    if how == "leave":
        await until(lambda: len(rec.left) == 1, what="on_avatar_left")
        assert rec.at_left is not None
        return rec.at_left
    assert ending is not None
    await ending
    return streams_now(participant)


ENDINGS = pytest.mark.parametrize("how", ["leave", "close"])


@ENDINGS
async def test_a_byte_stream_opening_as_the_session_ends_is_closed_before_it_ends(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The send passed the liveness check and is waiting for its writer when the session ends."""
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", LONG)
    participant = GatedOpenParticipant()
    rec = LeaveSnapshotRecorder(participant)
    rig = Rig(participant=participant, recorder=rec)
    session = await rig.launch()
    gate = participant.open_gate = asyncio.Event()
    sending = asyncio.ensure_future(session.send_audio(CHUNK, 24000, 1))
    ending: asyncio.Future[None] | None = None
    try:
        await until(lambda: len(participant.streams) == 1, what="the open to be requested")
        ending = await end_the_session(how, rig, session)

        gate.set()
        await sending
        at_end = await finish_ending(how, rec, participant, ending)

        assert streams_now(participant) == at_end, (
            f"a byte stream was opened after the session ended ({how}): "
            f"{at_end}, now {streams_now(participant)}"
        )
        assert at_end == Streams(opened=1, unclosed=0), (
            f"the stream opening as the session ended was still open when it ended ({how}): {at_end}"
        )
        assert participant.max_concurrent_open_streams == 1
    finally:
        gate.set()
        await settle(sending)
        if ending is not None:
            await settle(ending)
        await session.aclose()


@ENDINGS
async def test_a_send_waiting_behind_a_close_as_the_session_ends_leaves_no_byte_stream_open(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The most adversarial order: the next utterance's first chunk passes the liveness check while the
    previous utterance's close holds the streamer, and only gets to open its byte stream after the leave or
    the close has begun. That stream must still be closed before the session ends."""
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", LONG)
    participant = FakeLocalParticipant(BOT)
    rec = LeaveSnapshotRecorder(participant)
    rig = Rig(participant=participant, recorder=rec)
    session = await rig.launch()
    await session.send_audio(CHUNK, 24000, 1)  # utterance 1
    release = hold_after_the_close_request(participant.streams[0])
    closing_utterance = asyncio.ensure_future(session.end_utterance())
    sending: asyncio.Future[None] | None = None
    ending: asyncio.Future[None] | None = None
    try:
        await until(lambda: participant.streams[0].closed, what="utterance 1's close to be requested")
        sending = asyncio.ensure_future(session.send_audio(CHUNK, 24000, 1))  # utterance 2
        # One scheduler turn, not a wait for anything timed: asyncio runs ready callbacks in order, so the
        # send's first step (the liveness check, then its place behind the close) runs before this line.
        await asyncio.sleep(0)
        ending = await end_the_session(how, rig, session)

        release.set()
        await closing_utterance
        await sending
        at_end = await finish_ending(how, rec, participant, ending)

        assert streams_now(participant) == at_end, (
            f"a byte stream was opened after the session ended ({how}): "
            f"{at_end}, now {streams_now(participant)}"
        )
        assert at_end.unclosed == 0, (
            f"a byte stream opened after the session stopped being alive was left open ({how}): {at_end}"
        )
        assert participant.max_concurrent_open_streams == 1
        # The interleaving under test happened: utterance 2's stream was opened after the session had
        # stopped being alive. If a change makes that impossible, this guard is what fails.
        assert at_end.opened == 2, f"the queued send opened no stream; the race was not built: {at_end}"
    finally:
        release.set()
        await settle(closing_utterance)
        if sending is not None:
            await settle(sending)
        if ending is not None:
            await settle(ending)
        await session.aclose()
