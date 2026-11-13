"""The next byte stream after a close that outlived its wait: never open alongside the one being closed.

A close is a *detached close*: past `STREAM_CLOSE_SECS`, or when its caller is cancelled (the service's
interruption cancels its send loop), the streamer stops waiting and the next chunk may open the next
utterance's byte stream. What this file pins (*one-open-stream rule*):

* For a live writer, the close request has already gone out by then, because rtc 1.1.19's `aclose()` sends
  it in its first step (see `_slow_byte_streams.py`). So the next byte stream is requested only after the
  previous one's close: after an `end`, on a format change mid-utterance, and after a cancelled close
  followed by an interruption.
* Not pinned: an *orphaned open* (an interruption cancelled the send loop while it was opening) slower than
  the close budget. Its close cannot be requested until the open returns a writer, so the next utterance's
  byte stream can open alongside it. That is the one accepted exception to the rule, stated with it in
  docs/architecture.md.

Measured once by hand against livekit-server 1.9.1, with rtc 1.1.19 sending and receiving (2026-10-10):
with the close budget at 0, each stream's end still reached the receiver before the next stream's header;
with an orphaned open held past the budget, the next stream's header and end both arrived before the
orphaned stream's end. `FakeAvatarWorker.overlapping_streams` cannot make that distinction: it also
counts a header that arrives after the previous stream ended but before its serial reader has finished it.

`LiveKitAvatarSession` is internal, but it is the unit seam for the avatar session (see `_session_rig.py`).
`_types.STREAM_CLOSE_SECS` is shrunk only where the test is about the budget running out.
"""

from __future__ import annotations

import asyncio

import pytest

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.protocol import RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE
from tests.fakes.fake_rtc import FakeLocalParticipant
from tests.support.waiting import until
from tests.unit._session_rig import BOT, Rig, settle
from tests.unit._slow_byte_streams import hold_after_the_close_request

CHUNK = b"\x01\x02" * 480  # 20 ms of 24 kHz mono s16le
#: The close budget, shrunk: these tests are about it running out.
CLOSE_BUDGET = 0.05


async def opened_or_gave_up(participant: FakeLocalParticipant, sending: asyncio.Future[None]) -> None:
    """Wait until the next utterance's first chunk has opened a byte stream, or its send has finished
    without one. Either is allowed: it may be sent once nothing else is open, or dropped; it must never be
    opened alongside another stream. A send that waits without a bound (against *bounded waits*) fails
    here, on `until()`'s backstop."""
    await until(
        lambda: len(participant.streams) == 2 or sending.done(),
        what="the next utterance's first chunk to open its byte stream or give up",
    )


@pytest.mark.parametrize("variant", ["next-utterance", "format-change", "cancelled-close"])
async def test_a_close_still_unacknowledged_is_requested_before_the_next_byte_stream_opens(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Utterance 1's close stays unacknowledged until the end of the test, and the streamer has stopped
    waiting for it: its budget ran out, or (`cancelled-close`) the interruption cancelled the send loop that
    was waiting, and the interruption's own close found nothing open.

    In the cancelled case the cancel arrives after the close task's first step, which is when rtc sends
    the request: the close task is scheduled before the send loop can be woken by any cancel."""
    participant = FakeLocalParticipant(BOT)
    rig = Rig(participant=participant)
    session = await rig.launch()
    participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    await session.send_audio(CHUNK, 24000, 1)  # utterance 1
    release = hold_after_the_close_request(participant.streams[0])
    ending: asyncio.Future[None] | None = None
    sending: asyncio.Future[None] | None = None
    try:
        if variant == "next-utterance":
            monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", CLOSE_BUDGET)
            await session.end_utterance()  # stops waiting once the budget runs out
            sending = asyncio.ensure_future(session.send_audio(CHUNK, 24000, 1))
        elif variant == "format-change":
            monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", CLOSE_BUDGET)
            # closes utterance 1's stream, then opens one in the new format
            sending = asyncio.ensure_future(session.send_audio(CHUNK, 16000, 1))
        else:
            ending = asyncio.ensure_future(session.end_utterance())
            await until(lambda: participant.streams[0].closed, what="utterance 1's close to be requested")
            ending.cancel()
            await asyncio.wait({ending})
            # Shrunk only now, so that it is the cancel, not the budget, that ended the wait above.
            monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", CLOSE_BUDGET)
            await session.interrupt()  # clear buffer, then a close that finds nothing open
            sending = asyncio.ensure_future(session.send_audio(CHUNK, 24000, 1))
        await opened_or_gave_up(participant, sending)

        assert not release.is_set(), "utterance 1's close was still unacknowledged throughout"
        assert participant.max_concurrent_open_streams == 1, (
            f"the next byte stream was opened before utterance 1's close was requested: {participant!r}"
        )
    finally:
        release.set()
        for task in (ending, sending):
            if task is not None:
                await settle(task)
        await session.aclose()

    assert participant.unclosed_streams() == []
