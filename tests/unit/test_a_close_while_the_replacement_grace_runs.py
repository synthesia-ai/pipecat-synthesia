"""A close reports no leave, even when a *replacement grace* started before it ends while the close is still
closing the byte stream.

The replacement grace (1 s) starts on a video unpublish or a disconnect with no stated reason, and reports
`on_avatar_left` when it ends without the avatar participant back. `aclose()` stops being alive at once and
then closes the open byte stream, which waits for rtc to acknowledge the close, up to `STREAM_CLOSE_SECS`.
A close is the end of the session, not a leave: `test_livekit_session.py` pins that a grace pending at a
fast close reports nothing afterwards. This file pins the same for a close whose byte-stream close outlasts
the grace.

It drives `LiveKitAvatarSession` on the session rig (`_session_rig.py`): the replacement grace and the close
are the session's own, and both room adapters only build its `RoomHandle` and hand over.
`hold_after_the_close_request` holds rtc's acknowledgement of the byte stream's close, as a stalled data
channel would; the grace is shrunk to `GRACE`, and the hold lasts `GRACE_MARGIN`, ten times longer.
"""

from __future__ import annotations

import asyncio

import pytest

from pipecat_synthesia.livekit import _types
from tests.support.waiting import until
from tests.unit._session_rig import GRACE, GRACE_MARGIN, VIDEO_SID, Rig, settle
from tests.unit._slow_byte_streams import hold_after_the_close_request

CHUNK = b"\x01\x02" * 480  # 20 ms of 24 kHz mono s16le
#: Far longer than the test: the held acknowledgement, not the close budget, decides when the close ends.
LONG = 60.0


def unpublish_video(rig: Rig) -> None:
    assert rig.avatar is not None
    rig.unpublish(rig.avatar, VIDEO_SID)


def disconnect_without_a_reason(rig: Rig) -> None:
    assert rig.avatar is not None
    rig.depart(rig.avatar, reason=None)


@pytest.mark.parametrize(
    "start_the_grace",
    [unpublish_video, disconnect_without_a_reason],
    ids=["video-unpublished", "disconnect-without-a-reason"],
)
async def test_a_replacement_grace_that_ends_during_the_close_reports_no_leave(
    start_the_grace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grace starts with an utterance's byte stream open; the close starts before the grace ends, and
    its byte-stream close is still waiting for rtc when the grace ends. No `on_avatar_left` is delivered,
    during the close or after it."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", LONG)
    rig = Rig()
    session = await rig.launch()
    release: asyncio.Event | None = None
    closing: asyncio.Future[None] | None = None
    try:
        await session.send_audio(CHUNK, 24000, 1)
        (writer,) = rig.participant.streams
        release = hold_after_the_close_request(writer)

        start_the_grace(rig)
        closing = asyncio.ensure_future(session.aclose())
        await until(lambda: writer.closed, what="the close to request the byte stream's close")
        await asyncio.sleep(GRACE_MARGIN)  # the acknowledgement outlasts the grace
        release.set()
        await until(closing.done, what="the close to finish once rtc acknowledges")
        await asyncio.sleep(GRACE_MARGIN)  # and nothing happened
    finally:
        if release is not None:
            release.set()
        if closing is not None:
            await settle(closing)
        await session.aclose()

    assert rig.rec.left == [], "the close reported the avatar left"
    rig.assert_torn_down()
