"""A close reports no leave for a leave signal that arrives while it is still closing the byte stream.

`test_a_close_while_the_replacement_grace_runs.py` pins that a *replacement grace* started before the close
reports nothing. This file pins the same for the signals that arrive during the close, while it waits for
rtc to acknowledge the byte stream's close: a video unpublish and a disconnect with no stated reason (each
starts a grace), and a disconnect with a stated reason (a leave at once). A close is the end of the
session, not a leave.

Same rig and holds as that file: `LiveKitAvatarSession` on the session rig (`_session_rig.py`),
`hold_after_the_close_request` holding rtc's acknowledgement, the grace shrunk to `GRACE`, and the hold
lasting `GRACE_MARGIN`, ten times longer.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit import rtc as lk_rtc

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
    rig.leave(rig.avatar, reason=None)


def disconnect_with_a_reason(rig: Rig) -> None:
    assert rig.avatar is not None
    rig.leave(rig.avatar, reason=lk_rtc.DisconnectReason.CLIENT_INITIATED)


@pytest.mark.parametrize(
    "signal",
    [unpublish_video, disconnect_without_a_reason, disconnect_with_a_reason],
    ids=["video-unpublished", "disconnect-without-a-reason", "disconnect-with-a-reason"],
)
async def test_a_leave_signal_during_the_close_reports_no_leave(
    signal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The close starts with an utterance's byte stream open, and the signal arrives while the close waits
    for rtc to acknowledge that stream's close, for longer than a grace. No `on_avatar_left` is delivered,
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

        closing = asyncio.ensure_future(session.aclose())
        await until(lambda: writer.closed, what="the close to request the byte stream's close")
        signal(rig)
        await asyncio.sleep(GRACE_MARGIN)  # the acknowledgement outlasts a grace started by the signal
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


@pytest.mark.parametrize(
    "signal",
    [unpublish_video, disconnect_without_a_reason],
    ids=["video-unpublished", "disconnect-without-a-reason"],
)
async def test_a_grace_signal_as_the_close_ends_leaves_no_grace_running(
    signal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The signal arrives during the close, just before rtc acknowledges the byte stream's close. Once the
    close has returned, nothing the session started is still running: no grace outlives the session. The
    grace is `GRACE_MARGIN` here, so one that started would still be running."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE_MARGIN)
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", LONG)
    rig = Rig()
    session = await rig.launch()
    release: asyncio.Event | None = None
    closing: asyncio.Future[None] | None = None
    try:
        await session.send_audio(CHUNK, 24000, 1)
        (writer,) = rig.participant.streams
        release = hold_after_the_close_request(writer)

        before = asyncio.all_tasks()
        closing = asyncio.ensure_future(session.aclose())
        await until(lambda: writer.closed, what="the close to request the byte stream's close")
        signal(rig)
        release.set()
        await closing
        running = [task for task in asyncio.all_tasks() - before if not task.done()]
    finally:
        if release is not None:
            release.set()
        if closing is not None:
            await settle(closing)
        await session.aclose()

    assert running == [], f"still running after the close: {running}"
    assert rig.rec.left == [], "the close reported the avatar left"
