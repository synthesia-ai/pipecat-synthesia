"""No playback event is delivered after `on_avatar_left`, even while the leave is still aborting the
utterance in flight.

What this file pins (`AvatarCallbacks`: "nothing is delivered after it"): the leave first aborts the open
utterance and only then delivers `on_avatar_left`, so there is a window in which the avatar has been
declared left but the service has not yet been told. Playback events that arrive in that window, or after
`on_avatar_left`, are never delivered. A delivery counts from the moment the callback is entered, which is
how `AvatarCallbacks` defines it: callbacks run concurrently and may interleave at any `await`, so a
`lk.playback_started` delivered before the leave may still be running when `on_avatar_left` is delivered.

`LiveKitAvatarSession` is internal, but it is the unit seam for the avatar session (see `_session_rig.py`).
`_types.STREAM_CLOSE_SECS` is raised far past `until()`'s 5 s, so the held abort cannot run out its budget.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from livekit import rtc as lk_rtc

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.protocol import RPC_HANDLER_RESPONSE, RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED
from tests.support.callbacks import CallbackRecorder
from tests.support.waiting import until
from tests.unit._session_rig import AVATAR, Rig
from tests.unit._slow_byte_streams import hold_after_the_close_request

CHUNK = b"\x01\x02" * 480  # 20 ms of 24 kHz mono s16le
CLIENT_INITIATED = lk_rtc.DisconnectReason.CLIENT_INITIATED
FINISHED = json.dumps({"playback_position": 1.0, "interrupted": False})
#: Far longer than `until()`'s 5 s: the held abort must not run out its budget.
LONG = 60.0


class EntryRecorder(CallbackRecorder):
    """Records each delivery as it is entered, before any `await`; can hold `on_playback_started` open."""

    def __init__(self) -> None:
        super().__init__()
        self.entered: list[str] = []
        self.hold_started: asyncio.Event | None = None
        #: Set once a held `on_playback_started` has been released and has finished.
        self.held_finished = False

    async def on_playback_started(self) -> None:
        self.entered.append("playback_started")
        hold = self.hold_started
        if hold is not None:
            await hold.wait()
        await super().on_playback_started()
        if hold is not None:
            self.held_finished = True

    async def on_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self.entered.append("playback_finished")
        await super().on_playback_finished(playback_position, interrupted)

    async def on_avatar_left(self, reason: str, humans_present: bool | None) -> None:
        self.entered.append("left")
        await super().on_avatar_left(reason, humans_present)


async def both_playback_events(rig: Rig) -> None:
    """The worker's two playback RPCs, each answered as a success (the handlers never raise)."""
    for method, payload in ((RPC_PLAYBACK_STARTED, ""), (RPC_PLAYBACK_FINISHED, FINISHED)):
        answer = await rig.participant.invoke(method, caller_identity=AVATAR, payload=payload)
        assert answer == RPC_HANDLER_RESPONSE


async def test_no_playback_event_is_delivered_after_on_avatar_left(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_types, "STREAM_CLOSE_SECS", LONG)
    rec = EntryRecorder()
    rig = Rig(recorder=rec)
    session = await rig.launch()
    assert rig.avatar is not None
    in_flight = rec.hold_started = asyncio.Event()
    release_abort: asyncio.Event | None = None
    try:
        await session.send_audio(CHUNK, 24000, 1)  # an utterance in flight, for the leave to abort

        # A playback_started delivered before the leave, still running when the leave is reported.
        await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)
        await until(lambda: "playback_started" in rec.entered, what="the in-flight playback_started")
        rec.hold_started = None

        # The leave, held inside its abort of the open utterance: declared, but not yet delivered.
        release_abort = hold_after_the_close_request(rig.participant.streams[0])
        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        await until(lambda: rig.participant.streams[0].closed, what="the leave's abort to be under way")
        assert rec.left == []
        await both_playback_events(rig)  # in the window

        release_abort.set()
        await until(lambda: len(rec.left) == 1, what="on_avatar_left")
        await both_playback_events(rig)  # after it

        # Releasing the in-flight callback last: its wake-up is queued behind every playback delivery
        # started above, so once it has finished, each of them has had its turn.
        in_flight.set()
        await until(lambda: rec.held_finished, what="the in-flight playback_started to finish")

        assert rec.entered.count("left") == 1
        after_left = rec.entered[rec.entered.index("left") + 1 :]
        assert after_left == [], f"delivered after on_avatar_left: {after_left} (all: {rec.entered})"
    finally:
        in_flight.set()
        if release_abort is not None:
            release_abort.set()
        await session.aclose()
