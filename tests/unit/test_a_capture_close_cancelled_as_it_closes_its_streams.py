"""A track capture's close cancelled at any loop turn leaves no `stream.aclose()` coroutine un-awaited.

`AvatarTrackCapture.aclose()` runs as the session's close hook, under `CLOSE_HOOK_SECS`, so its caller can
cancel it at any point. Past its consumers, it closes their streams, each under `STREAM_CLOSE_SECS`. A
cancel that lands after those closes are scheduled and before they take their first step must not leave a
`stream.aclose()` coroutine that was created and never run: Python reports it as "coroutine ... was never
awaited" in the bot's log, at whatever moment the collector finds it.

No seam sits between scheduling the closes and their first step, so the cancel is swept across every loop
turn of the close, from its first step to one past its last. (A close cancelled before its first step
stops nothing, by contract.) `asyncio.sleep(0)` here steps the loop one turn at a time for that sweep; it
synchronises nothing. Each turn gets a fresh capture.

`AvatarTrackCapture` is driven directly, on its two stream-factory seams, because `PrivateLiveKitRoom`
builds it on the real `rtc` streams.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import warnings

from livekit import rtc as lk_rtc
from loguru import logger

from pipecat_synthesia.livekit.track_capture import AvatarTrackCapture
from tests.fakes.fake_media import FakeSubscribedPublication, StreamFactory
from tests.fakes.fake_rtc import FakeRemoteParticipant, FakeRoom
from tests.support.waiting import until

AVATAR = "synthesia-avatar-agent"
#: More loop turns than a close of two consumers takes; the sweep checks that its last turn is past it.
TURNS = 40


async def _ignore(data: bytes, a: int, b: int) -> None:
    return None


async def close_cancelled_after(turns: int) -> bool:
    """Capture both of the avatar's tracks, then cancel `aclose()` after `turns` loop turns.

    Returns whether the close had already finished when the cancel came.
    """
    video, audio = StreamFactory("video"), StreamFactory("audio")
    publications = {
        "TR_video": FakeSubscribedPublication("TR_video", kind=int(lk_rtc.TrackKind.KIND_VIDEO)),
        "TR_audio": FakeSubscribedPublication("TR_audio", kind=int(lk_rtc.TrackKind.KIND_AUDIO)),
    }
    room = FakeRoom("private-room")
    room.add_participant(FakeRemoteParticipant(AVATAR, track_publications=dict(publications)))
    capture = AvatarTrackCapture(
        room,  # type: ignore[arg-type] -- the fake satisfies RoomLike (fake_rtc's guard)
        avatar_identity=AVATAR,
        audio_sample_rate=24000,
        audio_channels=1,
        on_video_frame=_ignore,
        on_audio_frame=_ignore,
        log=logger.bind(component="test"),
        video_stream_factory=video,
        audio_stream_factory=audio,
    )
    before = asyncio.all_tasks()
    capture.start()
    await until(lambda: bool(video.streams) and bool(audio.streams), what="both consumers' streams")

    closing = asyncio.ensure_future(capture.aclose())
    for _ in range(turns):
        await asyncio.sleep(0)
    finished = closing.done()
    closing.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await closing
    await until(lambda: asyncio.all_tasks() <= before, what="every task of the close to end")
    return finished


async def test_a_capture_close_cancelled_at_any_loop_turn_leaves_no_close_unawaited() -> None:
    unawaited: dict[int, list[str]] = {}
    finished = False
    for turns in range(1, TURNS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finished = await close_cancelled_after(turns)
            gc.collect()  # a coroutine dropped without running is reported when it is collected
        messages = [str(w.message) for w in caught if "was never awaited" in str(w.message)]
        if messages:
            unawaited[turns] = messages
        if finished:
            break

    assert finished, f"the sweep stopped before the close finished: raise TURNS past {TURNS}"
    assert unawaited == {}, f"a cancelled close left coroutines un-awaited, by loop turn: {unawaited}"
