"""`AvatarTrackCapture` on the two stream-factory seams.

What these protect: the SDK contract the capture is built on (`format=RGB24` and a bounded capacity for
video, the pipeline's own rate for audio), one consumer per track kind, and the teardown order that keeps
a parked video consumer from outliving the session: cancel every task first, close every stream after.

The factory kwargs are asserted by name and value because they are an SDK contract, not an
implementation detail. Frames arriving at a callback are awaited with `until()`; no test sleeps.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from livekit import rtc as lk_rtc
from loguru import logger

from pipecat_synthesia.livekit.track_capture import AvatarTrackCapture
from tests.fakes.fake_media import (
    RGB24,
    FakeAudioFrame,
    FakeAudioFrameEvent,
    FakeSubscribedPublication,
    FakeTrack,
    FakeVideoFrame,
    FakeVideoFrameEvent,
    StreamFactory,
)
from tests.fakes.fake_rtc import FakeRemoteParticipant, FakeRoom
from tests.support.waiting import until

AVATAR = "synthesia-avatar-agent"
SAMPLE_RATE = 24000
KIND_VIDEO = int(lk_rtc.TrackKind.KIND_VIDEO)
KIND_AUDIO = int(lk_rtc.TrackKind.KIND_AUDIO)
I420 = lk_rtc.VideoBufferType.I420
#: How long the slow-to-die consumer holds its cancellation: an input, never asserted. It only has to
#: outlast the instant it takes the test to cancel `aclose()` from outside.
LINGER = 5.0


class Rig:
    """A room, a capture wired to both factories, and the frames its callbacks received."""

    def __init__(
        self,
        *,
        avatar_present: bool = True,
        video_publication: FakeSubscribedPublication | None = None,
        audio_publication: FakeSubscribedPublication | None = None,
        audio_channels: int = 1,
        on_video: Callable[[bytes, int, int], Awaitable[None]] | None = None,
    ) -> None:
        #: One list for both factories and both consumers: "cancelled" and "closed" in real order.
        self.order: list[str] = []
        self.video_factory = StreamFactory("video", order=self.order)
        self.audio_factory = StreamFactory("audio", order=self.order)
        self.video: list[tuple[bytes, int, int]] = []
        self.audio: list[tuple[bytes, int, int]] = []
        self.room = FakeRoom("private-room")
        self.avatar: FakeRemoteParticipant | None = None
        if avatar_present:
            publications: dict[str, Any] = {}
            for publication in (video_publication, audio_publication):
                if publication is not None:
                    publications[publication.sid] = publication
            self.avatar = self.room.add_participant(
                FakeRemoteParticipant(AVATAR, track_publications=publications)
            )
        self.capture = AvatarTrackCapture(
            self.room,  # type: ignore[arg-type] -- the fake satisfies RoomLike (fake_rtc's guard)
            avatar_identity=AVATAR,
            audio_sample_rate=SAMPLE_RATE,
            audio_channels=audio_channels,
            on_video_frame=on_video or self._on_video,
            on_audio_frame=self._on_audio,
            log=logger.bind(component="test"),
            video_stream_factory=self.video_factory,
            audio_stream_factory=self.audio_factory,
        )

    async def _on_video(self, data: bytes, width: int, height: int) -> None:
        self.video.append((data, width, height))

    async def _on_audio(self, data: bytes, sample_rate: int, channels: int) -> None:
        self.audio.append((data, sample_rate, channels))

    def subscribe(
        self, sid: str, kind: int, *, identity: str = AVATAR
    ) -> tuple[FakeTrack, FakeSubscribedPublication]:
        """Emit `track_subscribed` the way rtc does: (track, publication, participant)."""
        publication = FakeSubscribedPublication(sid, kind=kind)
        participant = FakeRemoteParticipant(identity)
        self.room.emit("track_subscribed", publication.track, publication, participant)
        return publication.track, publication


def video_publication(sid: str = "TR_video") -> FakeSubscribedPublication:
    return FakeSubscribedPublication(sid, kind=KIND_VIDEO)


def audio_publication(sid: str = "TR_audio") -> FakeSubscribedPublication:
    return FakeSubscribedPublication(sid, kind=KIND_AUDIO)


# --------------------------------------------------------------------------- picking up tracks


async def test_existing_subscribed_tracks_are_picked_up_at_start() -> None:
    """Tracks already subscribed at `start()` are captured: with `auto_subscribe=True`, the avatar's
    tracks can be subscribed before we arrive."""
    video, audio = video_publication(), audio_publication()
    rig = Rig(video_publication=video, audio_publication=audio)
    rig.capture.start()
    try:
        await until(
            lambda: bool(rig.video_factory.streams) and bool(rig.audio_factory.streams),
            what="both consumers to open their stream",
        )
        assert rig.video_factory.calls[0]["track"] is video.track
        assert rig.audio_factory.calls[0]["track"] is audio.track
    finally:
        await rig.capture.aclose()


@pytest.mark.parametrize("unreadable", ["unsubscribed", "subscribed-without-a-track"])
async def test_unsubscribed_publication_is_not_consumed(unreadable: str) -> None:
    """A publication we have not subscribed to yet, or one subscribed with no track object, has no track
    to read, so capture waits for the event.

    The audio track is the barrier: once *its* consumer has run, a video consumer that was going to
    start would have started too, so "no video stream" is an outcome rather than a race.
    """
    video: Any = video_publication()
    if unreadable == "unsubscribed":
        video.subscribed = False
    else:
        video.track = None
    rig = Rig(video_publication=video, audio_publication=audio_publication())
    rig.capture.start()
    try:
        await until(lambda: bool(rig.audio_factory.streams), what="the audio consumer")
        assert rig.video_factory.calls == []
    finally:
        await rig.capture.aclose()


async def test_later_track_subscribed_starts_consumption() -> None:
    """A track that turns up after `start()` is captured through the `track_subscribed` listener."""
    rig = Rig()
    rig.capture.start()
    try:
        track, _ = rig.subscribe("TR_video", KIND_VIDEO)
        await until(lambda: bool(rig.video_factory.streams), what="the video consumer")
        assert rig.video_factory.calls[0]["track"] is track
        rig.subscribe("TR_audio", KIND_AUDIO)
        await until(lambda: bool(rig.audio_factory.streams), what="the audio consumer")
    finally:
        await rig.capture.aclose()


async def test_other_identities_are_ignored() -> None:
    """Only the avatar's tracks are captured; a human's camera must never reach the pipeline."""
    rig = Rig()
    rig.capture.start()
    try:
        rig.subscribe("TR_human", KIND_VIDEO, identity="human")
        # The avatar's own audio track is the barrier: it went through the same listener, after.
        rig.subscribe("TR_audio", KIND_AUDIO)
        await until(lambda: bool(rig.audio_factory.streams), what="the audio consumer")
        assert rig.video_factory.calls == []
    finally:
        await rig.capture.aclose()


async def test_second_video_track_is_ignored_with_one_info(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """One consumer per track kind: a second video track is logged once and dropped."""
    rig = Rig()
    rig.capture.start()
    try:
        rig.subscribe("TR_video", KIND_VIDEO)
        await until(lambda: bool(rig.video_factory.streams), what="the video consumer")
        rig.subscribe("TR_video2", KIND_VIDEO)
        rig.subscribe("TR_video3", KIND_VIDEO)
        rig.subscribe("TR_audio", KIND_AUDIO)  # the barrier, emitted after both extra tracks
        await until(lambda: bool(rig.audio_factory.streams), what="the audio consumer")
        assert len(rig.video_factory.calls) == 1
        extra = [
            record
            for record in caplog_loguru.records
            if record.levelno == logging.INFO and "extra avatar video track" in record.getMessage()
        ]
        assert len(extra) == 1, f"expected one INFO about the extra track, got {len(extra)}"
    finally:
        await rig.capture.aclose()


# --------------------------------------------------------------------------- the SDK contract


async def test_stream_factory_kwargs() -> None:
    """The streams ask the FFI for RGB24 video, a bounded capacity, and audio at the pipeline's own rate."""
    rig = Rig(video_publication=video_publication(), audio_publication=audio_publication(), audio_channels=1)
    rig.capture.start()
    try:
        await until(
            lambda: bool(rig.video_factory.calls) and bool(rig.audio_factory.calls),
            what="both factories",
        )
        video_kwargs = dict(rig.video_factory.calls[0])
        video_kwargs.pop("track")
        assert video_kwargs == {"format": RGB24, "capacity": 8}

        audio_kwargs = dict(rig.audio_factory.calls[0])
        audio_kwargs.pop("track")
        assert audio_kwargs == {"sample_rate": SAMPLE_RATE, "num_channels": 1, "capacity": 50}
    finally:
        await rig.capture.aclose()


async def test_frames_reach_the_callbacks_without_converting_rgb24() -> None:
    """RGB24 frames go straight through: `FakeVideoFrame.convert` raises if it is ever called."""
    rig = Rig(video_publication=video_publication(), audio_publication=audio_publication())
    rig.capture.start()
    try:
        await until(
            lambda: bool(rig.video_factory.streams) and bool(rig.audio_factory.streams),
            what="both consumers",
        )
        frame = FakeVideoFrame(width=6, height=4, data=b"rgb-bytes", type=RGB24)
        rig.video_factory.stream.push(FakeVideoFrameEvent(frame))
        await until(lambda: bool(rig.video), what="the video callback")
        assert rig.video == [(b"rgb-bytes", 6, 4)]
        assert frame.convert_calls == []

        rig.audio_factory.stream.push(
            FakeAudioFrameEvent(FakeAudioFrame(b"pcm", sample_rate=SAMPLE_RATE, num_channels=1))
        )
        await until(lambda: bool(rig.audio), what="the audio callback")
        assert rig.audio == [(b"pcm", SAMPLE_RATE, 1)]
    finally:
        await rig.capture.aclose()


async def test_non_rgb24_frame_is_converted() -> None:
    """The belt-and-braces path: a frame that arrives in its native type is converted first."""
    rig = Rig(video_publication=video_publication())
    rig.capture.start()
    try:
        await until(lambda: bool(rig.video_factory.streams), what="the video consumer")
        converted = FakeVideoFrame(width=2, height=2, data=b"converted", type=RGB24)
        native = FakeVideoFrame(width=2, height=2, data=b"i420", type=I420, converts_to=converted)
        rig.video_factory.stream.push(FakeVideoFrameEvent(native))
        await until(lambda: bool(rig.video), what="the video callback")
        assert rig.video == [(b"converted", 2, 2)]
        assert native.convert_calls == [RGB24]
    finally:
        await rig.capture.aclose()


async def test_first_video_frame_is_logged_once(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The first video frame is logged once at INFO with its dimensions: the line an integrator looks for
    when nothing renders."""
    rig = Rig(video_publication=video_publication())
    rig.capture.start()
    try:
        await until(lambda: bool(rig.video_factory.streams), what="the video consumer")
        for _ in range(3):
            rig.video_factory.stream.push(
                FakeVideoFrameEvent(FakeVideoFrame(width=512, height=512, data=b"x"))
            )
        await until(lambda: len(rig.video) == 3, what="three video frames")
        first = [
            record
            for record in caplog_loguru.records
            if record.levelno == logging.INFO and "first avatar video frame" in record.getMessage()
        ]
        assert len(first) == 1
        assert "512x512" in first[0].getMessage()
    finally:
        await rig.capture.aclose()


# --------------------------------------------------------------------------- failure and teardown


async def test_stream_failure_ends_its_task_with_one_log_line(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A consumer that raises logs once and stops; the other track keeps flowing."""
    rig = Rig(video_publication=video_publication(), audio_publication=audio_publication())
    rig.capture.start()
    try:
        await until(
            lambda: bool(rig.video_factory.streams) and bool(rig.audio_factory.streams),
            what="both consumers",
        )
        rig.video_factory.stream.fail(RuntimeError("ffi exploded"))
        await until(
            lambda: (
                len(
                    [
                        record
                        for record in caplog_loguru.records
                        if record.levelno >= logging.ERROR and "video capture stopped" in record.getMessage()
                    ]
                )
                == 1
            ),
            what="the one failure line",
        )
        # The audio consumer is untouched.
        rig.audio_factory.stream.push(FakeAudioFrameEvent(FakeAudioFrame(b"pcm")))
        await until(lambda: bool(rig.audio), what="the audio callback")
    finally:
        await rig.capture.aclose()


async def test_aclose_cancels_tasks_before_closing_streams(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """`aclose()` cancels every consumer task before it closes any stream: a video consumer parked in
    `__anext__` is freed only by the cancel, and the close afterwards releases the FFI handle. A clean
    close warns about nothing."""
    rig = Rig(video_publication=video_publication(), audio_publication=audio_publication())
    rig.capture.start()
    await until(
        lambda: bool(rig.video_factory.streams) and bool(rig.audio_factory.streams),
        what="both consumers",
    )
    # Both consumers are now parked in `__anext__`, which is where a real one spends its life:
    # nothing has been cancelled or closed yet, so every entry `rig.order` grows from here is
    # teardown, in teardown order.
    assert rig.order == []

    await rig.capture.aclose()

    assert rig.video_factory.stream.closed and rig.audio_factory.stream.closed
    cancels = [index for index, entry in enumerate(rig.order) if entry.endswith(":cancelled")]
    closes = [index for index, entry in enumerate(rig.order) if entry.endswith(":closed")]
    assert len(cancels) == 2 and len(closes) == 2, rig.order
    assert max(cancels) < min(closes), f"a stream was closed before its task was cancelled: {rig.order}"
    assert "track_subscribed" not in rig.room.listeners
    assert [r.getMessage() for r in caplog_loguru.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize(("failing", "other"), [("video", "audio"), ("audio", "video")])
async def test_one_stream_failing_to_close_does_not_stop_the_other(
    caplog_loguru: pytest.LogCaptureFixture, failing: str, other: str
) -> None:
    """`aclose()` still returns, closes the other stream, and warns naming the failing kind."""
    rig = Rig(video_publication=video_publication(), audio_publication=audio_publication())
    rig.capture.start()
    await until(
        lambda: bool(rig.video_factory.streams) and bool(rig.audio_factory.streams),
        what="both consumers",
    )
    factories = {"video": rig.video_factory, "audio": rig.audio_factory}

    async def broken_close() -> None:
        raise RuntimeError("the FFI handle is already gone")

    factories[failing].stream.aclose = broken_close  # type: ignore[method-assign]

    await rig.capture.aclose()  # returns normally

    assert factories[other].stream.closed is True
    warnings = [r.getMessage() for r in caplog_loguru.records if r.levelno >= logging.WARNING]
    assert [message for message in warnings if failing in message], caplog_loguru.text


async def test_aclose_is_idempotent_and_safe_before_start() -> None:
    """`aclose()` is safe before `start()` and idempotent: it runs as a close hook on paths where
    `start()` never happened."""
    rig = Rig(video_publication=video_publication())
    await rig.capture.aclose()  # never started
    await rig.capture.aclose()

    other = Rig(video_publication=video_publication())
    other.capture.start()
    await until(lambda: bool(other.video_factory.streams), what="the video consumer")
    await other.capture.aclose()
    await other.capture.aclose()
    assert other.video_factory.stream.close_count == 1


async def test_start_after_aclose_does_nothing() -> None:
    """A late `track_subscribed` must not resurrect capture after the session closed."""
    rig = Rig(video_publication=video_publication())
    rig.capture.start()
    await until(lambda: bool(rig.video_factory.streams), what="the video consumer")
    await rig.capture.aclose()

    rig.capture.start()
    # The listener is gone, so the emit reaches nobody: no wait is needed, and none would help.
    assert "track_subscribed" not in rig.room.listeners
    rig.subscribe("TR_video2", KIND_VIDEO)
    assert len(rig.video_factory.calls) == 1


async def test_aclose_cancelled_from_outside_propagates_and_closes_nothing() -> None:
    """Cancelling `aclose()` from outside propagates and closes nothing: the close hook running out of
    time must not be mistaken for a consumer task ending.

    `aclose()` cancels its tasks and then waits for them, so a `CancelledError` during that wait means
    either that a cancelled task ended or that `aclose()` itself was cancelled. Conflating the two would
    swallow the caller's cancellation and close both streams under a live consumer.

    The slow-to-die consumer is an input: its callback holds the cancel for `LINGER`, which keeps
    `aclose()` inside the wait long enough for the outer cancel to land.
    """
    linger = asyncio.Event()

    async def slow_to_die(data: bytes, width: int, height: int) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            linger.set()
            await asyncio.sleep(LINGER)  # models a callback that does not die on the first cancel
            raise

    rig = Rig(
        video_publication=video_publication(),
        audio_publication=audio_publication(),
        on_video=slow_to_die,
    )
    rig.capture.start()
    await until(lambda: bool(rig.video_factory.streams), what="the video consumer")
    rig.video_factory.stream.push(FakeVideoFrameEvent(FakeVideoFrame()))
    await until(lambda: bool(rig.audio_factory.streams), what="the audio consumer")

    closing = asyncio.ensure_future(rig.capture.aclose())
    await until(linger.is_set, what="aclose() to reach the wait")
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert not rig.video_factory.stream.closed, "a stream was closed under a live consumer"
    assert not rig.audio_factory.stream.closed
