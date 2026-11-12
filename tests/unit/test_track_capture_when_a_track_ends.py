"""Track capture when a captured track ends: which events move capture on, and which change nothing.

`test_track_capture_across_replacement.py` pins the full same-identity replacement, where rtc reports the
old tracks unsubscribed and then the old avatar participant disconnected. This file pins the two edges
of that, each of which the full sequence would hide:
- rtc 1.1.19 skips `track_unsubscribed` when a racing disconnect has already removed the participant, so
  the disconnect alone must move capture on to the next avatar participant's tracks;
- the unsubscribe alone, with no disconnect yet, must move capture on too;
- an ignored extra track ending must leave the captured track's consumer running.

`AvatarTrackCapture` is driven directly, on its two stream-factory seams, because `PrivateLiveKitRoom`
builds it on the real `rtc` streams.
"""

from __future__ import annotations

from livekit import rtc as lk_rtc
from loguru import logger

from pipecat_synthesia.livekit.track_capture import AvatarTrackCapture
from tests.fakes.fake_media import (
    FakeSubscribedPublication,
    FakeVideoFrame,
    FakeVideoFrameEvent,
    StreamFactory,
)
from tests.fakes.fake_rtc import FakeRemoteParticipant, FakeRoom
from tests.support.waiting import until

AVATAR = "synthesia-avatar-agent"
KIND_VIDEO = int(lk_rtc.TrackKind.KIND_VIDEO)


class Rig:
    """A private room holding the avatar participant with one video track, and a capture on it."""

    def __init__(self) -> None:
        self.factory = StreamFactory("video")
        self.received: list[bytes] = []
        self.room = FakeRoom("private-room")
        self.publication = FakeSubscribedPublication("TR_video_old", kind=KIND_VIDEO)
        self.avatar = self.room.add_participant(
            FakeRemoteParticipant(AVATAR, track_publications={self.publication.sid: self.publication})
        )
        self.capture = AvatarTrackCapture(
            self.room,  # type: ignore[arg-type] -- the fake satisfies RoomLike (fake_rtc's guard)
            avatar_identity=AVATAR,
            audio_sample_rate=24000,
            audio_channels=1,
            on_video_frame=self._on_video,
            on_audio_frame=self._on_audio,
            log=logger.bind(component="test"),
            video_stream_factory=self.factory,
            audio_stream_factory=StreamFactory("audio"),
        )

    async def _on_video(self, data: bytes, width: int, height: int) -> None:
        self.received.append(data)

    async def _on_audio(self, data: bytes, sample_rate: int, channels: int) -> None:
        pass

    async def frame_reaches_the_callback(self, track: object, data: bytes) -> None:
        stream = next((s for s in self.factory.streams if s.track is track), None)
        assert stream is not None, f"no stream was opened on {track!r}"
        stream.push(FakeVideoFrameEvent(FakeVideoFrame(width=2, height=2, data=data)))
        await until(lambda: data in self.received, what=f"{data!r} at the video callback")


async def test_a_disconnect_alone_moves_capture_to_the_next_avatar_participants_track() -> None:
    """With no `track_unsubscribed` (rtc skipped it), the disconnect retires the old consumer, the
    replacement's track is captured, and `aclose()` closes both streams."""
    rig = Rig()
    rig.capture.start()
    try:
        old_track = rig.publication.track
        await until(lambda: bool(rig.factory.streams), what="the first consumer")
        await rig.frame_reaches_the_callback(old_track, b"before")

        # rtc removes the participant before it reports the disconnect; a replacement states no reason.
        rig.room.remove_participant(AVATAR)
        rig.avatar.set_disconnect_reason(None)
        rig.room.emit("participant_disconnected", rig.avatar)
        new = rig.room.add_participant(FakeRemoteParticipant(AVATAR))
        rig.room.emit("participant_connected", new)
        replacement = FakeSubscribedPublication("TR_video_new", kind=KIND_VIDEO)
        new.publish(replacement.sid, replacement)
        rig.room.emit("track_subscribed", replacement.track, replacement, new)

        await until(lambda: len(rig.factory.streams) == 2, what="a consumer on the replacement's track")
        await rig.frame_reaches_the_callback(replacement.track, b"after")
    finally:
        await rig.capture.aclose()

    assert [stream for stream in rig.factory.streams if not stream.closed] == [], "a stream outlived aclose()"


async def test_an_unsubscribe_alone_moves_capture_to_the_next_track() -> None:
    """The captured track's unsubscribe retires its consumer before any disconnect, so the next track of
    that kind is captured even while the avatar participant is still in the room."""
    rig = Rig()
    rig.capture.start()
    try:
        old_track = rig.publication.track
        await until(lambda: bool(rig.factory.streams), what="the first consumer")
        await rig.frame_reaches_the_callback(old_track, b"before")

        rig.publication.track = None  # type: ignore[assignment] -- rtc clears it before the emit
        rig.publication.subscribed = False
        rig.room.emit("track_unsubscribed", old_track, rig.publication, rig.avatar)
        following = FakeSubscribedPublication("TR_video_next", kind=KIND_VIDEO)
        rig.avatar.publish(following.sid, following)
        rig.room.emit("track_subscribed", following.track, following, rig.avatar)

        await until(lambda: len(rig.factory.streams) == 2, what="a consumer on the next track")
        await rig.frame_reaches_the_callback(following.track, b"after")
    finally:
        await rig.capture.aclose()

    assert [stream for stream in rig.factory.streams if not stream.closed] == [], "a stream outlived aclose()"


async def test_an_ignored_extra_track_ending_leaves_the_captured_track_flowing() -> None:
    """A second video track is ignored while the first is captured; its unsubscribe retires nothing."""
    rig = Rig()
    rig.capture.start()
    try:
        captured = rig.publication.track
        await until(lambda: bool(rig.factory.streams), what="the first consumer")
        extra = FakeSubscribedPublication("TR_video_extra", kind=KIND_VIDEO)
        rig.avatar.publish(extra.sid, extra)
        rig.room.emit("track_subscribed", extra.track, extra, rig.avatar)
        rig.room.emit("track_unsubscribed", extra.track, extra, rig.avatar)

        await rig.frame_reaches_the_callback(captured, b"still flowing")
        assert len(rig.factory.calls) == 1, "the extra track was captured"
        assert not rig.factory.stream.closed
    finally:
        await rig.capture.aclose()
