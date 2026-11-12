"""Track capture of a track that ends, or is closed, before its consumer has run: no stream outlives aclose().

A consumer task opens its stream on its first step, which runs on a later loop turn than the room
event that created it. rtc 1.1.19's `Room._listen_task` handles every FFI event already queued without
yielding to the loop in between (`asyncio.Queue.get()` returns at once when an item is waiting, and the
room queue's `join()` waits only while a `publish_track` or `unpublish_track` is in flight, which the bot
never does in the private room). So a `track_subscribed` and the end of that same track, its
`track_unsubscribed` or its participant's `participant_disconnected`, can both be delivered before the
new consumer has run, for example when an avatar participant is replaced or leaves just as its track is
subscribed. Each case here delivers the two events with no await between them, as rtc does then.

What is pinned is the capture's own guarantee: every stream it opened is closed by the time `aclose()`
returns, and the next track of that kind is still captured. The second test is the same guarantee when
`aclose()` itself runs before a just-created consumer.

`AvatarTrackCapture` is driven directly, on its two stream-factory seams, because `PrivateLiveKitRoom`
builds it on the real `rtc` streams.
"""

from __future__ import annotations

import pytest
from livekit import rtc as lk_rtc
from loguru import logger

from pipecat_synthesia.livekit.track_capture import AvatarTrackCapture
from tests.fakes.fake_media import (
    FakeAudioFrame,
    FakeAudioFrameEvent,
    FakeMediaStream,
    FakeSubscribedPublication,
    FakeVideoFrame,
    FakeVideoFrameEvent,
    StreamFactory,
)
from tests.fakes.fake_rtc import FakeRemoteParticipant, FakeRoom
from tests.support.waiting import until

AVATAR = "synthesia-avatar-agent"
SAMPLE_RATE = 24000
KIND = {"video": int(lk_rtc.TrackKind.KIND_VIDEO), "audio": int(lk_rtc.TrackKind.KIND_AUDIO)}


class Rig:
    """A private room holding an avatar participant with no tracks yet, and a capture on it."""

    def __init__(self) -> None:
        self.factories = {"video": StreamFactory("video"), "audio": StreamFactory("audio")}
        self.received: dict[str, list[bytes]] = {"video": [], "audio": []}
        self.room = FakeRoom("private-room")
        self.avatar = self.room.add_participant(FakeRemoteParticipant(AVATAR))
        self.capture = AvatarTrackCapture(
            self.room,  # type: ignore[arg-type] -- the fake satisfies RoomLike (fake_rtc's guard)
            avatar_identity=AVATAR,
            audio_sample_rate=SAMPLE_RATE,
            audio_channels=1,
            on_video_frame=self._on_video,
            on_audio_frame=self._on_audio,
            log=logger.bind(component="test"),
            video_stream_factory=self.factories["video"],
            audio_stream_factory=self.factories["audio"],
        )

    async def _on_video(self, data: bytes, width: int, height: int) -> None:
        self.received["video"].append(data)

    async def _on_audio(self, data: bytes, sample_rate: int, channels: int) -> None:
        self.received["audio"].append(data)

    def subscribe(self, kind: str, sid: str, participant: FakeRemoteParticipant) -> FakeSubscribedPublication:
        publication = FakeSubscribedPublication(sid, kind=KIND[kind])
        participant.publish(publication.sid, publication)
        self.room.emit("track_subscribed", publication.track, publication, participant)
        return publication

    def unsubscribe(self, publication: FakeSubscribedPublication) -> None:
        # rtc clears the publication's track, then emits `track_unsubscribed` with the old track.
        track, publication.track = publication.track, None  # type: ignore[assignment] -- as rtc does
        publication.subscribed = False
        self.room.emit("track_unsubscribed", track, publication, self.avatar)

    def disconnect_avatar(self) -> FakeRemoteParticipant:
        """The avatar participant leaves; rtc removes it before the emit. Returns the replacement."""
        self.room.remove_participant(AVATAR)
        self.avatar.set_disconnect_reason(None)
        self.room.emit("participant_disconnected", self.avatar)
        self.avatar = self.room.add_participant(FakeRemoteParticipant(AVATAR))
        self.room.emit("participant_connected", self.avatar)
        return self.avatar

    def opened(self) -> list[FakeMediaStream]:
        return [stream for factory in self.factories.values() for stream in factory.streams]

    async def frame_reaches_the_callback(self, kind: str, track: object, data: bytes) -> None:
        stream = next((s for s in self.factories[kind].streams if s.track is track), None)
        assert stream is not None, f"no {kind} stream was opened on {track!r}"
        if kind == "video":
            stream.push(FakeVideoFrameEvent(FakeVideoFrame(width=2, height=2, data=data)))
        else:
            stream.push(FakeAudioFrameEvent(FakeAudioFrame(data, sample_rate=SAMPLE_RATE, num_channels=1)))
        await until(lambda: data in self.received[kind], what=f"{data!r} at the {kind} callback")


@pytest.mark.parametrize("ending", ["track_unsubscribed", "participant_disconnected"])
@pytest.mark.parametrize("kind", ["video", "audio"])
async def test_a_track_that_ends_in_the_turn_it_is_subscribed_leaves_no_stream_open(
    kind: str, ending: str
) -> None:
    """The track is subscribed and ends before its consumer runs; the next track of that kind is still
    captured, and every stream the capture opened is closed once `aclose()` returns."""
    rig = Rig()
    rig.capture.start()
    try:
        # Two FFI events queued together: rtc's listen task delivers both with no await in between.
        first = rig.subscribe(kind, f"TR_{kind}_first", rig.avatar)
        if ending == "track_unsubscribed":
            rig.unsubscribe(first)
            owner = rig.avatar
        else:
            owner = rig.disconnect_avatar()

        following = rig.subscribe(kind, f"TR_{kind}_following", owner)
        await until(
            lambda: any(s.track is following.track for s in rig.factories[kind].streams),
            what=f"a consumer on the following {kind} track",
        )
        await rig.frame_reaches_the_callback(kind, following.track, b"after")
    finally:
        await rig.capture.aclose()

    unclosed = [stream.track for stream in rig.opened() if not stream.closed]
    assert unclosed == [], f"a stream outlived aclose(), on {unclosed}"


@pytest.mark.parametrize("kind", ["video", "audio"])
async def test_a_close_in_the_turn_a_track_is_subscribed_leaves_no_stream_open(kind: str) -> None:
    """`aclose()` runs before the consumer a `track_subscribed` just created; whatever that consumer
    opened is closed once `aclose()` returns."""
    rig = Rig()
    rig.capture.start()
    try:
        rig.subscribe(kind, f"TR_{kind}_first", rig.avatar)
    finally:
        await rig.capture.aclose()

    unclosed = [stream.track for stream in rig.opened() if not stream.closed]
    assert unclosed == [], f"a stream outlived aclose(), on {unclosed}"
