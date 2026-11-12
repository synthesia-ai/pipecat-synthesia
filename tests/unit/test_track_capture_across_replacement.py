"""Track capture across a same-identity replacement: the replacement's tracks are the ones captured.

A same-identity replacement (a launch retried with the same avatar token, whose avatar worker replaces
the one serving the session) leaves the session active: the *replacement grace* treats the old avatar
participant's disconnect and the new one's join as one avatar, and the session carries on. What this file
pins is the relayed join's half of that: once the replacement's tracks are subscribed, its video and its
audio reach the frame callbacks. That must hold whichever way the old consumer ended up: a video consumer
stays parked, because `rtc.VideoStream` never wakes it at end of stream, while an audio consumer exits,
because `rtc.AudioStream` does (see `tests/fakes/fake_media.py`). Every stream the capture opened, the
replaced ones included, is closed by `aclose()`.

The room events are the ones rtc delivers to the bot for a replacement, in rtc's order: the old tracks
unsubscribed and unpublished, the old participant removed and then reported disconnected with no reason,
a new participant object with the same identity, and its tracks published and subscribed.

`AvatarTrackCapture` is driven directly, on its two stream-factory seams. No public route reaches the
capture with stream fakes, because `PrivateLiveKitRoom` builds it on the real `rtc` streams. The
public-route version of the same claim, against a real livekit-server and two fake avatar workers, is
`tests/integration/test_relayed_join_media_across_replacement.py`.
"""

from __future__ import annotations

from typing import Any

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
    """A private room holding the avatar participant, and a capture wired to both stream factories."""

    def __init__(self) -> None:
        self.factories = {"video": StreamFactory("video"), "audio": StreamFactory("audio")}
        #: What each frame callback received, by kind: the payload bytes only.
        self.received: dict[str, list[bytes]] = {"video": [], "audio": []}
        self.room = FakeRoom("private-room")
        self.publications = {
            "video": FakeSubscribedPublication("TR_video_old", kind=KIND["video"]),
            "audio": FakeSubscribedPublication("TR_audio_old", kind=KIND["audio"]),
        }
        self.avatar = self.room.add_participant(
            FakeRemoteParticipant(
                AVATAR, track_publications={pub.sid: pub for pub in self.publications.values()}
            )
        )
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

    def stream_on(self, kind: str, track: Any) -> FakeMediaStream | None:
        """The stream the capture opened on this track, if it opened one."""
        return next((s for s in self.factories[kind].streams if s.track is track), None)

    def replace_avatar(self) -> dict[str, FakeSubscribedPublication]:
        """Replace the avatar participant as rtc reports it to the bot; return the new publications."""
        old = self.avatar
        for kind, publication in self.publications.items():
            # rtc clears the publication's track, then emits `track_unsubscribed` with the old track.
            track, publication.track = publication.track, None  # type: ignore[assignment] -- as rtc does
            publication.subscribed = False
            self.room.emit("track_unsubscribed", track, publication, old)
            if kind == "audio":
                # `rtc.AudioStream` delivers end of stream; `rtc.VideoStream` never does.
                stream = self.stream_on(kind, track)
                assert stream is not None
                stream.end()
        for publication in self.publications.values():
            old.unpublish(publication.sid)
            self.room.emit("track_unpublished", publication, old)
        # rtc removes the participant before it reports the disconnect; a replacement states no reason.
        self.room.remove_participant(AVATAR)
        old.set_disconnect_reason(None)
        self.room.emit("participant_disconnected", old)

        # rtc builds a new participant object on every join, with the same identity.
        new = self.room.add_participant(FakeRemoteParticipant(AVATAR))
        self.room.emit("participant_connected", new)
        replacements = {
            "video": FakeSubscribedPublication("TR_video_new", kind=KIND["video"]),
            "audio": FakeSubscribedPublication("TR_audio_new", kind=KIND["audio"]),
        }
        for publication in replacements.values():
            new.publish(publication.sid, publication)
            self.room.emit("track_published", publication, new)
        for publication in replacements.values():
            self.room.emit("track_subscribed", publication.track, publication, new)
        self.avatar = new
        return replacements


def push(stream: FakeMediaStream, kind: str, data: bytes) -> None:
    if kind == "video":
        stream.push(FakeVideoFrameEvent(FakeVideoFrame(width=2, height=2, data=data)))
    else:
        stream.push(FakeAudioFrameEvent(FakeAudioFrame(data, sample_rate=SAMPLE_RATE, num_channels=1)))


@pytest.mark.parametrize("kind", ["video", "audio"])
async def test_the_replacements_tracks_reach_the_callbacks(kind: str) -> None:
    """After a same-identity replacement, the new avatar participant's frames reach the callbacks.

    The old participant's frames arrive first, so the capture is known to be working before the
    replacement. Video is the case where the old consumer is still parked; audio the one where it exited.
    """
    rig = Rig()
    rig.capture.start()
    try:
        await until(
            lambda: all(rig.stream_on(k, rig.publications[k].track) for k in ("video", "audio")),
            what="both consumers on the first avatar participant's tracks",
        )
        old_stream = rig.stream_on(kind, rig.publications[kind].track)
        assert old_stream is not None
        push(old_stream, kind, b"before")
        await until(lambda: b"before" in rig.received[kind], what=f"the first participant's {kind}")

        replacements = rig.replace_avatar()

        try:
            await until(
                lambda: rig.stream_on(kind, replacements[kind].track) is not None,
                what=f"a consumer on the replacement's {kind} track",
            )
        except AssertionError as exc:
            opened = [stream.track for stream in rig.factories[kind].streams]
            raise AssertionError(f"{exc} ({kind} streams were opened only on {opened})") from None
        new_stream = rig.stream_on(kind, replacements[kind].track)
        assert new_stream is not None
        push(new_stream, kind, b"after")
        await until(lambda: b"after" in rig.received[kind], what=f"the replacement's {kind}")
    finally:
        await rig.capture.aclose()

    opened = [stream for factory in rig.factories.values() for stream in factory.streams]
    assert [stream for stream in opened if not stream.closed] == [], "a stream outlived aclose()"
