"""The `LiveKitAvatarSession` rig on the unit fakes: one `FakeRoom` with a `FakeLocalParticipant`, a
`RoomHandle` over it, a `FakeSessionApiClient` and a `CallbackRecorder`, everything `launch()` takes.
`SessionRigRoom` puts a session launched on the rig behind the service, for the service tests that need
the real session.

Not a test module (no `test_` prefix). `LiveKitAvatarSession` is internal, but it is the unit seam for
the avatar session: both room adapters only build its `RoomHandle` and hand over. Every helper that moves
the fake room does it in rtc's order: a participant is added before `participant_connected` is emitted,
and removed (with its reason set) before `participant_disconnected` is.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncIterator, Mapping
from time import monotonic
from typing import Any, cast

from livekit import rtc as lk_rtc

from pipecat_synthesia.livekit._types import AvatarTokenSource, RoomHandle
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME
from pipecat_synthesia.protocol import ATTR_AVATAR_READY, ATTR_AVATAR_STATUS
from pipecat_synthesia.room import AvatarRoom, AvatarSession, JoinMode, LaunchContext
from tests.fakes.fake_rtc import (
    FakeLocalParticipant,
    FakeRemoteParticipant,
    FakeRoom,
    FakeTrackPublication,
)
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.support.callbacks import CallbackRecorder, make_launch_context

AVATAR = DEFAULT_AVATAR_IDENTITY
BOT = "bot"
ROOM = "fake-room"
URL = "wss://livekit.example"
AVATAR_IDS = ("av_a", "av_b")
SESSION_ID = "ses_fake"  # FakeSessionApiClient's default
VIDEO_SID = "TR_video"
AUDIO_SID = "TR_audio"
STANDARD = int(lk_rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)
AGENT = int(lk_rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
KIND_VIDEO = int(lk_rtc.TrackKind.KIND_VIDEO)
KIND_AUDIO = int(lk_rtc.TrackKind.KIND_AUDIO)

#: The replacement grace, shrunk only in the tests that are about it elapsing; 10x is the
#: "and nothing happened" wait.
GRACE = 0.05
GRACE_MARGIN = 10 * GRACE


def status_attr(**states: str) -> str:
    """`avatar_status` as the worker writes it: unprefixed avatar ids, compact JSON."""
    return json.dumps({raw: {"state": state} for raw, state in states.items()}, separators=(",", ":"))


PREPARING = {ATTR_AVATAR_STATUS: status_attr(a="preparing", b="preparing")}
ALL_READY = {ATTR_AVATAR_STATUS: status_attr(a="ready", b="ready"), ATTR_AVATAR_READY: "true"}
A_READY_B_PREPARING = {ATTR_AVATAR_STATUS: status_attr(a="ready", b="preparing")}


class StubTokenSource(AvatarTokenSource):
    """Returns a fixed string and records what it was asked for."""

    def __init__(self, token: str = "stub-avatar-token") -> None:
        self.token = token
        self.calls: list[dict[str, Any]] = []

    def resolve(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.token


def video_publication() -> FakeTrackPublication:
    return FakeTrackPublication(VIDEO_SID, kind=KIND_VIDEO, name="avatar_video")


def make_avatar(
    *,
    attributes: Mapping[str, str] | None = None,
    video: bool = True,
    audio: bool = True,
    disconnect_reason: Any = None,
) -> FakeRemoteParticipant:
    """An avatar participant; `video=False, audio=False` is one rtc has announced before its tracks."""
    publications: dict[str, Any] = {}
    if video:
        publications[VIDEO_SID] = video_publication()
    if audio:
        publications[AUDIO_SID] = FakeTrackPublication(AUDIO_SID, kind=KIND_AUDIO, name="avatar_audio")
    return FakeRemoteParticipant(
        AVATAR,
        kind=AGENT,
        attributes=attributes,
        track_publications=publications,
        disconnect_reason=disconnect_reason,
    )


class Rig:
    """One fake room, its handle, a fake API and a callback recorder."""

    def __init__(
        self,
        *,
        owns_room: bool = False,
        avatar_present: bool = True,
        attributes: Mapping[str, str] | None = None,
        launch_timeout: float = 5.0,
        api: FakeSessionApiClient | None = None,
        participant: FakeLocalParticipant | None = None,
        recorder: CallbackRecorder | None = None,
        token_source: AvatarTokenSource | None = None,
        avatar_name: str = DEFAULT_AVATAR_NAME,
    ) -> None:
        self.participant = participant or FakeLocalParticipant(BOT)
        self.room = FakeRoom(ROOM, local_participant=self.participant)
        self.delete_calls = 0

        async def delete_room() -> None:
            self.delete_calls += 1

        self.handle = RoomHandle(
            room=cast(Any, self.room),
            bot_identity=BOT,
            livekit_url=URL,
            room_name=ROOM,
            owns_room=owns_room,
            release_room=delete_room if owns_room else None,
            connected_at=monotonic(),
        )
        self.api = api or FakeSessionApiClient()
        self.rec = recorder or CallbackRecorder()
        self.ctx = make_launch_context(
            self.api, avatar_ids=AVATAR_IDS, launch_timeout=launch_timeout, recorder=self.rec
        )
        self.tokens = token_source or StubTokenSource()
        self.avatar_name = avatar_name
        self.avatar: FakeRemoteParticipant | None = None
        if avatar_present:
            self.avatar = self.room.add_participant(make_avatar(attributes=attributes))

    async def launch(self) -> LiveKitAvatarSession:
        return await LiveKitAvatarSession.launch(
            self.handle,
            token_source=self.tokens,
            avatar_identity=AVATAR,
            avatar_name=self.avatar_name,
            ctx=self.ctx,
        )

    def launch_task(self) -> asyncio.Task[LiveKitAvatarSession]:
        return asyncio.ensure_future(self.launch())

    def join(self, participant: FakeRemoteParticipant | None = None) -> FakeRemoteParticipant:
        participant = participant or make_avatar()
        self.room.add_participant(participant)
        self.room.emit("participant_connected", participant)
        return participant

    def leave(self, participant: FakeRemoteParticipant, *, reason: Any = None) -> None:
        """Only the `participant_disconnected` half of a departure; `depart` is what a server sends."""
        self.room.remove_participant(participant.identity)
        participant.set_disconnect_reason(reason)
        self.room.emit("participant_disconnected", participant)

    def depart(self, participant: FakeRemoteParticipant, *, reason: Any = None) -> None:
        """A departure as livekit-server reports it to another participant (measured on 1.9.1):
        `track_unpublished` for each track, then `participant_disconnected`. A fake must not be kinder
        than what it stands in for."""
        for sid in (VIDEO_SID, AUDIO_SID):
            if sid in participant.track_publications:
                self.unpublish(participant, sid)
        self.leave(participant, reason=reason)

    def unpublish(self, participant: FakeRemoteParticipant, sid: str) -> None:
        publication = participant.unpublish(sid)
        self.room.emit("track_unpublished", publication, participant)

    def publish_video(self, participant: FakeRemoteParticipant) -> None:
        publication = video_publication()
        participant.publish(VIDEO_SID, publication)
        self.room.emit("track_published", publication, participant)

    def change_attributes(self, participant: FakeRemoteParticipant, attributes: Mapping[str, str]) -> None:
        participant.set_attributes(attributes)
        self.room.emit("participant_attributes_changed", dict(attributes), participant)

    def assert_torn_down(self) -> None:
        assert self.participant.rpc_methods == {}, "playback handlers still registered"
        assert self.room.listeners == {}, f"listeners still installed: {self.room.listeners}"


class SessionRigRoom(AvatarRoom):
    """A direct join whose `launch()` is a real `LiveKitAvatarSession` on the session rig.

    No public route puts a real session on the fake room, because both room adapters build their own
    `RoomHandle`. This one launches it with the service's own launch context, its API swapped for the rig's
    fake.
    """

    join_mode = JoinMode.DIRECT

    def __init__(self, rig: Rig) -> None:
        self.rig = rig

    @property
    def avatar_identity(self) -> str:
        return AVATAR

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        rig = self.rig
        return await LiveKitAvatarSession.launch(
            rig.handle,
            token_source=rig.tokens,
            avatar_identity=AVATAR,
            avatar_name=rig.avatar_name,
            # the service's callbacks, the rig's fake Synthesia API
            ctx=dataclasses.replace(ctx, api=rig.api),
        )


@contextlib.asynccontextmanager
async def launched(**kwargs: Any) -> AsyncIterator[tuple[Rig, LiveKitAvatarSession]]:
    """A launched session that is closed however the test ends, so nothing is left running."""
    rig = Rig(**kwargs)
    session = await rig.launch()
    try:
        yield rig, session
    finally:
        await session.aclose()


async def settle(task: asyncio.Future[Any]) -> None:
    """Cancel `task` if a failed assertion left it running, and wait for it, so nothing outlives the test."""
    if not task.done():
        task.cancel()
    await asyncio.wait({task})
    if not task.cancelled():
        task.exception()  # retrieved, so a failure the test already reported is not logged again at GC
