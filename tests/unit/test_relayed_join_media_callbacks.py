"""A relayed join launched without the media callbacks fails with a `SynthesiaError`, having created nothing.

`PrivateLiveKitRoom` is public, and its `launch()` takes a `LaunchContext` whose `on_video_frame` and
`on_audio_frame` the service sets only for a room whose join mode is relayed. A context built by hand, or by
the service for a room that reports the direct join mode, carries neither. What this file pins: such a
launch raises a `SynthesiaError` of type `UNKNOWN` with the `APPLICATION` category, rather than an
`AssertionError` that `python -O` would strip, and it does so before anything exists: no LiveKit API client,
no private room, no bot connect and no Synthesia session request, so no billed session and nothing to roll
back.

No service is in the loop (only the service could build a `TaskManager` for `assert_no_dangling`), so
"nothing created" is read from the fakes' recorders instead.
"""

from __future__ import annotations

from typing import Any

import pytest
from pipecat.utils.errors import ErrorCategory

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit.private_room import PrivateLiveKitRoom
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY
from tests.fakes.fake_lk_api import LiveKitApiFactory, RoomFactory
from tests.fakes.fake_rtc import FakeRemoteParticipant
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.support.callbacks import make_launch_context
from tests.support.secrets import LK_API_SECRET

ROOM = "private-room"
#: The LiveKit API key id: an identifier, not the secret.
LIVEKIT_ID = "APIlivekit"


@pytest.mark.unit
async def test_a_relayed_join_without_media_callbacks_fails_before_creating_anything() -> None:
    rooms, apis = RoomFactory(), LiveKitApiFactory()
    room = PrivateLiveKitRoom(
        url="wss://livekit.example",
        api_key=LIVEKIT_ID,
        api_secret=LK_API_SECRET,
        room_name=ROOM,
        room_factory=rooms,
        api_factory=apis,
    )
    session_api = FakeSessionApiClient()

    # Lets a launch that wrongly gets as far as the session request complete it, so such a launch fails
    # on the recorders below rather than on the join wait.
    async def avatar_joins(*_args: Any) -> None:
        lk_room = rooms.room
        lk_room.emit(
            "participant_connected",
            lk_room.add_participant(FakeRemoteParticipant(DEFAULT_AVATAR_IDENTITY)),
        )

    session_api.on_create = avatar_joins
    ctx = make_launch_context(session_api)  # the direct join's callbacks: no media callbacks
    assert ctx.callbacks.on_video_frame is None and ctx.callbacks.on_audio_frame is None

    with pytest.raises(SynthesiaError) as caught:
        await room.launch(ctx)

    assert caught.value.type is ErrorType.UNKNOWN
    assert caught.value.category is ErrorCategory.APPLICATION
    assert apis.instances == [], "a LiveKit API client was built (it owns an aiohttp session)"
    assert rooms.rooms == [], "a room object was built, so the bot was about to connect"
    assert session_api.calls == [], "a Synthesia session was requested: a billed session for nothing"
