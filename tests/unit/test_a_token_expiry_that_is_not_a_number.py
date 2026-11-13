"""A LiveKit token whose `exp` claim is not a number is reported as an invalid token, never as a `TypeError`.

A provided avatar token, or a direct join's bot token, comes from the developer's own minting code, so its
claims are untrusted input. A non-numeric `exp` must be refused like any other malformed token: by
`ProvidedAvatarToken` as a `SynthesiaError` of type `INVALID_ROOM_TOKEN`, and by `LiveKitTransportRoom`
either at construction, as a misuse error (`ValueError`), or at launch, as `INVALID_ROOM_TOKEN` before any
session request. Both places are correct: construction is where *misuse vs session errors* puts every
check that does not depend on the time, and the launch is where the expiry itself is checked. What this
file pins is that it is never a `TypeError` from comparing the claim with the clock, which reaches the
developer as an `UNKNOWN` launch failure.

The tokens are signed by `pyjwt` directly, because `livekit.api.AccessToken` always writes a numeric `exp`.
"""

from __future__ import annotations

import time
from typing import Any, cast

import jwt
import pytest
from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._types import LiveKitCredentials, ProvidedAvatarToken
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from pipecat_synthesia.protocol import TOKEN_ATTR_PUBLISH_ON_BEHALF
from tests.fakes.fake_rtc import FakeLocalParticipant, FakeRoom
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.fakes.fake_transport import FAKE_AGENT_TOKEN, FakeLiveKitTransport
from tests.harness import bidirectional_pipeline, make_service
from tests.support.callbacks import make_launch_context
from tests.support.secrets import LK_API_SECRET

BOT = "pipecat-bot"  # FAKE_AGENT_TOKEN's identity
ROOM = "fake-room"  # and its room
AVATAR = DEFAULT_AVATAR_IDENTITY
CREDENTIALS = LiveKitCredentials(api_key="fake-lk-key", api_secret=LK_API_SECRET)

NON_NUMERIC_EXPIRIES = pytest.mark.parametrize(
    "exp",
    [
        pytest.param(lambda: "soon", id="word"),
        pytest.param(lambda: str(int(time.time()) + 3600), id="digits-in-a-string"),
        pytest.param(lambda: [int(time.time()) + 3600], id="list"),
        pytest.param(lambda: {"seconds": 3600}, id="object"),
    ],
)


def signed(claims: dict[str, Any]) -> str:
    return jwt.encode({"iss": CREDENTIALS.api_key, **claims}, LK_API_SECRET, algorithm="HS256")


def avatar_token(exp: object) -> str:
    """An avatar token that passes every construction-time check, with `exp` as given."""
    return signed(
        {
            "sub": AVATAR,
            "kind": "agent",
            "video": {"roomJoin": True, "room": ROOM},
            "attributes": {TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT},
            "exp": exp,
        }
    )


def bot_token(exp: object) -> str:
    """An agent-kind bot token naming `BOT` in `ROOM`, which the constructor accepts, with `exp` as given."""
    return signed(
        {"sub": BOT, "kind": "agent", "video": {"roomJoin": True, "room": ROOM, "agent": True}, "exp": exp}
    )


def build_room(which: str, exp: object) -> LiveKitTransportRoom | None:
    """The direct join with the malformed token, or None if its constructor refuses it (a misuse error)."""
    room = FakeRoom(ROOM, local_participant=FakeLocalParticipant(BOT))
    try:
        if which == "bot-token":
            transport = FakeLiveKitTransport(room=room, token=bot_token(exp), participant_id="")
            return LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS)
        transport = FakeLiveKitTransport(room=room, token=FAKE_AGENT_TOKEN, participant_id="")
        return LiveKitTransportRoom(cast(Any, transport), avatar_token=avatar_token(exp))
    except ValueError:
        return None


WHICH_TOKEN = pytest.mark.parametrize("which", ["bot-token", "avatar-token"])


@NON_NUMERIC_EXPIRIES
def test_a_provided_avatar_token_with_a_non_numeric_expiry_is_an_invalid_room_token(exp) -> None:
    with pytest.raises(SynthesiaError) as excinfo:
        source = ProvidedAvatarToken(avatar_token(exp()), bot_identity=BOT, room_name=ROOM)
        source.resolve(
            bot_identity=BOT,
            room_name=ROOM,
            avatar_identity=AVATAR,
            avatar_name=DEFAULT_AVATAR_NAME,
            client=build_client_info(),
        )

    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN


@WHICH_TOKEN
@NON_NUMERIC_EXPIRIES
async def test_a_direct_join_refuses_a_token_with_a_non_numeric_expiry_before_any_request(which, exp) -> None:
    adapter = build_room(which, exp())
    if adapter is None:
        return  # refused at construction, as a misuse error
    api = FakeSessionApiClient()

    with pytest.raises(SynthesiaError) as excinfo:
        await adapter.launch(make_launch_context(api, avatar_ids=("av_a",)))

    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN
    assert api.calls == [], "a session was requested with an unusable token"


@WHICH_TOKEN
async def test_the_developer_is_told_invalid_room_token_for_a_non_numeric_expiry(which: str) -> None:
    """With `required=False` the bot continues voice-only, and the error it reports says why. A token
    refused at construction never reaches the service: the developer sees the `ValueError` at startup."""
    room = build_room(which, "soon")
    if room is None:
        return  # refused at construction, as a misuse error
    service = make_service(room, required=False)

    async with bidirectional_pipeline(service) as h:
        error = await h.frame(ErrorFrame, direction=FrameDirection.UPSTREAM)
        await h.finish()

    assert isinstance(error.exception, SynthesiaError), repr(error.exception)
    assert error.exception.type is ErrorType.INVALID_ROOM_TOKEN, f"{error.exception.type}: {error.exception}"
