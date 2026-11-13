"""`LiveKitTransportRoom`: the direct join, on a real `LiveKitTransport` subclass.

What these protect: the construction-time checks (*misuse vs session errors*), including the
*agent-kind requirement*; the *borrowed-room rule*; and the *self-hearing guard*: the bot neither hears nor
downloads the avatar, and `mute_participant` is a coroutine at both layers, so it must be awaited.

The construction-time tests are synchronous on purpose: a constructor that raises in a test with no event
loop and no pipeline has provably raised before any pipeline existed.

These tests drive the real `LiveKitAvatarSession` over `FakeRoom` and `FakeLiveKitTransport` rather than a
patched `launch` spy: the handle is observable through the public `session.handle`, the URL through what
`FakeSessionApiClient` was sent, and running the real session is what makes "nothing ever disconnects a
borrowed room" worth asserting. `LiveKitAvatarSession.launch`, a classmethod of ours, is the place to
patch if a later test needs to.

`mute_calls` is asserted by equality for the count and by containment for the identity. The agent-kind
message is asserted by text only for the substring the developer has to act on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Mapping
from datetime import timedelta
from time import monotonic
from typing import Any, cast
from unittest.mock import MagicMock

import jwt
import pytest
from livekit import api as lk_api
from livekit import rtc as lk_rtc

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia._compat import PIPECAT_VERSION
from pipecat_synthesia.errors import ErrorType, SynthesiaError, error_from_problem
from pipecat_synthesia.livekit import token as livekit_token
from pipecat_synthesia.livekit._types import LiveKitCredentials
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import (
    DEFAULT_AVATAR_IDENTITY,
    DEFAULT_AVATAR_NAME,
    DEFAULT_TOKEN_TTL_SECONDS,
    decode_unsigned,
    mint_avatar_token,
)
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from pipecat_synthesia.protocol import (
    TOKEN_ATTR_CLIENT,
    TOKEN_ATTR_LAUNCH,
    TOKEN_ATTR_PROTOCOL,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
)
from pipecat_synthesia.room import JoinMode
from tests.fakes.fake_rtc import (
    FakeLocalParticipant,
    FakeRemoteParticipant,
    FakeRoom,
    FakeTrackPublication,
)
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.fakes.fake_transport import (
    FAKE_AGENT_TOKEN,
    FAKE_STANDARD_TOKEN,
    FAKE_URL,
    FakeLiveKitTransport,
)
from tests.harness import make_service
from tests.support.callbacks import CallbackRecorder, make_launch_context
from tests.support.secrets import LK_API_SECRET, LK_TOKEN_MARKER, assert_no_secrets
from tests.support.waiting import until

AVATAR = DEFAULT_AVATAR_IDENTITY
BOT = "pipecat-bot"
ROOM = "fake-room"
CREDENTIALS = LiveKitCredentials(api_key="fake-lk-key", api_secret=LK_API_SECRET)
VIDEO_SID = "TR_video"
AUDIO_SID = "TR_audio"
KIND_VIDEO = int(lk_rtc.TrackKind.KIND_VIDEO)
KIND_AUDIO = int(lk_rtc.TrackKind.KIND_AUDIO)
#: The substring of the agent-kind message a developer has to act on; the rest of its wording is free.
KIND_HINT = 'with_kind("agent")'
#: The Synthesia API's 503 text, and the substring of the hint a direct join adds to it.
SERVICE_UNAVAILABLE_DETAIL = "The session could not be started. Retry shortly."
ROOM_PROBLEM_HINT = "could not join your LiveKit room"
#: The substring of the API-key mismatch warning a test looks for.
KEY_MISMATCH_HINT = "different LiveKit API key"
#: Two distinct LiveKit API keys, so "neither key is logged, not even in part" can be asserted.
BOT_API_KEY = "APIbotKEY7w3q"
MINTING_API_KEY = "APImintKEY2k9x"


def avatar_token(
    *, identity: str = AVATAR, bot_identity: str = BOT, room: str = ROOM, ttl_seconds: int = 3600
) -> str:
    """A caller-minted avatar token, by the product's own recipe, for the given bot and room."""
    return mint_avatar_token(
        credentials=CREDENTIALS,
        room_name=room,
        avatar_identity=identity,
        avatar_name=LK_TOKEN_MARKER,
        bot_identity=bot_identity,
        client=build_client_info(),
        ttl_seconds=ttl_seconds,
    )


def hand_minted_avatar_token(
    *, identity: str = AVATAR, room_join: bool = True, on_behalf: str | None = BOT
) -> str:
    """An avatar token a caller built by hand, which the product's own recipe cannot get wrong."""
    token = (
        lk_api.AccessToken(CREDENTIALS.api_key, LK_API_SECRET)
        .with_identity(identity)
        .with_kind("agent")
        .with_grants(lk_api.VideoGrants(room_join=room_join, room=ROOM, can_update_own_metadata=True))
        .with_ttl(timedelta(hours=1))
    )
    if on_behalf is not None:
        token = token.with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: on_behalf})
    return token.to_jwt()


def bot_token_signed_by(api_key: str) -> str:
    """An agent-kind bot token for `BOT` in `ROOM`, signed with `api_key` (its `iss` claim)."""
    return (
        lk_api.AccessToken(api_key, LK_API_SECRET)
        .with_identity(BOT)
        .with_name(f"{BOT} {LK_TOKEN_MARKER}")
        .with_kind("agent")
        .with_grants(lk_api.VideoGrants(room_join=True, room=ROOM, agent=True))
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )


def make_avatar(identity: str = AVATAR, *, attributes: Mapping[str, str] | None = None):
    """The avatar as the worker leaves it in the room: one video and one audio publication."""
    return FakeRemoteParticipant(
        identity,
        attributes=attributes,
        track_publications={
            VIDEO_SID: FakeTrackPublication(VIDEO_SID, kind=KIND_VIDEO, name="avatar_video"),
            AUDIO_SID: FakeTrackPublication(AUDIO_SID, kind=KIND_AUDIO, name="avatar_audio"),
        },
    )


def publication(avatar: FakeRemoteParticipant, sid: str) -> FakeTrackPublication:
    return cast(FakeTrackPublication, avatar.track_publications[sid])


class Rig:
    """A connected bot transport, its room, the avatar in it, and the adapter under test."""

    def __init__(
        self,
        *,
        participant_id: str = BOT,
        token: str | None = FAKE_AGENT_TOKEN,
        avatar_present: bool = True,
        transport: FakeLiveKitTransport | None = None,
        **adapter_kwargs: Any,
    ) -> None:
        self.participant = FakeLocalParticipant(BOT)
        self.room = FakeRoom(ROOM, local_participant=self.participant)
        self.transport = transport or FakeLiveKitTransport(
            room=self.room, token=token, participant_id=participant_id
        )
        self.api = FakeSessionApiClient()
        self.rec = CallbackRecorder()
        self.ctx = make_launch_context(self.api, avatar_ids=("av_a",), recorder=self.rec)
        self.avatar = self.room.add_participant(make_avatar()) if avatar_present else None
        kwargs: dict[str, Any] = dict(adapter_kwargs)
        if "avatar_token" not in kwargs:
            kwargs.setdefault("credentials", CREDENTIALS)
        self.adapter = LiveKitTransportRoom(cast(Any, self.transport), **kwargs)

    async def launch(self) -> LiveKitAvatarSession:
        """A direct join always yields the LiveKit session; the cast lets a test read `.handle`."""
        return cast(LiveKitAvatarSession, await self.adapter.launch(self.ctx))

    def sent_token_claims(self) -> dict[str, Any]:
        """The claims of the avatar token the Synthesia API was sent, exactly as sent."""
        assert len(self.api.calls) == 1, f"expected one create_session call, got {len(self.api.calls)}"
        return jwt.decode(self.api.calls[0]["livekit_token"], options={"verify_signature": False})

    def assert_torn_down(self) -> None:
        """Nothing of ours is left on the borrowed room or the transport."""
        assert self.room.disconnect_calls == 0, "a borrowed room was disconnected (borrowed-room rule)"
        assert self.room.listeners == {}, f"room listeners left behind: {self.room.listeners}"
        assert self.participant.rpc_methods == {}, "playback RPC handlers left registered"
        assert self.transport.handlers_for("on_audio_track_subscribed") == []
        assert self.transport.handlers_for("on_video_track_subscribed") == []


@contextlib.asynccontextmanager
async def launched(**kwargs: Any) -> AsyncIterator[tuple[Rig, LiveKitAvatarSession]]:
    """A launched direct-join session, closed however the test ends, so nothing is left running."""
    rig = Rig(**kwargs)
    session = await rig.launch()
    try:
        yield rig, session
    finally:
        await session.aclose()


def warnings_with(caplog: pytest.LogCaptureFixture, keyword: str) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and keyword in record.getMessage()
    ]


# ---------------------------------------------------------------------------- constructor validation


@pytest.mark.unit
def test_ctor_rejects_both_token_and_credentials() -> None:
    """An avatar token and credentials together are ambiguous, so they are rejected, not merged."""
    transport = FakeLiveKitTransport(participant_id=BOT)

    with pytest.raises(ValueError, match="pass either avatar_token or credentials, not both"):
        LiveKitTransportRoom(cast(Any, transport), avatar_token=FAKE_AGENT_TOKEN, credentials=CREDENTIALS)


@pytest.mark.unit
def test_ctor_without_either_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither given: `LiveKitCredentials.from_env()`, and the default avatar identity."""
    transport = FakeLiveKitTransport(participant_id=BOT)
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", LK_API_SECRET)

    room = LiveKitTransportRoom(cast(Any, transport))

    assert room.avatar_identity == DEFAULT_AVATAR_IDENTITY
    assert room.join_mode is JoinMode.DIRECT


@pytest.mark.unit
def test_ctor_without_credentials_or_environment_names_both_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no credentials and no environment, the error names the two environment variables to set."""
    transport = FakeLiveKitTransport(participant_id=BOT)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with pytest.raises(ValueError) as excinfo:
        LiveKitTransportRoom(cast(Any, transport))

    assert "LIVEKIT_API_KEY" in str(excinfo.value)
    assert "LIVEKIT_API_SECRET" in str(excinfo.value)


@pytest.mark.unit
def test_ctor_rejects_an_undecodable_avatar_token() -> None:
    """An undecodable avatar token raises `ValueError`, not `SynthesiaError` (*misuse vs session
    errors*), keeping the decoder's message as the explanation."""
    transport = FakeLiveKitTransport(participant_id=BOT)

    with pytest.raises(ValueError) as excinfo:
        LiveKitTransportRoom(cast(Any, transport), avatar_token="not-a-jwt")

    assert str(excinfo.value).startswith("avatar_token: ")
    assert "Must be a LiveKit access token" in str(excinfo.value)
    assert not isinstance(excinfo.value, SynthesiaError)


@pytest.mark.unit
def test_ctor_takes_the_avatar_identity_from_the_token() -> None:
    """With a caller-supplied token, the avatar identity comes from its `sub`, not from the default."""
    token = avatar_token(identity="avatar-from-token")
    transport = FakeLiveKitTransport(participant_id=BOT)

    room = LiveKitTransportRoom(cast(Any, transport), avatar_token=token)

    assert room.avatar_identity == "avatar-from-token" == decode_unsigned(token).identity


@pytest.mark.unit
def test_ctor_rejects_an_identity_that_contradicts_the_token() -> None:
    """A mismatch is caught at construction; the worker would otherwise join under one identity
    while we watched for another."""
    token = avatar_token(identity="avatar-from-token")
    transport = FakeLiveKitTransport(participant_id=BOT)

    with pytest.raises(ValueError) as excinfo:
        LiveKitTransportRoom(cast(Any, transport), avatar_token=token, avatar_identity="something-else")

    assert "'something-else' does not match the token identity 'avatar-from-token'" in str(excinfo.value)


@pytest.mark.unit
def test_ctor_accepts_an_explicit_identity_with_credentials() -> None:
    """With credentials, an explicit avatar identity is used as given."""
    transport = FakeLiveKitTransport(participant_id=BOT)

    room = LiveKitTransportRoom(
        cast(Any, transport), credentials=CREDENTIALS, avatar_identity="custom-avatar"
    )

    assert room.avatar_identity == "custom-avatar"


@pytest.mark.unit
@pytest.mark.parametrize(
    "transport_url,livekit_url,source",
    [
        ("ws://fake.livekit", None, "the LiveKitTransport's URL"),
        (FAKE_URL, "ws://public.livekit.example", "livekit_url"),
        (FAKE_URL, "https://", "livekit_url"),
    ],
    ids=["transport-ws", "override-ws", "override-no-host"],
)
def test_ctor_rejects_a_url_the_launch_would_refuse(
    transport_url: str, livekit_url: str | None, source: str
) -> None:
    """The URL the avatar worker is told, the override or else the transport's own, is checked at
    construction: a `ws://` URL, or one with no host, is a misuse error naming where the URL came from
    (*misuse vs session errors*)."""
    transport = FakeLiveKitTransport(url=transport_url, participant_id=BOT)

    with pytest.raises(ValueError) as excinfo:
        LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS, livekit_url=livekit_url)

    assert not isinstance(excinfo.value, SynthesiaError)
    assert str(excinfo.value).startswith(f"{source} ")
    assert "the Synthesia API accepts only wss://" in str(excinfo.value)


@pytest.mark.unit
def test_ctor_accepts_a_ws_url_only_through_the_test_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """The integration suite's local LiveKit server speaks `ws://`; the test seam is the one way to use it."""
    transport = FakeLiveKitTransport(url="ws://localhost:7880", participant_id=BOT)

    monkeypatch.setattr(livekit_token, "_ALLOW_INSECURE_WS", True)
    LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS)


@pytest.mark.unit
@pytest.mark.parametrize("subject", [object, MagicMock], ids=["plain-object", "magicmock"])
def test_ctor_rejects_a_non_livekit_transport(subject: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Daily or SmallWebRTC transport is the wrong join mode, so the constructor says which adapter to
    use. It is checked first: no credentials are needed to be told the transport is wrong."""
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with pytest.raises(TypeError) as excinfo:
        LiveKitTransportRoom(cast(Any, subject()))

    assert "PrivateLiveKitRoom" in str(excinfo.value)


@pytest.mark.unit
def test_ctor_rejects_a_transport_whose_internals_moved() -> None:
    """A `LiveKitTransport` from a Pipecat that moved what we read fails at construction, naming the
    Pipecat version, rather than a launch later with an `AttributeError`."""
    transport = FakeLiveKitTransport(participant_id=BOT)
    transport._client = object()  # type: ignore[assignment]

    with pytest.raises(TypeError) as excinfo:
        LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS)

    assert PIPECAT_VERSION in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
def test_bot_kind_rejected_at_construction(required: bool) -> None:
    """A bot token without the top-level `kind="agent"` claim is refused by the constructor, whatever
    `required` says: `required` decides only what a failed launch does, and this is a configuration
    mistake, not a launch failure (*agent-kind requirement*). The room is built before the service, so
    no session request, and no billed session, can happen."""
    transport = FakeLiveKitTransport(token=FAKE_STANDARD_TOKEN)

    with pytest.raises(ValueError) as excinfo:
        make_service(LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS), required=required)

    message = str(excinfo.value)
    assert KIND_HINT in message
    assert "never end the session when the real human leaves" in message


@pytest.mark.unit
@pytest.mark.parametrize("token", [None, "", "not-a-jwt"], ids=["none", "empty", "undecodable"])
def test_unreadable_bot_token_rejected_at_construction(token: str | None) -> None:
    """A transport with no usable token could never connect, and leaves nothing to check the bot's kind
    against, so it is refused at construction."""
    transport = FakeLiveKitTransport(token=token)

    with pytest.raises(ValueError, match="LiveKitTransport"):
        LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS)


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_token, expected",
    [
        pytest.param(
            lambda: avatar_token(room="another-room"),
            f"avatar_token: room grant is 'another-room' but the bot's room is {ROOM!r}",
            id="wrong-room",
        ),
        pytest.param(
            lambda: avatar_token(bot_identity="another-bot"),
            f"avatar_token: lk.publish_on_behalf is 'another-bot' but the bot identity is {BOT!r}",
            id="wrong-publish-on-behalf",
        ),
        pytest.param(
            lambda: hand_minted_avatar_token(on_behalf=None),
            "avatar_token: Must carry a non-empty lk.publish_on_behalf attribute set to the identity of the "
            "agent the avatar publishes for",
            id="no-publish-on-behalf",
        ),
        pytest.param(
            lambda: hand_minted_avatar_token(room_join=False),
            "avatar_token: token has no roomJoin grant",
            id="no-room-join-grant",
        ),
    ],
)
def test_ctor_checks_a_provided_avatar_token_against_the_bot(make_token: Any, expected: str) -> None:
    """A provided avatar token for another room or another bot is refused at construction, checked
    against the bot token's own `sub` and room. The transport has not connected (`participant_id` is
    empty), so nothing but the bot token can have supplied the bot's identity. Either mistake would
    otherwise surface only as an avatar that never joins."""
    transport = FakeLiveKitTransport(participant_id="")

    with pytest.raises(ValueError) as excinfo:
        LiveKitTransportRoom(cast(Any, transport), avatar_token=make_token())

    assert str(excinfo.value) == expected
    assert not isinstance(excinfo.value, SynthesiaError)


@pytest.mark.unit
async def test_expired_provided_avatar_token_is_caught_at_launch() -> None:
    """Expiry is the one check left to the launch, because a token can expire between startup and
    launch. It still fails before a session is requested: the Synthesia API would report it as a 503."""
    rig = Rig(avatar_token=avatar_token(ttl_seconds=-60))  # the constructor accepts it

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.launch()

    assert str(excinfo.value) == "avatar_token: token is expired"
    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN
    assert rig.api.calls == [], "a session was requested with an expired avatar token"
    assert rig.room.disconnect_calls == 0, "a borrowed room was disconnected"


@pytest.mark.unit
async def test_a_valid_provided_avatar_token_is_sent_unchanged() -> None:
    """The launch also waits for the token's own identity, not the default one."""
    token = hand_minted_avatar_token(identity="provided-avatar")
    rig = Rig(avatar_present=False, avatar_token=token)
    rig.room.add_participant(make_avatar("provided-avatar"))

    session = await rig.launch()
    try:
        assert [call["livekit_token"] for call in rig.api.calls] == [token]
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_the_minted_avatar_token_is_minted_for_this_bot_and_room() -> None:
    """What the direct join feeds the mint; the grant set itself is pinned in test_token.py."""
    async with launched() as (rig, _session):
        claims = rig.sent_token_claims()

    assert claims["video"]["room"] == ROOM
    assert claims["kind"] == "agent"
    assert claims["sub"] == DEFAULT_AVATAR_IDENTITY
    assert claims["name"] == DEFAULT_AVATAR_NAME
    assert claims["iss"] == CREDENTIALS.api_key
    launch_id = claims["attributes"].pop(TOKEN_ATTR_LAUNCH)  # random per launch (*early start*)
    assert len(launch_id) == 16 and int(launch_id, 16) >= 0
    assert claims["attributes"] == {
        TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT,
        TOKEN_ATTR_PROTOCOL: "1",
        TOKEN_ATTR_CLIENT: build_client_info().token_client_attr,
    }
    # Both bounds are stamped from one clock read, so the difference is the lifetime to within rounding.
    assert abs((claims["exp"] - claims["nbf"]) - DEFAULT_TOKEN_TTL_SECONDS) <= 1


@pytest.mark.unit
async def test_api_key_mismatch_warns_once_without_either_key(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """A bot token signed with a different API key than the credentials that mint the avatar token draws
    one warning, at construction, and none again at launch. A warning, not an error: one LiveKit project
    can have several keys. Neither key appears in any log line, not even in part (*redaction rule*)."""
    room = FakeRoom(ROOM, local_participant=FakeLocalParticipant(BOT))
    transport = FakeLiveKitTransport(room=room, token=bot_token_signed_by(BOT_API_KEY), participant_id=BOT)
    rig = Rig(transport=transport, credentials=LiveKitCredentials(MINTING_API_KEY, LK_API_SECRET))
    rig.room, rig.participant = room, cast(FakeLocalParticipant, room.local_participant)
    rig.room.add_participant(make_avatar())

    assert len(warnings_with(caplog_loguru, KEY_MISMATCH_HINT)) == 1, caplog_loguru.text
    session = await rig.launch()
    await session.aclose()

    assert len(warnings_with(caplog_loguru, KEY_MISMATCH_HINT)) == 1, caplog_loguru.text
    fragments = [key[:7] for key in (BOT_API_KEY, MINTING_API_KEY)] + [
        key[-4:] for key in (BOT_API_KEY, MINTING_API_KEY)
    ]
    assert_no_secrets(caplog_loguru.text, secrets=(BOT_API_KEY, MINTING_API_KEY, *fragments))
    assert_no_secrets(caplog_loguru.text)


@pytest.mark.unit
@pytest.mark.parametrize("route", ["same-key", "provided-token"])
def test_no_api_key_warning_when_nothing_differs(route: str, caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The same key that signed the bot token, or a provided avatar token (nothing is minted), draws no
    warning."""
    transport = FakeLiveKitTransport(token=bot_token_signed_by(BOT_API_KEY))
    if route == "same-key":
        LiveKitTransportRoom(cast(Any, transport), credentials=LiveKitCredentials(BOT_API_KEY, LK_API_SECRET))
    else:
        LiveKitTransportRoom(cast(Any, transport), avatar_token=avatar_token())

    assert warnings_with(caplog_loguru, KEY_MISMATCH_HINT) == [], caplog_loguru.text


# ---------------------------------------------------------------------------------- handle and URL


@pytest.mark.unit
async def test_launch_handle_and_url() -> None:
    """`launch()` waits for Pipecat's own connect, then hands the session a borrowed-room handle with a
    normalised URL, the URL the Synthesia API is told to join. (That the session request does not wait for
    the connect is `test_connect_overlap.py`'s.)"""
    rig = Rig(participant_id="")  # the transport has not connected yet: the normal case
    before = monotonic()
    task = asyncio.ensure_future(rig.launch())
    await until(lambda: rig.transport.handlers_for("on_connected") != [], what="launch() to wait for connect")
    assert not task.done(), "launch() ran on before the transport connected"

    rig.transport.set_connected(BOT)
    await rig.transport.fire("on_connected")
    session = await task
    try:
        handle = session.handle
        assert handle.owns_room is False, "a direct join must never own the room (borrowed-room rule)"
        assert handle.release_room is None
        assert handle.room is rig.room
        assert handle.room_name == ROOM
        assert handle.bot_identity == BOT
        assert handle.livekit_url == "wss://fake.livekit", "https:// was not normalised to wss://"
        assert handle.connected_at is not None and handle.connected_at >= before
        # What the API was sent is the wire artefact that matters, not just the handle.
        assert rig.api.calls[0]["livekit_url"] == "wss://fake.livekit"
    finally:
        await session.aclose()
    rig.assert_torn_down()


@pytest.mark.unit
async def test_launch_prefers_an_explicit_livekit_url() -> None:
    """`livekit_url=` overrides the transport's own URL, still trimmed and normalised: the bot may reach
    the server at a private address the avatar worker cannot."""
    async with launched(livekit_url="  https://public.livekit.example/ \n") as (rig, session):
        assert session.handle.livekit_url == "wss://public.livekit.example"
        assert rig.api.calls[0]["livekit_url"] == "wss://public.livekit.example"


# ----------------------------------------------------------------------- self-hearing and bandwidth


@pytest.mark.unit
async def test_launch_mutes_and_unsubscribes() -> None:
    """*Self-hearing guard*: the avatar's audio is muted for the bot on launch and on every later audio
    subscription, its video is unsubscribed, and both handlers come off with the session."""
    async with launched() as (rig, session):
        avatar = cast(FakeRemoteParticipant, rig.avatar)
        # The immediate call, awaited: the fake records `mute_participant` only when it is awaited.
        assert rig.transport.mute_calls == [AVATAR]
        # Video off: the bot never renders it, and it costs up to 4 Mbps.
        assert publication(avatar, VIDEO_SID).subscribed_calls == [False]
        # The audio publication is left to `mute_participant`; we never touch it directly.
        assert publication(avatar, AUDIO_SID).subscribed_calls == []

        # A subscription that lands after launch, the usual order because the worker publishes later.
        await rig.transport.fire("on_audio_track_subscribed", AVATAR)
        assert len(rig.transport.mute_calls) == 2
        assert AVATAR in rig.transport.mute_calls

        await rig.transport.fire("on_video_track_subscribed", AVATAR)
        assert publication(avatar, VIDEO_SID).subscribed_calls == [False, False]

        # Somebody else's tracks are never touched: muting the human would end the conversation.
        await rig.transport.fire("on_audio_track_subscribed", "human-1")
        await rig.transport.fire("on_video_track_subscribed", "human-1")
        assert len(rig.transport.mute_calls) == 2
        assert publication(avatar, VIDEO_SID).subscribed_calls == [False, False]
        assert rig.transport.unmute_calls == [], "nothing in the plugin may unmute"

    # The close hook removed both handlers: the events are now inert.
    await rig.transport.fire("on_audio_track_subscribed", AVATAR)
    await rig.transport.fire("on_video_track_subscribed", AVATAR)
    assert len(rig.transport.mute_calls) == 2
    assert publication(cast(FakeRemoteParticipant, rig.avatar), VIDEO_SID).subscribed_calls == [
        False,
        False,
    ]
    rig.assert_torn_down()


@pytest.mark.unit
async def test_launch_survives_a_failing_unsubscribe(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """A bandwidth optimisation must not destroy a working avatar session: a publication whose
    `set_subscribed` raises is one WARNING, and the session still comes back."""

    class Exploding(FakeTrackPublication):
        def set_subscribed(self, subscribed: bool) -> None:
            raise RuntimeError("ffi gone")

    rig = Rig(avatar_present=False)
    avatar = make_avatar()
    avatar.publish("TR_bad", Exploding("TR_bad", kind=KIND_VIDEO))
    rig.avatar = rig.room.add_participant(avatar)

    session = await rig.launch()
    try:
        assert session.is_alive
        assert rig.transport.mute_calls == [AVATAR]
        assert warnings_with(caplog_loguru, "unsubscribe the avatar's video"), caplog_loguru.text
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_failing_mute_still_unsubscribes_video(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """The mute and the unsubscribe are independent. A mute that raises must not cost the bot the video
    unsubscribe as well: by this point the avatar's video is already subscribed, so the
    `on_video_track_subscribed` handler would never fire to make up for it and the bot would
    download the whole call's video (up to 4 Mbps) for nothing."""

    class ExplodingMute(FakeLiveKitTransport):
        async def mute_participant(self, participant_id: str) -> None:
            raise RuntimeError("ffi gone")

    room = FakeRoom(ROOM, local_participant=FakeLocalParticipant(BOT))
    rig = Rig(transport=ExplodingMute(room=room, participant_id=BOT), avatar_present=False)
    rig.room = room
    rig.participant = cast(FakeLocalParticipant, room.local_participant)
    avatar = rig.room.add_participant(make_avatar())
    rig.avatar = avatar

    session = await rig.launch()
    try:
        assert session.is_alive, "a failed mute must not kill the session"
        assert publication(avatar, VIDEO_SID).subscribed_calls == [False], (
            "the video unsubscribe was skipped because the audio mute failed"
        )
        assert warnings_with(caplog_loguru, "mute the avatar's audio"), caplog_loguru.text
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_launch_cancelled_after_session_launch_closes_the_session() -> None:
    """The service's launch timeout can cancel us after the avatar is live. Nothing else holds a
    reference to that session, so `launch()` closes it before re-raising, and still never disconnects
    the borrowed room."""
    gate = asyncio.Event()
    entered = asyncio.Event()

    class BlockingMute(FakeLiveKitTransport):
        async def mute_participant(self, participant_id: str) -> None:
            entered.set()
            await gate.wait()
            await super().mute_participant(participant_id)

    room = FakeRoom(ROOM, local_participant=FakeLocalParticipant(BOT))
    rig = Rig(transport=BlockingMute(room=room, participant_id=BOT))
    rig.room = room
    rig.participant = cast(FakeLocalParticipant, room.local_participant)
    rig.room.add_participant(make_avatar())

    task = asyncio.ensure_future(rig.launch())
    await until(entered.is_set, what="the mute call to be in flight")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert rig.api.calls != [], "the avatar had already been launched"
    rig.assert_torn_down()


# -------------------------------------------------------------------------- the room stays borrowed


@pytest.mark.unit
async def test_borrowed_room_never_disconnected() -> None:
    """`aclose()` unwinds everything we installed and leaves the room connected (*borrowed-room rule*):
    Pipecat's own reference counting owns it."""
    rig = Rig()
    session = await rig.launch()
    assert rig.room.listeners != {}, "the session installed no watchers at all"

    await session.aclose()
    await session.aclose()  # idempotent: a second close must not start disconnecting either

    assert rig.room.disconnect_calls == 0
    assert rig.room.isconnected() is True
    rig.assert_torn_down()


@pytest.mark.unit
async def test_launch_logs_no_secrets(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The direct join is the only code that reads the bot's token, so this leak path exists nowhere
    else: neither that token, the LiveKit secret, nor a decoded token claim may reach a log line."""
    async with launched(avatar_name=LK_TOKEN_MARKER) as (_rig, _session):
        pass

    assert_no_secrets(caplog_loguru.text)
    assert_no_secrets(caplog_loguru.text, secrets=(FAKE_AGENT_TOKEN,))


# ------------------------------------------------------------------------- the Synthesia API's errors


@pytest.mark.unit
async def test_service_unavailable_names_the_likely_room_problem() -> None:
    """A direct join's 503 `service_unavailable` is how an avatar worker that could not join the room
    reaches us, so the message says so. The typed error is otherwise the one the API client raised."""
    rig = Rig()
    original = error_from_problem(
        503, {"code": "service_unavailable", "detail": SERVICE_UNAVAILABLE_DETAIL}, {"Request-Id": "req_503"}
    )
    rig.api.error = original

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.launch()

    err = excinfo.value
    assert err is original
    assert ROOM_PROBLEM_HINT in str(err)
    assert "different LiveKit project" in str(err)
    assert SERVICE_UNAVAILABLE_DETAIL in str(err)
    assert str(err).startswith("[req_503]")
    assert (err.type, err.status, err.code) == (ErrorType.CONNECTION, 503, "service_unavailable")
    rig.assert_torn_down()
