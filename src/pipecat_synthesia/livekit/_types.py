"""The LiveKit-side time bounds (*bounded waits*), the room handle, and the avatar token sources."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pipecat_synthesia.livekit.token import check_not_expired, mint_avatar_token, validate_avatar_token

if TYPE_CHECKING:
    from livekit import rtc as lk_rtc

    from pipecat_synthesia._client_info import ClientInfo

# Every LiveKit-side time bound, in one place (*bounded waits*).
TASK_CANCEL_SECS = 1.0
STREAM_CLOSE_SECS = 2.0
# The worker's deadline for clear buffer is rtc's 1 s floor (see RPC_ACK_SECS: half of 2 s is 1 s). That is
# enough: the handler does in-memory work and schedules a task, and live runs show the round trip at about
# 0.15 s.
CLEAR_BUFFER_SECS = 2.0
# Our RPCs give rtc `max_round_trip_latency = min(RPC_ACK_SECS, timeout / 2)`: how long the worker's
# acknowledgement may take. rtc (livekit-rpc `client.rs`, rtc 1.1.19) sends the avatar worker's handler a
# deadline of max(response_timeout - max_round_trip_latency, 1 s), and the worker's rtc answers
# RESPONSE_TIMEOUT once it passes; passing the timeout as both values made that deadline 1 s, whatever the
# timeout. So the worker is given the timeout minus 2 s (at least half of it, and at least 1 s): 13 s of the
# default 15 s swap_timeout. An acknowledgement normally takes tens of milliseconds; one slower than 2 s (a
# rare relay retry in LiveKit Cloud, for which rtc's own default allows 7 s) fails the RPC with
# CONNECTION_TIMEOUT.
RPC_ACK_SECS = 2.0
ROOM_CONNECT_SECS = 15.0  # the bot's connect to a private room
ROOM_CREATE_SECS = 10.0
ROOM_DISCONNECT_SECS = 5.0
ROOM_DELETE_SECS = 5.0
CLOSE_HOOK_SECS = 5.0
# backstop wait for the avatar participant after the 201 (the 201 means tracks are published)
POST_CREATE_JOIN_WAIT_SECS = 30.0
# see *replacement grace*
REPLACEMENT_GRACE_SECS = 1.0
# See *no-humans-left window*. When the last human hangs up, the avatar worker leaves at once, and its leave
# can reach the bot about a second before the bot's own participant_disconnected for that human (measured in
# a live call on 2026-11-10 UTC). Twice that, so a hang-up is not taken for an unexpected
# leave.
NO_HUMANS_LEFT_WINDOW_SECS = 2.0


@dataclass
class RoomHandle:
    """A room an adapter borrowed (direct join) or created (relayed join), plus what releasing it needs."""

    room: lk_rtc.Room
    bot_identity: str
    livekit_url: str  # wss://… exactly as sent to Synthesia
    room_name: str
    # True: the session releases it (relayed join). False: never touched (*borrowed-room rule*).
    owns_room: bool
    # how to release an owned room; None for a borrowed one
    release_room: Callable[[], Awaitable[None]] | None = None
    # monotonic() when the bot's room connect was observed
    connected_at: float | None = None


@dataclass(frozen=True)
class LiveKitCredentials:
    """A LiveKit API key/secret pair. Both fields are secrets: never repr'd, never logged."""

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)

    @classmethod
    def from_env(cls) -> LiveKitCredentials:
        """Read LIVEKIT_API_KEY / LIVEKIT_API_SECRET; ValueError naming both if missing."""
        api_key = (os.environ.get("LIVEKIT_API_KEY") or "").strip()
        api_secret = (os.environ.get("LIVEKIT_API_SECRET") or "").strip()
        if not api_key or not api_secret:
            raise ValueError("LiveKit credentials are required: set LIVEKIT_API_KEY and LIVEKIT_API_SECRET")
        return cls(api_key=api_key, api_secret=api_secret)


class AvatarTokenSource(ABC):
    """Supplies the avatar token: the LiveKit access token the avatar worker joins the room with."""

    @abstractmethod
    def resolve(
        self,
        *,
        bot_identity: str,
        room_name: str,
        avatar_identity: str,
        avatar_name: str,
        client: ClientInfo,
        launch_id: str | None = None,
    ) -> str:
        """The avatar token. A source that can, puts ``launch_id`` on it as the ``synthesia.launch``
        attribute (*early start*); one that can't ignores it, and the launch then waits for the 201."""


class MintedAvatarToken(AvatarTokenSource):
    """Mints a fresh avatar token from LiveKit API credentials."""

    def __init__(self, credentials: LiveKitCredentials, *, ttl_seconds: int = 6 * 3600) -> None:
        self._credentials = credentials
        self._ttl_seconds = ttl_seconds

    def resolve(
        self,
        *,
        bot_identity: str,
        room_name: str,
        avatar_identity: str,
        avatar_name: str,
        client: ClientInfo,
        launch_id: str | None = None,
    ) -> str:
        return mint_avatar_token(
            credentials=self._credentials,
            room_name=room_name,
            avatar_identity=avatar_identity,
            avatar_name=avatar_name,
            bot_identity=bot_identity,
            client=client,
            ttl_seconds=self._ttl_seconds,
            launch_id=launch_id,
        )


class ProvidedAvatarToken(AvatarTokenSource):
    """Uses a caller-supplied avatar token, validated against the bot identity and room when constructed.

    Only the expiry waits until `resolve()`, at launch: a token can expire between startup and launch.
    """

    def __init__(self, token: str, *, bot_identity: str, room_name: str | None) -> None:
        """Validate ``token`` against the bot it publishes for and the room it joins.

        Args:
            token: The caller-minted avatar token.
            bot_identity: The bot's LiveKit identity; the token's publish-on-behalf must equal it.
            room_name: The bot's room; the token's room grant must equal it. ``None`` skips that check.

        Raises:
            SynthesiaError: ``INVALID_ROOM_TOKEN``, from `validate_avatar_token`.
        """
        self._token = token
        self._claims = validate_avatar_token(token, bot_identity=bot_identity, room_name=room_name)

    @property
    def identity(self) -> str:  # claims["sub"]
        return self._claims.identity

    def resolve(
        self,
        *,
        bot_identity: str,
        room_name: str,
        avatar_identity: str,
        avatar_name: str,
        client: ClientInfo,
        launch_id: str | None = None,
    ) -> str:
        # The avatar identity is the token's own `sub`: `LiveKitTransportRoom` rejects a contradicting one
        # at construction. The caller minted it, so it carries no launch id: the launch waits for the 201.
        check_not_expired(self._claims)
        return self._token
