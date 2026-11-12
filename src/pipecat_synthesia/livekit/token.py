"""LiveKit access tokens: minting the avatar and bot tokens, and decoding and validating a provided one."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import jwt
from livekit import api as lk_api

from pipecat_synthesia import _once
from pipecat_synthesia._log import get_logger
from pipecat_synthesia._redact import redact_jwt
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.protocol import (
    PROTOCOL_VERSION,
    TOKEN_ATTR_CLIENT,
    TOKEN_ATTR_JOIN_MODE,
    TOKEN_ATTR_LAUNCH,
    TOKEN_ATTR_PROTOCOL,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
)

if TYPE_CHECKING:
    from pipecat_synthesia._client_info import ClientInfo
    from pipecat_synthesia.livekit._types import LiveKitCredentials

DEFAULT_AVATAR_IDENTITY = "synthesia-avatar-agent"  # the same default as the LiveKit plugin
DEFAULT_AVATAR_NAME = "Synthesia avatar"
DEFAULT_TOKEN_TTL_SECONDS = 6 * 3600
DEFAULT_BOT_IDENTITY = "pipecat-bot"  # relayed join only
DEFAULT_BOT_NAME = "Pipecat bot"

_log = get_logger()


@dataclass(frozen=True)
class TokenClaims:
    """The claims of a LiveKit access token, decoded without verifying the signature."""

    identity: str
    name: str | None
    kind: str | None
    # `iss`: the API key that signed the token. Never logged (*redaction rule*).
    issuer: str | None
    room: str | None
    room_join: bool
    attributes: Mapping[str, str]
    exp: int | None
    video: Mapping[str, Any]


def decode_unsigned(token: str) -> TokenClaims:
    """Decode a LiveKit access token's claims without verifying its signature or expiry.

    Raises ``SynthesiaError`` (``INVALID_ROOM_TOKEN``, the Synthesia API's own wording) for anything that
    is not a LiveKit token: undecodable, no ``sub``, a ``video``/``attributes`` claim that is not an
    object, or an ``exp`` that is not a finite number. Uses pyjwt rather than ``livekit.api.TokenVerifier``,
    whose claims drop ``kind``.
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
    except jwt.PyJWTError:
        payload = None
    if not isinstance(payload, dict) or "sub" not in payload:
        raise SynthesiaError(
            "avatar_token: Must be a LiveKit access token", type=ErrorType.INVALID_ROOM_TOKEN
        )
    video = payload.get("video") or {}
    attributes = payload.get("attributes") or {}
    # A token LiveKit minted never has a non-object "video"/"attributes" claim, but a provided token comes
    # from some other process: a claim of the wrong shape must fail like an undecodable token, not raise
    # AttributeError below.
    if not isinstance(video, dict) or not isinstance(attributes, dict):
        raise SynthesiaError(
            "avatar_token: Must be a LiveKit access token", type=ErrorType.INVALID_ROOM_TOKEN
        )
    # `exp` likewise: a provided token's is untrusted, and one that is not a number, a numeric string
    # included, must fail here rather than raise TypeError where the expiry is checked. A bool is not a
    # number here. A fractional one is kept to the whole second, so `TokenClaims.exp` is an int.
    exp = payload.get("exp")
    if exp is not None:
        if isinstance(exp, bool) or not isinstance(exp, int | float) or not math.isfinite(exp):
            raise SynthesiaError(
                "avatar_token: Must be a LiveKit access token", type=ErrorType.INVALID_ROOM_TOKEN
            )
        exp = int(exp)
    kind = payload.get("kind")
    if kind is not None and not isinstance(kind, str):
        kind = str(kind)
    issuer = payload.get("iss")
    return TokenClaims(
        identity=payload["sub"],
        name=payload.get("name"),
        kind=kind,
        issuer=issuer if isinstance(issuer, str) else None,
        room=video.get("room"),
        room_join=bool(video.get("roomJoin")),
        attributes={str(key): str(value) for key, value in attributes.items()},
        exp=exp,
        video=video,
    )


def mint_avatar_token(
    *,
    credentials: LiveKitCredentials,
    room_name: str,
    avatar_identity: str,
    avatar_name: str,
    bot_identity: str,
    client: ClientInfo,
    ttl_seconds: int,
    launch_id: str | None = None,
) -> str:
    """Mint the avatar token: kind "agent", scoped to the room, published on behalf of the bot.

    It also carries the ``synthesia.*`` client identification attributes (*installed-version channels*),
    with the join mode once the service has set it, but never the user transport: participant attributes
    are visible to everyone in the room. With ``launch_id``, it carries that too, so the session can
    recognise this launch's avatar before the 201 (*early start*).
    """
    attributes = {
        TOKEN_ATTR_PUBLISH_ON_BEHALF: bot_identity,
        TOKEN_ATTR_PROTOCOL: PROTOCOL_VERSION,
        TOKEN_ATTR_CLIENT: client.token_client_attr,
    }
    if client.join_mode is not None:
        attributes[TOKEN_ATTR_JOIN_MODE] = client.join_mode
    if launch_id is not None:
        attributes[TOKEN_ATTR_LAUNCH] = launch_id
    return (
        lk_api.AccessToken(credentials.api_key, credentials.api_secret)
        .with_identity(avatar_identity)
        .with_name(avatar_name)
        .with_kind("agent")
        .with_grants(
            lk_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                # Without this, livekit-server silently discards every attribute the worker sets on itself:
                # no error, no log, and avatar_ready / avatar_status never arrive (measured against
                # livekit-server 1.9.1).
                can_update_own_metadata=True,
            )
        )
        .with_attributes(attributes)
        .with_ttl(timedelta(seconds=ttl_seconds))
        .to_jwt()
    )


def mint_bot_token(
    *,
    credentials: LiveKitCredentials,
    room_name: str,
    bot_identity: str,
    bot_name: str,
    ttl_seconds: int,
) -> str:
    """Mint the bot's token for a private room: a standard participant, with no attributes.

    Neither the agent kind nor the agent grant, so that the bot keeps its private room open (see
    *agent-kind requirement*). A direct join's bot token is the developer's own, and must be an agent.
    """
    return (
        lk_api.AccessToken(credentials.api_key, credentials.api_secret)
        .with_identity(bot_identity)
        .with_name(bot_name)
        .with_grants(
            lk_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(timedelta(seconds=ttl_seconds))
        .to_jwt()
    )


def validate_avatar_token(token: str, *, bot_identity: str, room_name: str | None) -> TokenClaims:
    """Validate a provided avatar token against the bot and its room, and return its claims.

    Checks, in order, raising ``SynthesiaError`` (``INVALID_ROOM_TOKEN``) at the first failure: it decodes;
    it has a room-join grant; its room is the bot's room; it carries a non-empty publish-on-behalf
    attribute; that attribute is the bot's identity; it has the ``canUpdateOwnMetadata`` grant. A direct join
    runs these at construction. The expiry is :func:`check_not_expired`, run at launch instead, because a
    token can expire between startup and launch. A token without ``kind="agent"`` only draws a warning,
    because the avatar's own kind does not affect the worker's no-humans-left rule. Missing ``synthesia.*``
    attributes are not an error. The token itself is never logged.
    """
    claims = decode_unsigned(token)  # check 1: decode_unsigned raises the same message on failure

    if not claims.room_join:
        raise SynthesiaError("avatar_token: token has no roomJoin grant", type=ErrorType.INVALID_ROOM_TOKEN)

    if room_name is not None and claims.room != room_name:
        raise SynthesiaError(
            f"avatar_token: room grant is {claims.room!r} but the bot's room is {room_name!r}",
            type=ErrorType.INVALID_ROOM_TOKEN,
        )

    pob = claims.attributes.get(TOKEN_ATTR_PUBLISH_ON_BEHALF, "").strip()
    if not pob:
        raise SynthesiaError(
            "avatar_token: Must carry a non-empty lk.publish_on_behalf attribute set to the identity of the "
            "agent the avatar publishes for",
            type=ErrorType.INVALID_ROOM_TOKEN,
        )

    if pob != bot_identity:
        raise SynthesiaError(
            f"avatar_token: lk.publish_on_behalf is {pob!r} but the bot identity is {bot_identity!r}",
            type=ErrorType.INVALID_ROOM_TOKEN,
        )

    # The avatar worker reports readiness by setting its own attributes, which livekit-server drops, with no
    # error and no log, unless its token has this grant (see `mint_avatar_token`).
    if claims.video.get("canUpdateOwnMetadata") is not True:
        raise SynthesiaError(
            "avatar_token: token has no canUpdateOwnMetadata grant; without it the avatar cannot report "
            "readiness, so on_avatar_ready never fires",
            type=ErrorType.INVALID_ROOM_TOKEN,
        )

    if (claims.kind or "").lower() != "agent":
        _log.warning(
            'avatar_token: kind="agent" is recommended for the avatar token (the LiveKit plugin sets it; '
            "without it LiveKit clients list the avatar as a standard participant)"
        )

    if TOKEN_ATTR_PROTOCOL not in claims.attributes or TOKEN_ATTR_CLIENT not in claims.attributes:
        _once.debug_once(
            _log,
            "token:missing-synthesia-attrs",
            "avatar_token: caller-provided avatar token is missing synthesia.* attributes",
        )

    _log.debug("validated caller-provided avatar token {token}", token=redact_jwt(token))
    return claims


def check_not_expired(claims: TokenClaims, *, what: str = "avatar_token: token") -> None:
    """Raise ``SynthesiaError`` (``INVALID_ROOM_TOKEN``) if a token has expired; ``what`` names it.

    Checked at launch, because a token can expire between startup and launch. A provided avatar token: the
    Synthesia API reports an expired one as a misleading 503. In a direct join, the bot's own token: an
    expired one cannot connect, while the avatar worker's join, on its own token, would still succeed.
    """
    if claims.exp is not None and claims.exp <= time.time():
        raise SynthesiaError(f"{what} is expired", type=ErrorType.INVALID_ROOM_TOKEN)


def bot_token_is_agent_kind(token: str) -> bool:
    """Whether ``token`` carries the top-level ``kind="agent"`` claim (*agent-kind requirement*)."""
    try:
        claims = decode_unsigned(token)
    except SynthesiaError:
        return False
    return (claims.kind or "").lower() == "agent"


def normalize_livekit_url(url: str, *, allow_insecure_ws: bool = False) -> str:
    """Return ``url`` as the ``wss://`` URL the Synthesia API expects, without a trailing slash.

    ``https://`` becomes ``wss://`` and ``http://`` becomes ``ws://``. ``ws://`` is accepted only with
    ``allow_insecure_ws`` (the test suite's local LiveKit server; the Synthesia API rejects it), with one
    warning. Anything else, or a URL with no host, raises ``SynthesiaError`` (``INVALID_SESSION_REQUEST``,
    the Synthesia API's own wording). Localhost is allowed, because a local development server is
    legitimately there.
    """
    url = url.strip()
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]

    parts = urlsplit(url)
    if not parts.hostname:
        raise SynthesiaError("livekit_url: Missing host", type=ErrorType.INVALID_SESSION_REQUEST)

    if parts.scheme == "ws":
        if not allow_insecure_ws:
            raise SynthesiaError("livekit_url: Must be a wss:// URL", type=ErrorType.INVALID_SESSION_REQUEST)
        _once.warn_once(
            _log,
            "token:ws-insecure",
            "ws:// LiveKit URLs are accepted only for local development; Synthesia's API rejects them",
        )
    elif parts.scheme != "wss":
        raise SynthesiaError("livekit_url: Must be a wss:// URL", type=ErrorType.INVALID_SESSION_REQUEST)

    return url.removesuffix("/")


#: Test seam, not public API: the integration tests' local livekit-server speaks only ``ws://``, so they
#: set this to let the room adapters accept it. The Synthesia API rejects ``ws://``, so nothing else can
#: use it.
_ALLOW_INSECURE_WS = False


def checked_livekit_url(url: str, *, source: str) -> str:
    """`normalize_livekit_url` for a room adapter's constructor, where a URL it rejects is a misuse error.

    Raises ``ValueError`` naming ``source``, where the URL came from (*misuse vs session errors*).
    """
    try:
        return normalize_livekit_url(url, allow_insecure_ws=_ALLOW_INSECURE_WS)
    except SynthesiaError as err:
        raise ValueError(
            f"{source} must be a wss:// or https:// URL with a host, because the Synthesia API accepts only "
            f"wss:// ({err})"
        ) from err
