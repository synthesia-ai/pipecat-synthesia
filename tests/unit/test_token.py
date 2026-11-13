"""LiveKit token minting, decoding, validation and URL normalisation.

Message wording is usually not part of the contract, but these user-facing token and URL errors are, so
they are matched exactly. Tokens are real `livekit.api.AccessToken` and `pyjwt` throughout; nothing fakes
the JWT itself.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import re
import time
from datetime import timedelta

import jwt
import pytest
from livekit import api as lk_api

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia._log import get_logger
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._types import LiveKitCredentials, MintedAvatarToken, ProvidedAvatarToken
from pipecat_synthesia.livekit.token import (
    DEFAULT_TOKEN_TTL_SECONDS,
    bot_token_is_agent_kind,
    check_not_expired,
    decode_unsigned,
    mint_avatar_token,
    mint_bot_token,
    normalize_livekit_url,
    validate_avatar_token,
)
from pipecat_synthesia.protocol import TOKEN_ATTR_CLIENT, TOKEN_ATTR_PROTOCOL, TOKEN_ATTR_PUBLISH_ON_BEHALF
from tests.fakes.fake_transport import fake_bot_token
from tests.support.secrets import LK_API_SECRET, LK_TOKEN_MARKER, assert_no_secrets

CREDENTIALS = LiveKitCredentials(api_key="test-lk-key", api_secret=LK_API_SECRET)
BOT_IDENTITY = "pipecat-bot-under-test"
AVATAR_IDENTITY = "synthesia-avatar-under-test"
ROOM_NAME = "room-under-test"
CLIENT = build_client_info()


def _unsigned(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False, "verify_exp": False})


def _minted_avatar_token(**overrides) -> str:
    kwargs = dict(
        credentials=CREDENTIALS,
        room_name=ROOM_NAME,
        avatar_identity=AVATAR_IDENTITY,
        avatar_name=LK_TOKEN_MARKER,
        bot_identity=BOT_IDENTITY,
        client=CLIENT,
        ttl_seconds=DEFAULT_TOKEN_TTL_SECONDS,
    )
    kwargs.update(overrides)
    return mint_avatar_token(**kwargs)


def _hand_built_token(payload: dict) -> str:
    """An unsigned JWT `decode_unsigned` can read without needing a real signature."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}."


# --- decode_unsigned tolerates malformed-but-decodable claims --------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"sub": "x", "video": {"room": "r"}, "attributes": ["not", "a", "dict"]}, id="list-attrs"
        ),
        pytest.param({"sub": "x", "video": "not-a-dict", "attributes": {}}, id="string-video"),
    ],
)
def test_decode_unsigned_rejects_wrong_typed_claims(payload: dict) -> None:
    """A caller-supplied token (`ProvidedAvatarToken`) is untrusted input: a claim of the wrong JSON type
    must raise the same `SynthesiaError` an undecodable token does, never an `AttributeError`."""
    with pytest.raises(SynthesiaError, match=re.escape("avatar_token: Must be a LiveKit access token")):
        decode_unsigned(_hand_built_token(payload))


@pytest.mark.unit
def test_decode_unsigned_reads_the_kind_claim() -> None:
    """No `kind` reads as None (not the text "None"); a non-string `kind` is coerced rather than raised.

    `bot_token_is_agent_kind` and the kind check inside `validate_avatar_token` both do
    `(claims.kind or "").lower()`, which would raise `AttributeError` on a non-string `kind` if this
    weren't coerced first.
    """
    assert decode_unsigned(_hand_built_token({"sub": "x", "video": {"room": ROOM_NAME}})).kind is None
    assert decode_unsigned(_hand_built_token({"sub": "x", "kind": "agent"})).kind == "agent"
    assert decode_unsigned(_hand_built_token({"sub": "x", "kind": 7})).kind == "7"
    claims = decode_unsigned(_hand_built_token({"sub": "x", "kind": 5, "video": {}, "attributes": {}}))
    assert claims.kind == "5"

    non_string_kind_token = _hand_built_token(
        {
            "sub": AVATAR_IDENTITY,
            "kind": 5,
            "video": {"roomJoin": True, "room": ROOM_NAME, "canUpdateOwnMetadata": True},
            "attributes": {TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY},
        }
    )
    assert bot_token_is_agent_kind(non_string_kind_token) is False
    validate_avatar_token(non_string_kind_token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)


# --- mint_avatar_token -----------------------------------------------------------------------------


@pytest.mark.unit
def test_mint_avatar_token_claims() -> None:
    token = _minted_avatar_token(ttl_seconds=3600)
    claims = _unsigned(token)

    assert claims["sub"] == AVATAR_IDENTITY
    assert claims["name"] == LK_TOKEN_MARKER
    assert claims["kind"] == "agent"
    # The grant set is a locked contract. Without `canUpdateOwnMetadata`, livekit-server silently discards
    # the worker's own `set_attributes()` calls (`avatar_ready`, `avatar_status`).
    assert claims["video"] == {
        "roomJoin": True,
        "room": ROOM_NAME,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
        "canUpdateOwnMetadata": True,
    }
    assert claims["iss"] == CREDENTIALS.api_key
    assert claims["attributes"] == {
        TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY,
        TOKEN_ATTR_PROTOCOL: "1",
        TOKEN_ATTR_CLIENT: CLIENT.token_client_attr,
    }
    assert abs((claims["exp"] - claims["nbf"]) - 3600) <= 5


# --- mint_bot_token ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_mint_bot_token_claims() -> None:
    """The private room's bot is a standard participant: no agent kind and no agent grant.

    Either would make livekit-server count the bot as dependent, and close the private room, which then
    holds only dependent participants, `empty_timeout` after its creation however busy it is.
    """
    token = mint_bot_token(
        credentials=CREDENTIALS,
        room_name=ROOM_NAME,
        bot_identity=BOT_IDENTITY,
        bot_name=LK_TOKEN_MARKER,
        ttl_seconds=DEFAULT_TOKEN_TTL_SECONDS,
    )
    claims = _unsigned(token)

    assert "kind" not in claims, claims.get("kind")
    assert claims["video"] == {
        "roomJoin": True,
        "room": ROOM_NAME,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
    }
    assert (claims["sub"], claims["name"], claims["iss"]) == (
        BOT_IDENTITY,
        LK_TOKEN_MARKER,
        CREDENTIALS.api_key,
    )
    assert "attributes" not in claims


# --- validate_avatar_token ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_ok(caplog_loguru: pytest.LogCaptureFixture) -> None:
    token = _minted_avatar_token()

    claims = validate_avatar_token(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)

    assert claims.identity == AVATAR_IDENTITY
    assert not any(record.levelname == "WARNING" for record in caplog_loguru.records)


def _token_missing_room_join() -> str:
    return (
        lk_api.AccessToken(CREDENTIALS.api_key, CREDENTIALS.api_secret)
        .with_identity(AVATAR_IDENTITY)
        .with_name(LK_TOKEN_MARKER)
        .with_kind("agent")
        .with_grants(lk_api.VideoGrants(room=ROOM_NAME, can_publish=True))
        .with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY})
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )


def _token_with_attributes(attributes: dict[str, str]) -> str:
    return (
        lk_api.AccessToken(CREDENTIALS.api_key, CREDENTIALS.api_secret)
        .with_identity(AVATAR_IDENTITY)
        .with_name(LK_TOKEN_MARKER)
        .with_kind("agent")
        .with_grants(
            lk_api.VideoGrants(room_join=True, room=ROOM_NAME, can_publish=True, can_update_own_metadata=True)
        )
        .with_attributes(attributes)
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )


def _token_with_grants(grants: lk_api.VideoGrants) -> str:
    return (
        lk_api.AccessToken(CREDENTIALS.api_key, CREDENTIALS.api_secret)
        .with_identity(AVATAR_IDENTITY)
        .with_kind("agent")
        .with_grants(grants)
        .with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY})
        .to_jwt()
    )


def _expired_token() -> str:
    now = int(time.time())
    payload = {
        "sub": AVATAR_IDENTITY,
        "name": LK_TOKEN_MARKER,
        "kind": "agent",
        "video": {"roomJoin": True, "room": ROOM_NAME, "canUpdateOwnMetadata": True},
        "attributes": {TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY},
        "nbf": now - 100,
        "exp": now - 1,
    }
    return _hand_built_token(payload)


VALIDATE_FAILURES = [
    pytest.param(
        lambda: "not-a-jwt-at-all",
        "avatar_token: Must be a LiveKit access token",
        id="check1-undecodable",
    ),
    pytest.param(
        _token_missing_room_join,
        "avatar_token: token has no roomJoin grant",
        id="check2-no-room-join",
    ),
    pytest.param(
        lambda: _minted_avatar_token(room_name="a-different-room"),
        f"avatar_token: room grant is 'a-different-room' but the bot's room is {ROOM_NAME!r}",
        id="check3-wrong-room",
    ),
    pytest.param(
        lambda: _token_with_attributes({}),
        "avatar_token: Must carry a non-empty lk.publish_on_behalf attribute set to the identity of the "
        "agent the avatar publishes for",
        id="check4-blank-pob",
    ),
    pytest.param(
        lambda: _token_with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: "   "}),
        "avatar_token: Must carry a non-empty lk.publish_on_behalf attribute set to the identity of the "
        "agent the avatar publishes for",
        id="check4-whitespace-pob",
    ),
    pytest.param(
        lambda: _token_with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: "someone-else"}),
        f"avatar_token: lk.publish_on_behalf is 'someone-else' but the bot identity is {BOT_IDENTITY!r}",
        id="check5-wrong-pob",
    ),
    pytest.param(
        lambda: _token_with_grants(lk_api.VideoGrants(room_join=True, room=ROOM_NAME, can_publish=True)),
        "avatar_token: token has no canUpdateOwnMetadata grant; without it the avatar cannot report "
        "readiness",
        id="check6-no-metadata-grant",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("make_token, expected_message", VALIDATE_FAILURES)
def test_validate_failures(make_token, expected_message: str) -> None:
    with pytest.raises(SynthesiaError, match=re.escape(expected_message)) as excinfo:
        validate_avatar_token(make_token(), bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)
    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN


@pytest.mark.unit
def test_validate_leaves_expiry_to_check_not_expired() -> None:
    """`validate_avatar_token` runs at construction and leaves the expiry alone, because a token can
    expire between startup and launch; `check_not_expired` is the launch-time check."""
    claims = validate_avatar_token(_expired_token(), bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)

    with pytest.raises(SynthesiaError, match=re.escape("avatar_token: token is expired")) as excinfo:
        check_not_expired(claims)
    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN
    check_not_expired(decode_unsigned(_minted_avatar_token()))


@pytest.mark.unit
def test_check_not_expired_names_the_token_it_checks() -> None:
    """A direct join checks the bot's own token's expiry too; the message names that token, not
    `avatar_token`, because that is the one the developer must re-mint."""
    with pytest.raises(SynthesiaError) as excinfo:
        check_not_expired(decode_unsigned(_expired_token()), what="the LiveKitTransport's token")

    assert str(excinfo.value) == "the LiveKitTransport's token is expired"
    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN


@pytest.mark.unit
def test_validate_kind_missing_warns_not_raises(caplog_loguru: pytest.LogCaptureFixture) -> None:
    token = (
        lk_api.AccessToken(CREDENTIALS.api_key, CREDENTIALS.api_secret)
        .with_identity(AVATAR_IDENTITY)
        .with_name(LK_TOKEN_MARKER)
        .with_grants(
            lk_api.VideoGrants(room_join=True, room=ROOM_NAME, can_publish=True, can_update_own_metadata=True)
        )
        .with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY})
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )

    claims = validate_avatar_token(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)

    assert claims.identity == AVATAR_IDENTITY
    warnings = [r for r in caplog_loguru.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, warnings
    assert 'kind="agent" is recommended' in warnings[0].message


@pytest.mark.unit
def test_validate_tolerates_missing_synthesia_attrs(caplog_loguru: pytest.LogCaptureFixture) -> None:
    token = _token_with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY})

    claims = validate_avatar_token(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)

    assert claims.identity == AVATAR_IDENTITY
    assert not any(r.levelname == "WARNING" for r in caplog_loguru.records)
    debugs = [r for r in caplog_loguru.records if r.levelname == "DEBUG"]
    assert any("missing synthesia.*" in r.message for r in debugs), debugs


@pytest.mark.unit
def test_validate_order_first_failure_wins() -> None:
    """The checks run in a fixed order and the first failure wins: a token with no roomJoin grant and a
    blank publish_on_behalf gets the roomJoin message."""
    token = (
        lk_api.AccessToken(CREDENTIALS.api_key, CREDENTIALS.api_secret)
        .with_identity(AVATAR_IDENTITY)
        .with_name(LK_TOKEN_MARKER)
        .with_kind("agent")
        .with_grants(lk_api.VideoGrants(room=ROOM_NAME, can_publish=True))  # room_join defaults False
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )

    with pytest.raises(SynthesiaError, match=re.escape("avatar_token: token has no roomJoin grant")):
        validate_avatar_token(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)


# --- bot_token_is_agent_kind -------------------------------------------------------------------------


@pytest.mark.unit
def test_bot_token_kind() -> None:
    pipecat_style = (
        lk_api.AccessToken(CREDENTIALS.api_key, CREDENTIALS.api_secret)
        .with_identity(BOT_IDENTITY)
        .with_name(LK_TOKEN_MARKER)
        .with_grants(lk_api.VideoGrants(room_join=True, room=ROOM_NAME, agent=True))
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )
    assert bot_token_is_agent_kind(pipecat_style) is False

    assert bot_token_is_agent_kind(fake_bot_token(kind="agent")) is True

    assert bot_token_is_agent_kind("not-a-jwt") is False


# --- normalize_livekit_url ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_url() -> None:
    assert normalize_livekit_url("https://x.livekit.cloud") == "wss://x.livekit.cloud"
    assert normalize_livekit_url("wss://x/") == "wss://x"

    with pytest.raises(SynthesiaError, match=re.escape("livekit_url: Must be a wss:// URL")):
        normalize_livekit_url("ws://localhost:7880")
    assert normalize_livekit_url("ws://localhost:7880", allow_insecure_ws=True) == "ws://localhost:7880"

    with pytest.raises(SynthesiaError, match=re.escape("livekit_url: Must be a wss:// URL")):
        normalize_livekit_url("http://localhost:7880")
    assert normalize_livekit_url("http://localhost:7880", allow_insecure_ws=True) == "ws://localhost:7880"

    with pytest.raises(SynthesiaError, match=re.escape("livekit_url: Must be a wss:// URL")):
        normalize_livekit_url("ftp://x")

    with pytest.raises(SynthesiaError, match=re.escape("livekit_url: Missing host")):
        normalize_livekit_url("wss://")


@pytest.mark.unit
def test_normalize_url_insecure_ws_warns_once(caplog_loguru: pytest.LogCaptureFixture) -> None:
    normalize_livekit_url("ws://localhost:7880", allow_insecure_ws=True)
    normalize_livekit_url("ws://localhost:9999", allow_insecure_ws=True)

    warnings = [r for r in caplog_loguru.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, warnings
    assert "ws:// LiveKit URLs are accepted only for local development" in warnings[0].message


@pytest.mark.unit
def test_normalize_url_all_error_types_are_invalid_session_request() -> None:
    for url in ("ftp://x", "wss://"):
        with pytest.raises(SynthesiaError) as excinfo:
            normalize_livekit_url(url)
        assert excinfo.value.type is ErrorType.INVALID_SESSION_REQUEST


# --- ProvidedAvatarToken ----------------------------------------------------------------------------


@pytest.mark.unit
def test_provided_token_resolve_returns_token_unchanged() -> None:
    token = _minted_avatar_token()
    source = ProvidedAvatarToken(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)
    assert source.identity == AVATAR_IDENTITY

    resolved = source.resolve(
        bot_identity=BOT_IDENTITY,
        room_name=ROOM_NAME,
        avatar_identity=AVATAR_IDENTITY,
        avatar_name="whatever",
        client=CLIENT,
    )
    assert resolved == token


@pytest.mark.unit
def test_provided_token_undecodable_raises_synthesia_error() -> None:
    expected = re.escape("avatar_token: Must be a LiveKit access token")
    with pytest.raises(SynthesiaError, match=expected) as excinfo:
        ProvidedAvatarToken("garbage-not-a-jwt", bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)
    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN


@pytest.mark.unit
def test_provided_token_checks_the_bot_when_constructed_and_the_expiry_when_resolved() -> None:
    """The checks against the bot run when the source is built; only the expiry waits for `resolve()`."""
    with pytest.raises(SynthesiaError, match="but the bot identity is 'another-bot'"):
        ProvidedAvatarToken(_minted_avatar_token(), bot_identity="another-bot", room_name=ROOM_NAME)

    source = ProvidedAvatarToken(_expired_token(), bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)
    with pytest.raises(SynthesiaError, match=re.escape("avatar_token: token is expired")):
        source.resolve(
            bot_identity=BOT_IDENTITY,
            room_name=ROOM_NAME,
            avatar_identity=AVATAR_IDENTITY,
            avatar_name="whatever",
            client=CLIENT,
        )


@pytest.mark.unit
def test_minted_avatar_token_source_resolves() -> None:
    source = MintedAvatarToken(CREDENTIALS, ttl_seconds=1800)

    token = source.resolve(
        bot_identity=BOT_IDENTITY,
        room_name=ROOM_NAME,
        avatar_identity=AVATAR_IDENTITY,
        avatar_name=LK_TOKEN_MARKER,
        client=CLIENT,
    )

    claims = decode_unsigned(token)
    assert claims.identity == AVATAR_IDENTITY
    assert claims.attributes[TOKEN_ATTR_PUBLISH_ON_BEHALF] == BOT_IDENTITY


# --- secrets never leak ------------------------------------------------------------------------------


@pytest.mark.unit
def test_no_token_in_messages_or_logs(caplog_loguru: pytest.LogCaptureFixture) -> None:
    token = _minted_avatar_token()
    validate_avatar_token(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)

    texts = [caplog_loguru.text]

    for param in VALIDATE_FAILURES:
        make_token, _ = param.values
        try:
            validate_avatar_token(make_token(), bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)
        except SynthesiaError as err:
            texts.append(str(err))
            texts.append(repr(err))
        try:
            ProvidedAvatarToken(make_token(), bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)
        except SynthesiaError as err:
            texts.append(str(err))
            texts.append(repr(err))
    try:
        check_not_expired(decode_unsigned(_expired_token()))
    except SynthesiaError as err:
        texts.append(str(err))
        texts.append(repr(err))

    texts.append(caplog_loguru.text)
    assert_no_secrets(*texts)


@pytest.mark.unit
def test_validate_debug_log_does_not_include_raw_token(caplog_loguru: pytest.LogCaptureFixture) -> None:
    log = get_logger()
    log.debug("sanity check that DEBUG logs are captured")
    token = _minted_avatar_token()

    validate_avatar_token(token, bot_identity=BOT_IDENTITY, room_name=ROOM_NAME)

    assert token not in caplog_loguru.text
    assert "validated caller-provided avatar token" in caplog_loguru.text


# --- LiveKitCredentials ------------------------------------------------------------------------------


@pytest.mark.unit
def test_credentials_from_env_reads_and_trims_both_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", f"  {CREDENTIALS.api_key}\n")
    monkeypatch.setenv("LIVEKIT_API_SECRET", f"\t{LK_API_SECRET} ")

    credentials = LiveKitCredentials.from_env()

    assert isinstance(credentials, LiveKitCredentials)
    assert (credentials.api_key, credentials.api_secret) == (CREDENTIALS.api_key, LK_API_SECRET)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, "", "   ", "\n\t"], ids=["unset", "empty", "spaces", "whitespace"])
@pytest.mark.parametrize("which", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"])
def test_credentials_from_env_rejects_either_variable_missing_or_blank(
    monkeypatch: pytest.MonkeyPatch, which: str, bad: str | None
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", CREDENTIALS.api_key)
    monkeypatch.setenv("LIVEKIT_API_SECRET", LK_API_SECRET)
    if bad is None:
        monkeypatch.delenv(which)
    else:
        monkeypatch.setenv(which, bad)

    with pytest.raises(ValueError) as caught:
        LiveKitCredentials.from_env()

    message = str(caught.value)
    assert "LIVEKIT_API_KEY" in message and "LIVEKIT_API_SECRET" in message
    assert_no_secrets(message, secrets=[CREDENTIALS.api_key, LK_API_SECRET])


@pytest.mark.unit
def test_credentials_hide_both_values_and_are_immutable() -> None:
    """One instance is shared by every token source built from it."""
    credentials = LiveKitCredentials(api_key=CREDENTIALS.api_key, api_secret=LK_API_SECRET)

    assert_no_secrets(repr(credentials), str(credentials), secrets=[credentials.api_key, LK_API_SECRET])
    with pytest.raises(dataclasses.FrozenInstanceError):
        credentials.api_secret = "replaced"  # type: ignore[misc]
