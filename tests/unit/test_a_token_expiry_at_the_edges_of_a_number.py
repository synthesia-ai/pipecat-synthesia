"""A provided avatar token's `exp` at the edges of "a number": what is refused, and what is still a time.

`test_a_token_expiry_that_is_not_a_number.py` pins that a string, a list or an object is refused as an
invalid token. This file pins the edges of the same rule: a bool (a Python `int`, but not a time) and a
non-finite number are refused at construction too, while a fractional expiry is still an expiry, checked
against the clock when the token is resolved.

The tokens are signed by `pyjwt` directly, because `livekit.api.AccessToken` always writes a whole-second
`exp`; the default JSON encoder writes NaN and Infinity, and pyjwt reads them back.
"""

from __future__ import annotations

import math
import time
from typing import Any

import jwt
import pytest

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._types import ProvidedAvatarToken
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME
from pipecat_synthesia.protocol import TOKEN_ATTR_PUBLISH_ON_BEHALF
from tests.support.secrets import LK_API_SECRET

BOT = "pipecat-bot"
ROOM = "fake-room"
AVATAR = DEFAULT_AVATAR_IDENTITY


def avatar_token(exp: Any) -> str:
    """An avatar token that passes every construction-time check, with `exp` as given."""
    claims = {
        "iss": "fake-lk-key",
        "sub": AVATAR,
        "kind": "agent",
        "video": {"roomJoin": True, "room": ROOM, "canUpdateOwnMetadata": True},
        "attributes": {TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT},
        "exp": exp,
    }
    return jwt.encode(claims, LK_API_SECRET, algorithm="HS256")


def resolve(source: ProvidedAvatarToken) -> str:
    return source.resolve(
        bot_identity=BOT,
        room_name=ROOM,
        avatar_identity=AVATAR,
        avatar_name=DEFAULT_AVATAR_NAME,
        client=build_client_info(),
    )


@pytest.mark.unit
@pytest.mark.parametrize("exp", [True, math.nan, math.inf], ids=["bool", "nan", "infinity"])
def test_an_expiry_that_is_not_a_finite_number_is_refused_at_construction(exp: Any) -> None:
    with pytest.raises(SynthesiaError) as excinfo:
        ProvidedAvatarToken(avatar_token(exp), bot_identity=BOT, room_name=ROOM)

    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN


@pytest.mark.unit
def test_a_fractional_expiry_in_the_future_is_accepted() -> None:
    token = avatar_token(time.time() + 3600.5)

    assert resolve(ProvidedAvatarToken(token, bot_identity=BOT, room_name=ROOM)) == token


@pytest.mark.unit
def test_a_fractional_expiry_in_the_past_is_expired_when_resolved() -> None:
    source = ProvidedAvatarToken(avatar_token(time.time() - 3600.5), bot_identity=BOT, room_name=ROOM)

    with pytest.raises(SynthesiaError) as excinfo:
        resolve(source)

    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN
