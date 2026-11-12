"""Redaction helpers, so secrets never reach a log line (*redaction rule*)."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

_REDACTED = "<redacted>"
_JWT_UNPARSEABLE = "<jwt unparseable>"


def redact_secret(value: str | None) -> str:
    """Return a constant placeholder for a secret.

    Never echoes any part of ``value`` -- not a prefix, not a suffix, not its length.
    """
    return _REDACTED


def describe_exception(exc: BaseException) -> str:
    """``TypeName: message``, for a log line or an error message about ``exc``.

    Never ``repr(exc)``: aiohttp's ``ClientResponseError`` repr echoes the request headers, Authorization
    included, and the LiveKit API client runs on aiohttp (*redaction rule*).
    """
    return f"{type(exc).__name__}: {exc}"


def redact_jwt(token: str | None) -> str:
    """Summarise a JWT as ``<jwt sub=... room=... kind=... exp=...>`` without echoing it.

    Returns ``<jwt unparseable>`` for anything that is not one. Decodes the payload by hand rather than
    with pyjwt, because the core package imports without the ``[livekit]`` extra (*LiveKit-free core*).
    """
    if not token:
        return _JWT_UNPARSEABLE
    try:
        segment = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    except Exception:
        return _JWT_UNPARSEABLE
    if not isinstance(payload, dict):
        return _JWT_UNPARSEABLE
    # A LiveKit token carries the room name under the "video" grant; a bare top-level "room"
    # claim is accepted too so a hand-built token summarises the same way.
    video = payload.get("video")
    room = payload.get("room") or (video.get("room") if isinstance(video, dict) else None)
    return f"<jwt sub={payload.get('sub')} room={room} kind={payload.get('kind')} exp={payload.get('exp')}>"


def redact_mapping(
    body: Mapping[str, Any],
    *,
    keys: Sequence[str] = ("livekitToken", "token", "authorization", "api_key", "apiKey"),
) -> dict[str, Any]:
    """Return a NEW dict with every key in ``keys`` replaced by the redaction placeholder.

    Values under other keys are carried over untouched; ``body`` itself is never mutated.
    """
    return {key: (redact_secret(value) if key in keys else value) for key, value in body.items()}
