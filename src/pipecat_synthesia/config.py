"""Service configuration, and the avatar-id helpers (*unprefixed ids on the wire*)."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pipecat_synthesia.protocol import (
    AVATAR_ID_PATTERN,
    AVATAR_ID_PREFIX,
    DEFAULT_API_URL,
    MAX_AVATAR_IDS,
)

ENV_API_KEY = "SYNTHESIA_API_KEY"
ENV_API_URL = "SYNTHESIA_API_URL"
ENV_ALLOW_ANY_API_HOST = "SYNTHESIA_ALLOW_ANY_API_HOST"

#: The domain the Synthesia API URL's host must be, or be under, unless ``allow_any_api_host`` is set.
SYNTHESIA_DOMAIN = "synthesia.io"

# The values of SYNTHESIA_ALLOW_ANY_API_HOST, after strip() and lower(); any other value is a ValueError.
_ENV_TRUE = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE = frozenset({"", "0", "false", "no", "off"})


def normalize_avatar_id(avatar_id: object) -> str:
    """Return ``avatar_id`` stripped and with the ``av_`` prefix added if missing.

    Raises ``ValueError`` if it is not a non-empty string, or not ``av_`` followed by 1-64 of ``A-Za-z0-9_-``.
    """
    if not isinstance(avatar_id, str) or not avatar_id.strip():
        raise ValueError("avatar id must be a non-empty string")
    normalized = avatar_id.strip()
    if not normalized.startswith(AVATAR_ID_PREFIX):
        normalized = AVATAR_ID_PREFIX + normalized
    if re.fullmatch(AVATAR_ID_PATTERN, normalized) is None:
        raise ValueError(
            f"{avatar_id!r} is not a valid avatar id (expected av_ followed by 1-64 of A-Za-z0-9_-)"
        )
    return normalized


def unprefixed_avatar_id(avatar_id: str) -> str:
    """``avatar_id`` without its ``av_`` prefix, for the wire. No validation; idempotent."""
    return avatar_id.removeprefix(AVATAR_ID_PREFIX)


def normalize_avatar_ids(avatar_ids: object) -> tuple[str, ...]:
    """Normalise 1 to ``MAX_AVATAR_IDS`` avatar ids, keeping their order.

    Raises ``ValueError`` for a single string instead of a list, for too few or too many ids, and for
    duplicates after normalising; ``TypeError`` for something that is not iterable.
    """
    if isinstance(avatar_ids, str | bytes):
        raise ValueError("avatar_ids must be a list of ids, not a single string")
    if avatar_ids is None:
        items: tuple[object, ...] = ()
    elif isinstance(avatar_ids, Iterable):
        items = tuple(avatar_ids)
    else:
        raise TypeError(f"avatar_ids must be an iterable of avatar ids, got {type(avatar_ids).__name__}")
    if not items or len(items) > MAX_AVATAR_IDS:
        raise ValueError(f"avatar_ids must contain between 1 and {MAX_AVATAR_IDS} ids, got {len(items)}")
    normalized = tuple(normalize_avatar_id(item) for item in items)
    dups = {value for value in normalized if normalized.count(value) > 1}
    if dups:
        raise ValueError(f"avatar_ids contains duplicates: {sorted(dups)}")
    return normalized


def _check_api_url(url: str, *, source: str, allow_any_host: bool) -> None:
    """Raise ``ValueError`` unless ``url`` is ``https://`` to ``synthesia.io`` or a host under it, or
    ``http://`` or ``https://`` to a loopback host. With ``allow_any_host``, ``https://`` to any host passes.

    The service sends the API key and a LiveKit token to this URL, so a host outside synthesia.io is
    refused unless the developer opts in. The match is on whole DNS labels: ``evilsynthesia.io`` and
    ``synthesia.io.example.com`` fail. A backslash fails too, so that this check and the HTTP client can
    never disagree on the host, whatever the yarl version: ``urlsplit`` reads one host from it, older yarl
    read another, and newer yarl refuses it. A user name or password fails as well: aiohttp would refuse
    to send one alongside the ``Authorization`` header, and only once the launch had started.

    Checked at construction (*misuse vs session errors*): a scheme-less or malformed URL would otherwise
    fail inside aiohttp as a network error, and be retried as one. ``http://`` is for a local server, such
    as the tests' in-process one. The message names where the URL came from but never echoes it, in case
    a secret was pasted into the wrong setting (*redaction rule*).
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # raises ValueError for a bad port
    except ValueError:  # an unbalanced "[" in the host, or a port that is not a number
        host = None
    if host and "\\" not in url and parts.username is None and parts.password is None:
        if _is_loopback(host) and parts.scheme in ("http", "https"):
            return
        if parts.scheme == "https" and (allow_any_host or is_synthesia_host(host)):
            return
    raise ValueError(
        f"the Synthesia API URL from {source} must be an https:// URL to {SYNTHESIA_DOMAIN} or a host under "
        f"it, with no user name or password (http:// only for localhost): pass api_url or set "
        f"{ENV_API_URL}, or set neither to use {DEFAULT_API_URL}. To send the API key to an https:// host "
        f"outside {SYNTHESIA_DOMAIN}, pass allow_any_api_host=True or set {ENV_ALLOW_ANY_API_HOST}=1"
    )


def is_synthesia_host(host: str) -> bool:
    """Whether ``host`` (lower case, as ``urlsplit`` gives it) is ``synthesia.io`` or a host under it. An
    absolute name's trailing dot is ignored: ``api.synthesia.io.`` is the same host."""
    host = host.removesuffix(".")
    return host == SYNTHESIA_DOMAIN or host.endswith("." + SYNTHESIA_DOMAIN)


def api_url_outside_synthesia(url: str) -> bool:
    """Whether a URL that passed ``_check_api_url`` is let through only by ``allow_any_api_host``: its
    host is neither under synthesia.io nor loopback."""
    host = urlsplit(url).hostname or ""
    return not (is_synthesia_host(host) or _is_loopback(host))


def _allow_any_api_host(value: bool | None) -> bool:
    """``value``, or ``SYNTHESIA_ALLOW_ANY_API_HOST`` when it is None (unset is False).

    The message for a value that is neither true nor false names the variable but never echoes it
    (*redaction rule*).
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_ALLOW_ANY_API_HOST, "").strip().lower()
    if raw in _ENV_TRUE:
        return True
    if raw in _ENV_FALSE:
        return False
    raise ValueError(
        f"{ENV_ALLOW_ANY_API_HOST} must be 1, true, yes or on to allow a Synthesia API URL outside "
        f"{SYNTHESIA_DOMAIN}, or 0, false, no, off or unset to refuse one"
    )


def _is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ServiceConfig:
    """The service's configuration, built once from the constructor arguments.

    Only the constructor arguments are public; the other fields are fixed defaults. Every field is read from
    ``self._config`` where it is used, never copied, so tests can replace the config before ``setup()``.
    """

    avatar_ids: tuple[str, ...]
    api_key: str = field(repr=False)
    api_url: str
    launch_timeout: float  # seconds for the whole launch (default 90)
    required: bool
    swap_timeout: float = 15.0
    bot_stopped_hysteresis_secs: float = 0.35
    bot_speaking_interval_secs: float = 0.2
    prelaunch_max_seconds: float = 60.0
    stop_drain_timeout: float = 5.0

    @classmethod
    def from_kwargs(
        cls, *, avatar_ids, api_key, api_url, launch_timeout, required, allow_any_api_host=None
    ) -> ServiceConfig:
        """Build the config from the constructor arguments.

        ``avatar_ids`` is normalised (prefixed, de-duplicated, 1..MAX_AVATAR_IDS); the API key falls back
        to ``SYNTHESIA_API_KEY`` and is required; the URL falls back to ``SYNTHESIA_API_URL`` then
        ``DEFAULT_API_URL``, loses any trailing slash, and must be ``https://`` to synthesia.io or a host
        under it (any host if ``allow_any_api_host``, else ``SYNTHESIA_ALLOW_ANY_API_HOST``, allows it), or
        ``http://`` or ``https://`` to a loopback host; ``launch_timeout`` must be positive.
        Every failure is a ``ValueError``, never a ``SynthesiaError`` (*misuse vs session errors*).
        """
        ids = normalize_avatar_ids(avatar_ids)
        key = api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise ValueError(f"a Synthesia API key is required: pass api_key or set {ENV_API_KEY}")
        if api_url:
            url, source = api_url, "api_url"
        elif os.environ.get(ENV_API_URL):
            url, source = os.environ[ENV_API_URL], ENV_API_URL
        else:
            url, source = DEFAULT_API_URL, "DEFAULT_API_URL"
        url = url.strip().rstrip("/")
        _check_api_url(url, source=source, allow_any_host=_allow_any_api_host(allow_any_api_host))
        if not launch_timeout > 0:  # not `<= 0`: NaN fails every comparison, so this rejects it too
            raise ValueError("launch_timeout must be positive")
        return cls(
            avatar_ids=ids,
            api_key=key,
            api_url=url,
            launch_timeout=float(launch_timeout),
            required=bool(required),
        )
