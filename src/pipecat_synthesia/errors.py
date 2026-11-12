"""The error types, and `SynthesiaError`: the one exception type this package raises once running.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Mapping, Sequence
from typing import Any

import aiohttp
from pipecat.utils.errors import ErrorCategory, classify_http_status_code

from pipecat_synthesia._redact import describe_exception, redact_mapping, redact_secret
from pipecat_synthesia.config import ENV_API_URL
from pipecat_synthesia.protocol import REQUEST_ID_HEADER


# str + Enum rather than StrEnum (UP042): StrEnum would change what str() and f-strings print for a member.
class ErrorType(str, enum.Enum):  # noqa: UP042
    """The Synthesia error type. The first eleven match the LiveKit plugin's names and values exactly; the
    last three are this package's additions."""

    AUTH = "auth"
    FEATURE_NOT_IN_PLAN = "feature_not_in_plan"
    INVALID_ROOM_TOKEN = "invalid_room_token"
    LIVEKIT_CREDENTIALS_REJECTED = "livekit_credentials_rejected"
    INVALID_SESSION_REQUEST = "invalid_session_request"
    UNKNOWN_AVATAR = "unknown_avatar"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    CONCURRENCY_LIMIT = "concurrency_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    # --- additions (the LiveKit plugin reports these cases with type=None) ---
    UNKNOWN = "unknown"  # server code not in CODE_TABLE, or no code
    CLIENT_UPGRADE_REQUIRED = "client_upgrade_required"
    AVATAR_SWAP_FAILED = "avatar_swap_failed"


RETRYABLE_TYPES = frozenset(
    {ErrorType.RATE_LIMITED, ErrorType.CONCURRENCY_LIMIT, ErrorType.TIMEOUT, ErrorType.CONNECTION}
)

# Used only when there is no HTTP status. A test checks that every ErrorType has an entry.
DEFAULT_CATEGORY: Mapping[ErrorType, ErrorCategory] = {
    ErrorType.AUTH: ErrorCategory.AUTHENTICATION,
    ErrorType.FEATURE_NOT_IN_PLAN: ErrorCategory.AUTHORIZATION,
    ErrorType.INVALID_ROOM_TOKEN: ErrorCategory.INVALID_REQUEST,
    ErrorType.LIVEKIT_CREDENTIALS_REJECTED: ErrorCategory.AUTHENTICATION,
    ErrorType.INVALID_SESSION_REQUEST: ErrorCategory.INVALID_REQUEST,
    ErrorType.UNKNOWN_AVATAR: ErrorCategory.INVALID_REQUEST,
    ErrorType.QUOTA_EXCEEDED: ErrorCategory.QUOTA,
    ErrorType.RATE_LIMITED: ErrorCategory.RATE_LIMIT,
    ErrorType.CONCURRENCY_LIMIT: ErrorCategory.RATE_LIMIT,
    ErrorType.TIMEOUT: ErrorCategory.CONNECTIVITY,
    ErrorType.CONNECTION: ErrorCategory.CONNECTIVITY,
    ErrorType.UNKNOWN: ErrorCategory.UNKNOWN,
    ErrorType.CLIENT_UPGRADE_REQUIRED: ErrorCategory.INVALID_REQUEST,
    ErrorType.AVATAR_SWAP_FAILED: ErrorCategory.APPLICATION,
}


class SynthesiaError(Exception):
    """The one exception this package raises once running (*misuse vs session errors*).

    `str(err)` is the message. The message, the repr and `body` never contain the API key, a LiveKit token
    or a LiveKit secret (*redaction rule*): `body` is redacted before it is stored. The attributes are
    plain and writable, so `with_session()`, `with_request()` and `with_hint()` can add to them later.
    """

    def __init__(
        self,
        message: str,
        *,
        type: ErrorType = ErrorType.UNKNOWN,
        # None => derived, in this order: (1) type is CLIENT_UPGRADE_REQUIRED -> INVALID_REQUEST regardless of
        # status (Pipecat maps 426/410 to UNKNOWN, which would make the version gate non-permanent; see
        # *installed-version channels*);
        # (2) classify_http_status_code(status) if status is not None; (3) DEFAULT_CATEGORY[type]
        category: ErrorCategory | None = None,
        retryable: bool | None = None,  # None => type in RETRYABLE_TYPES
        retry_after: float | None = None,  # seconds; from Retry-After (integer) or body retryAfter
        status: int | None = None,
        code: str | None = None,  # server `code` verbatim
        # Two meanings. An error from the Synthesia API: the failing request's id (Request-Id header, else
        # body requestId), also prefixed in square brackets to the message. Any other error raised after the
        # session was created (the avatar left, a swap failed, the launch timed out): the request id of the
        # session's create call, the 201, added by `with_request`; the message then has no bracket prefix.
        request_id: str | None = None,
        session_id: str | None = None,
        detail: str | None = None,  # server `detail` verbatim
        body: Mapping[str, Any] | None = None,  # parsed problem body, redacted
    ) -> None:
        super().__init__(message)
        if category is None:
            if type is ErrorType.CLIENT_UPGRADE_REQUIRED:
                category = ErrorCategory.INVALID_REQUEST
            elif status is not None:
                category = classify_http_status_code(status)
            else:
                category = DEFAULT_CATEGORY[type]
        self.message = message
        self.type = type
        self.category = category
        self.retryable = (type in RETRYABLE_TYPES) if retryable is None else retryable
        self.retry_after = retry_after
        self.status = status
        self.code = code
        self.request_id = request_id
        self.session_id = session_id
        self.detail = detail
        self.body = body

    @property
    def permanent(self) -> bool:
        return self.category.is_permanent

    def with_session(self, session_id: str | None) -> SynthesiaError:
        """Set ``session_id`` unless one is already set, and return the error.

        One of the changes made to an error after it is built (the others are `with_request` and
        `with_hint`): the session and the service stamp errors raised by code that does not know the session
        id, such as the RPC and readiness helpers.
        """
        if self.session_id is None and session_id is not None:
            self.session_id = session_id
        return self

    def with_request(self, request_id: str | None) -> SynthesiaError:
        """Set ``request_id`` unless one is already set, and return the error: the same rule as
        `with_session`, for the request id of the Synthesia API's 201."""
        if self.request_id is None and request_id is not None:
            self.request_id = request_id
        return self

    def with_hint(self, hint: str) -> SynthesiaError:
        """Append ``hint``, one or more sentences, to the message, and return the error.

        For the layer that knows what a generic Synthesia API error most likely means: the API client knows
        the avatar ids it sent, and the session knows whether the room is borrowed. Nothing else changes, so
        the Synthesia API's own ``detail`` and the ``Request-Id`` prefix stay in the message.
        """
        separator = " " if self.message.endswith((".", "!", "?")) else ". "
        self.message = f"{self.message}{separator}{hint}"
        self.args = (self.message,)
        return self

    def __repr__(self) -> str:
        return (
            f"SynthesiaError({self.message!r}, type={self.type.value!r}, status={self.status!r}, "
            f"code={self.code!r}, request_id={self.request_id!r}, session_id={self.session_id!r})"
        )


# The Synthesia API's error codes. A code not listed here maps to UNKNOWN, with the server's `detail` kept as
# it is.
CODE_TABLE: Mapping[str, ErrorType] = {
    "bad_request": ErrorType.INVALID_SESSION_REQUEST,
    "validation_error": ErrorType.INVALID_SESSION_REQUEST,
    "not_found": ErrorType.INVALID_SESSION_REQUEST,
    "method_not_allowed": ErrorType.INVALID_SESSION_REQUEST,
    "conflict": ErrorType.INVALID_SESSION_REQUEST,
    "payload_too_large": ErrorType.INVALID_SESSION_REQUEST,
    "unsupported_media_type": ErrorType.INVALID_SESSION_REQUEST,
    "unauthenticated": ErrorType.AUTH,
    "forbidden": ErrorType.AUTH,
    "insufficient_scope": ErrorType.AUTH,
    "feature_not_in_plan": ErrorType.FEATURE_NOT_IN_PLAN,
    "payment_required": ErrorType.QUOTA_EXCEEDED,
    "quota_exceeded": ErrorType.QUOTA_EXCEEDED,
    "unknown_reference": ErrorType.UNKNOWN_AVATAR,
    "concurrency_limit": ErrorType.CONCURRENCY_LIMIT,
    "rate_limited": ErrorType.RATE_LIMITED,
    "internal": ErrorType.CONNECTION,
    "service_unavailable": ErrorType.CONNECTION,
    "timeout": ErrorType.TIMEOUT,
    "client_upgrade_required": ErrorType.CLIENT_UPGRADE_REQUIRED,
}

# How much of a body that is not problem details goes into the message.
_EXCERPT_CHARS = 200

# Types for which a Retry-After / retryAfter value is meaningful and therefore kept on the error, 5xx
# included (*installed-version channels*).
_RETRY_AFTER_TYPES = frozenset(
    {ErrorType.RATE_LIMITED, ErrorType.CONCURRENCY_LIMIT, ErrorType.CONNECTION, ErrorType.TIMEOUT}
)


def _status_fallback(status: int) -> ErrorType:
    """The type of an error response with no ``code``.

    The Synthesia API always sends a code, so this is a response from something else: a wrong
    ``api_url``, a proxy, or a CDN page. A 404 or 403 from one of those says nothing about avatars or API
    keys, so most statuses give ``UNKNOWN``. Two keep their meaning whoever sends them: a 429 is rate
    limiting (``RATE_LIMITED``, with its ``Retry-After`` kept), and a 5xx is a server or gateway failure
    that may pass (``TIMEOUT`` for a 504, else ``CONNECTION``), so both stay retryable.
    """
    if status == 429:
        return ErrorType.RATE_LIMITED
    if status == 504:
        return ErrorType.TIMEOUT
    if 500 <= status < 600:
        return ErrorType.CONNECTION
    return ErrorType.UNKNOWN


def _scrub(text: str, secrets: Sequence[str]) -> str:
    """``text`` with each of ``secrets`` replaced by the redaction placeholder (*redaction rule*)."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, redact_secret(secret))
    return text


def _scrub_deep(value: Any, secrets: Sequence[str]) -> Any:
    """``value`` with every string in it scrubbed of ``secrets``, at any depth, keys included.

    A new structure: mappings become dicts, lists and tuples become lists, and ``value`` itself is never
    mutated. Anything else, such as a number, passes through unchanged.
    """
    if isinstance(value, str):
        return _scrub(value, secrets)
    if isinstance(value, Mapping):
        return {_scrub_deep(key, secrets): _scrub_deep(item, secrets) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub_deep(item, secrets) for item in value]
    return value


def _excerpt(text: str | None) -> str | None:
    """The start of ``text``, on one line and at most ``_EXCERPT_CHARS`` long; None if it is blank."""
    flat = " ".join((text or "").split())
    if not flat:
        return None
    return flat if len(flat) <= _EXCERPT_CHARS else flat[: _EXCERPT_CHARS - 1] + "…"


def _codeless_message(status: int, summary: str | None, text: str | None) -> str:
    """The message for an error response with no ``code``: the status, and what the body said.

    ``summary`` is the body's ``detail`` or ``title``, if it has either. Otherwise the body is not problem
    details, and its start goes into the message instead.
    """
    message = f"Synthesia request failed (HTTP {status}, with no Synthesia error code)"
    if summary:
        message += f": {summary}"
    elif excerpt := _excerpt(text):
        message += f"; the response began: {excerpt}"
    if not 500 <= status < 600:
        message += (
            ". The Synthesia API always sends a code, so something else most likely answered: check "
            f"api_url (or {ENV_API_URL}), and any proxy between the bot process and the Synthesia API"
        )
    return message


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive lookup; `headers` may be a plain dict (tests) or aiohttp's own mapping."""
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _str_field(body: Mapping[str, Any] | None, key: str) -> str | None:
    if not isinstance(body, Mapping):
        return None
    value = body.get(key)
    return value if isinstance(value, str) else None


def _parse_retry_after(headers: Mapping[str, str], body: Mapping[str, Any] | None) -> float | None:
    raw = _header(headers, "Retry-After")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            return None  # HTTP-date form: dropped, not parsed
    if isinstance(body, Mapping):
        for key in ("retryAfter", "retry_after"):
            value = body.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    return None


def error_from_problem(
    status: int,
    body: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    *,
    session_id: str | None = None,
    text: str | None = None,
    secrets: Sequence[str] = (),
) -> SynthesiaError:
    """Build the error for a non-2xx response.

    ``body`` is the parsed JSON object, if the body was one, and ``text`` the body as text, which a
    response with no ``code`` quotes from. ``secrets`` are the values sent with the request; they are
    scrubbed from every string in the body, at any depth, and from the quoted text, in case whatever
    answered echoed them (*redaction rule*). That applies to a response with a code too: the Synthesia API
    has no reason to echo a secret, but the body is stored on the error, parts of it reach the message and
    a WARNING log line, and scrubbing costs nothing.
    """
    if isinstance(body, Mapping):
        # Scrubbed first, and every field below is read from the scrubbed copy, so no key the secret is
        # echoed under, however deep, can carry it into the message, the error or its `body`.
        body = _scrub_deep(body, secrets)
    code = _str_field(body, "code")
    type_ = CODE_TABLE.get(code, ErrorType.UNKNOWN) if code else _status_fallback(status)

    request_id = _header(headers, REQUEST_ID_HEADER) or _str_field(body, "requestId")

    detail = _str_field(body, "detail")
    title = _str_field(body, "title")
    if code:
        message = detail or title or f"Synthesia request failed ({code})"
    else:
        # scrubbed before the excerpt is cut, so that not even the start of a secret survives
        message = _codeless_message(status, detail or title, _scrub(text, secrets) if text else None)

    if type_ is ErrorType.FEATURE_NOT_IN_PLAN:
        features = body.get("features") if isinstance(body, Mapping) else None
        if isinstance(features, list):
            message += f" (missing features: {', '.join(str(f) for f in features)})"
    elif type_ is ErrorType.CLIENT_UPGRADE_REQUIRED:
        # the version gate, which must tell a stale install what to upgrade to (*installed-version channels*)
        min_version = body.get("min_supported_version") if isinstance(body, Mapping) else None
        upgrade_url = body.get("upgrade_url") if isinstance(body, Mapping) else None
        if min_version or upgrade_url:
            parts = []
            if min_version:
                parts.append(f"minimum supported version {min_version}")
            if upgrade_url:
                parts.append(f"see {upgrade_url}")
            message += " — " + "; ".join(parts)

    if request_id:
        message = f"[{request_id}] {message}"

    retry_after = _parse_retry_after(headers, body)
    if type_ not in _RETRY_AFTER_TYPES:
        retry_after = None

    redacted_body = redact_mapping(body) if isinstance(body, Mapping) else None

    return SynthesiaError(
        message,
        type=type_,
        retry_after=retry_after,
        status=status,
        code=code,
        request_id=request_id,
        session_id=session_id,
        detail=detail,
        body=redacted_body,
    )


def error_from_unread_body(status: int, headers: Mapping[str, str], exc: BaseException) -> SynthesiaError:
    """Build the error for a non-2xx response whose body could not be read (``exc`` says why).

    Only the status line and headers arrived, so the type falls back on the status, as for a response with
    no ``code``, and the request id and ``Retry-After`` come from the headers. The message says the body
    was lost rather than pointing at ``api_url``: a body that broke off says nothing about who sent it.
    """
    type_ = _status_fallback(status)
    request_id = _header(headers, REQUEST_ID_HEADER)
    message = (
        f"Synthesia request failed (HTTP {status}), and its response body could not be read "
        f"({describe_exception(exc)})"
    )
    if request_id:
        message = f"[{request_id}] {message}"
    retry_after = _parse_retry_after(headers, None) if type_ in _RETRY_AFTER_TYPES else None
    return SynthesiaError(message, type=type_, retry_after=retry_after, status=status, request_id=request_id)


def error_from_exception(
    exc: BaseException, *, message: str, session_id: str | None = None
) -> SynthesiaError:
    full_message = f"{message}: {type(exc).__name__}: {exc}"
    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return SynthesiaError(full_message, type=ErrorType.TIMEOUT, session_id=session_id)
    if isinstance(exc, aiohttp.ClientError | OSError):
        return SynthesiaError(full_message, type=ErrorType.CONNECTION, session_id=session_id)
    return SynthesiaError(
        full_message, type=ErrorType.UNKNOWN, category=ErrorCategory.APPLICATION, session_id=session_id
    )
