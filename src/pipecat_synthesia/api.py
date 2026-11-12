"""The Synthesia API client: creates sessions, with retries, advisories and error mapping.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp

from pipecat_synthesia._client_info import ClientInfo
from pipecat_synthesia._log import get_logger
from pipecat_synthesia._once import warn_once
from pipecat_synthesia.errors import (
    ErrorType,
    SynthesiaError,
    error_from_exception,
    error_from_problem,
    error_from_unread_body,
)
from pipecat_synthesia.protocol import REQUEST_ID_HEADER, SESSIONS_PATH

if TYPE_CHECKING:
    from multidict import MultiMapping

# The advisories are these headers, a deprecation `Link` and `warnings[]` (*installed-version channels*)
_ADVISORY_HEADERS = ("Deprecation", "Sunset", "Warning")
# A `rel` link parameter (RFC 8288): its name in any case, its value a token or a quoted string of
# space-separated relation types. URIs are removed first, so a ";rel=" inside one cannot match.
_LINK_URI = re.compile(r"<[^>]*>")
_LINK_REL = re.compile(r';\s*rel\s*=\s*(?:"([^"]*)"|([^\s;,"]+))', re.IGNORECASE)

# Reserved for an API-version header, once the Synthesia API defines one (*installed-version channels*).
# This is the only place it would be added: no constructor parameter and no environment variable until then.
API_VERSION_HEADER: tuple[str, str] | None = None
RETRYABLE_STATUSES = frozenset({500, 502, 503, 504})

# Added to an UNKNOWN_AVATAR message, after the ids that were sent. It points at the README rather than
# quoting a stock avatar id: those are product data, not a stable contract.
_UNKNOWN_AVATAR_HINT = (
    'An avatar id from Synthesia Studio is not an Interactive Avatar id; see "Avatar ids" in the '
    "pipecat-synthesia README for where Interactive Avatar ids come from."
)


def _log_advisories(headers: MultiMapping[str], log) -> None:
    """Warn once per process per (header, value) for the Synthesia API's advisory headers
    (*installed-version channels*). Every field of a repeated header is read."""
    for header in _ADVISORY_HEADERS:
        for value in headers.getall(header, ()):
            warn_once(log, f"api:advisory:{header}:{value}", f"Synthesia API advisory {header}: {value}")
    for link in headers.getall("Link", ()):
        if _has_deprecation_rel(link):
            warn_once(log, f"api:advisory:Link:{link}", f"Synthesia API advisory Link: {link}")


def _has_deprecation_rel(field: str) -> bool:
    """Whether a `Link` field has a link whose relation types include `deprecation`."""
    return any(
        "deprecation" in (quoted or token).lower().split()
        for quoted, token in _LINK_REL.findall(_LINK_URI.sub("", field))
    )


def _log_body_warnings(body: Mapping[str, Any], log) -> None:
    """Warn once per process per warning code for a successful response's `warnings[]`
    (*installed-version channels*)."""
    warnings_list = body.get("warnings")
    if not isinstance(warnings_list, list):
        return
    for item in warnings_list:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        message = item.get("message")
        sunset = item.get("sunset")
        url = item.get("url")
        warn_once(
            log,
            f"api:warning:{code}",
            f"Synthesia API warning {code}: {message} (sunset {sunset}, {url})",
        )


def _log_outcome(
    log,
    *,
    outcome: str,
    session_id: str | None,
    request_id: str | None,
    status: int | None,
    attempt: int,
    duration_ms: float,
    avatar_ids: Sequence[str],
    code: str | None = None,
    detail: str | None = None,
    error_type: str | None = None,
) -> None:
    if error_type is not None:
        log.warning(
            "synthesia session create {outcome} session_id={session_id} request_id={request_id} "
            "status={status} attempt={attempt} duration_ms={duration_ms:.1f} avatar_ids={avatar_ids} "
            "code={code} detail={detail} type={type}",
            outcome=outcome,
            session_id=session_id,
            request_id=request_id,
            status=status,
            attempt=attempt,
            duration_ms=duration_ms,
            avatar_ids=list(avatar_ids),
            code=code,
            detail=detail,
            type=error_type,
        )
    else:
        log.info(
            "synthesia session create {outcome} session_id={session_id} request_id={request_id} "
            "status={status} attempt={attempt} duration_ms={duration_ms:.1f} avatar_ids={avatar_ids}",
            outcome=outcome,
            session_id=session_id,
            request_id=request_id,
            status=status,
            attempt=attempt,
            duration_ms=duration_ms,
            avatar_ids=list(avatar_ids),
        )


@dataclass(frozen=True)
class SessionCreated:
    """The parsed 201 body of a session create."""

    session_id: str  # body["id"]; non-empty str or SynthesiaError(UNKNOWN)
    request_id: str | None  # headers[REQUEST_ID_HEADER] else body.get("requestId")


class SessionApiClient:
    """Owns the HTTP conversation with the Synthesia public API. Never owns its aiohttp session."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        client: ClientInfo,
        http_session: aiohttp.ClientSession,
        request_timeout: float = 80.0,
        max_attempts: int = 3,
        backoff_seconds: Sequence[float] = (0.5, 1.0, 2.0),
        retry_after_cap: float = 10.0,
        log=None,
    ) -> None:
        # request_timeout is 80 s, longer than the Synthesia API's own 75 s deadline, so that the API's 504
        # response, with its code and Request-Id, arrives rather than a bare client timeout.
        self._api_key = api_key
        self._api_url = api_url
        self._client = client
        self._http_session = http_session
        self._request_timeout = request_timeout
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._retry_after_cap = retry_after_cap
        self._log = log

    def headers(self) -> dict[str, str]:
        """``{'Authorization': f'Bearer {api_key}', 'User-Agent': client.user_agent,
        'X-Synthesia-Client': client.as_header_json(),
        'Accept': 'application/json, application/problem+json',
        'Content-Type': 'application/json'}`` + ``API_VERSION_HEADER`` if set.

        ``User-Agent`` and ``X-Synthesia-Client`` are client identification (*installed-version channels*).
        """
        result = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self._client.user_agent,
            "X-Synthesia-Client": self._client.as_header_json(),
            "Accept": "application/json, application/problem+json",
            "Content-Type": "application/json",
        }
        if API_VERSION_HEADER is not None:
            key, value = API_VERSION_HEADER
            result[key] = value
        return result

    async def create_session(
        self,
        *,
        avatar_ids: Sequence[str],
        livekit_url: str,
        livekit_token: str,
        on_retry: Callable[[], None] | None = None,
    ) -> SessionCreated:
        """Create a session: POST ``{"avatarIds", "livekitUrl", "livekitToken"}`` to the sessions endpoint.

        Retries, with the same token, on a 5xx in ``RETRYABLE_STATUSES`` or a connection error or timeout,
        never on a 4xx, waiting ``Retry-After`` (capped) or the next backoff step. The status decides, even
        when the body that follows it cannot be read; a 2xx whose body is lost is retried like a connection
        error. A redirect is not followed: a 3xx fails like a 4xx. A launch may cancel this
        at any point, on the service's launch timeout or a direct join's failed connect; a cancelled
        request is simply abandoned.

        ``on_retry`` is called once per retry, before its wait: the failed attempt may still have launched an
        avatar worker, and the retry, with the same token, may launch a second one as the same identity
        (*early start* stops trusting the avatar in the room then).

        Advisory headers and response ``warnings[]`` are logged once each. Every failed attempt is logged
        at WARNING with the server's ``code`` and ``detail`` as sent, plus our error type; an
        ``UNKNOWN_AVATAR`` error's message also names the avatar ids sent. Raises ``SynthesiaError``
        only (``retryable=True`` once the retries run out), never an aiohttp or asyncio exception except
        ``CancelledError``.
        """
        body = {"avatarIds": list(avatar_ids), "livekitUrl": livekit_url, "livekitToken": livekit_token}
        url = self._api_url + SESSIONS_PATH
        log = self._log if self._log is not None else get_logger()

        for attempt in range(self._max_attempts):
            start = time.monotonic()
            try:
                async with self._http_session.post(
                    url,
                    json=body,
                    headers=self.headers(),
                    timeout=aiohttp.ClientTimeout(total=self._request_timeout),
                    # A redirect is not followed: a 307 or 308 would resend the body, LiveKit token included,
                    # to whatever host it names. A 3xx is an error response like any other: not retried.
                    allow_redirects=False,
                ) as resp:
                    status = resp.status
                    headers = resp.headers
                    # Why the body could not be read, when it could not: the status and headers are in.
                    unread: TimeoutError | aiohttp.ClientError | None = None
                    try:
                        resp_body = await resp.json(content_type=None)
                    except (aiohttp.ContentTypeError, ValueError):
                        resp_body = None
                    except (TimeoutError, aiohttp.ClientError) as exc:
                        if 200 <= status < 300:
                            # A 2xx whose body is lost: the session may exist, but without its id it cannot
                            # be used. Retried like a connection failure, with the same token, so a second
                            # avatar worker may join as the same identity: a same-identity replacement,
                            # which the session survives (the replacement grace, and in a relayed join
                            # track capture). Not retrying would fail a launch that a retry can still save.
                            raise
                        resp_body, unread = None, exc
                    _log_advisories(headers, log)
                    duration_ms = (time.monotonic() - start) * 1000

                    if 200 <= status < 300:
                        raw: Mapping[str, Any] = resp_body if isinstance(resp_body, Mapping) else {}
                        _log_body_warnings(raw, log)
                        session_id = raw.get("id")
                        request_id = headers.get(REQUEST_ID_HEADER) or (
                            raw.get("requestId") if isinstance(raw.get("requestId"), str) else None
                        )
                        if not isinstance(session_id, str) or not session_id:
                            err = SynthesiaError(
                                "Synthesia response did not contain a session id",
                                type=ErrorType.UNKNOWN,
                                status=status,
                                request_id=request_id,
                            )
                            _log_outcome(
                                log,
                                outcome="error",
                                session_id=None,
                                request_id=request_id,
                                status=status,
                                attempt=attempt + 1,
                                duration_ms=duration_ms,
                                avatar_ids=avatar_ids,
                                code=None,
                                detail=None,
                                error_type=err.type.value,
                            )
                            raise err
                        _log_outcome(
                            log,
                            outcome="ok",
                            session_id=session_id,
                            request_id=request_id,
                            status=status,
                            attempt=attempt + 1,
                            duration_ms=duration_ms,
                            avatar_ids=avatar_ids,
                        )
                        return SessionCreated(session_id=session_id, request_id=request_id)

                    # non-2xx: the status decides the retry, even when the body is lost.
                    if unread is not None:
                        err = error_from_unread_body(status, headers, unread)
                    else:
                        # The text is for a body that is not problem details, which the message quotes.
                        try:
                            text: str | None = await resp.text(errors="replace")
                        except LookupError:  # a charset Python does not know
                            text = None
                        err = error_from_problem(
                            status, resp_body, headers, text=text, secrets=(self._api_key, livekit_token)
                        )
                    _log_outcome(
                        log,
                        outcome="error",
                        session_id=None,
                        request_id=err.request_id,
                        status=status,
                        attempt=attempt + 1,
                        duration_ms=duration_ms,
                        avatar_ids=avatar_ids,
                        code=err.code,
                        detail=err.detail,
                        error_type=err.type.value,
                    )
                    if status in RETRYABLE_STATUSES and attempt < self._max_attempts - 1:
                        # Retry-After is honoured on a 5xx too (*installed-version channels*)
                        wait = (
                            min(err.retry_after, self._retry_after_cap)
                            if err.retry_after is not None
                            else self._backoff_seconds[min(attempt, len(self._backoff_seconds) - 1)]
                        )
                        if on_retry is not None:
                            on_retry()
                        await asyncio.sleep(wait)
                        continue
                    if status in RETRYABLE_STATUSES:
                        err.retryable = True
                    if err.type is ErrorType.UNKNOWN_AVATAR:
                        err.with_hint(f"Avatar ids sent: {', '.join(avatar_ids)}. {_UNKNOWN_AVATAR_HINT}")
                    raise err
            except (TimeoutError, aiohttp.ClientError) as exc:
                duration_ms = (time.monotonic() - start) * 1000
                error_type = (
                    ErrorType.TIMEOUT
                    if isinstance(exc, asyncio.TimeoutError | TimeoutError)
                    else ErrorType.CONNECTION
                )
                _log_outcome(
                    log,
                    outcome="error",
                    session_id=None,
                    request_id=None,
                    status=None,
                    attempt=attempt + 1,
                    duration_ms=duration_ms,
                    avatar_ids=avatar_ids,
                    code=None,
                    detail=str(exc),
                    error_type=error_type.value,
                )
                if attempt < self._max_attempts - 1:
                    wait = self._backoff_seconds[min(attempt, len(self._backoff_seconds) - 1)]
                    if on_retry is not None:
                        on_retry()
                    await asyncio.sleep(wait)
                    continue
                raise error_from_exception(
                    exc, message=f"Synthesia session request failed after {self._max_attempts} attempts"
                ) from exc

        raise AssertionError("unreachable: create_session retry loop exited without returning or raising")
