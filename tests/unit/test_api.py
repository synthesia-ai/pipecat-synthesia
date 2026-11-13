"""`api.py`: `SessionApiClient.headers()` and `create_session()`."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import warnings

import aiohttp
import pytest

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient, SessionCreated
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.protocol import SESSIONS_PATH
from tests.fakes.fake_api import RecordedRequest, ScriptedResponse
from tests.support.secrets import API_KEY, LK_TOKEN_MARKER, assert_no_secrets
from tests.support.waiting import until

CLIENT = build_client_info()
USER_AGENT_PATTERN = r"^pipecat-synthesia/\S+ pipecat-ai/\S+ python/\S+ \(.+\)$"


def _make_client(http_session: aiohttp.ClientSession, api_url: str, **overrides) -> SessionApiClient:
    kwargs = dict(
        api_key=API_KEY,
        api_url=api_url,
        client=CLIENT,
        http_session=http_session,
        request_timeout=5.0,
        max_attempts=3,
        backoff_seconds=(0.01, 0.01, 0.01),
        retry_after_cap=10.0,
    )
    kwargs.update(overrides)
    return SessionApiClient(**kwargs)


async def _create(client: SessionApiClient) -> SessionCreated:
    return await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")


def _held_open(gate: asyncio.Event) -> ScriptedResponse:
    """Not answered until `gate` is set, so any request timeout beats it, however short."""

    async def hold(_request: RecordedRequest) -> None:
        await gate.wait()

    return ScriptedResponse(201, {"id": "ses_held"}, before=hold)


# --- headers() -----------------------------------------------------------------------------------


@pytest.mark.unit
async def test_headers_exact() -> None:
    async with aiohttp.ClientSession() as session:
        headers = _make_client(session, "https://example.invalid").headers()

    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert re.match(USER_AGENT_PATTERN, headers["User-Agent"]), headers["User-Agent"]
    parsed = json.loads(headers["X-Synthesia-Client"])
    assert len(parsed) == 8, parsed
    assert headers["Accept"] == "application/json, application/problem+json"
    assert headers["Content-Type"] == "application/json"
    # API_VERSION_HEADER is None -> no other headers
    assert set(headers) == {"Authorization", "User-Agent", "X-Synthesia-Client", "Accept", "Content-Type"}


# --- request body ----------------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_session_body_exact(fake_api_factory) -> None:
    async with fake_api_factory([ScriptedResponse(201, {"id": "ses_1"})]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            await client.create_session(
                avatar_ids=["av_a", "av_b"], livekit_url="wss://x", livekit_token="tok"
            )

    assert len(api.requests) == 1
    recorded: RecordedRequest = api.requests[0]
    assert recorded.path == SESSIONS_PATH
    assert recorded.json == {"avatarIds": ["av_a", "av_b"], "livekitUrl": "wss://x", "livekitToken": "tok"}


# --- success parsing ---------------------------------------------------------------------------------


@pytest.mark.unit
async def test_success_parse_id_and_request_id(fake_api_factory) -> None:
    """The session id is read from `id`; other keys, known or not, don't get in the way."""
    body = {
        "id": "ses_1",
        "requestId": "req_1",
        "warnings": [{"code": "w1", "message": "m"}, "not-a-dict"],
        "features": {"f": 1},
        "somethingNew": [1, 2],
    }
    async with fake_api_factory([ScriptedResponse(201, body)]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            result = await client.create_session(
                avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok"
            )

    assert result.session_id == "ses_1"
    assert result.request_id == "req_1"


@pytest.mark.unit
@pytest.mark.parametrize("status", [200, 201, 202, 299])
async def test_every_2xx_status_with_a_session_id_is_a_created_session(fake_api_factory, status: int) -> None:
    async with fake_api_factory([ScriptedResponse(status, {"id": "ses_1"})]) as api:
        async with aiohttp.ClientSession() as session:
            result = await _create(_make_client(session, api.base_url))

    assert result.session_id == "ses_1"
    assert len(api.requests) == 1


@pytest.mark.unit
async def test_a_300_status_is_an_error_even_with_a_session_id_in_the_body(fake_api_factory) -> None:
    async with fake_api_factory([ScriptedResponse(300, {"id": "ses_1"})]) as api:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(SynthesiaError) as excinfo:
                await _create(_make_client(session, api.base_url))

    assert excinfo.value.status == 300
    assert len(api.requests) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="missing"),
        pytest.param({"id": ""}, id="empty-id"),
        pytest.param({"id": "", "session_id": ""}, id="empty-id-and-empty-fallback"),
        pytest.param({"id": None}, id="null"),
        pytest.param({"id": 123}, id="number"),
        pytest.param({"id": ["ses_1"]}, id="list"),
        pytest.param({"session_id": 123}, id="number-in-fallback-field"),
        pytest.param([{"id": "ses_1"}], id="body-is-a-list"),
        pytest.param("this is not json", id="body-is-not-json"),
    ],
)
async def test_a_201_without_a_usable_session_id_is_an_unknown_error_and_not_retried(
    fake_api_factory, body: object
) -> None:
    script = [ScriptedResponse(201, body, headers={"Request-Id": "req_from_header"})]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(SynthesiaError) as excinfo:
                await _create(_make_client(session, api.base_url))

    assert excinfo.value.type is ErrorType.UNKNOWN
    assert "session id" in str(excinfo.value)
    assert excinfo.value.status == 201
    assert excinfo.value.request_id == "req_from_header"
    assert len(api.requests) == 1


# --- retry policy ------------------------------------------------------------------------------------


@pytest.mark.unit
async def test_no_retry_on_4xx(fake_api_factory) -> None:
    async with fake_api_factory([ScriptedResponse(429, {"code": "concurrency_limit", "detail": "d"})]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    assert len(api.requests) == 1
    assert excinfo.value.type is ErrorType.CONCURRENCY_LIMIT
    assert excinfo.value.retryable is True


@pytest.mark.unit
@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 409, 429, 501, 505])
async def test_any_other_error_status_is_not_retried(fake_api_factory, status: int) -> None:
    script = [ScriptedResponse(status), ScriptedResponse(201, {"id": "ses_1"})]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(SynthesiaError) as excinfo:
                await _create(_make_client(session, api.base_url, max_attempts=4))

    assert len(api.requests) == 1
    assert excinfo.value.status == status


@pytest.mark.unit
async def test_retry_on_5xx_same_token(fake_api_factory) -> None:
    script = [ScriptedResponse(502), ScriptedResponse(500), ScriptedResponse(201, {"id": "ses_1"})]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            result = await client.create_session(
                avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok-same"
            )

    assert result.session_id == "ses_1"
    assert len(api.requests) == 3
    assert {req.json["livekitToken"] for req in api.requests} == {"tok-same"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script", "retries"),
    [
        ([ScriptedResponse(201, {"id": "ses_1"})], 0),
        ([ScriptedResponse(502), ScriptedResponse(201, {"id": "ses_1"})], 1),
        ([ScriptedResponse(502), ScriptedResponse(500), ScriptedResponse(201, {"id": "ses_1"})], 2),
    ],
    ids=["no-retry", "one-5xx", "two-5xx"],
)
async def test_on_retry_is_called_once_per_retry(fake_api_factory, script, retries: int) -> None:
    """`on_retry` marks each retry, before its wait: the failed attempt may have launched a worker that
    the retry's worker, with the same token, replaces (*early start* stops trusting it)."""
    calls: list[int] = []
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            result = await client.create_session(
                avatar_ids=["av_a"],
                livekit_url="wss://x",
                livekit_token="tok",
                on_retry=lambda: calls.append(len(api.requests)),
            )

    assert result.session_id == "ses_1"
    assert calls == list(range(1, retries + 1)), "once per retry, after the failed attempt, before the next"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script", "max_attempts", "retries"),
    [
        ([ScriptedResponse(429, {"code": "concurrency_limit", "detail": "d"})], 3, 0),
        ([ScriptedResponse(503)], 2, 1),
    ],
    ids=["4xx-not-retried", "the-last-failed-attempt-is-not-a-retry"],
)
async def test_on_retry_is_not_called_for_a_failure_that_is_not_retried(
    fake_api_factory, script, max_attempts: int, retries: int
) -> None:
    calls: list[int] = []
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, max_attempts=max_attempts)
            with pytest.raises(SynthesiaError):
                await client.create_session(
                    avatar_ids=["av_a"],
                    livekit_url="wss://x",
                    livekit_token="tok",
                    on_retry=lambda: calls.append(1),
                )

    assert len(calls) == retries


@pytest.mark.unit
async def test_on_retry_is_called_for_a_request_timeout(fake_api_factory) -> None:
    gate = asyncio.Event()
    calls: list[int] = []
    script = [_held_open(gate), ScriptedResponse(201, {"id": "ses_ok"})]
    async with fake_api_factory(script) as api:
        try:
            async with aiohttp.ClientSession() as session:
                client = _make_client(session, api.base_url, request_timeout=1.0, max_attempts=3)
                await client.create_session(
                    avatar_ids=["av_a"],
                    livekit_url="wss://x",
                    livekit_token="tok",
                    on_retry=lambda: calls.append(1),
                )
        finally:
            gate.set()

    assert calls == [1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (500, None, ErrorType.CONNECTION),
        (500, {"code": "internal", "detail": "d"}, ErrorType.CONNECTION),
        (502, None, ErrorType.CONNECTION),
        (503, None, ErrorType.CONNECTION),
        (504, None, ErrorType.TIMEOUT),
        (504, {"code": "timeout", "detail": "d"}, ErrorType.TIMEOUT),
    ],
)
async def test_a_retryable_status_is_retried_until_the_attempts_are_spent(
    fake_api_factory, status: int, body: dict | None, expected: ErrorType
) -> None:
    async with fake_api_factory([ScriptedResponse(status, body)]) as api:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(SynthesiaError) as excinfo:
                await _create(_make_client(session, api.base_url, max_attempts=4))

    assert len(api.requests) == 4
    assert excinfo.value.status == status
    assert excinfo.value.type is expected
    assert excinfo.value.retryable is True


@pytest.mark.unit
async def test_a_backoff_list_shorter_than_the_attempts_is_reused_after_a_retryable_status(
    fake_api_factory,
) -> None:
    async with fake_api_factory([ScriptedResponse(503)]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, backoff_seconds=(0.01,), max_attempts=5)
            with pytest.raises(SynthesiaError) as excinfo:
                await _create(client)

    assert len(api.requests) == 5
    assert excinfo.value.status == 503


@pytest.mark.unit
async def test_a_backoff_list_shorter_than_the_attempts_is_reused_after_a_request_timeout(
    fake_api_factory,
) -> None:
    gate = asyncio.Event()
    async with fake_api_factory([_held_open(gate)]) as api:
        try:
            async with aiohttp.ClientSession() as session:
                client = _make_client(
                    session, api.base_url, request_timeout=0.05, backoff_seconds=(0.01,), max_attempts=4
                )
                with pytest.raises(SynthesiaError) as excinfo:
                    await _create(client)
        finally:
            gate.set()

    assert len(api.requests) == 4
    assert excinfo.value.type is ErrorType.TIMEOUT


@pytest.mark.unit
async def test_a_request_timeout_is_retried_and_a_later_success_returns_the_session(fake_api_factory) -> None:
    gate = asyncio.Event()
    script = [_held_open(gate), ScriptedResponse(201, {"id": "ses_ok"})]
    async with fake_api_factory(script) as api:
        try:
            async with aiohttp.ClientSession() as session:
                # The held request times out after a second; the answered one has that full second to arrive.
                result = await _create(
                    _make_client(session, api.base_url, request_timeout=1.0, max_attempts=3)
                )
        finally:
            gate.set()

    assert result.session_id == "ses_ok"
    assert len(api.requests) == 2


@pytest.mark.unit
async def test_the_next_attempt_waits_for_the_backoff_after_a_request_timeout(fake_api_factory) -> None:
    """The backoff is a huge input; the assertion is only that no second request arrives while it runs."""
    gate = asyncio.Event()
    script = [_held_open(gate), ScriptedResponse(201, {"id": "ses_1"})]
    async with fake_api_factory(script) as api:
        try:
            async with aiohttp.ClientSession() as session:
                client = _make_client(
                    session, api.base_url, request_timeout=0.05, backoff_seconds=(60.0,), max_attempts=2
                )
                task = asyncio.ensure_future(_create(client))
                try:
                    await until(lambda: len(api.requests) == 1, timeout=5.0, what="the first request")
                    done, _pending = await asyncio.wait({task}, timeout=0.5)
                    assert not done, "the client retried without waiting for its backoff"
                    assert len(api.requests) == 1
                finally:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
        finally:
            gate.set()


@pytest.mark.unit
async def test_retry_exhaustion_forces_retryable_even_for_a_normally_non_retryable_type(
    fake_api_factory,
) -> None:
    """A 503 whose body `code` maps to a non-retryable type is still retried by *status*, and the
    error raised on exhaustion is still `retryable=True` — the client tried and gave up, regardless
    of what the mapped `ErrorType`'s own default retryability would otherwise say."""
    script = [ScriptedResponse(503, {"code": "bad_request", "detail": "d"})]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, max_attempts=3)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    assert len(api.requests) == 3
    assert excinfo.value.type is ErrorType.INVALID_SESSION_REQUEST
    assert excinfo.value.retryable is True


@pytest.mark.unit
async def test_retry_after_honoured_and_capped(fake_api_factory) -> None:
    script = [
        ScriptedResponse(503, {"code": "service_unavailable", "detail": "d"}, headers={"Retry-After": "30"}),
        ScriptedResponse(201, {"id": "ses_1"}),
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, backoff_seconds=(0.01,), retry_after_cap=0.01)
            task = asyncio.ensure_future(
                client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")
            )
            # Durations are inputs, never assertions: if the 30 s were honoured verbatim, this `until`
            # would time out well before pytest's own 30 s backstop.
            await until(lambda: len(api.requests) == 2, timeout=5.0, what="a second (capped) request")
            result = await task

    assert result.session_id == "ses_1"


@pytest.mark.unit
async def test_retry_stops_when_cancelled_mid_backoff(fake_api_factory) -> None:
    script = [
        ScriptedResponse(503, {"code": "service_unavailable", "detail": "d"}),
        ScriptedResponse(201, {"id": "ses_1"}),
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, backoff_seconds=(5.0,), max_attempts=3)
            task = asyncio.ensure_future(
                client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")
            )
            await until(lambda: len(api.requests) == 1, timeout=5.0, what="the first request")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(AssertionError):
                await api.wait_for_requests(2, timeout=0.2)

    assert len(api.requests) == 1


# --- exceptions --------------------------------------------------------------------------------------


@pytest.mark.unit
async def test_timeout_maps_to_TIMEOUT(fake_api_factory) -> None:
    async with fake_api_factory([ScriptedResponse(201, {"id": "ses_1"}, delay=1.0)]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, request_timeout=0.2, max_attempts=1)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    assert excinfo.value.type is ErrorType.TIMEOUT


@pytest.mark.unit
async def test_cancellation_propagates(fake_api_factory) -> None:
    async with fake_api_factory([ScriptedResponse(201, {"id": "ses_1"}, delay=5.0)]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, request_timeout=10.0, max_attempts=1)
            task = asyncio.ensure_future(
                client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")
            )
            await until(lambda: len(api.requests) == 1, timeout=5.0, what="the request to arrive")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# --- advisories and logging --------------------------------------------------------------------------


@pytest.mark.unit
async def test_advisories_logged_once(fake_api_factory, caplog_loguru) -> None:
    body = {
        "id": "ses_1",
        "warnings": [
            {"code": "w1", "message": "deprecated field", "sunset": "2027-01-01", "url": "https://x"}
        ],
    }
    script = [ScriptedResponse(201, body, headers={"Deprecation": "true"}) for _ in range(2)]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    assert recorded == []
    advisory_lines = [r.message for r in caplog_loguru.records if "advisory" in r.message]
    warning_lines = [r.message for r in caplog_loguru.records if "w1" in r.message]
    assert len(advisory_lines) == 1, advisory_lines
    assert len(warning_lines) == 1, warning_lines


@pytest.mark.unit
async def test_a_deprecation_link_header_raises_an_advisory_and_another_link_does_not(
    fake_api_factory, caplog_loguru
) -> None:
    script = [
        ScriptedResponse(201, {"id": "ses_1"}, headers={"Link": '<https://x/next>; rel="next"'}),
        ScriptedResponse(201, {"id": "ses_1"}, headers={"Link": '<https://x/migrate>; rel="deprecation"'}),
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            await _create(client)
            before = [r for r in caplog_loguru.records if "advisory" in r.getMessage()]
            await _create(client)

    after = [
        r for r in caplog_loguru.records if r.levelno == logging.WARNING and "advisory" in r.getMessage()
    ]
    assert before == []
    assert len(after) == 1


@pytest.mark.unit
async def test_outcome_log_carries_code_and_detail(fake_api_factory, caplog_loguru) -> None:
    script = [ScriptedResponse(400, {"code": "some_unknown_code", "detail": "a very particular detail"})]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, max_attempts=1)
            with pytest.raises(SynthesiaError):
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    text = caplog_loguru.text
    assert "some_unknown_code" in text
    assert "a very particular detail" in text
    assert "type=unknown" in text


@pytest.mark.unit
async def test_failed_attempts_log_at_warning_and_success_at_info(fake_api_factory, caplog_loguru) -> None:
    """A failed session create is worth a developer's attention, so every failed attempt is a WARNING;
    the successful one stays INFO."""
    script = [
        ScriptedResponse(503, {"code": "service_unavailable", "detail": "d"}),
        ScriptedResponse(201, {"id": "ses_1"}),
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    outcomes = {
        record.getMessage().split()[3]: record.levelno
        for record in caplog_loguru.records
        if record.getMessage().startswith("synthesia session create")
    }
    assert outcomes == {"error": logging.WARNING, "ok": logging.INFO}, caplog_loguru.text


@pytest.mark.unit
async def test_unknown_avatar_message_names_the_ids_and_the_studio_trap(fake_api_factory) -> None:
    """The most likely setup mistake gets an actionable message: the ids that were sent, and that a Studio
    id is not an Interactive Avatar id. Everything else about the typed error is what the Synthesia API
    said."""
    detail = "One or more of avatarIds does not exist or is not accessible."
    script = [
        ScriptedResponse(
            404, {"code": "unknown_reference", "detail": detail}, headers={"Request-Id": "req_404"}
        )
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(
                    avatar_ids=["av_first", "av_second"], livekit_url="wss://x", livekit_token="tok"
                )

    err = excinfo.value
    message = str(err)
    assert "av_first" in message
    assert "av_second" in message
    assert "Studio" in message
    assert "Interactive Avatar id" in message
    assert message.startswith("[req_404]")
    assert detail in message
    assert err.message == message
    assert err.type is ErrorType.UNKNOWN_AVATAR
    assert (err.status, err.code, err.detail, err.request_id) == (404, "unknown_reference", detail, "req_404")
    assert err.retryable is False
    assert len(api.requests) == 1


# --- a response with no problem-details code: something other than the Synthesia API answered -------


@pytest.mark.unit
@pytest.mark.parametrize(
    "status,body",
    [
        (404, "<html><head><title>404 Not Found</title></head><body>nginx</body></html>"),
        (404, None),
        (403, "<html><body><h1>403 Forbidden</h1>Request blocked.</body></html>"),
        (403, None),
    ],
)
async def test_codeless_4xx_is_unknown_with_the_status_and_no_avatar_hint(
    fake_api_factory, status, body
) -> None:
    """A wrong api_url, a proxy or a CDN page answers without a code. That is not an unknown avatar or a
    bad API key, so the error is UNKNOWN, names the status and quotes the body, carries no Studio-avatar
    hint, and is not retried."""
    async with fake_api_factory([ScriptedResponse(status, body)]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    err = excinfo.value
    assert err.type is ErrorType.UNKNOWN
    assert err.status == status
    assert f"HTTP {status}" in str(err)
    if body is not None:
        assert "Not Found" in str(err) or "Forbidden" in str(err)
    assert "api_url" in str(err)
    assert "Studio" not in str(err)
    assert "av_a" not in str(err)
    assert err.retryable is False
    assert len(api.requests) == 1


@pytest.mark.unit
@pytest.mark.parametrize("status", [502, 503])
async def test_codeless_5xx_with_a_non_json_body_is_still_connection_and_retried(
    fake_api_factory, status: int
) -> None:
    async with fake_api_factory([ScriptedResponse(status, "<html>Service Unavailable</html>")]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, max_attempts=3)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    assert len(api.requests) == 3
    assert (excinfo.value.status, excinfo.value.type) == (status, ErrorType.CONNECTION)
    assert excinfo.value.retryable is True
    assert excinfo.value.body is None
    assert f"HTTP {status}" in str(excinfo.value)


@pytest.mark.unit
async def test_codeless_429_stays_rate_limited_with_its_retry_after(fake_api_factory) -> None:
    """A 429 means rate limiting whoever sends it, so a code-less one keeps RATE_LIMITED and its
    Retry-After. Like any 4xx it is not retried here."""
    script = [ScriptedResponse(429, "Too Many Requests", headers={"Retry-After": "7"})]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    err = excinfo.value
    assert err.type is ErrorType.RATE_LIMITED
    assert err.retryable is True
    assert err.retry_after == 7.0
    assert "HTTP 429" in str(err)
    assert len(api.requests) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "echo",
    [
        f"Authorization: Bearer {API_KEY}\nlivekitToken={LK_TOKEN_MARKER}",
        {"detail": f"Authorization: Bearer {API_KEY}", "title": f"livekitToken={LK_TOKEN_MARKER}"},
    ],
    ids=["text", "json-detail"],
)
async def test_codeless_body_that_echoes_the_request_leaks_nothing(
    fake_api_factory, caplog_loguru, echo
) -> None:
    """Something that echoes the request back (a debugging proxy, say) returns the API key and the LiveKit
    token in its body, as text or as a JSON `detail`. The message, `detail`, the body and the WARNING
    session-create line (which carries `detail`) scrub both (*redaction rule*)."""
    async with fake_api_factory([ScriptedResponse(400, echo)]) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(
                    avatar_ids=["av_a"], livekit_url="wss://x", livekit_token=LK_TOKEN_MARKER
                )

    err = excinfo.value
    assert "Authorization: Bearer" in str(err), "the body is quoted"
    assert "synthesia session create error" in caplog_loguru.text, "the WARNING line was written"
    assert_no_secrets(
        caplog_loguru.text,
        str(err),
        repr(err),
        repr(err.body),
        err.detail or "",
        secrets=(API_KEY, LK_TOKEN_MARKER),
    )


@pytest.mark.unit
async def test_no_secrets_in_logs_or_errors(fake_api_factory, caplog_loguru) -> None:
    script = [
        ScriptedResponse(400, {"code": "bad_request", "detail": "d"}),
        ScriptedResponse(201, {"id": "ses_1"}),
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = _make_client(session, api.base_url, max_attempts=1)
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(
                    avatar_ids=["av_a"], livekit_url="wss://x", livekit_token=LK_TOKEN_MARKER
                )
            await client.create_session(
                avatar_ids=["av_a"], livekit_url="wss://x", livekit_token=LK_TOKEN_MARKER
            )

    err = excinfo.value
    assert_no_secrets(
        caplog_loguru.text, str(err), repr(err), repr(err.body), secrets=(API_KEY, LK_TOKEN_MARKER)
    )
