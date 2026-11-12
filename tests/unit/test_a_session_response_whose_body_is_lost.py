"""A session response whose status arrives but whose body cannot be read: the status decides.

`test_session_request_body_read_failures.py` pins that a 4xx whose body stalls is final and keeps its
status and `Request-Id`. This file pins the rest of the same rule:
- the error says the body was lost, and keeps a 429's `Retry-After`, rather than sending the developer to
  check `api_url` as a response with no Synthesia error code does;
- a retryable 5xx whose body stalls is still retried, and ends as that status, retryable;
- a 2xx whose body stalls is retried with the same token, as a connection failure is: without the session
  id the session cannot be used, and a second avatar worker on the same identity is a same-identity
  replacement, which the session survives.

The stalled body is built as in that file: a `Content-Length` header larger than the body
`FakeSynthesiaApi` sends. `request_timeout` is an input, shrunk so the stalled read ends quickly.
"""

from __future__ import annotations

import aiohttp
import pytest

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from tests.fakes.fake_api import FakeSynthesiaApi, ScriptedResponse
from tests.support.secrets import API_KEY

#: Far more than any body below, so the read waits for bytes that never come.
_STALLS = {"Content-Length": "4096"}
ATTEMPTS = 3


def _client(api: FakeSynthesiaApi, session: aiohttp.ClientSession) -> SessionApiClient:
    return SessionApiClient(
        api_key=API_KEY,
        api_url=api.base_url,
        client=build_client_info(),
        http_session=session,
        request_timeout=0.2,
        max_attempts=ATTEMPTS,
        backoff_seconds=(0.01,),
        retry_after_cap=0.01,
    )


async def _create(api: FakeSynthesiaApi) -> SynthesiaError:
    async with aiohttp.ClientSession() as session:
        with pytest.raises(SynthesiaError) as excinfo:
            await _client(api, session).create_session(
                avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok"
            )
    return excinfo.value


@pytest.mark.unit
async def test_a_429_whose_body_stalls_says_so_and_keeps_its_retry_after(fake_api_factory) -> None:
    script = [
        ScriptedResponse(
            429, {"code": "rate_limited", "detail": "slow down"}, headers={**_STALLS, "Retry-After": "7"}
        )
    ]
    async with fake_api_factory(script) as api:
        err = await _create(api)

    assert len(api.requests) == 1
    assert err.type is ErrorType.RATE_LIMITED
    assert err.retry_after == 7.0
    assert "could not be read" in str(err)
    assert "api_url" not in str(err), "a lost body sent the developer to check api_url"


@pytest.mark.unit
async def test_a_503_whose_body_stalls_is_retried_and_ends_as_a_503(fake_api_factory) -> None:
    script = [ScriptedResponse(503, {"code": "unavailable"}, headers={**_STALLS, "Request-Id": "req_503"})]
    async with fake_api_factory(script) as api:
        err = await _create(api)

    assert len(api.requests) == ATTEMPTS
    assert err.status == 503
    assert err.request_id == "req_503"
    assert err.retryable is True


@pytest.mark.unit
async def test_a_2xx_whose_body_stalls_is_retried_with_the_same_token(fake_api_factory) -> None:
    script = [
        ScriptedResponse(201, {"id": "ses_lost"}, headers=_STALLS),
        ScriptedResponse(201, {"id": "ses_second"}),
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            created = await _client(api, session).create_session(
                avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok"
            )

    assert created.session_id == "ses_second"
    assert len(api.requests) == 2
    assert api.requests[0].json == api.requests[1].json, "the retry did not resend the same request"
