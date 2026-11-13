"""A redirect in answer to the session request is not followed: it fails the request, and is not retried.

The request carries the API key in its `Authorization` header and a LiveKit token in its body. Following a
3xx would send the request on to whatever host its `Location` names, the body too for a 307 or 308. So a
3xx is an error response like a 4xx: a redirect has no Synthesia error code, so it is `UNKNOWN`, names the
status, and is not retried; and nothing reaches the host it points at.

`SessionApiClient` is internal; it is tested directly, against two scripted HTTP servers, because the
public route to it is a room adapter's launch, which would add a LiveKit room and nothing to what is
checked here.
"""

from __future__ import annotations

import aiohttp
import pytest

from pipecat_synthesia import ErrorType, SynthesiaError
from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from pipecat_synthesia.protocol import SESSIONS_PATH
from tests.fakes.fake_api import ScriptedResponse
from tests.support.secrets import API_KEY, LK_TOKEN_MARKER

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_a_redirect_fails_the_request_and_nothing_reaches_its_target(
    fake_api_factory, status: int
) -> None:
    async with fake_api_factory([ScriptedResponse(201, {"id": "ses_elsewhere"})]) as target:
        location = {"Location": target.base_url + SESSIONS_PATH}
        async with fake_api_factory([ScriptedResponse(status, headers=location)]) as api:
            async with aiohttp.ClientSession() as session:
                client = SessionApiClient(
                    api_key=API_KEY,
                    api_url=api.base_url,
                    client=build_client_info(),
                    http_session=session,
                    max_attempts=3,
                    backoff_seconds=(0.01,),
                )
                with pytest.raises(SynthesiaError) as excinfo:
                    await client.create_session(
                        avatar_ids=["av_a"], livekit_url="wss://x", livekit_token=LK_TOKEN_MARKER
                    )

    assert target.requests == [], "the redirect was followed"
    err = excinfo.value
    assert err.type is ErrorType.UNKNOWN
    assert err.status == status
    assert f"HTTP {status}" in str(err)
    assert err.retryable is False
    assert len(api.requests) == 1, "the redirect was retried"
