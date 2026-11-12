"""A session request whose error response arrives but whose body cannot be read in full.

Pins that the status decides the retry, not the body: once the Synthesia API's status line and headers
have arrived, a 4xx is final. A body that stalls part-way (the response promises more bytes than it
sends, so the read runs into the request timeout) must not turn that 4xx into a connection failure or a
timeout, must not be retried, and must keep the status and the `Request-Id` the response carried.

The stalled body is built with `FakeSynthesiaApi`'s own inputs: a `Content-Length` header larger than the
body the fake sends, on a keep-alive connection. `FakeSynthesiaApi` has no way to drop the connection
part-way through a body, so the other way a body read fails (an `aiohttp.ClientPayloadError`) is not
exercised here; `create_session` handles both in the same `except` branch. `request_timeout` is an
input, shrunk so the stalled read ends quickly; nothing asserts on a duration.
"""

from __future__ import annotations

import aiohttp
import pytest

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from pipecat_synthesia.errors import SynthesiaError
from tests.fakes.fake_api import ScriptedResponse
from tests.support.secrets import API_KEY

#: Far more than any body below, so the read waits for bytes that never come.
_PROMISED_LENGTH = "4096"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(401, {"code": "unauthorized", "detail": "invalid api key"}, id="401"),
        pytest.param(404, {"code": "unknown_reference", "detail": "no such avatar"}, id="404"),
        pytest.param(429, {"code": "rate_limited", "detail": "slow down"}, id="429"),
    ],
)
async def test_a_4xx_whose_body_stalls_is_not_retried_and_keeps_its_status(
    fake_api_factory, status: int, body: dict
) -> None:
    # one entry, so a retry would get the same stalled response again
    script = [
        ScriptedResponse(
            status,
            body,
            headers={"Content-Length": _PROMISED_LENGTH, "Request-Id": "req_stalled"},
        )
    ]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = SessionApiClient(
                api_key=API_KEY,
                api_url=api.base_url,
                client=build_client_info(),
                http_session=session,
                request_timeout=0.2,
                max_attempts=3,
                backoff_seconds=(0.01,),
            )
            with pytest.raises(SynthesiaError) as excinfo:
                await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    err = excinfo.value
    assert len(api.requests) == 1, f"a {status} was retried after its body stalled: {err!r}"
    assert err.status == status
    assert err.request_id == "req_stalled"
