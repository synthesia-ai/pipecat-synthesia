"""An error response that echoes the request's secrets, under any key and at any depth, leaks neither.

What this pins (*redaction rule*): every string in a problem-details body, keys included, is scrubbed of
the Synthesia API key and the LiveKit token the request carried before the body is stored on
`SynthesiaError.body`, and before any part of it reaches the message or the WARNING session-create log
line. That covers the body fields the message quotes beyond `detail` and `title`: the missing features of a
`FEATURE_NOT_IN_PLAN` error and the upgrade URL of a `CLIENT_UPGRADE_REQUIRED` one. The existing masking of
known secret keys stays, and the rest of the body keeps its shape.

`SessionApiClient` is internal; it is tested directly, against the scripted HTTP server, because the
public route to it is a room adapter's launch, which would add a LiveKit room and nothing to what is
checked here.
"""

from __future__ import annotations

import aiohttp
import pytest

from pipecat_synthesia import ErrorType, SynthesiaError
from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from tests.fakes.fake_api import ScriptedResponse
from tests.support.secrets import API_KEY, LK_TOKEN_MARKER, assert_no_secrets

pytestmark = pytest.mark.unit

SECRETS = (API_KEY, LK_TOKEN_MARKER)


async def _fail_with(fake_api_factory, body: dict) -> SynthesiaError:
    """The error `create_session` raises for one 4xx answering with `body`; the request carries both
    secrets, the API key in its header and the LiveKit token in its body."""
    async with fake_api_factory([ScriptedResponse(400, body)]) as api:
        async with aiohttp.ClientSession() as session:
            client = SessionApiClient(
                api_key=API_KEY,
                api_url=api.base_url,
                client=build_client_info(),
                http_session=session,
                max_attempts=1,
            )
            with pytest.raises(SynthesiaError) as caught:
                await client.create_session(
                    avatar_ids=["av_a"], livekit_url="wss://x", livekit_token=LK_TOKEN_MARKER
                )
    return caught.value


async def test_a_secret_under_an_unexpected_key_and_nested_is_scrubbed_from_the_stored_body(
    fake_api_factory, caplog_loguru
) -> None:
    body = {
        "code": "bad_request",
        "detail": "the request was invalid",
        "echo": API_KEY,
        "token": "masked-by-key",
        "errors": [{"field": "livekitToken", "received": f"value {LK_TOKEN_MARKER} rejected"}],
        "debug": {"request": {"headers": {"Authorization": f"Bearer {API_KEY}"}}},
        f"seen-{API_KEY}": True,
    }

    err = await _fail_with(fake_api_factory, body)

    assert err.type is ErrorType.INVALID_SESSION_REQUEST
    assert "synthesia session create error" in caplog_loguru.text, "the WARNING line was written"
    assert_no_secrets(caplog_loguru.text, str(err), repr(err), repr(err.body), secrets=SECRETS)
    # The shape survives the scrub, so the body is still useful in a report.
    assert err.body is not None
    assert err.body["errors"][0]["field"] == "livekitToken"
    assert "Bearer" in err.body["debug"]["request"]["headers"]["Authorization"]
    assert err.detail == "the request was invalid"
    # Known secret keys are still masked whatever they hold.
    assert err.body["token"] != "masked-by-key"


@pytest.mark.parametrize(
    "body",
    [
        {"code": "feature_not_in_plan", "detail": "missing", "features": ["avatars", f"x-{LK_TOKEN_MARKER}"]},
        {
            "code": "client_upgrade_required",
            "detail": "upgrade",
            "min_supported_version": "9.9",
            "upgrade_url": f"https://example.invalid/?k={API_KEY}",
        },
    ],
    ids=["missing-features", "upgrade-url"],
)
async def test_body_fields_the_message_quotes_are_scrubbed_too(fake_api_factory, body: dict) -> None:
    err = await _fail_with(fake_api_factory, body)

    assert_no_secrets(str(err), repr(err), repr(err.body), secrets=SECRETS)
    assert "example.invalid" in str(err) or "avatars" in str(err), "the field is still quoted"
