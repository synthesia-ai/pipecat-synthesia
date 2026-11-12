"""Advisories from a response that carries several `Link` header fields: a deprecation link is logged
wherever it is.

HTTP lets a response repeat `Link` (RFC 9110 field lines, RFC 8288 links), and the aiohttp client keeps each
field line as its own entry. The deprecation `Link` is one of the *installed-version channels*, which an
installed version can never learn to read later, so it must be found whether it is the first `Link` field,
a later one, or one of several links in a single field. It is found however RFC 8288 lets it be written
(an unquoted value, several relation types in one `rel`, the parameter name in any case), and not where the
word is only part of a URI or of another relation type. Every `Warning` field is an advisory too.

Driven through `SessionApiClient.create_session` against the fake Synthesia API. The fake hands its scripted
headers straight to aiohttp's `web.Response`, which keeps repeated fields when given a `CIMultiDict`.
"""

from __future__ import annotations

import logging
from typing import cast

import aiohttp
import pytest
from multidict import CIMultiDict

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from tests.fakes.fake_api import ScriptedResponse
from tests.support.secrets import API_KEY

OTHER_LINK = '<https://x/docs>; rel="describedby"'
DEPRECATION_LINK = '<https://x/migrate>; rel="deprecation"'

LINK_FIELDS = {
    "deprecation-in-the-first-field": [DEPRECATION_LINK, OTHER_LINK],
    "deprecation-in-a-later-field": [OTHER_LINK, DEPRECATION_LINK],
    "deprecation-later-in-one-field": [f"{OTHER_LINK}, {DEPRECATION_LINK}"],
    # RFC 8288: a token value needs no quotes, `rel` may list several relation types, and parameter
    # names are case-insensitive.
    "deprecation-unquoted": [OTHER_LINK, "<https://x/migrate>; rel=deprecation"],
    "deprecation-among-several-relation-types": [
        OTHER_LINK,
        '<https://x/migrate>; rel="alternate deprecation"',
    ],
    "deprecation-with-an-uppercase-parameter-name": [OTHER_LINK, '<https://x/migrate>; REL="deprecation"'],
}
#: Links that are not deprecation links, though the word appears in them.
NOT_DEPRECATION_FIELDS = {
    "deprecation-only-in-the-uri": ['<https://x/a;rel=deprecation;v=1>; rel="next"'],
    "another-relation-type-that-contains-the-word": ['<https://x/a>; rel="deprecation-notice"'],
}


async def _advisories_for(fake_api_factory, caplog_loguru, headers: CIMultiDict[str]) -> list[str]:
    """Create a session whose 201 carries `headers`; return the advisory lines logged at WARNING."""
    script = [ScriptedResponse(201, {"id": "ses_1"}, headers=cast("dict[str, str]", headers))]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = SessionApiClient(
                api_key=API_KEY,
                api_url=api.base_url,
                client=build_client_info(),
                http_session=session,
                request_timeout=5.0,
                max_attempts=1,
                backoff_seconds=(0.01,),
                retry_after_cap=10.0,
            )
            await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")
    return [
        r.getMessage()
        for r in caplog_loguru.records
        if r.levelno == logging.WARNING and "advisory" in r.getMessage()
    ]


@pytest.mark.unit
@pytest.mark.parametrize("fields", NOT_DEPRECATION_FIELDS.values(), ids=NOT_DEPRECATION_FIELDS.keys())
async def test_a_link_with_no_deprecation_relation_is_no_advisory(
    fake_api_factory, caplog_loguru, fields: list[str]
) -> None:
    headers = CIMultiDict(("Link", value) for value in fields)
    assert await _advisories_for(fake_api_factory, caplog_loguru, headers) == []


@pytest.mark.unit
async def test_every_warning_field_is_an_advisory(fake_api_factory, caplog_loguru) -> None:
    fields = ['299 - "the first warning"', '299 - "the second warning"']
    headers = CIMultiDict(("Warning", value) for value in fields)
    advisories = await _advisories_for(fake_api_factory, caplog_loguru, headers)
    for field in fields:
        assert any(field in line for line in advisories), f"{field!r} was not logged: {advisories}"


@pytest.mark.unit
@pytest.mark.parametrize("fields", LINK_FIELDS.values(), ids=LINK_FIELDS.keys())
async def test_a_deprecation_link_is_an_advisory_whichever_link_field_carries_it(
    fake_api_factory, caplog_loguru, fields: list[str]
) -> None:
    headers = CIMultiDict(("Link", value) for value in fields)
    # The fake declares `dict[str, str]`, and passes the object to aiohttp unchanged.
    script = [ScriptedResponse(201, {"id": "ses_1"}, headers=cast("dict[str, str]", headers))]
    async with fake_api_factory(script) as api:
        async with aiohttp.ClientSession() as session:
            client = SessionApiClient(
                api_key=API_KEY,
                api_url=api.base_url,
                client=build_client_info(),
                http_session=session,
                request_timeout=5.0,
                max_attempts=1,
                backoff_seconds=(0.01,),
                retry_after_cap=10.0,
            )
            await client.create_session(avatar_ids=["av_a"], livekit_url="wss://x", livekit_token="tok")

    advisories = [
        r for r in caplog_loguru.records if r.levelno == logging.WARNING and "advisory" in r.getMessage()
    ]
    assert advisories, f"no advisory was logged for the Link fields {fields}"
