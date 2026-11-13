"""`errors.py`: CODE_TABLE, the fallback for a response with no code, and the two parsers."""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import pytest
from pipecat.frames.frames import ErrorFrame
from pipecat.utils.errors import ErrorCategory

from pipecat_synthesia.errors import (
    CODE_TABLE,
    DEFAULT_CATEGORY,
    ErrorType,
    SynthesiaError,
    error_from_exception,
    error_from_problem,
)
from tests.support.secrets import API_KEY, LK_API_SECRET, LK_TOKEN_MARKER, assert_no_secrets

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_PATH = REPO_ROOT / "tools" / "errors.json"


# --- catalogue coverage ----------------------------------------------------------------------------


@pytest.mark.unit
def test_catalogue_coverage() -> None:
    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    catalogue_codes = {entry["code"] for entry in catalogue["errors"]}

    missing_from_table = catalogue_codes - set(CODE_TABLE)
    assert not missing_from_table, f"catalogue codes missing from CODE_TABLE: {missing_from_table}"

    extra_in_table = set(CODE_TABLE) - catalogue_codes - {"client_upgrade_required"}
    assert not extra_in_table, (
        f"CODE_TABLE codes not in the catalogue (and not client_upgrade_required): {extra_in_table}"
    )

    unmapped = [member for member in ErrorType if member not in DEFAULT_CATEGORY]
    assert not unmapped, f"ErrorType members with no DEFAULT_CATEGORY entry: {unmapped}"


# --- error_from_problem: one table literal driving type/category/retryable/permanent/retry_after ----


# (status, body, headers) -> (type, category, retryable, permanent, retry_after)
TABLE: list[tuple[int, dict | None, dict[str, str], tuple]] = [
    (
        400,
        {"code": "bad_request", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    (
        400,
        {"code": "validation_error", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    (
        404,
        {"code": "not_found", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    (
        405,
        {"code": "method_not_allowed", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.UNKNOWN, False, False, None),
    ),
    (
        409,
        {"code": "conflict", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.UNKNOWN, False, False, None),
    ),
    (
        413,
        {"code": "payload_too_large", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.UNKNOWN, False, False, None),
    ),
    (
        415,
        {"code": "unsupported_media_type", "detail": "d"},
        {},
        (ErrorType.INVALID_SESSION_REQUEST, ErrorCategory.UNKNOWN, False, False, None),
    ),
    (
        401,
        {"code": "unauthenticated", "detail": "d"},
        {},
        (ErrorType.AUTH, ErrorCategory.AUTHENTICATION, False, True, None),
    ),
    (
        403,
        {"code": "forbidden", "detail": "d"},
        {},
        (ErrorType.AUTH, ErrorCategory.AUTHORIZATION, False, True, None),
    ),
    (
        403,
        {"code": "insufficient_scope", "detail": "d"},
        {},
        (ErrorType.AUTH, ErrorCategory.AUTHORIZATION, False, True, None),
    ),
    (
        402,
        {"code": "feature_not_in_plan", "detail": "d", "features": ["x"]},
        {},
        (ErrorType.FEATURE_NOT_IN_PLAN, ErrorCategory.QUOTA, False, False, None),
    ),
    (
        402,
        {"code": "payment_required", "detail": "d"},
        {},
        (ErrorType.QUOTA_EXCEEDED, ErrorCategory.QUOTA, False, False, None),
    ),
    (
        402,
        {"code": "quota_exceeded", "detail": "d"},
        {},
        (ErrorType.QUOTA_EXCEEDED, ErrorCategory.QUOTA, False, False, None),
    ),
    (
        404,
        {"code": "unknown_reference", "detail": "d"},
        {},
        (ErrorType.UNKNOWN_AVATAR, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    (
        429,
        {"code": "concurrency_limit", "detail": "d"},
        {},
        (ErrorType.CONCURRENCY_LIMIT, ErrorCategory.RATE_LIMIT, True, False, None),
    ),
    (
        429,
        {"code": "rate_limited", "detail": "d"},
        {"Retry-After": "7"},
        (ErrorType.RATE_LIMITED, ErrorCategory.RATE_LIMIT, True, False, 7.0),
    ),
    (
        429,
        {"code": "rate_limited", "detail": "d"},
        {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
        (ErrorType.RATE_LIMITED, ErrorCategory.RATE_LIMIT, True, False, None),
    ),
    (
        500,
        {"code": "internal", "detail": "d"},
        {},
        (ErrorType.CONNECTION, ErrorCategory.SERVER, True, False, None),
    ),
    (
        503,
        {"code": "service_unavailable", "detail": "d"},
        {},
        (ErrorType.CONNECTION, ErrorCategory.SERVER, True, False, None),
    ),
    (
        504,
        {"code": "timeout", "detail": "d"},
        {},
        (ErrorType.TIMEOUT, ErrorCategory.SERVER, True, False, None),
    ),
    (
        426,
        {
            "code": "client_upgrade_required",
            "detail": "d",
            "min_supported_version": "2.0",
            "upgrade_url": "https://x",
        },
        {},
        (ErrorType.CLIENT_UPGRADE_REQUIRED, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    # unknown code -> UNKNOWN, never the status fallback
    (
        400,
        {"code": "some_new_code", "detail": "d"},
        {},
        (ErrorType.UNKNOWN, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    # no code: not the Synthesia API, so UNKNOWN for a 4xx, with the category still from the status; a 429
    # stays RATE_LIMITED with its Retry-After, and a 5xx keeps its retryable type
    (400, None, {}, (ErrorType.UNKNOWN, ErrorCategory.INVALID_REQUEST, False, True, None)),
    (401, None, {}, (ErrorType.UNKNOWN, ErrorCategory.AUTHENTICATION, False, True, None)),
    (403, None, {}, (ErrorType.UNKNOWN, ErrorCategory.AUTHORIZATION, False, True, None)),
    (404, None, {}, (ErrorType.UNKNOWN, ErrorCategory.INVALID_REQUEST, False, True, None)),
    (
        404,
        {"message": "Not Found"},
        {},
        (ErrorType.UNKNOWN, ErrorCategory.INVALID_REQUEST, False, True, None),
    ),
    (429, None, {"Retry-After": "7"}, (ErrorType.RATE_LIMITED, ErrorCategory.RATE_LIMIT, True, False, 7.0)),
    (418, None, {}, (ErrorType.UNKNOWN, ErrorCategory.UNKNOWN, False, False, None)),
    (502, None, {}, (ErrorType.CONNECTION, ErrorCategory.SERVER, True, False, None)),
    (503, None, {}, (ErrorType.CONNECTION, ErrorCategory.SERVER, True, False, None)),
    (504, None, {}, (ErrorType.TIMEOUT, ErrorCategory.SERVER, True, False, None)),
    # 429 concurrency_limit with no Retry-After -> retry_after is None
    (
        429,
        {"code": "concurrency_limit"},
        {},
        (ErrorType.CONCURRENCY_LIMIT, ErrorCategory.RATE_LIMIT, True, False, None),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("status,body,headers,expected", TABLE)
def test_error_from_problem(status, body, headers, expected) -> None:
    exp_type, exp_category, exp_retryable, exp_permanent, exp_retry_after = expected
    err = error_from_problem(status, body, headers)
    assert err.type is exp_type
    assert err.category is exp_category
    assert err.retryable is exp_retryable
    assert err.permanent is exp_permanent
    assert err.retry_after == exp_retry_after
    assert err.status == status


@pytest.mark.unit
@pytest.mark.parametrize("status", [403, 404])
def test_codeless_4xx_message_names_the_status_quotes_the_body_and_points_at_api_url(status: int) -> None:
    """A 4xx with no code came from something other than the Synthesia API (a wrong host, a proxy, a CDN
    page), so the message says what did answer, and where to look; it is never an avatar or auth error."""
    page = (
        "<!DOCTYPE html>\n<html>\n  <head><title>Not Found</title></head>\n  <body>" + "x" * 500 + "</body>"
    )
    err = error_from_problem(status, None, {}, text=page)

    assert err.type is ErrorType.UNKNOWN
    assert f"HTTP {status}" in err.message
    assert "<!DOCTYPE html> <html> <head><title>Not Found</title></head>" in err.message  # one line
    assert "x" * 300 not in err.message, "the body excerpt is capped"
    assert "api_url" in err.message
    assert "SYNTHESIA_API_URL" in err.message

    no_body = error_from_problem(status, None, {})
    assert no_body.type is ErrorType.UNKNOWN
    assert f"HTTP {status}" in no_body.message
    assert "api_url" in no_body.message


@pytest.mark.unit
def test_codeless_body_excerpt_never_carries_a_secret_that_was_sent() -> None:
    """Whatever answered may echo the request back; the values sent are scrubbed from the quote before it
    is cut to length, so not even a prefix of one survives (*redaction rule*)."""
    echo = json.dumps(
        {"headers": {"Authorization": f"Bearer {API_KEY}"}, "json": {"livekitToken": LK_TOKEN_MARKER}}
    )
    err = error_from_problem(404, None, {}, text=echo, secrets=(API_KEY, LK_TOKEN_MARKER))

    assert "Authorization" in err.message, "the excerpt is there"
    assert_no_secrets(str(err), repr(err))

    cut_through = error_from_problem(404, None, {}, text="x" * 190 + API_KEY, secrets=(API_KEY,))
    assert API_KEY[:5] not in cut_through.message


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        {"detail": f"Bearer {API_KEY} rejected", "title": f"token {LK_TOKEN_MARKER}"},
        {"code": "bad_request", "detail": f"Bearer {API_KEY} rejected", "title": f"token {LK_TOKEN_MARKER}"},
    ],
    ids=["codeless", "with-code"],
)
def test_detail_and_title_never_carry_a_secret_that_was_sent(body: dict) -> None:
    """`detail` reaches the message, `err.detail`, `err.body` and the WARNING session-create log line, so
    the values sent are scrubbed from it and from `title`, with a code or without (*redaction rule*)."""
    err = error_from_problem(400, body, {}, secrets=(API_KEY, LK_TOKEN_MARKER))

    assert err.detail is not None and "rejected" in err.detail, "the rest of the detail is kept"
    assert_no_secrets(str(err), repr(err), err.detail, repr(err.body), secrets=(API_KEY, LK_TOKEN_MARKER))


@pytest.mark.unit
def test_codeless_5xx_keeps_no_api_url_hint() -> None:
    """A code-less 5xx is still a retryable CONNECTION, with the status in the message. It gets no api_url
    hint: a gateway in front of the Synthesia API answers with a bare 5xx too."""
    err = error_from_problem(503, None, {}, text="<html>Service Unavailable</html>")

    assert err.type is ErrorType.CONNECTION
    assert err.retryable is True
    assert "HTTP 503" in err.message
    assert "Service Unavailable" in err.message
    assert "api_url" not in err.message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        *[(status, ErrorType.CONNECTION) for status in (500, 501, 502, 503, 505, 599)],
        *[(status, ErrorType.UNKNOWN) for status in (200, 300, 400, 401, 402, 403, 404, 418, 499, 600, 601)],
        (429, ErrorType.RATE_LIMITED),
        (504, ErrorType.TIMEOUT),
    ],
)
def test_a_codeless_response_maps_by_status(status: int, expected: ErrorType) -> None:
    """The edges of the 5xx range in particular: 499 and 600 are not a CONNECTION."""
    assert error_from_problem(status, None, {}).type is expected
    assert error_from_problem(status, {"detail": "d"}, {}).type is expected


@pytest.mark.unit
def test_a_known_code_wins_over_the_status_and_an_unknown_code_does_not_fall_back_to_it() -> None:
    assert error_from_problem(503, {"code": "quota_exceeded"}, {}).type is ErrorType.QUOTA_EXCEEDED
    unknown = error_from_problem(503, {"code": "no_such_code"}, {})
    assert unknown.type is ErrorType.UNKNOWN
    assert unknown.code == "no_such_code"


@pytest.mark.unit
@pytest.mark.parametrize("body", [None, [], "text", 7, ["code"]])
def test_a_body_that_is_not_an_object_is_tolerated(body: object) -> None:
    err = error_from_problem(503, body, {})  # type: ignore[arg-type]

    assert err.type is ErrorType.CONNECTION
    assert err.code is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        *[{"code": bad} for bad in (42, None, True, ["quota_exceeded"], {"x": 1})],
        *[{"error": {"code": bad}} for bad in (42, None, True, ["quota_exceeded"], {"x": 1})],
        *[{"error": value} for value in ("boom", 5, ["quota_exceeded"], None)],
    ],
)
def test_only_a_top_level_string_code_is_read(body: dict) -> None:
    err = error_from_problem(400, body, {})

    assert err.type is ErrorType.UNKNOWN
    assert err.code is None


# --- retry_after -------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("key", ["retryAfter", "retry_after"])
@pytest.mark.parametrize(("value", "expected"), [(7, 7.0), (1.5, 1.5), (0, 0.0)])
def test_a_numeric_retry_after_in_the_body_is_read_as_seconds(
    key: str, value: float, expected: float
) -> None:
    err = error_from_problem(429, {"code": "rate_limited", key: value}, {})

    assert err.retry_after == expected
    assert isinstance(err.retry_after, float)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False, "7", None, [7], {"s": 7}])
def test_a_retry_after_in_the_body_that_is_not_a_number_is_ignored(value: object) -> None:
    assert error_from_problem(429, {"code": "rate_limited", "retryAfter": value}, {}).retry_after is None
    assert error_from_problem(429, {"code": "rate_limited", "retry_after": value}, {}).retry_after is None


@pytest.mark.unit
def test_the_retry_after_header_is_read_in_any_case_and_wins_over_the_body() -> None:
    body = {"code": "rate_limited", "retryAfter": 99}

    assert error_from_problem(429, {"code": "rate_limited"}, {"Retry-After": "7"}).retry_after == 7.0
    assert error_from_problem(429, {"code": "rate_limited"}, {"retry-after": "2.5"}).retry_after == 2.5
    assert error_from_problem(429, body, {"Retry-After": "3"}).retry_after == 3.0
    # An unparseable header is dropped, not replaced by the body's value.
    assert error_from_problem(429, body, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}).retry_after is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "code", "kept"),
    [
        (429, "rate_limited", True),
        (429, "concurrency_limit", True),
        (503, "service_unavailable", True),
        (504, "timeout", True),
        (400, "bad_request", False),
        (401, "unauthenticated", False),
        (402, "quota_exceeded", False),
        (404, "unknown_reference", False),
        (400, "no_such_code", False),
    ],
)
def test_retry_after_is_kept_only_for_types_where_waiting_helps(status: int, code: str, kept: bool) -> None:
    from_header = error_from_problem(status, {"code": code}, {"Retry-After": "5"})
    from_body = error_from_problem(status, {"code": code, "retryAfter": 5}, {})

    assert from_header.retry_after == (5.0 if kept else None)
    assert from_body.retry_after == (5.0 if kept else None)


@pytest.mark.unit
def test_a_client_upgrade_error_names_only_the_details_it_was_given() -> None:
    """Each detail the Synthesia API sent is named; one it did not send never shows as a placeholder
    ("None"), and with neither the message is the server's own detail and nothing more."""

    def upgrade(**details: str) -> SynthesiaError:
        body = {"code": "client_upgrade_required", "detail": "Please upgrade.", **details}
        return error_from_problem(426, body, {})

    both = upgrade(min_supported_version="1.2.0", upgrade_url="https://example.test/upgrade")
    assert both.type is ErrorType.CLIENT_UPGRADE_REQUIRED
    assert (both.permanent, both.retryable) == (True, False)
    for part in ("Please upgrade.", "1.2.0", "https://example.test/upgrade"):
        assert part in str(both)

    only_version = str(upgrade(min_supported_version="1.2.0"))
    assert "1.2.0" in only_version and "None" not in only_version

    only_url = str(upgrade(upgrade_url="https://example.test/upgrade"))
    assert "https://example.test/upgrade" in only_url and "None" not in only_url

    assert str(upgrade()) == "Please upgrade."


# --- message construction ---------------------------------------------------------------------------


@pytest.mark.unit
def test_message_features_and_client_upgrade_and_request_id_prefix() -> None:
    err = error_from_problem(
        402,
        {"code": "feature_not_in_plan", "detail": "no plan", "features": ["a", "b"]},
        {"Request-Id": "req-1"},
    )
    assert err.message.startswith("[req-1] ")
    assert "missing features: a, b" in err.message

    upgrade = error_from_problem(
        426,
        {
            "code": "client_upgrade_required",
            "detail": "upgrade",
            "min_supported_version": "2.0.0",
            "upgrade_url": "https://docs.synthesia.io/upgrade",
        },
        {},
    )
    assert "2.0.0" in upgrade.message
    assert "https://docs.synthesia.io/upgrade" in upgrade.message

    no_body = error_from_problem(503, None, {})
    assert "HTTP 503" in no_body.message

    with_code_no_detail = error_from_problem(400, {"code": "bad_request"}, {})
    assert with_code_no_detail.message == "Synthesia request failed (bad_request)"

    with_title = error_from_problem(400, {"code": "bad_request", "title": "Bad request"}, {})
    assert with_title.message == "Bad request"


@pytest.mark.unit
def test_request_id_header_takes_precedence_over_body() -> None:
    err = error_from_problem(
        400, {"code": "bad_request", "requestId": "from-body"}, {"Request-Id": "from-header"}
    )
    assert err.request_id == "from-header"

    err2 = error_from_problem(400, {"code": "bad_request", "requestId": "from-body"}, {})
    assert err2.request_id == "from-body"


@pytest.mark.unit
def test_body_redacted_and_session_id_propagated() -> None:
    err = error_from_problem(
        400,
        {"code": "bad_request", "detail": "d", "livekitToken": "SECRET-TOKEN"},
        {},
        session_id="ses_1",
    )
    assert err.body is not None
    assert err.body["livekitToken"] == "<redacted>"
    assert err.session_id == "ses_1"
    assert "SECRET-TOKEN" not in repr(err.body)


# --- error_from_exception ----------------------------------------------------------------------------


@pytest.mark.unit
def test_error_from_exception() -> None:
    timeout_err = error_from_exception(TimeoutError(), message="request failed")
    assert timeout_err.type is ErrorType.TIMEOUT
    assert timeout_err.retryable is True
    assert timeout_err.category is ErrorCategory.CONNECTIVITY
    assert "request failed" in str(timeout_err)

    client_err = error_from_exception(aiohttp.ClientConnectionError("boom"), message="request failed")
    assert client_err.type is ErrorType.CONNECTION
    assert client_err.retryable is True
    assert client_err.category is ErrorCategory.CONNECTIVITY

    other_err = error_from_exception(RuntimeError("boom"), message="request failed")
    assert other_err.type is ErrorType.UNKNOWN
    assert other_err.category is ErrorCategory.APPLICATION


# --- category derivation order ------------------------------------------------------------------------


@pytest.mark.unit
def test_category_derivation_order() -> None:
    upgrade_at_500 = SynthesiaError("x", type=ErrorType.CLIENT_UPGRADE_REQUIRED, status=500)
    assert upgrade_at_500.category is ErrorCategory.INVALID_REQUEST

    timeout_at_401 = SynthesiaError("x", type=ErrorType.TIMEOUT, status=401)
    assert timeout_at_401.category is ErrorCategory.AUTHENTICATION

    timeout_no_status = SynthesiaError("x", type=ErrorType.TIMEOUT, status=None)
    assert timeout_no_status.category is ErrorCategory.CONNECTIVITY


# --- no secrets leak through an error built from a hostile body ---------------------------------------


@pytest.mark.unit
def test_no_secrets_in_error_repr_or_body() -> None:
    err = error_from_problem(
        400,
        {
            "code": "bad_request",
            "detail": "rejected",
            "livekitToken": API_KEY,
            "token": LK_API_SECRET,
        },
        {},
    )
    assert_no_secrets(str(err), repr(err), repr(err.body), secrets=(API_KEY, LK_API_SECRET))


@pytest.mark.unit
def test_error_frame_keeps_the_request_id_prefix() -> None:
    """Pipecat 1.13+ masks credentials in `ErrorFrame.error`; the `[request_id]` prefix `errors.py` puts
    on a Synthesia API error message when the response carries a request id is what support asks for,
    and must survive the masking."""
    request_id = "req_01HZX5K2M9Q7T3W8Y4V6N0PABC"
    frame = ErrorFrame(error=f"[{request_id}] Synthesia API error 503 service_unavailable")
    assert request_id in frame.error
