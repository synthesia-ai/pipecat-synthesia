"""Constructor validation and `ServiceConfig.from_kwargs`.

Every failure here is constructor misuse, so it is a `ValueError` / `TypeError` with an exact message,
never a `SynthesiaError` (*misuse vs session errors*).
"""

from __future__ import annotations

import re

import pytest

from pipecat_synthesia.config import ENV_ALLOW_ANY_API_HOST, ENV_API_KEY, ENV_API_URL, ServiceConfig
from pipecat_synthesia.protocol import DEFAULT_API_URL, MAX_AVATAR_IDS
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState, SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import make_service
from tests.support.secrets import API_KEY, assert_no_secrets


@pytest.mark.unit
def test_ctor_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare string, count, duplicates, missing key, env fallback, keyword-only, launch_timeout <= 0."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    room = FakeAvatarRoom(JoinMode.DIRECT)

    with pytest.raises(ValueError, match=re.escape("avatar_ids must be a list of ids, not a single string")):
        SynthesiaVideoService(avatar_ids="av_a", room=room, api_key=API_KEY)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError, match=re.escape(f"avatar_ids must contain between 1 and {MAX_AVATAR_IDS} ids, got 0")
    ):
        SynthesiaVideoService(avatar_ids=[], room=room, api_key=API_KEY)
    too_many = [f"av_{i}" for i in range(MAX_AVATAR_IDS + 1)]
    with pytest.raises(
        ValueError,
        match=re.escape(f"avatar_ids must contain between 1 and {MAX_AVATAR_IDS} ids, got {len(too_many)}"),
    ):
        SynthesiaVideoService(avatar_ids=too_many, room=room, api_key=API_KEY)

    with pytest.raises(ValueError, match=re.escape("avatar_ids contains duplicates: ['av_a']")):
        SynthesiaVideoService(avatar_ids=["a", "av_a"], room=room, api_key=API_KEY)

    with pytest.raises(
        ValueError,
        match=re.escape("a Synthesia API key is required: pass api_key or set SYNTHESIA_API_KEY"),
    ):
        SynthesiaVideoService(avatar_ids=["av_a"], room=room)

    monkeypatch.setenv(ENV_API_KEY, API_KEY)
    service = SynthesiaVideoService(avatar_ids=["a"], room=room)
    assert service.state is ServiceState.INIT
    assert service.active_avatar_id == "av_a"
    assert service.avatar_identity == room.avatar_identity
    assert service.session_id is None

    with pytest.raises(TypeError):
        SynthesiaVideoService(["av_a"], room)  # type: ignore[misc]

    for bad in (0, -1, 0.0):
        with pytest.raises(ValueError, match=re.escape("launch_timeout must be positive")):
            SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY, launch_timeout=bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [5, 1.5, object(), True])
def test_avatar_ids_that_is_not_an_iterable_is_a_type_error(bad: object) -> None:
    with pytest.raises(TypeError, match="avatar_ids"):
        SynthesiaVideoService(avatar_ids=bad, room=FakeAvatarRoom(JoinMode.DIRECT), api_key=API_KEY)  # type: ignore[arg-type]


MAXIMUM = [f"av_{i}" for i in range(MAX_AVATAR_IDS)]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("make", "ids"),
    [
        (lambda: list(MAXIMUM), MAXIMUM),
        (lambda: (i for i in MAXIMUM), MAXIMUM),
        (lambda: ("av_0", "av_1"), ["av_0", "av_1"]),
        (lambda: iter(["av_0", "av_1"]), ["av_0", "av_1"]),
        (lambda: (i for i in ["av_0", "av_1"]), ["av_0", "av_1"]),
        (lambda: dict.fromkeys(["av_0", "av_1"]), ["av_0", "av_1"]),
    ],
    ids=["list-of-the-maximum", "generator-of-the-maximum", "tuple", "iterator", "generator", "dict-keys"],
)
def test_any_iterable_of_avatar_ids_is_accepted(make, ids: list[str]) -> None:
    """Every id is kept, in order, whatever the iterable; read through `make_service`, the seam."""
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT), avatar_ids=make())

    assert service._config.avatar_ids == tuple(ids)
    assert service.active_avatar_id == "av_0"


@pytest.mark.unit
def test_nan_launch_timeout_rejected() -> None:
    """NaN is not a positive number of seconds, though `nan <= 0` is False."""
    room = FakeAvatarRoom(JoinMode.DIRECT)
    with pytest.raises(ValueError, match=re.escape("launch_timeout must be positive")):
        SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY, launch_timeout=float("nan"))


@pytest.mark.unit
def test_from_kwargs_url_and_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_url falls back to the env then DEFAULT_API_URL and loses its trailing slash."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_API_URL, raising=False)

    cfg = ServiceConfig.from_kwargs(
        avatar_ids=["a", "av_b"], api_key=API_KEY, api_url=None, launch_timeout=90.0, required=True
    )
    assert cfg.avatar_ids == ("av_a", "av_b")
    assert cfg.api_url == DEFAULT_API_URL
    assert cfg.launch_timeout == 90.0
    assert cfg.required is True

    explicit = ServiceConfig.from_kwargs(
        avatar_ids=["av_a"],
        api_key=API_KEY,
        api_url="https://api.synthesia.io/",
        launch_timeout=1,
        required=False,
    )
    assert explicit.api_url == "https://api.synthesia.io"
    assert explicit.required is False

    monkeypatch.setenv(ENV_API_URL, "https://from-env.synthesia.io///")
    monkeypatch.setenv(ENV_API_KEY, API_KEY)
    from_env = ServiceConfig.from_kwargs(
        avatar_ids=["av_a"], api_key=None, api_url=None, launch_timeout=5.0, required=True
    )
    assert from_env.api_url == "https://from-env.synthesia.io"

    # *redaction rule*: the key is a repr=False field, so logging the config by accident does not leak it.
    assert_no_secrets(repr(from_env), str(from_env))


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_url",
    [
        "developers.synthesia.io",  # no scheme: aiohttp would reject it only at the first request
        "http://developers.synthesia.io",  # http:// to a host that is not loopback
        "https://",  # no host
        "https://[::1",  # malformed
        "https://api.example.test:port",  # a port that is not a number
    ],
)
def test_bad_api_url_raises_at_construction(monkeypatch: pytest.MonkeyPatch, bad_url: str) -> None:
    """A URL the Synthesia API client could never reach is a misuse error at construction, naming both
    ways to set it, rather than a network error retried at launch."""
    monkeypatch.delenv(ENV_API_URL, raising=False)
    room = FakeAvatarRoom(JoinMode.DIRECT)

    with pytest.raises(ValueError) as excinfo:
        SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY, api_url=bad_url)

    message = str(excinfo.value)
    assert "https://" in message
    assert "api_url" in message
    assert ENV_API_URL in message


@pytest.mark.unit
def test_bad_api_url_from_the_environment_is_named_and_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message names the environment variable the URL came from, and never echoes the value: a secret
    pasted into the wrong variable must not reach a log (*redaction rule*)."""
    monkeypatch.setenv(ENV_API_URL, API_KEY)

    with pytest.raises(ValueError) as excinfo:
        ServiceConfig.from_kwargs(
            avatar_ids=["av_a"], api_key=API_KEY, api_url=None, launch_timeout=5.0, required=True
        )

    assert f"from {ENV_API_URL}" in str(excinfo.value)
    assert_no_secrets(str(excinfo.value))


@pytest.mark.unit
@pytest.mark.parametrize(
    "good_url",
    [
        "https://api.synthesia.io",
        "https://synthesia.io",
        "https://developers.synthesia.io:8443",  # the port is kept
        "https://API.Synthesia.IO",  # a host name is case-insensitive
        "https://api.synthesia.io.",  # an absolute name: the same host
        "http://127.0.0.1:8080",
        "http://localhost",
        "http://localhost:8080",
        "http://[::1]:8080",
        "https://127.0.0.1:8443",
    ],
)
@pytest.mark.parametrize("via", ["keyword", "environment"])
def test_synthesia_and_loopback_api_urls_are_accepted(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, good_url: str, via: str
) -> None:
    """`https://` to synthesia.io or a host under it is accepted, as is `http://` or `https://` to a loopback
    host, for a local server such as the tests' in-process Synthesia API. Nothing to opt in to, so no
    warning."""
    monkeypatch.delenv(ENV_ALLOW_ANY_API_HOST, raising=False)
    room = FakeAvatarRoom(JoinMode.DIRECT)
    if via == "keyword":
        monkeypatch.delenv(ENV_API_URL, raising=False)
        service = SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY, api_url=good_url)
    else:
        monkeypatch.setenv(ENV_API_URL, good_url)
        service = SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY)

    assert service._config.api_url == good_url
    assert _warnings(caplog_loguru) == []


_OUTSIDE_SYNTHESIA = [
    "https://evilsynthesia.io",  # not a label match
    "https://synthesia.io.example.com",  # synthesia.io is not the end of the host
    "https://api.example.com",
    "https://api.synthesia.io@example.com",  # the host is example.com; the rest is a user name
    "https://example.com\\@api.synthesia.io",  # urlsplit and aiohttp read different hosts from it
]

# aiohttp refuses credentials in the URL alongside the Authorization header, and only at the request.
_WITH_USER_INFO = [
    "https://user:pw@api.synthesia.io",
    "https://example.com@api.synthesia.io",
    "http://user:pw@127.0.0.1:8080",
]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]


@pytest.mark.unit
@pytest.mark.parametrize("bad_url", [*_OUTSIDE_SYNTHESIA, *_WITH_USER_INFO, "http://api.synthesia.io"])
@pytest.mark.parametrize("via", ["keyword", "environment"])
def test_an_api_url_outside_synthesia_is_refused_and_not_echoed(
    monkeypatch: pytest.MonkeyPatch, bad_url: str, via: str
) -> None:
    """The service sends the API key and a LiveKit token to this URL, so a host outside synthesia.io is a
    misuse error at construction unless the developer opts in; so is a URL with a user name or password,
    which aiohttp would refuse only once the launch had started. The message says where the URL came from
    and how to opt in, and never echoes the URL (*redaction rule*)."""
    monkeypatch.delenv(ENV_ALLOW_ANY_API_HOST, raising=False)
    room = FakeAvatarRoom(JoinMode.DIRECT)

    with pytest.raises(ValueError) as excinfo:
        if via == "keyword":
            monkeypatch.delenv(ENV_API_URL, raising=False)
            SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY, api_url=bad_url)
        else:
            monkeypatch.setenv(ENV_API_URL, bad_url)
            SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY)

    message = str(excinfo.value)
    assert f"from {'api_url' if via == 'keyword' else ENV_API_URL}" in message
    assert "allow_any_api_host=True" in message
    assert ENV_ALLOW_ANY_API_HOST in message
    assert bad_url not in message
    assert "example" not in message
    assert "pw" not in message


@pytest.mark.unit
@pytest.mark.parametrize("via", ["keyword", "environment"])
def test_the_opt_in_accepts_an_https_host_outside_synthesia_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, via: str
) -> None:
    """With the opt-in, an `https://` host outside synthesia.io is accepted, and one WARNING at construction
    says so, without the URL."""
    monkeypatch.delenv(ENV_API_URL, raising=False)
    url = "https://api.example.com:8443"
    room = FakeAvatarRoom(JoinMode.DIRECT)
    if via == "keyword":
        monkeypatch.delenv(ENV_ALLOW_ANY_API_HOST, raising=False)
        service = SynthesiaVideoService(
            avatar_ids=["av_a"], room=room, api_key=API_KEY, api_url=url, allow_any_api_host=True
        )
    else:
        monkeypatch.setenv(ENV_ALLOW_ANY_API_HOST, " TRUE ")
        service = SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY, api_url=url)

    assert service._config.api_url == url
    warnings = _warnings(caplog_loguru)
    assert len(warnings) == 1, warnings
    assert "outside synthesia.io" in warnings[0]
    assert "example" not in caplog_loguru.text


@pytest.mark.unit
@pytest.mark.parametrize("url", [None, "https://api.synthesia.io", "http://127.0.0.1:8080"])
def test_the_opt_in_does_not_warn_for_a_url_it_was_not_needed_for(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, url: str | None
) -> None:
    """The opt-in left set for a Synthesia or loopback URL (or the default) is not an override: no warning."""
    monkeypatch.delenv(ENV_API_URL, raising=False)
    SynthesiaVideoService(
        avatar_ids=["av_a"],
        room=FakeAvatarRoom(JoinMode.DIRECT),
        api_key=API_KEY,
        api_url=url,
        allow_any_api_host=True,
    )

    assert _warnings(caplog_loguru) == []


@pytest.mark.unit
@pytest.mark.parametrize("bad_url", ["http://api.example.com", "http://api.synthesia.io"])
def test_the_opt_in_still_refuses_http_to_a_host_that_is_not_loopback(
    monkeypatch: pytest.MonkeyPatch, bad_url: str
) -> None:
    monkeypatch.delenv(ENV_API_URL, raising=False)

    with pytest.raises(ValueError) as excinfo:
        ServiceConfig.from_kwargs(
            avatar_ids=["av_a"],
            api_key=API_KEY,
            api_url=bad_url,
            launch_timeout=5.0,
            required=True,
            allow_any_api_host=True,
        )

    assert bad_url not in str(excinfo.value)


@pytest.mark.unit
def test_the_keyword_overrides_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`allow_any_api_host=False` refuses a host outside synthesia.io even with the variable set, as `api_url`
    wins over `SYNTHESIA_API_URL`."""
    monkeypatch.delenv(ENV_API_URL, raising=False)
    monkeypatch.setenv(ENV_ALLOW_ANY_API_HOST, "1")

    with pytest.raises(ValueError, match=re.escape("outside synthesia.io")):
        ServiceConfig.from_kwargs(
            avatar_ids=["av_a"],
            api_key=API_KEY,
            api_url="https://api.example.com",
            launch_timeout=5.0,
            required=True,
            allow_any_api_host=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "false", "No", "off", ""])
def test_a_false_environment_value_refuses(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.delenv(ENV_API_URL, raising=False)
    monkeypatch.setenv(ENV_ALLOW_ANY_API_HOST, value)

    with pytest.raises(ValueError, match=re.escape(f"set {ENV_ALLOW_ANY_API_HOST}=1")):
        ServiceConfig.from_kwargs(
            avatar_ids=["av_a"],
            api_key=API_KEY,
            api_url="https://api.example.com",
            launch_timeout=5.0,
            required=True,
        )


@pytest.mark.unit
def test_an_environment_value_that_is_neither_true_nor_false_is_named_and_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value the variable does not take is a misuse error, even for a Synthesia URL: a typo must not
    silently decide. The message never echoes it (*redaction rule*)."""
    monkeypatch.delenv(ENV_API_URL, raising=False)
    monkeypatch.setenv(ENV_ALLOW_ANY_API_HOST, API_KEY)

    with pytest.raises(ValueError) as excinfo:
        ServiceConfig.from_kwargs(
            avatar_ids=["av_a"], api_key=API_KEY, api_url=None, launch_timeout=5.0, required=True
        )

    assert ENV_ALLOW_ANY_API_HOST in str(excinfo.value)
    assert_no_secrets(str(excinfo.value))
