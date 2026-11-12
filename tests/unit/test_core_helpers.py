"""Contracts on the core helpers, which need nothing but the LiveKit-free core: wire strings only in
protocol.py, `SynthesiaError`'s defaults, the redaction helpers, the User-Agent format, the avatar-id
helpers, no `livekit` or `jwt` import in a core module, and `version.py` agreeing with `pyproject.toml`.

Moved unchanged from `test_scaffold.py`, which keeps the contracts that need the complete package (the
public exports, the livekit import guard, per-module importability and the LiveKit time bounds), so that
these can ship with the core on their own.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import json
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "pipecat_synthesia"
PROTOCOL_MODULE = PACKAGE_ROOT / "protocol.py"
LIVEKIT_SUBPACKAGE_DIR = "livekit"


def _package_files() -> list[Path]:
    """Every .py file shipped under src/pipecat_synthesia, sorted, build artefacts excluded."""
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_core(path: Path) -> bool:
    """True for a module outside pipecat_synthesia/livekit/, i.e. one importable without the extra."""
    return path.relative_to(PACKAGE_ROOT).parts[0] != LIVEKIT_SUBPACKAGE_DIR


# --- wire strings --------------------------------------------------------------------------------

# Wire names are verbatim and live only in protocol.py.
WIRE_PREFIX = "lk."
# Human-readable messages that merely *mention* lk.publish_on_behalf start with this.
MESSAGE_PREFIX = "avatar_token:"
WIRE_EXACT = frozenset(
    {"swapAvatar", "avatar_ready", "avatar_status", "synthesia.protocol", "synthesia.client"}
)


@pytest.mark.unit
def test_no_wire_literals_outside_protocol() -> None:
    """No wire string is spelled out anywhere but protocol.py."""
    offenders: list[str] = []
    for path in _package_files():
        if path == PROTOCOL_MODULE:
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if value.startswith(MESSAGE_PREFIX):
                continue
            if value.startswith(WIRE_PREFIX) or value in WIRE_EXACT:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {value!r}")
    assert not offenders, "wire literals belong in protocol.py only:\n  " + "\n  ".join(offenders)


# --- SynthesiaError ------------------------------------------------------------------------------


@pytest.mark.unit
def test_synthesia_error_defaults() -> None:
    """SynthesiaError's retryable/category derivation, its redacting repr, and `with_session`."""
    from pipecat.utils.errors import ErrorCategory

    from pipecat_synthesia.errors import DEFAULT_CATEGORY, ErrorType, SynthesiaError

    timeout = SynthesiaError("x", type=ErrorType.TIMEOUT)
    assert timeout.retryable is True
    assert timeout.category is ErrorCategory.CONNECTIVITY

    unauthorised = SynthesiaError("x", status=401)
    assert unauthorised.category is ErrorCategory.AUTHENTICATION
    assert unauthorised.permanent

    # CLIENT_UPGRADE_REQUIRED wins over the status: Pipecat maps 426 to UNKNOWN, which would make the
    # version gate non-permanent.
    upgrade = SynthesiaError("x", type=ErrorType.CLIENT_UPGRADE_REQUIRED, status=426)
    assert upgrade.category is ErrorCategory.INVALID_REQUEST

    shown = repr(SynthesiaError("m", body={"livekitToken": "secret"}, session_id="ses_1"))
    assert "secret" not in shown, shown
    assert "session_id='ses_1'" in shown, shown

    stamped = SynthesiaError("m", session_id="ses_1")
    assert stamped.with_session("ses_2") is stamped
    assert stamped.session_id == "ses_1"
    blank = SynthesiaError("m")
    assert blank.with_session("ses_3") is blank
    assert blank.session_id == "ses_3"

    unmapped = [member for member in ErrorType if member not in DEFAULT_CATEGORY]
    assert not unmapped, f"ErrorType members with no DEFAULT_CATEGORY entry: {unmapped}"


# --- redaction -----------------------------------------------------------------------------------


def _b64url_segment(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@pytest.mark.unit
def test_redact() -> None:
    """redact_secret / redact_jwt / redact_mapping never echo the value they are given (*redaction rule*)."""
    from pipecat_synthesia._redact import redact_jwt, redact_mapping, redact_secret

    assert redact_secret("abcdef") == "<redacted>"

    # An unsigned JWT assembled by hand as header.payload.signature: _redact must decode it with
    # base64 + json only, because core has to import without the [livekit] extra (no pyjwt).
    payload_segment = _b64url_segment({"sub": "synthesia-avatar-agent", "room": "room-1", "kind": "agent"})
    token = ".".join(
        (
            _b64url_segment({"alg": "none", "typ": "JWT"}),
            payload_segment,
            base64.urlsafe_b64encode(b"not-a-real-signature").decode("ascii").rstrip("="),
        )
    )
    described = redact_jwt(token)
    assert "sub=" in described, described
    assert token not in described, described
    assert payload_segment not in described, described

    assert redact_mapping({"livekitToken": "t", "x": 1}) == {"livekitToken": "<redacted>", "x": 1}


def _jwt(payload: object) -> str:
    return f"{_b64url_segment({'alg': 'none'})}.{_b64url_segment(payload)}.sig"


@pytest.mark.unit
@pytest.mark.parametrize(
    "token",
    [None, "", "garbage", "a.!!!.c", f"h.{_b64url_segment(['not', 'a', 'dict'])}.s"],
    ids=["none", "empty", "one-segment", "bad-base64", "not-an-object"],
)
def test_redact_jwt_summarises_an_unreadable_token_as_unparseable(token: str | None) -> None:
    from pipecat_synthesia._redact import redact_jwt

    assert redact_jwt(token) == "<jwt unparseable>"


@pytest.mark.unit
def test_redact_jwt_decodes_every_base64_padding_length() -> None:
    """The payload segment is stored unpadded, so each padding length has to be restored."""
    from pipecat_synthesia._redact import redact_jwt

    seen: set[int] = set()
    for size in range(1, 12):
        subject = "s" * size
        token = _jwt({"sub": subject, "room": "r"})
        seen.add(len(token.split(".")[1]) % 4)
        summary = redact_jwt(token)
        assert f"sub={subject} " in summary, summary
        assert "room=r " in summary, summary
    assert seen == {0, 2, 3}, "the payloads no longer cover every padding length"


@pytest.mark.unit
def test_redact_jwt_summary_fields_and_nothing_else() -> None:
    """Subject, room (from the `video` grant, else a bare top-level `room`), kind and expiry; never the
    token or any other claim."""
    from pipecat_synthesia._redact import redact_jwt

    token = _jwt(
        {"sub": "avatar", "kind": "agent", "exp": 1234, "name": "TOKENSECRET", "video": {"room": "room-1"}}
    )
    summary = redact_jwt(token)

    for field in ("sub=avatar", "room=room-1", "kind=agent", "exp=1234"):
        assert field in summary, summary
    assert token not in summary
    assert token.split(".")[1] not in summary
    assert "TOKENSECRET" not in summary
    assert "room=bare-room" in redact_jwt(_jwt({"sub": "avatar", "room": "bare-room"}))


# --- client identification -----------------------------------------------------------------------

USER_AGENT_PATTERN = r"^pipecat-synthesia/\S+ pipecat-ai/\S+ python/\S+ \(.+\)$"


@pytest.mark.unit
def test_client_info_user_agent_format() -> None:
    """build_client_info() is cached, and produces a User-Agent matching `USER_AGENT_PATTERN` and the
    eight-key header JSON."""
    from pipecat_synthesia._client_info import ClientInfo, build_client_info

    info = build_client_info()
    assert re.match(USER_AGENT_PATTERN, info.user_agent), info.user_agent

    header = json.loads(info.as_header_json())
    assert len(header) == 8, header
    assert set(header) == {field.name for field in dataclasses.fields(ClientInfo)}, header

    assert build_client_info() is build_client_info(), "build_client_info() must be cached"


# --- avatar ids ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_avatar_id_normalisation() -> None:
    """normalize_avatar_id / unprefixed_avatar_id / normalize_avatar_ids, and their error messages."""
    from pipecat_synthesia.config import normalize_avatar_id, normalize_avatar_ids, unprefixed_avatar_id
    from pipecat_synthesia.protocol import MAX_AVATAR_IDS

    assert normalize_avatar_id("foo") == "av_foo"
    assert normalize_avatar_id(" av_foo ") == "av_foo"
    assert unprefixed_avatar_id("av_foo") == "foo"
    assert unprefixed_avatar_id("foo") == "foo"

    blank = re.escape("avatar id must be a non-empty string")
    with pytest.raises(ValueError, match=blank):
        normalize_avatar_id("")
    with pytest.raises(ValueError, match=blank):
        normalize_avatar_id(None)
    with pytest.raises(ValueError, match=re.escape("is not a valid avatar id")):
        normalize_avatar_id("a b")

    single = re.escape("avatar_ids must be a list of ids, not a single string")
    with pytest.raises(ValueError, match=single):
        normalize_avatar_ids("av_foo")
    with pytest.raises(ValueError, match=single):
        normalize_avatar_ids(b"av_foo")

    def count_message(n: int) -> str:
        return re.escape(f"avatar_ids must contain between 1 and {MAX_AVATAR_IDS} ids, got {n}")

    with pytest.raises(ValueError, match=count_message(0)):
        normalize_avatar_ids([])
    with pytest.raises(ValueError, match=count_message(0)):
        normalize_avatar_ids(None)
    too_many = [f"av_id{i}" for i in range(MAX_AVATAR_IDS + 1)]
    with pytest.raises(ValueError, match=count_message(len(too_many))):
        normalize_avatar_ids(too_many)

    with pytest.raises(ValueError, match=re.escape("avatar_ids contains duplicates: ['av_foo']")):
        normalize_avatar_ids(["foo", "av_foo"])

    assert normalize_avatar_ids(["b", "a"]) == ("av_b", "av_a")


# --- LiveKit-free core ---------------------------------------------------------------------------

# The core package must install and import without the [livekit] extra, so neither dependency may be
# named anywhere outside pipecat_synthesia/livekit/ — not even inside a function or `if TYPE_CHECKING`.
FORBIDDEN_CORE_IMPORTS = frozenset({"jwt", "livekit"})


@pytest.mark.unit
def test_core_modules_do_not_import_jwt_or_livekit() -> None:
    """No core module imports jwt or livekit, at any nesting depth (*LiveKit-free core*)."""
    offenders: list[str] = []
    for path in _package_files():
        if not _is_core(path):
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which can never reach a third-party package.
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN_CORE_IMPORTS:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")
    assert not offenders, "core modules must import neither jwt nor livekit:\n  " + "\n  ".join(offenders)


# --- version agreement ---------------------------------------------------------------------------

PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.mark.unit
def test_version_py_matches_pyproject() -> None:
    """`version.py::__version__` equals `pyproject.toml` `project.version`.

    RELEASING.md's per-release steps bump both by hand, but release.yml checks only the tag against
    `project.version`. `_client_info.py` builds the User-Agent and the `synthesia.client` attribute from
    `__version__`, so a forgotten bump ships a distribution that misreports its own version to the server
    on every session. Nothing else checks this.
    """
    from pipecat_synthesia.version import __version__

    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert __version__ == declared, (
        f"src/pipecat_synthesia/version.py __version__ is {__version__!r} but pyproject.toml "
        f"project.version is {declared!r}; the release steps in RELEASING.md bump both together"
    )
