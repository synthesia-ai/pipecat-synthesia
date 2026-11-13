"""Shared secret sentinels and the grep-negative assertion for the *redaction rule*.

Every "no secret in any log line, ``str(err)``, ``repr(err)`` or ``err.body``" test imports these three
values rather than inventing its own, so a single :func:`assert_no_secrets` call covers every credential
the plugin handles, and a leak path found by one test is checked by every other for free.

The values are deliberately unmistakable: no real key looks like this, and each contains ``SECRET`` so
a human scanning CI output sees immediately what leaked.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "API_KEY",
    "DEFAULT_SECRETS",
    "LK_API_SECRET",
    "LK_TOKEN_MARKER",
    "assert_no_secrets",
]

#: The Synthesia API key every test passes to the service / ``SessionApiClient``.
API_KEY = "sk-TESTSECRET-9f3a"

#: The LiveKit API secret every test signs its tokens with.
LK_API_SECRET = "LKSECRET-7c1d"

#: Goes into the ``name`` claim of every test token, so it lands in the token's *payload* segment and
#: a log line that prints the raw JWT — or a ``redact_jwt`` that echoes a decoded claim it should not —
#: fails the grep. The signature segment cannot carry a chosen marker, so the claim is the seam.
LK_TOKEN_MARKER = "TOKENSECRET"

DEFAULT_SECRETS: tuple[str, ...] = (API_KEY, LK_API_SECRET, LK_TOKEN_MARKER)

_SENTINEL_NAMES = {
    API_KEY: "API_KEY",
    LK_API_SECRET: "LK_API_SECRET",
    LK_TOKEN_MARKER: "LK_TOKEN_MARKER",
}

_CONTEXT_CHARS = 60


def _masked(text: str, secret: str, index: int) -> str:
    """``text`` around ``index`` with the secret itself replaced by its sentinel name.

    The failure message must show *where* the leak is without reprinting the secret: a CI log that
    quotes the leaked value to prove a leak is its own small leak, and the habit matters once the same
    assertion runs against staging output.
    """
    name = _SENTINEL_NAMES.get(secret, "SECRET")
    start = max(0, index - _CONTEXT_CHARS)
    end = min(len(text), index + len(secret) + _CONTEXT_CHARS)
    body = text[start:index] + f"<{name}>" + text[index + len(secret) : end]
    return ("..." if start else "") + body + ("..." if end < len(text) else "")


def assert_no_secrets(*texts: str, secrets: Iterable[str] = DEFAULT_SECRETS) -> None:
    """Assert none of ``texts`` contains any of ``secrets`` (*redaction rule*).

    Pass whatever the customer could see: ``caplog_loguru.text``, ``str(err)``, ``repr(err)``,
    ``repr(err.body)``, a serialised request. Substring matching is on purpose — a secret truncated
    into a longer string, or embedded in a URL or a JSON blob, is still a leak.

    Args:
        *texts: Strings to check. ``None`` is accepted and skipped so a caller can pass an optional
            field without guarding it.
        secrets: Override only to check a value that is not one of the shared sentinels (a per-test
            token, say); the default covers all three.
    """
    secret_list = list(secrets)
    offenders: list[str] = []
    for position, text in enumerate(texts):
        if text is None:
            continue
        if not isinstance(text, str):
            raise TypeError(f"assert_no_secrets takes strings; argument {position} is {type(text).__name__}")
        for secret in secret_list:
            index = text.find(secret)
            if index >= 0:
                name = _SENTINEL_NAMES.get(secret, "a secret")
                offenders.append(f"argument {position} leaks {name}: {_masked(text, secret, index)}")
    assert not offenders, "secret material must never be exposed (redaction rule):\n  " + "\n  ".join(
        offenders
    )
