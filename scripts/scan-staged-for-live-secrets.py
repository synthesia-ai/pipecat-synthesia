#!/usr/bin/env python3
"""Refuse a commit that contains a value from this machine's .env, in any shape.

Pattern-based detectors miss the shapes that matter most in practice. gitleaks catches
``API_KEY = "..."`` and ``KEY=...``, but not a pasted debug log (``DEBUG connect api_key=...``),
a positional argument (``LiveKitAPI("<key>", "<secret>")``), or a secret quoted in prose. Those
are exactly the pasted-debugging artefacts that leak keys.

This hook does not guess. It reads the credential values this machine actually has and looks for
them verbatim in the staged content, so every shape is caught and there are no false positives.

It never prints a value -- only the variable name and the file it appeared in.

No .env, no work: developers and CI without one exit 0 immediately, so this costs nothing to
anyone who has not configured credentials.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Values that are not secret and would cause noisy false positives -- a project URL or an avatar
# id legitimately appears in docs and examples.
NOT_SECRET = frozenset(
    {
        "LIVEKIT_URL",
        "SYNTHESIA_API_URL",
        "SYNTHESIA_AVATAR_ID",
        "LIVEKIT_ROOM",
    }
)

# Below this length a value matches too much unrelated text to be usable as a needle.
MIN_LENGTH = 8


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def load_secrets(env_path: Path) -> dict[str, str]:
    """Parse KEY=VALUE without sourcing the file (no shell expansion, no code execution)."""
    secrets: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if name in NOT_SECRET or len(value) < MIN_LENGTH:
            continue
        secrets[name] = value
    return secrets


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def staged_content(path: str) -> str:
    """Read the staged blob, not the worktree file -- they can differ."""
    out = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=False)
    return out.stdout.decode("utf-8", errors="replace") if out.returncode == 0 else ""


def main() -> int:
    try:
        root = repo_root()
    except subprocess.CalledProcessError:
        return 0

    env_path = root / ".env"
    if not env_path.is_file():
        return 0

    secrets = load_secrets(env_path)
    if not secrets:
        return 0

    hits: list[tuple[str, str]] = []
    for path in staged_files():
        content = staged_content(path)
        if not content:
            continue
        hits.extend((path, name) for name, value in secrets.items() if value in content)

    if not hits:
        return 0

    print("ERROR: a live credential value from your .env is in the staged content.\n")
    for path, name in hits:
        print(f"  {path}: contains the value of {name}")
    print(
        "\nNever commit a credential: your LiveKit API key and secret, for one, are admin-equivalent\n"
        "for the whole project. Remove the value and stage again. If this is a false positive, the\n"
        "value in your .env is too generic to be used as a needle; shorten it or add it to\n"
        "NOT_SECRET in scripts/scan-staged-for-live-secrets.py."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
