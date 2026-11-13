#!/usr/bin/env python3
"""Report which credentials are configured, without ever printing a value.

    uv run python scripts/check_credentials.py

Reads `.env` from the repo root, the same way the examples do. Prints set/missing and a character
count per variable. The count is deliberate: it catches the common failure where a value was
pasted with surrounding quotes or a trailing newline, without revealing the secret itself.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Same as the examples: `.env` wins over a stale exported shell variable, so this reports what a
# run would actually use rather than what your shell happens to hold.
load_dotenv(override=True)

# (variable, required-for, note)
CHECKS: list[tuple[str, str, str]] = [
    ("SYNTHESIA_API_KEY", "always", "Synthesia developer portal"),
    ("SYNTHESIA_AVATAR_ID", "always", "av_-prefixed; the prefix is added for you if missing"),
    ("SYNTHESIA_API_URL", "optional", "override only for non-production environments"),
    ("SYNTHESIA_ALLOW_ANY_API_HOST", "optional", "1 lets SYNTHESIA_API_URL be a host outside synthesia.io"),
    ("LIVEKIT_URL", "always", "wss://<project>.livekit.cloud"),
    ("LIVEKIT_API_KEY", "always", ""),
    ("LIVEKIT_API_SECRET", "always", ""),
    ("OPENAI_API_KEY", "--s2s and the default pipeline", "your OpenAI account"),
    ("CARTESIA_API_KEY", "default pipeline only", "your Cartesia account; not needed for --s2s"),
]


def main() -> int:
    missing_required: list[str] = []
    width = max(len(name) for name, _, _ in CHECKS)

    for name, needed_for, note in CHECKS:
        raw = os.environ.get(name)
        if raw:
            flag = " (has surrounding whitespace!)" if raw != raw.strip() else ""
            status = f"set, {len(raw)} chars{flag}"
        else:
            status = "MISSING"
            if needed_for == "always":
                missing_required.append(name)
        detail = f"  [{needed_for}]" + (f" {note}" if note else "")
        print(f"  {name:<{width}}  {status}{detail}")

    print()
    if missing_required:
        print(f"Missing required: {', '.join(missing_required)}")
        return 1
    print("All always-required variables are set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
