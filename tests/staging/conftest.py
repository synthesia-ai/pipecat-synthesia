"""Fixtures for the staging tier: real Synthesia API, real LiveKit project.

**This tier is a smoke run, not a regression suite.** It is run by hand, or on a schedule, by
someone who has the credentials. It must never run on a pull request: it costs money, it consumes a
concurrent-session slot, and it talks to a shared LiveKit project. The `staging` marker keeps it out
of every other selection, and every fixture here skips loudly rather than failing when the
credentials are absent.

**Room hygiene.** The LiveKit credentials in use are typically *shared*, and a LiveKit API key
cannot be permission-scoped -- anyone holding it can administer any room in the project. So this
tier never enumerates rooms it did not create: `room_name` is unique per run, and the only LiveKit
admin calls made here are scoped to that exact name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4

import pytest

#: Every variable the relayed-join round trip needs. Missing any one skips the tier.
REQUIRED_VARS = (
    "SYNTHESIA_API_KEY",
    "SYNTHESIA_AVATAR_ID",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
)

#: Rooms this tier creates are named `<prefix><hex>`, so a leaked one is identifiable as ours.
ROOM_PREFIX = "pcs-staging-"


@dataclass(frozen=True)
class StagingEnv:
    """The credentials and ids for one staging run. Values are never logged."""

    api_key: str
    avatar_id: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    @property
    def api_url(self) -> str | None:
        """`SYNTHESIA_API_URL` if the run targets a non-production environment."""
        return os.environ.get("SYNTHESIA_API_URL") or None


@pytest.fixture
def staging_env() -> StagingEnv:
    """Credentials from the environment, or skip.

    Skipping names every missing variable at once: discovering them one run at a time, each costing
    a launch timeout, is the kind of friction that stops people running this tier at all.
    """
    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        pytest.skip(f"staging tier needs {', '.join(missing)} (see .env.example)")
    return StagingEnv(
        api_key=os.environ["SYNTHESIA_API_KEY"],
        avatar_id=os.environ["SYNTHESIA_AVATAR_ID"],
        livekit_url=os.environ["LIVEKIT_URL"],
        livekit_api_key=os.environ["LIVEKIT_API_KEY"],
        livekit_api_secret=os.environ["LIVEKIT_API_SECRET"],
    )


@pytest.fixture
def room_name() -> str:
    """A room name unique to this run, so teardown can delete by exact name and nothing else."""
    return f"{ROOM_PREFIX}{uuid4().hex[:12]}"
