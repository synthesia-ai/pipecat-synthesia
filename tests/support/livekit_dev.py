"""Dev-token and room helpers for the integration suite — the LiveKit SDK, and nothing of ours.

**Why this exists rather than ``pipecat_synthesia.livekit.token``.** The conformance tests must mint
tokens without importing the product's minter: otherwise a bug in the minter would make the *oracle*
wrong, and the conformance test would agree with itself. These helpers go straight to
``livekit.api.AccessToken``, so a token the fake worker accepts here was minted by the LiveKit SDK, not
by our own code.

Nothing here logs a token: a JWT in a CI log is a credential in a CI log even when the secret is
``devkey``/``secret``, and the habit is what the *redaction rule* is really protecting.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from livekit import api, rtc

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pytest
    from livekit.protocol.models import ParticipantInfo

__all__ = [
    "LiveKitDev",
    "connect_room",
    "delete_room",
    "list_participants",
    "list_rooms",
    "mint_dev_token",
    "server_api",
    "unique_room_name",
]

#: Room names are built from the test's node id, which for a parametrised test carries ``[`` and ``]``.
#: Everything outside this set is folded to ``-`` so the name stays a plain identifier; for an ordinary
#: test name the substitution is a no-op.
_SAFE_ROOM_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


@dataclass(frozen=True)
class LiveKitDev:
    """Where the integration suite's ``livekit-server`` is and how to authenticate to it.

    Yielded by the ``livekit_server`` fixture (``tests/integration/conftest.py``). Frozen because it is
    session-scoped shared state: a test that "fixed" its URL in place would silently redirect every
    later test in the session.

    Attributes:
        url: WebSocket URL for ``rtc.Room.connect`` (``ws://127.0.0.1:7880``).
        api_key: LiveKit API key (``devkey`` on a ``--dev`` server).
        api_secret: LiveKit API secret (``secret`` on a ``--dev`` server).
        http_url: HTTP base for ``LiveKitAPI`` / the readiness poll (``http://127.0.0.1:7880``).
    """

    url: str
    api_key: str
    api_secret: str
    http_url: str


def unique_room_name(request: pytest.FixtureRequest) -> str:
    """A room name no other test in this run — or any concurrent run — can collide with.

    A room left behind by a failed test is the classic way an integration suite starts failing tests
    that have nothing wrong with them. The test name is kept in
    the room name so ``ListRooms`` on a wedged server names the culprit; the uuid suffix is what makes
    a re-run of the same test safe.
    """
    stem = "".join(char if char in _SAFE_ROOM_CHARS else "-" for char in request.node.name[:20])
    return f"pcs-{stem}-{uuid4().hex[:8]}"


def mint_dev_token(
    dev: LiveKitDev,
    *,
    identity: str,
    room: str,
    kind: str | None = "agent",
    agent_grant: bool = False,
    attributes: dict[str, str] | None = None,
    name: str | None = None,
    ttl_seconds: int = 3600,
) -> str:
    """Mint a LiveKit JWT with the dev server's key pair.

    Two knobs matter, and they are genuinely different things: ``kind="agent"`` sets the top-level ``kind``
    claim, which is what makes the server report ``PARTICIPANT_KIND_AGENT``; ``agent_grant`` sets
    ``video.agent``, a permission to register as an Agent-framework worker that does **not** affect the
    reported kind. ``kind=None`` omits the claim and mints a STANDARD participant — the "the bot is
    STANDARD" and "a human joined" cases.

    Args:
        dev: The dev server to mint against.
        identity: Participant identity (the ``sub`` claim).
        room: Room the grant is scoped to.
        kind: Participant kind, or ``None`` for STANDARD.
        agent_grant: Set ``video.agent``. Rarely what you want; see above.
        attributes: Token attributes, e.g. ``{"lk.publish_on_behalf": bot_identity}``.
        name: Display name; defaults to ``identity``.
        ttl_seconds: Token lifetime.

    Returns:
        The signed JWT. Never log it.
    """
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        # Required for `set_attributes` to have any effect (measured against livekit-server 1.9.1).
        # Without it the server drops a participant's write to its own attributes *silently*:
        # `set_attributes` returns normally, no error reaches the client, and the attributes never change.
        # That is how the worker's `avatar_ready` / `avatar_status` would go missing with no diagnostic,
        # so the avatar token the product mints carries this grant too.
        can_update_own_metadata=True,
        # `or None` because `Claims.asdict` drops None but keeps False, and a literal `"agent": false`
        # in the grant is a claim we never want to assert against.
        agent=agent_grant or None,
    )
    token = (
        api.AccessToken(dev.api_key, dev.api_secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_grants(grants)
    )
    if kind is not None:
        # `with_kind` is typed as a Literal of the five kinds LiveKit defines. The parameter here stays a
        # plain `str` so a test can mint a deliberately bogus kind and watch the server reject it, which a
        # Literal would forbid at type-check time rather than at the layer being tested.
        token = token.with_kind(cast(Any, kind))
    if attributes:
        token = token.with_attributes(attributes)
    return token.with_ttl(timedelta(seconds=ttl_seconds)).to_jwt()


@contextlib.asynccontextmanager
async def connect_room(dev: LiveKitDev, token: str, *, timeout: float = 15.0) -> AsyncIterator[rtc.Room]:
    """Connect one ``rtc.Room`` and disconnect it however the body ends, so the test leaves nothing running.

    One context manager per *participant*: a test that needs a bot and a human opens two, because one
    ``rtc.Room`` is one participant. Both the connect and the disconnect are bounded — an unbounded
    ``room.connect`` against a server that accepted the TCP connection but never finished the join is
    a 120-second test timeout with no useful message.

    Disconnect errors are swallowed: the room may already be gone (the test deleted it, the fake
    disconnected us), and a teardown that raises hides the assertion that actually failed.
    """
    room = rtc.Room()
    try:
        await asyncio.wait_for(
            room.connect(dev.url, token, options=rtc.RoomOptions(auto_subscribe=True)), timeout
        )
        yield room
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(room.disconnect(), 5.0)


@contextlib.asynccontextmanager
async def server_api(dev: LiveKitDev) -> AsyncIterator[api.LiveKitAPI]:
    """A LiveKit server-API client, closed however the body ends.

    Each helper below opens its own rather than sharing one: ``LiveKitAPI`` owns an ``aiohttp`` session,
    and a session cached across tests would outlive the function-scoped event loop it was created on,
    and surface later as an unraisable ``ResourceWarning`` in an unrelated test.
    The per-call cost is a local TCP connection; the ``aclose`` guarantee has one owner here.
    """
    lkapi = api.LiveKitAPI(dev.http_url, dev.api_key, dev.api_secret)
    try:
        yield lkapi
    finally:
        await lkapi.aclose()


async def delete_room(dev: LiveKitDev, name: str) -> None:
    """Delete a room server-side, ignoring "it was never created".

    Called from the ``room_name`` fixture's ``finally`` for every test, so "not found" is the common
    case, not an error: a test that skipped before connecting never made the room exist.
    """
    async with server_api(dev) as lkapi:
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=name))
        except Exception as err:
            # `TwirpError.code == "not_found"` is the benign one; anything else during cleanup
            # is still not worth failing a passing test over, so it is swallowed too but not silently.
            if getattr(err, "code", None) != "not_found":
                print(f"delete_room({name!r}) failed: {type(err).__name__}: {err}")


async def list_rooms(dev: LiveKitDev) -> list[str]:
    """The names of every room that currently exists server-side.

    For the teardown test: "the adapter deleted the room it created" is only true if the *server* has
    stopped listing it. A client-side check would
    pass against a room the adapter merely disconnected from and left behind to time out.
    """
    async with server_api(dev) as lkapi:
        response: Any = await lkapi.room.list_rooms(api.ListRoomsRequest())
        return [room.name for room in response.rooms]


async def list_participants(dev: LiveKitDev, name: str) -> list[ParticipantInfo]:
    """Server-side participant list for ``name`` — the ground truth for identity and kind.

    The server's view is the one that matters for the *agent-kind requirement*: a bot that *believes* it
    is an agent but joined with a STANDARD token shows up here as STANDARD, and no client-side assertion
    would notice.
    """
    async with server_api(dev) as lkapi:
        response: Any = await lkapi.room.list_participants(api.ListParticipantsRequest(room=name))
        return list(response.participants)
