"""Mint the two LiveKit tokens `bot_direct_join.py` needs.

    uv run python examples/mint_tokens.py --room my-room

Prints a BOT token (for the Pipecat bot) and a HUMAN token (for you, in a browser).

Why this exists rather than `pipecat.runner.livekit.generate_token_with_agent`
-----------------------------------------------------------------------------
That helper sets the `video.agent` **grant**, which is what makes LiveKit clients render the
participant as an agent. It does NOT set the top-level `kind` **claim**. Those are different
fields, and the avatar worker reads `kind`. Pipecat's dev runner (`-t livekit`) mints its bot
token with that helper, so `LiveKitTransportRoom(transport)` refuses it at startup:

    ValueError: the bot's LiveKit token lacks the top-level kind="agent" claim
    (AccessToken.with_kind("agent")); the avatar worker would treat the bot as a human and
    never end the session when the real human leaves. ...

A session that never ends keeps billing, so this matters more than it looks. LiveKit Cloud's own
agent-dispatch tokens carry `kind`; hand-minted ones usually do not, which is the trap. We set
both: `.with_kind("agent")` and `VideoGrants(agent=True)`.

Note on running this
--------------------
Use `uv run python`, not a bare `python`: the `livekit` package lives in the project venv, and
bare `python` gives `ModuleNotFoundError: No module named 'livekit'`.

Do NOT pipe its output through a secret-scrubbing wrapper: the tokens it prints are JWTs, and any
such filter will replace them with a placeholder instead of the value you need. It reads
`LIVEKIT_*` from the environment or a `.env` in the repo root.

Tokens are bearer credentials scoped to one room. Default TTL is deliberately short.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from livekit import api


def _grants(room: str) -> dict:
    return dict(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )


def mint_bot_token(key: str, secret: str, room: str, identity: str, ttl_minutes: int) -> str:
    """The bot's token: agent grant AND agent kind. Both are required -- see the module docstring."""
    import datetime

    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_kind("agent")
        .with_grants(api.VideoGrants(agent=True, **_grants(room)))
        .with_ttl(datetime.timedelta(minutes=ttl_minutes))
        .to_jwt()
    )


def mint_human_token(key: str, secret: str, room: str, identity: str, ttl_minutes: int) -> str:
    """Your browser's token: a plain participant, no agent grant and no agent kind."""
    import datetime

    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(**_grants(room)))
        .with_ttl(datetime.timedelta(minutes=ttl_minutes))
        .to_jwt()
    )


def main() -> int:
    # override=True to match `bot_direct_join.py` and Pipecat's own runner. If one of these read the
    # file and the other let a stale exported shell variable win, you could mint a token for one room
    # and have the bot join another -- which fails as an unrelated-looking permissions error. First
    # thing in main(), before the parser reads LIVEKIT_ROOM, and not at module scope, so that
    # importing this file loads nothing.
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--room", default=os.environ.get("LIVEKIT_ROOM"), help="room name")
    parser.add_argument("--bot-identity", default="pipecat-bot")
    parser.add_argument("--human-identity", default="human")
    parser.add_argument("--ttl-minutes", type=int, default=60)
    args = parser.parse_args()

    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    url = os.environ.get("LIVEKIT_URL", "<LIVEKIT_URL not set>")
    if not key or not secret:
        print("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set (env or .env).", file=sys.stderr)
        return 1
    if not args.room:
        print("Pass --room, or set LIVEKIT_ROOM.", file=sys.stderr)
        return 1

    bot = mint_bot_token(key, secret, args.room, args.bot_identity, args.ttl_minutes)
    human = mint_human_token(key, secret, args.room, args.human_identity, args.ttl_minutes)

    print(f"# room: {args.room}   url: {url}   ttl: {args.ttl_minutes}m\n")
    print("# For examples/bot_direct_join.py -- add to .env or export:")
    print(f"LIVEKIT_ROOM={args.room}")
    print(f"BOT_TOKEN={bot}\n")
    print("# For you, in a browser (e.g. https://meet.livekit.io -> Custom):")
    print(f"HUMAN_TOKEN={human}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
