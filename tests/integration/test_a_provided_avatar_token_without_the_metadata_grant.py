"""A direct join with a provided avatar token: the avatar worker's readiness reaches the service, or the
token is refused.

The avatar worker reports readiness by writing `avatar_status` and `avatar_ready` to its own participant
attributes, with the avatar token it joined with. livekit-server applies a participant's write to its own
attributes only if that token carries `canUpdateOwnMetadata`. Without it the server answers the writer
`NOT_ALLOWED`, which livekit rtc 1.1.19's `set_attributes` does not raise, and tells nobody else. The minted
avatar token carries the grant. A provided one carries whatever the developer's minting code put in it.

So, against a real livekit-server, with `FakeAvatarWorker` joining on the very token the developer provided:
- with the grant, the session becomes active and `on_avatar_ready` fires;
- without it, the session becomes active and `on_avatar_ready` never fires, with no error anywhere, unless
  the direct join refuses the token, at construction (`ValueError`) or at launch (`INVALID_ROOM_TOKEN`).

The unit tier's `test_a_provided_avatar_token_without_the_metadata_grant.py` pins the refusal itself.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from livekit import api as lk_api
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.pipeline import Pipeline

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._transport_access import get_room
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from pipecat_synthesia.protocol import TOKEN_ATTR_PUBLISH_ON_BEHALF
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.integration._join_modes import (
    AVATAR_IDS,
    DIRECT_JOIN_BOT,
    WAIT,
    direct_join_bot_token,
    livekit_transport,
)
from tests.integration._launcher import Launcher
from tests.support.events import EventRecorder
from tests.support.livekit_dev import LiveKitDev
from tests.support.waiting import until


def provided_avatar_token(dev: LiveKitDev, *, room_name: str, metadata_grant: bool) -> str:
    """The avatar token a developer mints for a direct join with livekit-api: the LiveKit plugin's grant
    set, plus `canUpdateOwnMetadata` or not. Minted here, not by `mint_dev_token`, which always sets it."""
    grants = lk_api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        can_update_own_metadata=True if metadata_grant else None,
    )
    return (
        lk_api.AccessToken(dev.api_key, dev.api_secret)
        .with_identity(DEFAULT_AVATAR_IDENTITY)
        .with_name("Synthesia avatar")
        .with_kind("agent")
        .with_grants(grants)
        .with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: DIRECT_JOIN_BOT})
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )


def refusals(h: Harness) -> list[SynthesiaError]:
    return [
        frame.exception
        for frame in h.up
        if isinstance(frame, ErrorFrame)
        and isinstance(frame.exception, SynthesiaError)
        and frame.exception.type is ErrorType.INVALID_ROOM_TOKEN
    ]


@pytest.mark.integration
@pytest.mark.parametrize("metadata_grant", [True, False], ids=["with-the-grant", "without-the-grant"])
async def test_a_provided_avatar_token_lets_the_avatar_worker_report_readiness_or_is_refused(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher, metadata_grant: bool
) -> None:
    dev = livekit_server
    api = await launcher.api()
    transport = livekit_transport(
        dev, room_name=room_name, token=direct_join_bot_token(dev, room_name=room_name)
    )
    token = provided_avatar_token(dev, room_name=room_name, metadata_grant=metadata_grant)
    try:
        room = LiveKitTransportRoom(transport, avatar_token=token)
    except ValueError:
        return  # refused at construction, as a misuse error
    # required=False: a refusal at launch is an error the pipeline survives, so it can be read here
    service = make_service(room, avatar_ids=AVATAR_IDS, api_url=api.base_url, required=False)
    rec = EventRecorder(service)
    pipeline = Pipeline([transport.input(), room.input_filter(), service, transport.output()])

    async with bidirectional_pipeline(pipeline) as h:
        try:
            await until(
                lambda: rec.count("on_avatar_ready") > 0 or bool(refusals(h)),
                timeout=WAIT,
                what="on_avatar_ready, or the avatar token refused at launch",
            )
        except AssertionError:
            avatar = get_room(transport).remote_participants.get(DEFAULT_AVATAR_IDENTITY)
            seen = sorted(avatar.attributes) if avatar is not None else None
            errors = [str(frame.exception) for frame in h.up if isinstance(frame, ErrorFrame)]
            raise AssertionError(
                f"no readiness and no refusal within {WAIT}s: events {rec.names()}; errors pushed {errors}; "
                f"the avatar participant's attribute keys, as the bot sees them: {seen}\n"
                f"{launcher.debug_dump()}"
            ) from None
        await h.finish()

    if metadata_grant:
        assert not refusals(h), "a provided avatar token with every grant the avatar worker needs was refused"
        assert rec.count("on_avatar_ready") == 1
