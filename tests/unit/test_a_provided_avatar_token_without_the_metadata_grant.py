"""A provided avatar token without the `canUpdateOwnMetadata` grant is refused, as an invalid room token.

The avatar worker reports readiness by writing `avatar_status` and `avatar_ready` to its own participant
attributes. livekit-server refuses those writes unless the worker's token, the avatar token, carries
`canUpdateOwnMetadata`: it answers the worker `NOT_ALLOWED`, which livekit rtc 1.1.19's `set_attributes`
does not raise, and tells nobody else. So `on_avatar_ready` never fires and nothing says why. The minted
avatar token carries the grant. A provided avatar token comes from the developer's own minting code; the
grant set the LiveKit plugin mints lacks it, and livekit-api leaves it out unless asked. So a direct join
must refuse such a token, like any other token the avatar worker cannot work with: at construction as a
misuse error (`ValueError`), or at launch as `INVALID_ROOM_TOKEN` before any session request. Either place
is accepted here; construction is where *misuse vs session errors* puts every check that does not depend on
the time.

`tests/integration/test_a_provided_avatar_token_without_the_metadata_grant.py` shows the same token against a
real livekit-server: the session becomes active, and readiness never arrives.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

import pytest
from livekit import api as lk_api

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from pipecat_synthesia.protocol import TOKEN_ATTR_PUBLISH_ON_BEHALF
from tests.fakes.fake_rtc import FakeLocalParticipant, FakeRoom
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.fakes.fake_transport import FAKE_AGENT_TOKEN, FakeLiveKitTransport
from tests.support.callbacks import make_launch_context
from tests.support.secrets import LK_API_SECRET

BOT = "pipecat-bot"  # FAKE_AGENT_TOKEN's identity
ROOM = "fake-room"  # and its room
AVATAR = DEFAULT_AVATAR_IDENTITY
#: An input. The bot never connects, so a launch that refuses nothing fails here, rather than after 30 s.
CONNECT_TIMEOUT = 0.2


def avatar_token(grants: lk_api.VideoGrants) -> str:
    """An avatar token as a developer would mint it with livekit-api: `kind="agent"`, the bot's room,
    publish-on-behalf the bot, and `grants`. It passes every check the constructor makes today."""
    return (
        lk_api.AccessToken("fake-lk-key", LK_API_SECRET)
        .with_identity(AVATAR)
        .with_name("Synthesia avatar")
        .with_kind("agent")
        .with_grants(grants)
        .with_attributes({TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT})
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )


WITHOUT_THE_GRANT = pytest.mark.parametrize(
    "grants",
    [
        # the grant set the LiveKit plugin mints its avatar token with
        pytest.param(
            lk_api.VideoGrants(
                room_join=True, room=ROOM, can_publish=True, can_subscribe=True, can_publish_data=True
            ),
            id="grant-absent",
        ),
        pytest.param(
            lk_api.VideoGrants(room_join=True, room=ROOM, can_update_own_metadata=False), id="grant-false"
        ),
    ],
)


@WITHOUT_THE_GRANT
async def test_a_direct_join_refuses_a_provided_avatar_token_without_the_metadata_grant(
    grants: lk_api.VideoGrants,
) -> None:
    room = FakeRoom(ROOM, local_participant=FakeLocalParticipant(BOT))
    transport = FakeLiveKitTransport(room=room, token=FAKE_AGENT_TOKEN, participant_id="")
    try:
        adapter = LiveKitTransportRoom(
            cast(Any, transport), avatar_token=avatar_token(grants), connect_timeout=CONNECT_TIMEOUT
        )
    except ValueError:
        return  # refused at construction, as a misuse error
    api = FakeSessionApiClient()

    with pytest.raises(SynthesiaError) as excinfo:
        await adapter.launch(make_launch_context(api, avatar_ids=("av_a",)))

    requested = len(api.calls)  # counted, never printed: each call carries the avatar token
    assert requested == 0, (
        f"a session was requested with an avatar token the avatar worker cannot report readiness with; the "
        f"launch then failed for another reason: {excinfo.value.type}: {excinfo.value}"
    )
    assert excinfo.value.type is ErrorType.INVALID_ROOM_TOKEN
