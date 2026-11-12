"""Client identification names the join mode and the user transport.

An installed version only ever sends the fields it was built with, so these are wire contracts, asserted
with equality. What this file pins: the session request carries both in `X-Synthesia-Client`, a minted
avatar token carries `synthesia.join_mode`, and the user transport comes from a closed set of values
read from the Pipecat package of the first output transport downstream.

The classification cases call `_compat.transport_name()` directly: through the public route, each value
would need a whole pipeline; the two end-to-end tests cover that route.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.heygen.transport import HeyGenOutputTransport
from pipecat.transports.livekit.transport import LiveKitOutputTransport
from pipecat.transports.smallwebrtc.transport import SmallWebRTCOutputTransport

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia._compat import transport_name
from pipecat_synthesia.livekit._types import LiveKitCredentials, MintedAvatarToken
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME, decode_unsigned
from pipecat_synthesia.room import JoinMode, LaunchContext
from tests.fakes.fake_api import RecordedRequest, ScriptedResponse
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.secrets import LK_API_SECRET


def developers_subclass(base: type[BaseOutputTransport]) -> BaseOutputTransport:
    """A developer's own subclass of an output transport, defined in this module, forwarding every frame.

    Only `BaseOutputTransport.__init__` runs: the real transports' constructors need a live client.
    """

    def init(self: BaseOutputTransport) -> None:
        BaseOutputTransport.__init__(self, TransportParams())

    async def process_frame(self: BaseOutputTransport, frame: Frame, direction: FrameDirection) -> None:
        await FrameProcessor.process_frame(self, frame, direction)
        await self.push_frame(frame, direction)

    namespace = {"__init__": init, "process_frame": process_frame}
    return type(f"Developers{base.__name__}", (base,), namespace)()


async def launch_and_record(fake_api_factory: Any, join_mode: JoinMode, wrap=None) -> RecordedRequest:
    """Launch a service whose room sends the session request with a minted avatar token, as an adapter's
    launch does, through the service's own API client and client identification."""

    async def request_the_session(_room: FakeAvatarRoom, ctx: LaunchContext) -> None:
        token = MintedAvatarToken(LiveKitCredentials(api_key="lk-key", api_secret=LK_API_SECRET)).resolve(
            bot_identity="bot",
            room_name="room",
            avatar_identity=DEFAULT_AVATAR_IDENTITY,
            avatar_name=DEFAULT_AVATAR_NAME,
            client=ctx.client,
        )
        await ctx.api.create_session(avatar_ids=ctx.avatar_ids, livekit_url="wss://lk", livekit_token=token)

    async with fake_api_factory([ScriptedResponse(201, {"id": "ses_ident"})]) as api:
        service = make_service(
            FakeAvatarRoom(join_mode, during_launch=request_the_session), api_url=api.base_url
        )
        rec = EventRecorder(service)
        async with bidirectional_pipeline(wrap(service) if wrap else service) as h:
            await rec.wait("on_avatar_connected")
            await h.finish()
    assert len(api.requests) == 1, api.requests
    return api.requests[0]


@pytest.mark.unit
async def test_a_direct_join_over_a_developers_livekit_transport(fake_api_factory: Any) -> None:
    """A real `Pipeline` links the service to the output transport before `setup()`, where it is read."""
    request = await launch_and_record(
        fake_api_factory,
        JoinMode.DIRECT,
        wrap=lambda service: Pipeline([service, developers_subclass(LiveKitOutputTransport)]),
    )

    header = json.loads(request.headers["X-Synthesia-Client"])
    assert (header["join_mode"], header["transport"]) == ("direct", "livekit")
    assert dict(decode_unsigned(request.json["livekitToken"]).attributes) == {
        "lk.publish_on_behalf": "bot",
        "synthesia.protocol": "1",
        "synthesia.client": build_client_info().token_client_attr,
        "synthesia.join_mode": "direct",
    }


@pytest.mark.unit
async def test_a_relayed_join_with_no_output_transport_reads_unknown(fake_api_factory: Any) -> None:
    request = await launch_and_record(fake_api_factory, JoinMode.RELAYED)

    header = json.loads(request.headers["X-Synthesia-Client"])
    assert (header["join_mode"], header["transport"]) == ("relayed", "unknown")
    assert decode_unsigned(request.json["livekitToken"]).attributes["synthesia.join_mode"] == "relayed"


@pytest.mark.unit
@pytest.mark.parametrize(
    "base, expected",
    [
        (LiveKitOutputTransport, "livekit"),
        (SmallWebRTCOutputTransport, "smallwebrtc"),
        (HeyGenOutputTransport, "other"),  # a Pipecat transport package outside the named set
        (BaseOutputTransport, "other"),  # a transport from no Pipecat transport package
    ],
)
def test_a_developers_subclass_reads_as_its_pipecat_package(
    base: type[BaseOutputTransport], expected: str
) -> None:
    upstream = FrameProcessor(name="upstream")
    upstream.link(developers_subclass(base))
    assert transport_name(upstream) == expected
