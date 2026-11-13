"""Who owns the LiveKit room, and who may close it (*borrowed-room rule*, *room ownership handoff*).

One rule, two opposite obligations, and getting either backwards is invisible until production:

* a room **we own** (the relayed join's private room) must be disconnected *and deleted* on the way
  out, or every session leaves a room behind on the developer's LiveKit project;
* a room we **borrowed** (the direct join's transport room) must never be touched, because Pipecat's
  transport reference-counts its own connect/disconnect and our closing it would break the pipeline.

The two halves use different rooms and are asserted against the *server*: a room the adapter merely
disconnected from and left to time out would pass any client-side check.

The direct-join half needs no spy on ``rtc.Room.disconnect``. The ``EndFrame`` reaches processors in
pipeline order and each one's ``stop()`` is awaited before the frame is pushed on, so a witness placed
between the service and ``transport.output()`` observes the room at the one moment that answers the
question: the service's teardown, ``session.aclose()`` and all, has finished, and the transport has not
started its own yet. If our session disconnected the borrowed room, the room is already gone by then.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pipecat.frames.frames import EndFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_synthesia.livekit._transport_access import get_room
from pipecat_synthesia.service import ServiceState
from tests.harness import bidirectional_pipeline, make_service
from tests.integration._join_modes import (
    AVATAR_IDS,
    WAIT,
    direct_join_bot_token,
    livekit_transport,
    private_room,
    transport_room,
)
from tests.integration._launcher import Launcher
from tests.support.events import EventRecorder
from tests.support.livekit_dev import LiveKitDev, list_rooms

POLL_SECS = 0.1


class EndFrameWitness(FrameProcessor):
    """Answers one question at the exact moment the ``EndFrame`` passes this point in the pipeline.

    Direct mode, so the observation happens inside the push from the processor upstream of it — which
    is what makes "after the service's ``stop()``, before the transport's" a fact about the pipeline
    rather than a hope about task scheduling.
    """

    def __init__(self, observe: Callable[[], object], **kwargs) -> None:
        super().__init__(enable_direct_mode=True, **kwargs)
        self._observe = observe
        #: One entry per ``EndFrame`` seen travelling downstream.
        self.observed: list[object] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame) and direction == FrameDirection.DOWNSTREAM:
            self.observed.append(self._observe())
        await self.push_frame(frame, direction)


async def _wait_rooms_lack(dev: LiveKitDev, name: str, *, timeout: float = WAIT) -> None:
    """Poll ``ListRooms`` until ``name`` is gone; ``AssertionError`` naming what is still there.

    The ``until()`` of ``tests/support/waiting.py`` takes a synchronous predicate and this question is
    an HTTP call, so the poll is spelled out. Same guarantee: the outcome is asserted, never a duration.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        rooms = await list_rooms(dev)
        if name not in rooms:
            return
        if loop.time() >= deadline:
            raise AssertionError(
                f"timed out after {timeout}s waiting for room {name!r} to be deleted; ListRooms: {rooms}"
            )
        await asyncio.sleep(POLL_SECS)


@pytest.mark.integration
async def test_teardown_ownership(livekit_server: LiveKitDev, room_name: str, launcher: Launcher) -> None:
    """An owned room is deleted on the way out; a borrowed room is never touched by us."""
    dev = livekit_server

    # ---------------------------------------------------------------- relayed join: owns its room
    owned_name = f"{room_name}-2a"
    api = await launcher.api()
    service = make_service(
        private_room(dev, room_name=owned_name), avatar_ids=AVATAR_IDS, api_url=api.base_url
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        assert owned_name in await list_rooms(dev), (
            "the adapter must have created the private room\n" + launcher.debug_dump()
        )
        await h.finish()

    assert service.state is ServiceState.ENDED
    assert rec.count("on_session_ended") == 1
    # The room, the bot inside it and the worker's session with it: all gone server-side.
    await _wait_rooms_lack(dev, owned_name)

    # ---------------------------------------------------------------- direct join: borrows its room
    api = await launcher.api()
    transport = livekit_transport(
        dev, room_name=room_name, token=direct_join_bot_token(dev, room_name=room_name)
    )
    room = transport_room(dev, transport)
    service = make_service(room, avatar_ids=AVATAR_IDS, api_url=api.base_url)
    rec = EventRecorder(service)

    # Asked when the EndFrame has been through the service and not yet through transport.output().
    witness = EndFrameWitness(lambda: get_room(transport).isconnected(), name="after-service")
    pipeline = Pipeline([transport.input(), room.input_filter(), service, witness, transport.output()])

    async with bidirectional_pipeline(pipeline) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        borrowed = get_room(transport)
        assert borrowed.isconnected() is True, launcher.debug_dump()
        assert room_name in await list_rooms(dev), launcher.debug_dump()
        await h.finish()

    assert service.state is ServiceState.ENDED
    assert len(witness.observed) == 1, (
        "the EndFrame never reached the witness, so this test proved nothing about ownership: the "
        "service's stop() must have raised (push_error swallows it and the frame is never pushed) or "
        "the pipeline ended some other way. Fix that first."
    )
    assert witness.observed == [True], (
        "borrowed-room rule: the transport's borrowed room was already disconnected by the time the "
        "EndFrame had cleared the service. Almost certainly our aclose() not honouring "
        "RoomHandle(owns_room=False) "
        "— but check the other possibility too: `transport.input()` sits upstream of us and calls "
        "`_client.disconnect()` first, and only Pipecat's connect refcount (input.setup + output.setup "
        f"= 2) stops that from closing the room. Witness saw isconnected()={witness.observed}."
    )
    # And the transport did close it, from its own stop(): the room is ours to leave alone, not to keep
    # alive. "Never touched" must not quietly become "kept open by us".
    assert borrowed.isconnected() is False, "the transport disconnects its own room on EndFrame"
    # Both halves really did launch a worker, each into its own room.
    assert [fake.room_name for fake in launcher.fakes] == [owned_name, room_name], launcher.debug_dump()
