"""A relayed join never cancels the bot's in-flight connect to its private room.

What this pins: cancelling the LiveKit SDK's `Room.connect()` can make the SDK send the bot process
SIGTERM. So when the launch is cancelled while the bot is connecting (a `CancelFrame` or `EndFrame`
mid-launch), even twice, or when the adapter's own connect budget runs out, `PrivateLiveKitRoom` leaves
the connect running and deletes the private room, and the launch ends only once the connect has ended
and, if it connected, the bot has left the room. A connect that fails instead is logged, and its failure
is never reported as unretrieved, even after the launch stopped waiting for it; one that connects after
that is still disconnected. And the service waits long enough for such a launch to end.

Through `PrivateLiveKitRoom` on its `room_factory` and `api_factory` seams, with the connect parked at a
gate the test opens. Budgets are inputs: `LONG` is far beyond anything `until()` here waits for.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import re
from typing import Any

import pytest
from livekit import rtc as lk_rtc

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._types import ROOM_DELETE_SECS, ROOM_DISCONNECT_SECS
from pipecat_synthesia.livekit.private_room import PrivateLiveKitRoom
from pipecat_synthesia.service import LAUNCH_CANCEL_SECS
from tests.fakes.fake_lk_api import LiveKitApiFactory, RoomFactory
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.support.callbacks import make_relayed_launch_context
from tests.support.secrets import LK_API_SECRET
from tests.support.waiting import until

ROOM = "private-room"
LONG = 60.0


def launch(rooms: RoomFactory, apis: LiveKitApiFactory, connect_timeout: float = LONG) -> asyncio.Task[Any]:
    room = PrivateLiveKitRoom(
        url="wss://livekit.example",
        api_key="APIlivekit",
        api_secret=LK_API_SECRET,
        room_name=ROOM,
        room_factory=rooms,
        api_factory=apis,
        connect_timeout=connect_timeout,
    )
    ctx = make_relayed_launch_context(FakeSessionApiClient(), launch_timeout=2 * LONG)
    return asyncio.ensure_future(room.launch(ctx))


async def launch_and_cancel(
    rooms: RoomFactory, apis: LiveKitApiFactory, *, turns: int = 0, cancels: int = 1
) -> asyncio.Task[Any]:
    """Park the connect, cancel the launch, and return it once the room is deleted and the connect parked."""
    task = launch(rooms, apis)
    await until(lambda: bool(rooms.rooms) and rooms.room.connect_started.is_set(), what="the bot's connect")
    for _ in range(turns):
        await asyncio.sleep(0)  # moves where the cancel lands; not a synchronisation
    task.cancel()
    await until(lambda: apis.api.room.deleted_names == [ROOM], what="the private room's delete")
    for _ in range(cancels - 1):
        task.cancel()
    assert not task.done(), "the launch ended before the bot's connect did"
    return task


@pytest.mark.parametrize(("turns", "cancels"), [(0, 1), (5, 1), (0, 2)], ids=["at-once", "deferred", "twice"])
async def test_a_cancelled_launch_leaves_the_connect_running_and_the_room_once_it_connects(
    turns: int, cancels: int
) -> None:
    gate = asyncio.Event()
    rooms, apis = RoomFactory(connect_gate=gate), LiveKitApiFactory()
    task = await launch_and_cancel(rooms, apis, turns=turns, cancels=cancels)

    gate.set()  # the connect now succeeds
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not rooms.room.connect_cancelled, "the bot's in-flight Room.connect() was cancelled"
    assert rooms.room.disconnect_calls == 1, "the bot stayed in the private room"


async def test_a_connect_past_the_budget_is_not_cancelled_and_leaves_the_room_once_it_connects() -> None:
    gate = asyncio.Event()
    rooms, apis = RoomFactory(connect_gate=gate), LiveKitApiFactory()
    task = launch(rooms, apis, connect_timeout=0.01)
    await until(lambda: bool(apis.instances) and apis.api.room.deleted_names == [ROOM], what="the delete")
    assert not task.done(), "the launch ended before the bot's connect did"

    gate.set()
    with pytest.raises(SynthesiaError) as caught:
        await task
    assert caught.value.type is ErrorType.CONNECTION
    assert re.search(r"did not connect to the private room within \d+\.\ds", str(caught.value)), caught.value
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert not rooms.room.connect_cancelled, "the bot's in-flight Room.connect() was cancelled"
    assert rooms.room.disconnect_calls == 1, "the bot stayed in the private room"


async def fail_late(let_go: bool, failure: Exception, caplog: pytest.LogCaptureFixture) -> RoomFactory:
    """A connect that fails once the launch has stopped waiting for it; its task is unreferenced on return."""
    gate = asyncio.Event()
    rooms, apis = RoomFactory(connect_gate=gate, connect_error=failure), LiveKitApiFactory()
    if let_go:  # the budget runs out, then the wait for the connect does: the launch lets it go
        task = launch(rooms, apis, connect_timeout=0.01)
        await until(task.done, what="the launch to let the connect go")
    else:
        task = await launch_and_cancel(rooms, apis)
    gate.set()  # the connect now fails
    await until(
        lambda: any(r.levelno >= logging.INFO and str(failure) in r.getMessage() for r in caplog.records),
        what="the connect's failure to be logged",
    )
    with pytest.raises((asyncio.CancelledError, SynthesiaError)):
        await task
    return rooms


@pytest.mark.parametrize("let_go", [False, True], ids=["waited-for", "let-go"])
async def test_a_connect_that_fails_after_the_launch_stopped_waiting_is_logged_and_retrieved(
    caplog_loguru: pytest.LogCaptureFixture, let_go: bool
) -> None:
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        rooms = await fail_late(let_go, lk_rtc.ConnectError("the private room was deleted"), caplog_loguru)
        gc.collect()  # asyncio reports an unretrieved failure when its task is collected
    finally:
        loop.set_exception_handler(previous)
    assert [c for c in reported if "never retrieved" in str(c.get("message"))] == [], reported
    assert not rooms.room.connect_cancelled, "the bot's in-flight Room.connect() was cancelled"


async def test_a_let_go_connect_that_succeeds_later_is_disconnected() -> None:
    gate = asyncio.Event()
    rooms, apis = RoomFactory(connect_gate=gate), LiveKitApiFactory()
    task = launch(rooms, apis, connect_timeout=0.01)
    with pytest.raises(SynthesiaError):
        await task  # the budget runs out, then the wait for the connect does: the launch lets it go
    assert rooms.room.disconnect_calls == 0

    gate.set()  # the connect now succeeds
    await until(lambda: rooms.room.disconnect_calls >= 1, what="the late connect to be disconnected")
    assert rooms.room.disconnect_calls == 1
    assert not rooms.room.connect_cancelled


def test_the_service_waits_long_enough_for_a_launch_cancelled_mid_connect() -> None:
    """Derived, not a value: the default connect_timeout, + 1 s for the leave task's wait past the SDK's
    own timeout, + its disconnect, + the release's delete."""
    connect_timeout = inspect.signature(PrivateLiveKitRoom).parameters["connect_timeout"].default
    assert connect_timeout + 1.0 + ROOM_DISCONNECT_SECS + ROOM_DELETE_SECS <= LAUNCH_CANCEL_SECS
