"""Stand-ins for `lk_api.LiveKitAPI` and a connectable `rtc.Room`, for `PrivateLiveKitRoom`.

They plug into the two constructor seams the adapter already exposes, `api_factory` and `room_factory`,
so no relayed-join test monkeypatches the SDK.

**They record; they do not decide.** What is modelled, and why:

* `LiveKitAPI` owns an `aiohttp.ClientSession` and must be `aclose()`d exactly once on every path;
  `close_count` is how a test sees a leak. That close *suspends*: the real one awaits
  `ClientSession.close()`, then `connector.close()`. This is modelled because a close with no
  suspension point can never be interrupted, and "a second cancel cannot abort the cleanup"
  (*cancel-proof cleanup*) would then be untestable. `close_gate` holds it open on demand and
  `close_started` says when it was entered; the counter is bumped after the suspension, so a close that
  was cut off does not count as one that happened.
* a room-service failure arrives as `api.TwirpError`, which in `livekit-api` 1.2.1 *is*
  `api.ServerError` (one class, two names), carrying `.code: str`, `.message` and `.status: int`.
  `server_error()` builds a real one, so the production `except` clause is exercised against the class
  it will actually meet.
* `Room.connect(url, token, options=...)` raises `rtc.ConnectError`; `FakeConnectingRoom` adds exactly
  that to `FakeRoom` and nothing else.
"""

from __future__ import annotations

import asyncio
from typing import Any

from livekit import api as lk_api

from tests.fakes.fake_rtc import FakeLocalParticipant, FakeRoom

__all__ = [
    "FakeConnectingRoom",
    "FakeLiveKitAPI",
    "FakeRoomService",
    "LiveKitApiFactory",
    "RoomFactory",
    "server_error",
]


def server_error(code: str, *, status: int, message: str = "nope") -> lk_api.TwirpError:
    """A real `api.TwirpError` — the class the production code catches, not a look-alike."""
    return lk_api.TwirpError(code, message, status=status)


class FakeRoomService:
    """`lkapi.room`: records `create_room` / `delete_room` and answers with whatever was scripted."""

    def __init__(self) -> None:
        self.create_calls: list[Any] = []
        self.delete_calls: list[Any] = []
        #: Raised by `create_room` / `delete_room` instead of answering (repeats; not consumed).
        self.create_error: BaseException | None = None
        self.delete_error: BaseException | None = None
        #: When set, the matching call waits on it — a hold for cancellation-ordering tests.
        self.create_gate: asyncio.Event | None = None

    async def create_room(self, request: Any) -> Any:
        self.create_calls.append(request)
        if self.create_gate is not None:
            await self.create_gate.wait()
        if self.create_error is not None:
            raise self.create_error
        return request

    async def delete_room(self, request: Any) -> Any:
        self.delete_calls.append(request)
        if self.delete_error is not None:
            raise self.delete_error
        return request

    @property
    def created_names(self) -> list[str]:
        return [request.name for request in self.create_calls]

    @property
    def deleted_names(self) -> list[str]:
        return [request.room for request in self.delete_calls]


class FakeLiveKitAPI:
    """What `api_factory(url, key, secret, timeout=...)` returns."""

    def __init__(
        self, url: str, api_key: str, api_secret: str, *, timeout: Any = None, **kwargs: Any
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.extra_kwargs = kwargs
        self.room = FakeRoomService()
        self.close_count = 0
        #: Set on *entering* `aclose()`, so a test can act while the close is in flight.
        self.close_started = asyncio.Event()
        #: When set, `aclose()` waits on it — a hold for cancellation-ordering tests.
        self.close_gate: asyncio.Event | None = None

    async def aclose(self) -> None:
        # Suspends, as the real close does, so a cancel can land mid-close (see the module docstring).
        # The counter is bumped after the yield, so a close that was cut off does not count.
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        else:
            await asyncio.sleep(0)
        self.close_count += 1

    def __repr__(self) -> str:
        return f"FakeLiveKitAPI(url={self.url!r}, closed={self.close_count})"


class LiveKitApiFactory:
    """The `api_factory` seam: one recorded `FakeLiveKitAPI` per call, configured up front.

    The errors are set here rather than on the instance because the adapter builds its client inside
    `launch()` — a test has nowhere to reach in afterwards without patching something that is not a seam.
    """

    def __init__(
        self,
        *,
        create_error: BaseException | None = None,
        delete_error: BaseException | None = None,
        create_gate: asyncio.Event | None = None,
    ) -> None:
        self.create_error = create_error
        self.delete_error = delete_error
        self.create_gate = create_gate
        self.instances: list[FakeLiveKitAPI] = []

    def __call__(self, url: str, api_key: str, api_secret: str, **kwargs: Any) -> FakeLiveKitAPI:
        api = FakeLiveKitAPI(url, api_key, api_secret, **kwargs)
        api.room.create_error = self.create_error
        api.room.delete_error = self.delete_error
        api.room.create_gate = self.create_gate
        self.instances.append(api)
        return api

    @property
    def api(self) -> FakeLiveKitAPI:
        """The one client the adapter built; fails loudly if there is not exactly one."""
        assert len(self.instances) == 1, f"expected one LiveKitAPI, got {len(self.instances)}"
        return self.instances[0]


class FakeConnectingRoom(FakeRoom):
    """`FakeRoom` plus the `connect()` a relayed join calls on its private room."""

    def __init__(self, name: str = "fake-room") -> None:
        super().__init__(name, local_participant=FakeLocalParticipant("pipecat-bot"))
        self.connect_calls: list[dict[str, Any]] = []
        #: Raised by `connect()` instead of connecting.
        self.connect_error: BaseException | None = None
        #: When set, `connect()` waits on it before returning — models a slow or hung join.
        self.connect_gate: asyncio.Event | None = None
        #: Set on *entering* `connect()`, so a test can cancel while the join is in flight.
        self.connect_started = asyncio.Event()
        #: Whether `connect()` was cancelled in flight, which a relayed join must never do.
        self.connect_cancelled = False

    async def connect(self, url: str, token: str, options: Any = None) -> None:
        self.connect_calls.append({"url": url, "token": token, "options": options})
        self.connect_started.set()
        if self.connect_gate is not None:
            try:
                await self.connect_gate.wait()
            except asyncio.CancelledError:
                self.connect_cancelled = True
                raise
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True


class RoomFactory:
    """The `room_factory` seam: one recorded `FakeConnectingRoom` per call, configured up front."""

    def __init__(
        self,
        *,
        connect_error: BaseException | None = None,
        connect_gate: asyncio.Event | None = None,
        name: str = "fake-room",
    ) -> None:
        self.connect_error = connect_error
        self.connect_gate = connect_gate
        self.name = name
        self.rooms: list[FakeConnectingRoom] = []

    def __call__(self) -> FakeConnectingRoom:
        room = FakeConnectingRoom(self.name)
        room.connect_error = self.connect_error
        room.connect_gate = self.connect_gate
        self.rooms.append(room)
        return room

    @property
    def room(self) -> FakeConnectingRoom:
        """The one room the adapter built; fails loudly if there is not exactly one."""
        assert len(self.rooms) == 1, f"expected one Room, got {len(self.rooms)}"
        return self.rooms[0]
