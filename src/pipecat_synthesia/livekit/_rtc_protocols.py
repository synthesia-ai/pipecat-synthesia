"""The narrow slice of the LiveKit rtc API the package uses, as Protocols the test fakes implement."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from livekit import rtc as lk_rtc


class ByteStreamWriterLike(Protocol):
    async def write(self, data: bytes) -> None: ...
    async def aclose(self) -> None: ...  # RuntimeError on second call; StreamError on transport failure


class LocalParticipantLike(Protocol):
    @property
    def identity(self) -> str: ...

    async def stream_bytes(
        self,
        name: str,
        *,
        topic: str,
        destination_identities: list[str],
        attributes: dict[str, str],
    ) -> ByteStreamWriterLike: ...

    # rtc's parameter is named `method_name` and returns the handler; always call it positionally.
    def register_rpc_method(
        self,
        method_name: str,
        handler: Callable[[lk_rtc.RpcInvocationData], Awaitable[str | None] | str | None],
        /,
    ) -> Any: ...

    def unregister_rpc_method(self, method: str, /) -> None: ...

    async def perform_rpc(
        self,
        *,
        destination_identity: str,
        method: str,
        payload: str,
        response_timeout: float | None = None,
        max_round_trip_latency: float | None = None,
    ) -> str: ...


class RemoteParticipantLike(Protocol):
    @property
    def identity(self) -> str: ...
    @property
    def kind(self) -> int: ...
    @property
    def attributes(self) -> Mapping[str, str]: ...
    @property
    def track_publications(self) -> Mapping[str, Any]: ...
    @property
    def disconnect_reason(self) -> Any: ...


class RoomLike(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def local_participant(self) -> LocalParticipantLike: ...
    @property
    def remote_participants(self) -> Mapping[str, RemoteParticipantLike]: ...

    # rtc types `event` as a Literal union; SYNC callbacks only
    def on(self, event: Any, callback: Callable[..., Any]) -> Any: ...
    def off(self, event: Any, callback: Callable[..., Any]) -> None: ...
    def isconnected(self) -> bool: ...
    async def disconnect(self) -> None: ...


# Static guard: pyright fails this module if real rtc ever stops satisfying the Protocols.
# Never delete; fix the Protocol instead.
if TYPE_CHECKING:

    def _assert_room(r: lk_rtc.Room) -> RoomLike:
        return r

    def _assert_local(p: lk_rtc.LocalParticipant) -> LocalParticipantLike:
        return p

    def _assert_remote(p: lk_rtc.RemoteParticipant) -> RemoteParticipantLike:
        return p

    def _assert_writer(w: lk_rtc.ByteStreamWriter) -> ByteStreamWriterLike:
        return w
