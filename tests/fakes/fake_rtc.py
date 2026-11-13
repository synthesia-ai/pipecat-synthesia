"""In-process stand-ins for the narrow `livekit.rtc` surface in `pipecat_synthesia.livekit._rtc_protocols`.

`FakeByteStreamWriter`, `FakeLocalParticipant`, `FakeRemoteParticipant`, `FakeRoom` and
`FakeTrackPublication` are enough to drive `UtteranceStreamer`, `PlaybackRpcHandlers`,
`perform_clear_buffer`, `perform_swap_avatar`, `AvatarReadiness` and `LiveKitAvatarSession` with no
LiveKit server. Read this docstring before using them.

**These fakes record; they do not decide.** The only behaviour they model is behaviour read in rtc's
source (livekit rtc 1.1.19), and each such rule has a comment beside it:

* a second `aclose()` on a writer raises `RuntimeError("Stream already closed")`, and a `write()` after
  close raises `RuntimeError`;
* `Room.on()` rejects coroutine functions with `ValueError`;
* `emit()` passes only as many positional arguments as the callback declares, and passes them all to a
  `*args` callback;
* re-registering an RPC method silently overwrites the handler;
* a handler that raises reaches the caller as `RpcError(APPLICATION_ERROR)`;
* `disconnect()` emits `disconnected(CLIENT_INITIATED)`, and returns at once on a room that is not
  connected;
* rtc fills a participant's `disconnect_reason` right before it emits `participant_disconnected`;
* a `perform_rpc` answer scripted with `reply_after(seconds, ...)` arrives only if it comes within the
  window rtc gives the worker's handler, max(response_timeout - max_round_trip_latency, 1 s), and is
  `RpcError(RESPONSE_TIMEOUT)` at the end of that window otherwise (livekit-rpc `client.rs`).

`tests/integration/test_wire_conformance.py` runs the same scenarios against real rtc and a real
`livekit-server` and compares the outcomes; any divergence it finds is fixed here or listed below.

**Deliberate divergences from real rtc:**

* real rtc keeps listeners in a `set`, so callback order is undefined; `FakeRoom` keeps a list, which is
  a *stronger* guarantee. Never assert listener order in a test: it would pass here and be a coin flip
  against a real server.
* real `emit()` logs and swallows a callback's `Exception` but re-raises `TypeError`; `FakeRoom.emit`
  does the same, and also records the exception in `callback_errors` so a test can see what was
  swallowed.
* the fakes are single-loop, in-process and instantaneous unless a delay is asked for: `open_delay`
  exists because the real open is not.
* `rtc.Room()` needs a running event loop in its constructor; `FakeRoom()` does not. No product code
  builds a `Room` outside a coroutine, and requiring a loop here would only make sync fixtures awkward,
  but a test that builds a real `Room` must be `async`.
* `FakeRoom.off()` drops an event's key once its last listener is removed, so a fully torn-down room
  reads `listeners == {}`. Real `EventEmitter.off()` keeps the empty set; nothing in the product
  observes either.

Against livekit-server 1.9.1 and livekit rtc 1.1.19 the conformance tests found nothing else:
`UtteranceStreamer` and `PlaybackRpcHandlers` produce the same streams and playback events against
`FakeLocalParticipant` as against a real `rtc.LocalParticipant` talking to `FakeAvatarWorker`.

**Known modelling gaps, left so the fake keeps recording rather than deciding:**

* `invoke()` puts `response_timeout` on the invocation data but does not enforce it, where real rtc
  answers `RpcError(RESPONSE_TIMEOUT)` when the caller's deadline passes. A handler that never returns
  hangs the call; bound it in the test.
* a `stream_bytes` that fails *after* being recorded (`fail_next_open`) leaves the stream recorded and
  counted as open, because rtc sends the open request before awaiting it, and the header may already
  be on the wire.
* `reply_after` acknowledges at once, so the acknowledgement timeout is not modelled: real rtc fails the
  call with `RpcError(CONNECTION_TIMEOUT)` when no acknowledgement comes within `max_round_trip_latency`.
* `never_returns` hangs past the worker-side deadline until the caller's own timeout, where real rtc's
  handler side answers `RpcError(RESPONSE_TIMEOUT)` once that deadline passes.

Past a *cancelled* open, sender and receiver count different things: `FakeLocalParticipant.streams`
records what the sender **requested** (which is the point of the counter), while a receiver only ever
sees streams whose header actually went out. The conformance tests cannot compare the two there.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from livekit import rtc as lk_rtc

# rtc reads this private asyncio sentinel defensively for the same reason we do: it is how
# `asyncio.iscoroutinefunction` recognised a tagged callable, and it may disappear in a later Python.
_ASYNCIO_COROUTINE_MARKER = getattr(asyncio.coroutines, "_is_coroutine", None)

if TYPE_CHECKING:
    from pipecat_synthesia.livekit._rtc_protocols import (
        ByteStreamWriterLike,
        LocalParticipantLike,
        RemoteParticipantLike,
        RoomLike,
    )

__all__ = [
    "FakeByteStreamWriter",
    "FakeLocalParticipant",
    "FakeRemoteParticipant",
    "FakeRoom",
    "FakeTrackPublication",
    "never_returns",
    "reply_after",
]


class _NeverReturns:
    """Sentinel: a `perform_rpc` that hangs until the caller's own timeout cancels it."""

    def __repr__(self) -> str:
        return "never_returns"


#: Script this for a `perform_rpc` that never answers (`script_rpc(method, never_returns)`).
never_returns = _NeverReturns()

# rtc's defaults when `perform_rpc` is given neither value (livekit-rpc `PerformRpcData`, rtc 1.1.19).
_RTC_RESPONSE_TIMEOUT_SECS = 15.0
_RTC_MAX_ROUND_TRIP_LATENCY_SECS = 7.0
# The least time rtc gives the worker's handler, however small the difference of the two values.
_RTC_MIN_HANDLER_SECS = 1.0


class _ReplyAfter:
    """A `perform_rpc` answer that the worker sends `seconds` after the request; see `reply_after`."""

    def __init__(self, seconds: float, response: str | BaseException) -> None:
        self.seconds = seconds
        self.response = response

    def __repr__(self) -> str:
        return f"reply_after({self.seconds}, {self.response!r})"


def reply_after(seconds: float, response: str | BaseException) -> _ReplyAfter:
    """Script a worker that answers `response` after `seconds` (`script_rpc(method, reply_after(5, ok))`).

    The caller hears it only if it comes within the window rtc gives the worker's handler:
    max(response_timeout - max_round_trip_latency, 1 s), and no later than `response_timeout`. Past
    that window the call raises `RpcError(RESPONSE_TIMEOUT)`, at the end of the window, as rtc does.
    """
    return _ReplyAfter(seconds, response)


class FakeByteStreamWriter:
    """One byte stream. Records what it was opened with and every chunk written to it."""

    def __init__(
        self,
        *,
        name: str,
        topic: str,
        destination_identities: list[str],
        attributes: dict[str, str],
        fail_on_write: BaseException | None = None,
        on_close: Callable[[FakeByteStreamWriter], None] | None = None,
    ) -> None:
        self.name = name
        self.topic = topic
        self.destination_identities = destination_identities
        self.attributes = attributes
        self.fail_on_write = fail_on_write
        self.writes: list[bytes] = []
        self.closed = False
        self.close_count = 0
        self._on_close = on_close

    @property
    def data(self) -> bytes:
        """Everything written, concatenated — what the worker's receiver would have read."""
        return b"".join(self.writes)

    async def write(self, data: bytes) -> None:
        if self.closed:
            # rtc raises on a write after close.
            raise RuntimeError("Cannot write after stream is closed")
        if self.fail_on_write is not None:
            raise self.fail_on_write
        self.writes.append(bytes(data))
        # rtc sends the chunk, then awaits the FFI's answer: the writer yields once per write, so whatever
        # was waiting can run between two chunks, as against the real SDK
        await asyncio.sleep(0)

    async def aclose(self) -> None:
        if self.closed:
            # rtc raises on a second close; the streamer must treat this as benign.
            raise RuntimeError("Stream already closed")
        self.closed = True
        self.close_count += 1
        if self._on_close is not None:
            self._on_close(self)

    def __repr__(self) -> str:
        return (
            f"FakeByteStreamWriter(name={self.name!r}, attributes={self.attributes!r}, "
            f"chunks={len(self.writes)}, closed={self.closed})"
        )


class FakeLocalParticipant:
    """The bot's own participant: opens byte streams, serves and performs RPCs.

    `streams` is appended to when `stream_bytes` is *requested*, before any `open_delay` elapses,
    because that is when rtc issues the FFI request and the header may reach the wire (the reason
    `UtteranceStreamer` keeps an `_orphaned_open`). `open_stream_count`, `max_concurrent_open_streams`
    and `unclosed_streams()` are counted at the same moment and decremented on `aclose()`, so a breach of
    the *one-open-stream rule* is observable at the instant it happens.
    """

    def __init__(self, identity: str = "bot", *, open_delay: float = 0.0) -> None:
        self._identity = identity
        #: Default delay between the request being recorded and the writer being returned.
        self.open_delay = open_delay
        self.streams: list[FakeByteStreamWriter] = []
        #: Raised by the next `stream_bytes`, then cleared.
        self.fail_next_open: BaseException | None = None
        #: Given to every writer this participant opens.
        self.fail_on_write: BaseException | None = None
        self.open_stream_count = 0
        self.max_concurrent_open_streams = 0
        self.rpc_methods: dict[str, Callable[..., Awaitable[str | None] | str | None]] = {}
        self.registration_order: list[str] = []
        self.unregistered: list[str] = []
        self.perform_rpc_calls: list[dict[str, Any]] = []
        self._scripted: dict[str, list[object]] = {}

    @property
    def identity(self) -> str:
        return self._identity

    def unclosed_streams(self) -> list[FakeByteStreamWriter]:
        """Every stream this participant opened and never closed (the *one-open-stream rule* leak check)."""
        return [stream for stream in self.streams if not stream.closed]

    # --- byte streams ----------------------------------------------------------------------------

    async def stream_bytes(
        self,
        name: str,
        *,
        topic: str,
        destination_identities: list[str],
        attributes: dict[str, str],
        open_delay: float | None = None,
    ) -> FakeByteStreamWriter:
        """Open a stream. `open_delay` (kwarg, else `self.open_delay`) sleeps AFTER recording the open.

        The extra keyword has a default, so this still satisfies `LocalParticipantLike`; production
        code never passes it.
        """
        writer = FakeByteStreamWriter(
            name=name,
            topic=topic,
            destination_identities=list(destination_identities),
            attributes=dict(attributes),
            fail_on_write=self.fail_on_write,
            on_close=self._on_stream_closed,
        )
        self.streams.append(writer)
        self.open_stream_count += 1
        self.max_concurrent_open_streams = max(self.max_concurrent_open_streams, self.open_stream_count)
        delay = self.open_delay if open_delay is None else open_delay
        if delay:
            await asyncio.sleep(delay)
        if self.fail_next_open is not None:
            failure, self.fail_next_open = self.fail_next_open, None
            # The stream stays recorded and counted as open. rtc sends the open request before awaiting
            # its result, so a header may be on the wire even though the caller got an exception instead
            # of a writer. Un-recording it would be the fake *deciding* that a failed open never happened,
            # and would hide the leak a test wants to see. The writer stays in `.streams` to be closed.
            raise failure
        return writer

    def _on_stream_closed(self, writer: FakeByteStreamWriter) -> None:
        self.open_stream_count -= 1

    # --- RPC: methods we serve -------------------------------------------------------------------

    def register_rpc_method(
        self, method_name: str, handler: Callable[..., Awaitable[str | None] | str | None], /
    ) -> Callable[..., Awaitable[str | None] | str | None]:
        # Re-registering the same name overwrites silently, as in rtc.
        self.rpc_methods[method_name] = handler
        self.registration_order.append(method_name)
        return handler

    def unregister_rpc_method(self, method: str, /) -> None:
        self.rpc_methods.pop(method, None)
        self.unregistered.append(method)

    async def invoke(
        self,
        method: str,
        *,
        caller_identity: str,
        payload: str = "",
        response_timeout: float = 10.0,
        request_id: str = "req-1",
    ) -> str:
        """Call a registered handler the way the worker would, and return its response string.

        Mirrors rtc's incoming side: an unregistered method is `UNSUPPORTED_METHOD` and a handler that
        raises anything other than `RpcError` reaches the caller as `APPLICATION_ERROR`. For
        `lk.playback_finished` that error ends the worker's session after three retries, so a test that
        sees it has found a real bug.

        `response_timeout` is put on the invocation data for the handler to read, but it is not enforced
        here, unlike real rtc, which answers `RpcError(RESPONSE_TIMEOUT)` when the caller's deadline
        passes: enforcing it would make the fake decide. A handler that never returns hangs this call;
        that is the test's own bug to bound.
        """
        handler = self.rpc_methods.get(method)
        if handler is None:
            raise lk_rtc.RpcError(
                lk_rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD, "Method not supported at destination"
            )
        data = lk_rtc.RpcInvocationData(
            request_id=request_id,
            caller_identity=caller_identity,
            payload=payload,
            response_timeout=response_timeout,
            method=method,
        )
        try:
            result = handler(data)
            if inspect.isawaitable(result):
                result = await result
        except lk_rtc.RpcError:
            raise
        except Exception as exc:
            raise lk_rtc.RpcError(
                lk_rtc.RpcError.ErrorCode.APPLICATION_ERROR, "Application error in method handler"
            ) from exc
        return "" if result is None else result

    # --- RPC: methods we perform -----------------------------------------------------------------

    def script_rpc(self, method: str, *responses: object) -> None:
        """Queue what the next `perform_rpc(method=...)` calls return.

        Each entry is a `str` to return, a `BaseException` to raise, `never_returns`, or `reply_after(...)`.
        The last entry repeats, so a single value scripts every call.
        """
        self._scripted.setdefault(method, []).extend(responses)

    async def perform_rpc(
        self,
        *,
        destination_identity: str,
        method: str,
        payload: str,
        response_timeout: float | None = None,
        max_round_trip_latency: float | None = None,
    ) -> str:
        self.perform_rpc_calls.append(
            {
                "destination_identity": destination_identity,
                "method": method,
                "payload": payload,
                "response_timeout": response_timeout,
                "max_round_trip_latency": max_round_trip_latency,
            }
        )
        queued = self._scripted.get(method)
        if not queued:
            raise AssertionError(
                f"perform_rpc({method!r}) was called with nothing scripted; "
                f"call script_rpc({method!r}, ...) in the test"
            )
        response = queued[0] if len(queued) == 1 else queued.pop(0)
        if isinstance(response, _ReplyAfter):
            response = await self._reply_after(response, response_timeout, max_round_trip_latency)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, _NeverReturns):
            await asyncio.Event().wait()  # cancelled by the caller's wait_for
        return str(response)

    @staticmethod
    async def _reply_after(
        reply: _ReplyAfter, response_timeout: float | None, max_round_trip_latency: float | None
    ) -> object:
        # livekit-rpc `client.rs` (rtc 1.1.19): the request tells the worker's handler it has
        # max(response_timeout - max_round_trip_latency, 1 s), and the worker's rtc answers RESPONSE_TIMEOUT
        # when that passes. The caller itself waits up to `response_timeout` for the answer, after the
        # acknowledgement, which the fake gives at once.
        wait = _RTC_RESPONSE_TIMEOUT_SECS if response_timeout is None else response_timeout
        ack = _RTC_MAX_ROUND_TRIP_LATENCY_SECS if max_round_trip_latency is None else max_round_trip_latency
        window = min(max(wait - ack, _RTC_MIN_HANDLER_SECS), wait)
        if reply.seconds > window:
            await asyncio.sleep(window)
            return lk_rtc.RpcError(lk_rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "Response timeout")
        await asyncio.sleep(reply.seconds)
        return reply.response

    def __repr__(self) -> str:
        return (
            f"FakeLocalParticipant(identity={self._identity!r}, streams={len(self.streams)}, "
            f"open={self.open_stream_count}, max_concurrent={self.max_concurrent_open_streams})"
        )


class FakeTrackPublication:
    """The slice of `rtc.RemoteTrackPublication` a room listener reads: `sid`, `kind`, `name`, `source`.

    `kind` is an `rtc.TrackKind` value (`KIND_AUDIO=1`, `KIND_VIDEO=2`).

    `set_subscribed` records instead of sending the request real rtc sends (synchronously, returning
    `None`, with no error path). It is what `mute_participant` and our video unsubscribe both come down
    to, so `subscribed_calls` is how a test sees either of them land.
    """

    def __init__(self, sid: str, *, kind: int, name: str = "", source: int = 0) -> None:
        self.sid = sid
        self.kind = kind
        self.name = name
        self.source = source
        #: Every `set_subscribed` argument, in order. A repeated `False` is harmless on real rtc.
        self.subscribed_calls: list[bool] = []

    @property
    def subscribed(self) -> bool:
        """The last `set_subscribed` value; `True` (rtc's `auto_subscribe`) if never called."""
        return self.subscribed_calls[-1] if self.subscribed_calls else True

    def set_subscribed(self, subscribed: bool) -> None:
        self.subscribed_calls.append(subscribed)

    def __repr__(self) -> str:
        return (
            f"FakeTrackPublication(sid={self.sid!r}, kind={self.kind}, name={self.name!r}, "
            f"subscribed={self.subscribed})"
        )


class FakeRemoteParticipant:
    """A remote participant. Every `RemoteParticipantLike` member is a property, as on real rtc."""

    def __init__(
        self,
        identity: str,
        *,
        kind: int = int(lk_rtc.ParticipantKind.PARTICIPANT_KIND_AGENT),
        attributes: Mapping[str, str] | None = None,
        track_publications: Mapping[str, Any] | None = None,
        disconnect_reason: Any = None,
    ) -> None:
        self._identity = identity
        self._kind = kind
        # rtc returns a fresh dict from `.attributes`; tests mutate this one to model an attribute
        # update, so keep it a plain writable dict.
        self._attributes: dict[str, str] = dict(attributes or {})
        self._track_publications: dict[str, Any] = dict(track_publications or {})
        self._disconnect_reason = disconnect_reason

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def kind(self) -> int:
        return self._kind

    @property
    def attributes(self) -> Mapping[str, str]:
        return dict(self._attributes)

    @property
    def track_publications(self) -> Mapping[str, Any]:
        return dict(self._track_publications)

    @property
    def disconnect_reason(self) -> Any:
        return self._disconnect_reason

    def set_attributes(self, attributes: Mapping[str, str]) -> None:
        """Replace the attribute map (the worker writes the whole map back)."""
        self._attributes = dict(attributes)

    def set_disconnect_reason(self, reason: Any) -> None:
        """What rtc does right before emitting `participant_disconnected`; call it just before `emit`."""
        self._disconnect_reason = reason

    def publish(self, sid: str, publication: Any) -> None:
        self._track_publications[sid] = publication

    def unpublish(self, sid: str) -> Any:
        return self._track_publications.pop(sid, None)

    def __repr__(self) -> str:
        return f"FakeRemoteParticipant(identity={self._identity!r}, kind={self._kind})"


class FakeRoom:
    """A room: listener registry, remote participants and a recordable `disconnect()`."""

    def __init__(
        self, name: str = "fake-room", *, local_participant: FakeLocalParticipant | None = None
    ) -> None:
        self._name = name
        self._local_participant = local_participant or FakeLocalParticipant()
        self._remote_participants: dict[str, FakeRemoteParticipant] = {}
        self.connected = True
        #: event -> callbacks, in registration order. Real rtc uses a `set`: never assert the order.
        self.listeners: dict[str, list[Callable[..., Any]]] = {}
        #: Exceptions `emit` swallowed, as real rtc's emitter does.
        self.callback_errors: list[BaseException] = []
        self.disconnect_calls = 0
        #: Set on *entering* `disconnect()`, so a test can act while the disconnect is in flight.
        self.disconnect_started = asyncio.Event()
        #: When set, `disconnect()` waits for it — a hold for teardown-ordering tests.
        self.disconnect_event: asyncio.Event | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def local_participant(self) -> FakeLocalParticipant:
        return self._local_participant

    @property
    def remote_participants(self) -> Mapping[str, FakeRemoteParticipant]:
        return self._remote_participants

    def add_participant(self, participant: FakeRemoteParticipant) -> FakeRemoteParticipant:
        """Put a participant in the room without emitting anything (models pre-existing state)."""
        self._remote_participants[participant.identity] = participant
        return participant

    def remove_participant(self, identity: str) -> FakeRemoteParticipant | None:
        """Take a participant out. rtc removes it *before* emitting `participant_disconnected`."""
        return self._remote_participants.pop(identity, None)

    def on(self, event: Any, callback: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(callback) or (
            _ASYNCIO_COROUTINE_MARKER is not None
            and getattr(callback, "_is_coroutine", None) is _ASYNCIO_COROUTINE_MARKER
        ):
            # Both halves of rtc's check, marker included. A sync-only registry is why every listener
            # schedules a task instead of awaiting.
            raise ValueError(
                "Cannot register an async callback with `.on()`. Use `asyncio.create_task` within "
                "your synchronous callback instead."
            )
        self.listeners.setdefault(str(event), []).append(callback)
        return callback

    def off(self, event: Any, callback: Callable[..., Any]) -> None:
        handlers = self.listeners.get(str(event))
        if handlers and callback in handlers:
            handlers.remove(callback)
        if handlers is not None and not handlers:
            # Drop the empty key so "every listener removed" reads as `listeners == {}`.
            del self.listeners[str(event)]

    def emit(self, event: Any, *args: Any) -> None:
        """Call every listener for `event`, trimming the arguments exactly as rtc's emitter does."""
        # `str(event)`, the same key `on()`/`off()` store under: rtc types events as a Literal union of
        # strings, but a caller passing anything else must still reach its listeners rather than
        # silently dispatching to none.
        for callback in list(self.listeners.get(str(event), ())):
            try:
                callback(*_trim_args(callback, args))
            except TypeError:
                raise  # rtc re-raises TypeError rather than swallowing it
            except Exception as exc:  # mirrors rtc's "log it and carry on"
                self.callback_errors.append(exc)

    def isconnected(self) -> bool:
        return self.connected

    async def disconnect(self) -> None:
        """Disconnect, and emit `disconnected` the way rtc does, so a listener still installed sees it.

        Real `Room.disconnect()` returns at once when the room is not connected, and otherwise emits
        `disconnected(reason)` on the way out. So a listener left installed across the owned-room
        disconnect is a mistake the fake can catch. The call is counted before the early return, so a
        second disconnect stays visible to a *single flight* assertion.
        """
        self.disconnect_calls += 1
        if not self.connected:
            return
        self.disconnect_started.set()
        if self.disconnect_event is not None:
            await self.disconnect_event.wait()
        self.connected = False
        self.emit("disconnected", lk_rtc.DisconnectReason.CLIENT_INITIATED)

    def __repr__(self) -> str:
        return f"FakeRoom(name={self._name!r}, participants={sorted(self._remote_participants)})"


def _trim_args(callback: Callable[..., Any], args: tuple[Any, ...]) -> tuple[Any, ...]:
    """rtc passes only as many positional args as the callback declares; `*args` gets them all."""
    params = list(inspect.signature(callback).parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return args
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return args[: min(len(args), len(positional))]


# Structural guard: pyright fails this module if a fake stops satisfying the Protocol the production
# code takes, which is the whole reason the unit tier can stand in for real rtc. Never delete; fix the
# fake. (`uv run pyright` covers `src` only, so run it on this file explicitly.)
if TYPE_CHECKING:

    def _assert_writer(w: FakeByteStreamWriter) -> ByteStreamWriterLike:
        return w

    def _assert_local(p: FakeLocalParticipant) -> LocalParticipantLike:
        return p

    def _assert_remote(p: FakeRemoteParticipant) -> RemoteParticipantLike:
        return p

    def _assert_room(r: FakeRoom) -> RoomLike:
        return r
