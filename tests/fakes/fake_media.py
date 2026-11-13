"""Stand-ins for `rtc.VideoStream` / `rtc.AudioStream` and the tracks they are opened on.

These let `AvatarTrackCapture` be driven through its two injected factories (`video_stream_factory` /
`audio_stream_factory`) with no LiveKit server and no FFI.

**They record; they do not decide.** The one behaviour modelled is the one the production code is built
around:

* a stream yields events until it is told to stop, and a consumer parked in `__anext__` is **not**
  woken by `aclose()`, only by having its task cancelled. That is exactly `rtc.VideoStream` (livekit rtc
  1.1.19): its `_run` loop `break`s on end of stream without enqueuing the `None` sentinel `__anext__`
  waits for, and `aclose()` only disposes the FFI handle and awaits `_run`. `rtc.AudioStream` *does*
  enqueue the sentinel, so `end()` models that half.
* `FakeVideoFrame.convert()` raises unless the test built the frame expecting a conversion, which is
  how "no `convert()` on an RGB24 frame" is asserted rather than assumed.

Nothing here decides *when* a frame arrives: the test pushes one and waits for the effect with
`until()`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from livekit import rtc as lk_rtc

__all__ = [
    "RGB24",
    "FakeAudioFrame",
    "FakeAudioFrameEvent",
    "FakeMediaStream",
    "FakeSubscribedPublication",
    "FakeTrack",
    "FakeVideoFrame",
    "FakeVideoFrameEvent",
    "StreamFactory",
]

RGB24 = lk_rtc.VideoBufferType.RGB24


class FakeTrack:
    """A subscribed remote track: the object a stream factory is opened on."""

    def __init__(self, sid: str = "TR_1", *, kind: int = int(lk_rtc.TrackKind.KIND_VIDEO)) -> None:
        self.sid = sid
        self.kind = kind

    def __repr__(self) -> str:
        return f"FakeTrack(sid={self.sid!r}, kind={self.kind})"


class FakeSubscribedPublication:
    """A `RemoteTrackPublication` that already carries its subscribed `track`.

    `tests/fakes/fake_rtc.py::FakeTrackPublication` models the publication a `track_published`
    listener sees, which has no `.track`; the `start()` scan needs one that does.
    """

    def __init__(
        self, sid: str, *, kind: int, track: FakeTrack | None = None, subscribed: bool = True
    ) -> None:
        self.sid = sid
        self.kind = kind
        self.track = track if track is not None else FakeTrack(sid, kind=kind)
        self.subscribed = subscribed
        self.name = ""
        self.source = 0

    def __repr__(self) -> str:
        return f"FakeSubscribedPublication(sid={self.sid!r}, kind={self.kind}, subscribed={self.subscribed})"


class FakeVideoFrame:
    """One RGB24 (or not) video frame. `convert()` raises unless the test asked for a conversion."""

    def __init__(
        self,
        *,
        width: int = 4,
        height: int = 2,
        data: bytes | None = None,
        type: int = RGB24,  # noqa: A002 -- rtc names the attribute `type`
        converts_to: FakeVideoFrame | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.data = data if data is not None else bytes(range(width * height * 3 % 256))
        self.type = type
        self._converts_to = converts_to
        self.convert_calls: list[int] = []

    def convert(self, fmt: int) -> FakeVideoFrame:
        self.convert_calls.append(fmt)
        if self._converts_to is None:
            raise AssertionError(
                "convert() was called on a frame the FFI already delivered as RGB24; "
                "VideoStream(format=RGB24) makes this conversion the Rust side's job"
            )
        return self._converts_to

    def __repr__(self) -> str:
        return f"FakeVideoFrame({self.width}x{self.height}, type={self.type})"


class FakeVideoFrameEvent:
    """`rtc.VideoFrameEvent`: the frame plus the timing fields we do not use."""

    def __init__(self, frame: FakeVideoFrame, *, timestamp_us: int = 0) -> None:
        self.frame = frame
        self.timestamp_us = timestamp_us
        self.rotation = 0
        self.metadata = None


class FakeAudioFrame:
    """One PCM frame as `rtc.AudioFrame` presents it: `.data`, `.sample_rate`, `.num_channels`."""

    def __init__(
        self, data: bytes = b"\x01\x02" * 240, *, sample_rate: int = 24000, num_channels: int = 1
    ) -> None:
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = len(data) // (2 * max(1, num_channels))

    def __repr__(self) -> str:
        return f"FakeAudioFrame({len(self.data)} bytes @ {self.sample_rate} Hz x{self.num_channels})"


class FakeAudioFrameEvent:
    """`rtc.AudioFrameEvent`: a frame and nothing else."""

    def __init__(self, frame: FakeAudioFrame) -> None:
        self.frame = frame


class FakeMediaStream:
    """One `VideoStream`/`AudioStream`: an async iterator a test feeds by hand.

    Nothing is scheduled: `push()` makes exactly one event available, `fail(exc)` makes the consumer
    raise it, and `end()` models the EOS sentinel only `AudioStream` enqueues. A consumer waiting on
    an empty stream stays waiting through `aclose()` — see the module docstring.
    """

    def __init__(
        self,
        track: Any,
        *,
        label: str,
        kwargs: dict[str, Any],
        order: list[str] | None = None,
    ) -> None:
        self.track = track
        self.label = label
        #: Exactly what the factory was called with, so a test can assert the keyword arguments by name.
        self.kwargs = kwargs
        self.closed = False
        self.close_count = 0
        self._order = order if order is not None else []
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    _EOS = object()

    def push(self, event: Any) -> None:
        self._queue.put_nowait(event)

    def fail(self, exc: BaseException) -> None:
        self._queue.put_nowait(exc)

    def end(self) -> None:
        """EOS as `rtc.AudioStream` delivers it; `rtc.VideoStream` never does this."""
        self._queue.put_nowait(self._EOS)

    def __aiter__(self) -> FakeMediaStream:
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._queue.get()
        except asyncio.CancelledError:
            # The only thing that ends a parked video consumer. Recorded so a test can prove the
            # cancel happened before the close.
            self._order.append(f"{self.label}:cancelled")
            raise
        if isinstance(item, BaseException):
            raise item
        if item is self._EOS:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self.closed = True
        self.close_count += 1
        self._order.append(f"{self.label}:closed")

    def __repr__(self) -> str:
        return f"FakeMediaStream({self.label}, kwargs={self.kwargs}, closed={self.closed})"


class StreamFactory:
    """A `video_stream_factory` / `audio_stream_factory` that records every call and its stream."""

    def __init__(self, label: str, *, order: list[str] | None = None) -> None:
        self.label = label
        self.order = order if order is not None else []
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeMediaStream] = []
        #: Raised by the next call instead of returning a stream, then cleared.
        self.fail_next: BaseException | None = None

    def __call__(self, track: Any, **kwargs: Any) -> FakeMediaStream:
        self.calls.append({"track": track, **kwargs})
        if self.fail_next is not None:
            failure, self.fail_next = self.fail_next, None
            raise failure
        stream = FakeMediaStream(track, label=self.label, kwargs=dict(kwargs), order=self.order)
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeMediaStream:
        """The one stream this factory opened; fails loudly if there is not exactly one."""
        assert len(self.streams) == 1, f"{self.label}: expected one stream, got {len(self.streams)}"
        return self.streams[0]

    def __repr__(self) -> str:
        return f"StreamFactory({self.label!r}, calls={len(self.calls)})"
