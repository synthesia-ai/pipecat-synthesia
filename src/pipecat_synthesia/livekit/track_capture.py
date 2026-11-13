"""Track capture: in a relayed join, turns the avatar's video and audio tracks into Pipecat frames.

One consumer task per track feeds one of the service's frame callbacks, with no lock or queue between
them (*state-before-await*). A slow callback slows only its own track, and the SDK's bounded buffer drops
the oldest frame rather than growing.

A consumer is stopped by cancelling its task before closing its stream. At end of stream,
`rtc.VideoStream` never wakes a consumer blocked in `__anext__` (unlike `rtc.AudioStream`), and
`VideoStream.aclose()` does not wake it either, so closing first could leave the video consumer parked
forever (livekit rtc 1.1.19). Cancelling first always ends the task; closing afterwards releases the FFI
handle.

A consumer stops in one of two ways:
- **Its track ends.** rtc reports the captured track unsubscribed, or the avatar participant
  disconnected. The consumer is *retired*: its handles are cleared at once, and it is stopped in a task
  of its own, which `aclose()` waits for. The next track of that kind is then captured, which is what
  carries the relayed join's media across a *same-identity replacement*. The disconnect is watched as
  well as the unsubscribe because rtc 1.1.19 skips `track_unsubscribed` when a racing disconnect has
  already removed the participant. Both are safe: rtc never holds two participants with one identity,
  so every consumer at a disconnect reads the departing participant's tracks.
- **`aclose()`** stops the rest.

Whether the avatar has *left* is not decided here: the session's leave detection owns that, and
retiring a consumer reports nothing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._rtc import lk_rtc

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pipecat_synthesia.livekit._rtc_protocols import RoomLike

__all__ = ["AUDIO_STREAM_CAPACITY", "VIDEO_STREAM_CAPACITY", "AvatarTrackCapture"]

#: Frames the SDK buffers for us before it starts dropping the oldest (`RingQueue`). Eight frames is
#: ~0.3 s at the worker's 25 fps: enough to ride out a slow callback, short enough that what we hand
#: the pipeline is still current. `0`, the SDK default, means unbounded.
VIDEO_STREAM_CAPACITY = 8

#: ~0.5 s of the worker's 10 ms audio frames, for the same reason.
AUDIO_STREAM_CAPACITY = 50


class AvatarTrackCapture:
    """One consumer task per avatar track, feeding the service's frame callbacks."""

    def __init__(
        self,
        room: lk_rtc.Room,
        *,
        avatar_identity: str,
        audio_sample_rate: int,
        audio_channels: int,
        on_video_frame: Callable[[bytes, int, int], Awaitable[None]],
        on_audio_frame: Callable[[bytes, int, int], Awaitable[None]],
        log,
        video_stream_factory=None,
        audio_stream_factory=None,
    ) -> None:
        # Two views of one object, as in `session.py`: the concrete rtc type the caller passes and
        # the narrow Protocol whose `on()`/`off()` accept any event name (and that the fakes satisfy).
        self._room: RoomLike = room
        self._avatar_identity = avatar_identity
        self._audio_sample_rate = audio_sample_rate
        self._audio_channels = audio_channels
        self._on_video_frame = on_video_frame
        self._on_audio_frame = on_audio_frame
        self._log = log
        self._video_stream_factory = video_stream_factory or lk_rtc.VideoStream
        self._audio_stream_factory = audio_stream_factory or lk_rtc.AudioStream
        self._video_task: asyncio.Task[None] | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._video_stream: Any = None
        self._audio_stream: Any = None
        #: The track each consumer reads, so that its end retires that consumer and no other.
        self._video_track: Any = None
        self._audio_track: Any = None
        #: Retired consumers still being stopped; held so they are not garbage-collected mid-flight.
        self._retiring: set[asyncio.Task[None]] = set()
        self._listening = False
        self._closed = False
        #: Kinds whose "a second track turned up" line has already been logged (per capture, not
        #: per process: `_once` is the wrong home for state that belongs to one session).
        self._extra_logged: set[str] = set()

    def start(self) -> None:
        """Watch for the avatar's tracks and pick up the ones already subscribed.

        Both halves are needed: `auto_subscribe=True` means a track published before we got here is
        already subscribed and will never emit `track_subscribed` again.
        """
        if self._closed or self._listening:
            return
        for event, listener in self._listeners():
            self._room.on(event, listener)
        self._listening = True
        participant = self._room.remote_participants.get(self._avatar_identity)
        if participant is None:
            return
        for publication in list(participant.track_publications.values()):
            track = getattr(publication, "track", None)
            if track is not None and getattr(publication, "subscribed", False):
                self._consume(track, getattr(publication, "kind", None))

    # ------------------------------------------------------------------ room listeners (sync)

    def _listeners(self) -> tuple[tuple[str, Callable[..., None]], ...]:
        return (
            ("track_subscribed", self._on_track_subscribed),
            ("track_unsubscribed", self._on_track_unsubscribed),
            ("participant_disconnected", self._on_participant_disconnected),
        )

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        if participant.identity != self._avatar_identity:
            return
        self._consume(track, getattr(publication, "kind", None))

    def _on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        """Retire the consumer reading `track`; an ignored extra track ending changes nothing."""
        if participant.identity != self._avatar_identity or track is None:
            return
        if track is self._video_track:
            self._retire("video")
        elif track is self._audio_track:
            self._retire("audio")

    def _on_participant_disconnected(self, participant: Any) -> None:
        """Retire both consumers: their tracks are the departing participant's (see the module docstring)."""
        if participant.identity != self._avatar_identity:
            return
        self._retire("video")
        self._retire("audio")

    def _consume(self, track: Any, kind: Any) -> None:
        """Start the consumer for `kind`, unless one is already running (then one INFO, ignored)."""
        if self._closed:
            return
        loop = asyncio.get_running_loop()
        if kind == lk_rtc.TrackKind.KIND_VIDEO:
            if self._video_task is not None:
                self._log_extra_track("video")
                return
            self._video_track = track
            self._video_task = loop.create_task(self._consume_video(track))
        elif kind == lk_rtc.TrackKind.KIND_AUDIO:
            if self._audio_task is not None:
                self._log_extra_track("audio")
                return
            self._audio_track = track
            self._audio_task = loop.create_task(self._consume_audio(track))

    def _take(self, kind: str) -> tuple[str, asyncio.Task[None], Any] | None:
        """Clear `kind`'s handles, cancel its consumer task, and return the task and its stream, if any.

        The stream is the one the task opened, or None if it never ran: the task opens its stream before
        its first `await`. The cancel is issued here, as the handles are cleared, because a task cancelled
        before it starts never runs at all, so it cannot open a stream after this has read None. It is
        issued once: `_stop()` only waits.
        """
        if kind == "video":
            task, stream = self._video_task, self._video_stream
            self._video_task = self._video_stream = self._video_track = None
        else:
            task, stream = self._audio_task, self._audio_stream
            self._audio_task = self._audio_stream = self._audio_track = None
        if task is not None:
            task.cancel()
        return None if task is None else (kind, task, stream)

    def _retire(self, kind: str) -> None:
        """Stop `kind`'s consumer because its track ended, so the next track of that kind is captured.

        Synchronous, as a room listener must be (*state-before-await*): the handles are clear before
        this returns, so a `track_subscribed` delivered next starts a fresh consumer.
        """
        if self._closed:
            return
        consumer = self._take(kind)
        if consumer is None:
            return
        self._log.info(f"avatar {kind} track ended; its capture stops until the next one")
        retiring = asyncio.get_running_loop().create_task(self._stop([consumer]))
        self._retiring.add(retiring)
        retiring.add_done_callback(self._retiring.discard)

    def _log_extra_track(self, kind: str) -> None:
        if kind in self._extra_logged:
            return
        self._extra_logged.add(kind)
        self._log.info(f"ignoring an extra avatar {kind} track; the first one is already captured")

    # ------------------------------------------------------------------ consumers

    async def _consume_video(self, track: Any) -> None:
        """RGB24 frames into `on_video_frame`; ended only by `aclose()` cancelling this task."""
        try:
            stream = self._video_stream_factory(
                track, format=lk_rtc.VideoBufferType.RGB24, capacity=VIDEO_STREAM_CAPACITY
            )
            self._video_stream = stream
            first = True
            async for event in stream:
                frame = event.frame
                # The FFI already converted for us (`format=RGB24`); this is the belt-and-braces path
                # for a build that hands back its native buffer type anyway.
                if frame.type != lk_rtc.VideoBufferType.RGB24:
                    frame = frame.convert(lk_rtc.VideoBufferType.RGB24)
                if first:
                    first = False
                    self._log.info(f"first avatar video frame {frame.width}x{frame.height}")
                await self._on_video_frame(bytes(frame.data), frame.width, frame.height)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Log once and stop: nothing resubscribes, and the session's `track_unpublished` watcher
            # is what turns a genuinely departed avatar into `on_avatar_left`.
            self._log.exception("avatar video capture stopped")

    async def _consume_audio(self, track: Any) -> None:
        """PCM at the pipeline's rate into `on_audio_frame`; the SDK does the resampling."""
        try:
            stream = self._audio_stream_factory(
                track,
                sample_rate=self._audio_sample_rate,
                num_channels=self._audio_channels,
                capacity=AUDIO_STREAM_CAPACITY,
            )
            self._audio_stream = stream
            first = True
            async for event in stream:
                frame = event.frame
                if first:
                    first = False
                    self._log.info(f"first avatar audio frame {frame.sample_rate} Hz x{frame.num_channels}")
                await self._on_audio_frame(bytes(frame.data), frame.sample_rate, frame.num_channels)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._log.exception("avatar audio capture stopped")

    # ------------------------------------------------------------------ teardown

    async def aclose(self) -> None:
        """Stop capturing: listeners off, then every consumer stopped, retired ones included. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._listening:
            self._listening = False
            for event, listener in self._listeners():
                try:
                    self._room.off(event, listener)
                except Exception as exc:
                    self._log.warning(f"could not remove the {event} listener: {exc!r}")

        # The close runs as a close hook under CLOSE_HOOK_SECS (5 s), and a hook cut off part way cannot be
        # retried, so it costs at most TASK_CANCEL_SECS + STREAM_CLOSE_SECS (3 s). A retired consumer's stop
        # started before this one, under the same two bounds, so waiting for it alongside adds nothing.
        consumers = [consumer for consumer in (self._take("video"), self._take("audio")) if consumer]
        retiring = tuple(self._retiring)
        waits: list[Awaitable[Any]] = [self._stop(consumers)]
        if retiring:
            # `asyncio.wait` neither cancels what it waits for nor raises on its behalf; a cancel of this
            # coroutine still propagates, and leaves the retired consumers' stops running.
            waits.append(asyncio.wait(retiring, timeout=_types.TASK_CANCEL_SECS + _types.STREAM_CLOSE_SECS))
        await asyncio.gather(*waits)
        if any(not task.done() for task in retiring):
            self._log.warning("a retired avatar capture was still stopping when the capture closed")
        self._log.debug("avatar track capture closed")

    async def _stop(self, consumers: list[tuple[str, asyncio.Task[None], Any]]) -> None:
        """Wait for the consumers' tasks, which `_take()` cancelled, then close their streams (see the module
        docstring).

        All tasks at once and then all streams at once, so this costs at most TASK_CANCEL_SECS +
        STREAM_CLOSE_SECS however many consumers it stops.
        """
        if not consumers:
            return
        # `asyncio.wait` never raises on behalf of the tasks it waits for, so the CancelledError of a task
        # `_take()` cancelled cannot be mistaken for *this* coroutine being cancelled -- which must still
        # propagate, because it means the close hook ran out of budget.
        await asyncio.wait([task for _, task, _ in consumers], timeout=_types.TASK_CANCEL_SECS)
        for kind, task, _ in consumers:
            if not task.done():
                self._log.warning(
                    f"avatar {kind} capture task did not stop within {_types.TASK_CANCEL_SECS}s"
                )
            elif not task.cancelled() and task.exception() is not None:
                self._log.warning(f"avatar {kind} capture task ended with {task.exception()!r}")

        # ... then the streams, so the FFI handles are released. Each `aclose()` coroutine is created inside
        # its task: a cancel before the task's first step must not leave one that was never awaited.
        async def close(stream: Any) -> None:
            await asyncio.wait_for(stream.aclose(), _types.STREAM_CLOSE_SECS)

        streams = [(kind, stream) for kind, _, stream in consumers if stream is not None]
        closes = await asyncio.gather(*(close(stream) for _, stream in streams), return_exceptions=True)
        for (kind, _), outcome in zip(streams, closes, strict=True):
            if isinstance(outcome, BaseException):
                self._log.warning(f"closing the avatar {kind} stream failed: {outcome!r}")
