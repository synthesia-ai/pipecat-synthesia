"""Sends the bot's audio to the avatar worker: one LiveKit byte stream per utterance.

livekit-agents and the worker call an utterance a "segment"; closing the byte stream is what ends one.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.protocol import (
    AUDIO_STREAM_NAME_PREFIX,
    AUDIO_STREAM_TOPIC,
    STREAM_ATTR_NUM_CHANNELS,
    STREAM_ATTR_SAMPLE_RATE,
)

if TYPE_CHECKING:
    from pipecat_synthesia.livekit._rtc_protocols import ByteStreamWriterLike, LocalParticipantLike

# How long without a chunk before the idle close ends the utterance. Tests pass a smaller value.
DEFAULT_IDLE_CLOSE_SECS = 1.0


class UtteranceStreamer:
    """One byte stream per utterance, under the *one-open-stream rule*.

    At most one stream is open at a time, and every stream that was opened is closed, including an
    orphaned open: rtc sends the open request before its callback returns, so a stream header may be on
    the wire even though no writer came back. Every public coroutine takes ``_lock``, so both hold under
    any interleaving of ``write``/``end``/``abort``/``aclose`` and the idle timer.

    Two consequences that are easy to miss:

    * A close must outlive the caller that started it. The service cancels its send and interrupt tasks
      within 1 s, less than ``STREAM_CLOSE_SECS``, so a cancel does arrive mid-close. Every close is
      therefore a detached close that callers only wait on: a cancel or a timeout drops the wait, never the
      close.
    * A writer whose ``write()`` failed is still open on the wire. Dropping it without closing it would
      leak one stream per chunk while the failure lasts, so the drop comes with a detached close.

    One overlap is accepted (see the rule in docs/architecture.md): an orphaned open slower than
    ``STREAM_CLOSE_SECS``. Its close cannot be requested until the open returns a writer, so once the wait
    for it stops, the next utterance's stream can open alongside it. The orphaned stream carries no audio,
    and it is still closed.
    """

    def __init__(
        self,
        participant: LocalParticipantLike,
        *,
        destination_identity: str,
        idle_close_secs: float,
        log,
    ) -> None:
        self._participant = participant
        self._destination_identity = destination_identity
        self._idle_close_secs = idle_close_secs
        self._log = log
        self._writer: ByteStreamWriterLike | None = None
        # the orphaned open, if any: kept until the next close or write closes it
        self._orphaned_open: asyncio.Future[ByteStreamWriterLike] | None = None
        self._open_rate: int | None = None
        self._open_channels: int | None = None
        self._idle_handle: asyncio.TimerHandle | None = None
        # Bumped every time the idle timer is armed or cancelled. The idle task cannot be cancelled once it
        # runs, and it races an incoming write() for the lock: without the generation, a write that won
        # would have its fresh stream closed by the stale timer, splitting the utterance mid-sentence.
        self._idle_generation = 0
        # detached closes and the running idle task: held so they are not garbage-collected mid-flight, and
        # so aclose() can wait for them
        self._detached_closes: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        # Set by aclose() only, under the lock: from then on write() opens nothing. Never by end() or
        # abort(), which end an utterance, not the streamer.
        self._closed = False
        self._failed_this_utterance = False
        # on the instance, not in `_once`: a format change belongs to this session, not the process
        self._warned_format_change = False

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    async def write(self, pcm_s16le: bytes, sample_rate: int, num_channels: int) -> None:
        """Append one chunk of PCM s16le to the current stream, opening one if needed.

        Swallows every data-stream failure (the utterance is dropped, not the pipeline); only
        ``CancelledError`` propagates, and then the pending open is remembered for the next close. A no-op
        once ``aclose()`` has begun: a write queued behind it opens no stream that nothing would close.
        """
        async with self._lock:
            if self._closed:
                return
            if self._orphaned_open is not None:
                await self._close_locked("orphaned open")
            if self._writer is not None and (
                self._open_rate != sample_rate or self._open_channels != num_channels
            ):
                if not self._warned_format_change:
                    self._warned_format_change = True
                    self._log.info(
                        f"audio format changed from {self._open_rate}Hz x{self._open_channels} to "
                        f"{sample_rate}Hz x{num_channels}; the stream attributes are per-stream, so "
                        f"the current stream is closed first"
                    )
                await self._close_locked("format change")
            try:
                writer = self._writer
                if writer is None:
                    writer = await self._open_locked(sample_rate, num_channels)
                await writer.write(pcm_s16le)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Drop the rest of this utterance; the next chunk opens a fresh stream. The broken stream is
                # still open on the wire, so it gets a detached close: this is the audio hot path.
                writer, self._writer = self._writer, None
                self._open_rate = self._open_channels = None
                if writer is not None:
                    self._close_detached(writer, "write failed")
                if not self._failed_this_utterance:
                    self._failed_this_utterance = True
                    self._log.warning(f"audio chunk dropped after data-stream failure: {exc!r}")
                return
            self._arm_idle_timer()

    async def end(self) -> None:
        """Close the current stream — the end-of-utterance marker the worker waits for.

        Raises nothing except `CancelledError`, and even then the close it started is already running
        detached, so the stream still gets closed.
        """
        async with self._lock:
            await self._close_locked("end")

    async def abort(self) -> None:
        """`end()`, logged as an interruption. Same raise contract."""
        async with self._lock:
            await self._close_locked("interruption")

    async def aclose(self) -> None:
        """`end()`, marking the streamer closed so a later `write()` is a no-op, then reap any close still
        running detached; idempotent."""
        async with self._lock:
            self._closed = True
            await self._close_locked("end")
        await self._reap_detached_closes()

    # --- internals (all called with `_lock` held) ------------------------------------------------

    async def _open_locked(self, rate: int, ch: int) -> ByteStreamWriterLike:
        """Open one byte stream. If the caller is cancelled, the future is kept as the orphaned open."""
        fut = asyncio.ensure_future(
            self._participant.stream_bytes(
                AUDIO_STREAM_NAME_PREFIX + uuid4().hex[:12],
                topic=AUDIO_STREAM_TOPIC,
                destination_identities=[self._destination_identity],
                # Required by the worker's receiver, as decimal strings: a missing or non-string value
                # raises ValueError there and ends the session.
                attributes={
                    STREAM_ATTR_SAMPLE_RATE: str(int(rate)),
                    STREAM_ATTR_NUM_CHANNELS: str(int(ch)),
                },
            )
        )
        try:
            writer = await asyncio.shield(fut)
        except asyncio.CancelledError:
            self._orphaned_open = fut
            raise
        self._writer = writer
        self._open_rate, self._open_channels = rate, ch
        self._log.debug(f"opened audio stream {rate}Hz x{ch} to {self._destination_identity}")
        return writer

    async def _close_locked(self, reason: str) -> None:
        """Close whatever is open: an orphaned open, a live writer, or both. Bounded; never raises.

        Each close runs in its own task that the caller only waits on, so a cancel or a timeout detaches the
        wait and never the close itself.
        """
        # Taken first: whatever happens below, this close supersedes any timer already scheduled.
        self._cancel_idle_timer()
        pending, self._orphaned_open = self._orphaned_open, None
        writer, self._writer = self._writer, None
        self._open_rate = self._open_channels = None
        self._failed_this_utterance = False

        if pending is not None:
            await self._await_close(
                asyncio.ensure_future(_drain_and_close(pending)),
                what=f"orphaned audio stream ({reason})",
            )
        if writer is not None:
            await self._await_close(asyncio.ensure_future(writer.aclose()), what=f"audio stream ({reason})")

    async def _await_close(self, closing: asyncio.Task[None], *, what: str) -> None:
        """Wait up to `STREAM_CLOSE_SECS` for a close task; on cancel or timeout let it finish alone."""
        try:
            await asyncio.wait_for(asyncio.shield(closing), _types.STREAM_CLOSE_SECS)
        except asyncio.CancelledError:
            self._run_detached(closing, what=what)
            raise
        except TimeoutError:
            self._run_detached(closing, what=what)
            self._log.warning(f"closing {what} is taking over {_types.STREAM_CLOSE_SECS}s; not waiting")
        except RuntimeError as exc:
            # "Stream already closed" is benign: someone else closed it first.
            if "already closed" in str(exc):
                self._log.debug(f"{what} was already closed")
            else:
                self._log.warning(f"could not close {what}: {exc!r}")
        except Exception as exc:
            self._log.warning(f"could not close {what}: {exc!r}")
        else:
            self._log.debug(f"closed {what}")

    def _close_detached(self, writer: ByteStreamWriterLike, reason: str) -> None:
        """Close a writer without waiting, for the audio hot path."""
        self._run_detached(asyncio.ensure_future(writer.aclose()), what=f"audio stream ({reason})")

    def _run_detached(self, closing: asyncio.Task[None], *, what: str) -> None:
        """Keep a close alive without a waiter: hold a reference and consume its result when it lands."""
        self._detached_closes.add(closing)

        def _done(task: asyncio.Task[None]) -> None:
            self._detached_closes.discard(task)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                self._log.debug(f"detached close of {what} failed: {exc!r}")

        closing.add_done_callback(_done)

    async def _reap_detached_closes(self) -> None:
        """Wait out any detached close, so a closed streamer leaves nothing running."""
        pending = tuple(self._detached_closes)
        if not pending:
            return
        _, still_running = await asyncio.wait(pending, timeout=_types.STREAM_CLOSE_SECS)
        if still_running:
            self._log.warning(f"{len(still_running)} audio stream close(s) still running after aclose()")

    def _arm_idle_timer(self) -> None:
        """(Re-)arm the idle close, so an utterance whose TTS never sends TTSStoppedFrame still ends."""
        self._cancel_idle_timer()
        loop = asyncio.get_running_loop()
        self._idle_handle = loop.call_later(self._idle_close_secs, self._on_idle, self._idle_generation)

    def _on_idle(self, generation: int) -> None:
        self._idle_handle = None
        self._run_detached(
            asyncio.ensure_future(self._end_if_still_idle(generation)), what="the idle fallback close"
        )

    async def _end_if_still_idle(self, generation: int) -> None:
        """Close the utterance, unless a write got the lock first and re-armed the timer behind us."""
        async with self._lock:
            if self._idle_generation != generation:
                return
            await self._close_locked("idle")

    def _cancel_idle_timer(self) -> None:
        self._idle_generation += 1
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None


async def _drain_and_close(pending: asyncio.Future[ByteStreamWriterLike]) -> None:
    """Wait for an open whose caller was cancelled, then close the stream it produced."""
    writer = await asyncio.shield(pending)
    await writer.aclose()
