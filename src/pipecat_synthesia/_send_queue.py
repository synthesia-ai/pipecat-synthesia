"""The send queue: a bounded FIFO of audio chunks and utterance ends, from the frame path to the send loop.

``process_frame`` must never await the network (*frame contract*), so it only puts, and the send loop
gets and does the I/O. ``put()`` is synchronous and cheap; ``get()`` parks while the queue is empty. Items
carry no generation: an interruption clears the queue synchronously, and late audio from an interrupted
TTS context is dropped before it reaches the queue.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioChunk:
    audio: bytes
    sample_rate: int
    num_channels: int
    context_id: str | None


@dataclass(frozen=True)
class UtteranceEnd:
    context_id: str | None
    # set on the utterance end that stop() queues: when the send loop reaches it, the queue has drained
    final: bool = False


class SendQueue:
    """FIFO of :class:`AudioChunk` / :class:`UtteranceEnd` bounded by queued audio bytes.

    When a ``put()`` would push the queued audio above ``max_bytes`` the oldest ``AudioChunk`` s are dropped
    until it fits (``UtteranceEnd`` s are never dropped: they carry the utterance boundaries the worker
    needs). The number of bytes dropped is returned so the caller can count it and warn once.
    """

    def __init__(self, *, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._items: deque[AudioChunk | UtteranceEnd] = deque()
        self._bytes = 0
        self._event = asyncio.Event()

    def put(self, item: AudioChunk | UtteranceEnd) -> int:
        """Append ``item``; return the number of audio bytes dropped to stay within ``max_bytes``."""
        self._items.append(item)
        if isinstance(item, AudioChunk):
            self._bytes += len(item.audio)
        dropped = 0
        if self._bytes > self._max_bytes:
            # Drop from the left, skipping UtteranceEnds, until the audio fits again.
            kept: deque[AudioChunk | UtteranceEnd] = deque()
            while self._items and self._bytes > self._max_bytes:
                oldest = self._items.popleft()
                if isinstance(oldest, AudioChunk):
                    self._bytes -= len(oldest.audio)
                    dropped += len(oldest.audio)
                else:
                    kept.append(oldest)
            # Re-insert the preserved boundaries ahead of whatever audio remains, in their original order.
            self._items.extendleft(reversed(kept))
        self._event.set()
        return dropped

    async def get(self) -> AudioChunk | UtteranceEnd:
        """Wait for and pop the oldest item, subtracting its audio bytes."""
        while not self._items:
            self._event.clear()
            await self._event.wait()
        item = self._items.popleft()
        if isinstance(item, AudioChunk):
            self._bytes -= len(item.audio)
        if not self._items:
            self._event.clear()
        return item

    def put_back(self, item: AudioChunk | UtteranceEnd) -> None:
        """Return the item ``get()`` just took to the front of the queue, as if it had never been taken."""
        self._items.appendleft(item)
        if isinstance(item, AudioChunk):
            self._bytes += len(item.audio)
        self._event.set()

    def clear(self) -> None:
        """Drop everything and reset the byte counter."""
        self._items.clear()
        self._bytes = 0
        self._event.clear()

    def take_all(self) -> list[AudioChunk | UtteranceEnd]:
        """Remove and return everything (used for the voice-only handover)."""
        items = list(self._items)
        self.clear()
        return items

    @property
    def bytes(self) -> int:
        """Queued, not-yet-sent audio bytes."""
        return self._bytes

    def truncate_after(self, item: AudioChunk | UtteranceEnd | None) -> int:
        """Drop everything queued after ``item``, found by identity; with ``None``, or if ``item`` is not
        queued, drop everything. Return the number of audio bytes dropped.

        Unlike an overflow, this drops utterance ends too. It is only used before the session is active (a
        later *pre-launch interruption*), when nothing has been sent: no byte stream is open at the worker,
        and an utterance dropped whole needs no end. Nor can it drop the final end a graceful stop queues,
        which is queued only once the session is active.
        """
        before = self._bytes
        while self._items and self._items[-1] is not item:
            dropped = self._items.pop()
            if isinstance(dropped, AudioChunk):
                self._bytes -= len(dropped.audio)
        if not self._items:
            self._event.clear()
        return before - self._bytes

    @property
    def last(self) -> AudioChunk | UtteranceEnd | None:
        """The item queued last, or None when the queue is empty."""
        return self._items[-1] if self._items else None

    @property
    def empty(self) -> bool:
        """True when no item is queued (UtteranceEnds count: ``bytes == 0`` does not imply ``empty``)."""
        return not self._items
