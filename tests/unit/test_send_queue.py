"""`pipecat_synthesia._send_queue.SendQueue`: FIFO order, byte accounting, drop-oldest bounded by audio
bytes with utterance boundaries preserved, and an event-driven `get()`."""

from __future__ import annotations

import asyncio

import pytest

from pipecat_synthesia._send_queue import AudioChunk, SendQueue, UtteranceEnd


def _audio(n: int, tag: int = 0) -> AudioChunk:
    return AudioChunk(bytes([tag]) * n, 24000, 1, "c")


@pytest.mark.unit
async def test_fifo_and_byte_accounting() -> None:
    q = SendQueue(max_bytes=10_000)
    assert q.empty and q.bytes == 0
    assert q.put(_audio(100, 1)) == 0
    assert q.put(UtteranceEnd("c")) == 0
    assert q.put(_audio(50, 2)) == 0
    assert q.bytes == 150 and not q.empty

    first = await q.get()
    assert isinstance(first, AudioChunk) and first.audio[0] == 1
    assert q.bytes == 50
    assert isinstance(await q.get(), UtteranceEnd)
    assert q.bytes == 50 and not q.empty  # boundaries carry no bytes, so bytes == 0 != empty
    third = await q.get()
    assert isinstance(third, AudioChunk) and third.audio[0] == 2
    assert q.empty and q.bytes == 0


@pytest.mark.unit
def test_drop_oldest_keeps_utterance_ends_and_reports_bytes() -> None:
    q = SendQueue(max_bytes=250)
    q.put(_audio(100, 1))
    q.put(UtteranceEnd("c1"))
    q.put(_audio(100, 2))
    # 300 > 250: the oldest AudioChunk goes, the boundary stays, the newest is kept
    assert q.put(_audio(100, 3)) == 100
    assert q.bytes == 200
    items = q.take_all()
    assert [type(i).__name__ for i in items] == ["UtteranceEnd", "AudioChunk", "AudioChunk"]
    assert [i.audio[0] for i in items if isinstance(i, AudioChunk)] == [2, 3]
    assert q.empty and q.bytes == 0


@pytest.mark.unit
def test_oversized_single_item_is_dropped_and_counted() -> None:
    q = SendQueue(max_bytes=10)
    assert q.put(_audio(4096)) == 4096
    assert q.empty and q.bytes == 0


@pytest.mark.unit
async def test_get_waits_for_put_and_clear_resets() -> None:
    q = SendQueue(max_bytes=1000)
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    assert not getter.done()
    q.put(_audio(10))
    item = await asyncio.wait_for(getter, 1.0)
    assert isinstance(item, AudioChunk)

    q.put(_audio(10))
    q.put(UtteranceEnd(None, final=True))
    q.clear()
    assert q.empty and q.bytes == 0
    waiting = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    assert not waiting.done(), "clear() must leave get() parked, not hand it a stale item"
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting


@pytest.mark.unit
def test_truncate_after_drops_what_follows_an_item_found_by_identity() -> None:
    q = SendQueue(max_bytes=10_000)
    q.put(_audio(100, 1))
    kept = UtteranceEnd("c1")
    q.put(kept)
    q.put(_audio(50, 2))
    q.put(UtteranceEnd("c1"))  # equal to `kept`, but not it
    q.put(_audio(30, 3))
    assert q.truncate_after(kept) == 80, "audio and ends alike are dropped after `kept`"
    assert q.last is kept and q.bytes == 100


@pytest.mark.unit
def test_truncate_after_none_or_an_item_not_queued_drops_everything() -> None:
    for item in (None, UtteranceEnd("elsewhere")):
        q = SendQueue(max_bytes=10_000)
        q.put(_audio(100, 1))
        q.put(UtteranceEnd("c1"))
        assert q.truncate_after(item) == 100
        assert q.empty and q.bytes == 0 and q.last is None


@pytest.mark.unit
def test_an_utterance_end_kept_through_an_overflow_still_marks_where_to_truncate() -> None:
    """Overflow drops the audio ahead of `kept`, never `kept` itself, so what came after it can still go."""
    q = SendQueue(max_bytes=150)
    q.put(_audio(100, 1))
    kept = UtteranceEnd("c1")
    q.put(kept)
    assert q.put(_audio(100, 2)) == 100
    assert q.truncate_after(kept) == 100
    assert q.last is kept and q.bytes == 0


@pytest.mark.unit
def test_truncate_after_the_last_item_drops_nothing() -> None:
    q = SendQueue(max_bytes=10_000)
    q.put(_audio(100, 1))
    kept = UtteranceEnd("c1")
    q.put(kept)
    assert q.truncate_after(kept) == 0
    assert q.last is kept and q.bytes == 100


@pytest.mark.unit
async def test_truncate_after_that_empties_the_queue_leaves_get_waiting() -> None:
    q = SendQueue(max_bytes=10_000)
    q.put(_audio(100, 1))
    q.truncate_after(None)
    getter = asyncio.ensure_future(q.get())
    await asyncio.sleep(0)
    assert not getter.done(), "get() returned from an emptied queue"
    q.put(_audio(10, 2))
    item = await asyncio.wait_for(getter, 1.0)
    assert isinstance(item, AudioChunk) and item.audio[0] == 2
