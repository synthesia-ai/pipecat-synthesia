"""Shared constants and small helpers for `tests/unit/test_service_*.py`.

Not a test module (no `test_` prefix). Everything here is a convenience over the public seams:
frame constructors with the pipeline's default 24 kHz mono rate, a `caplog` filter that matches a log
line by its level and one stable keyword rather than its full text, and two waits: for a callback the
fake emitted, and for every frame queued so far to have been processed.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from pipecat.frames.frames import TextFrame, TTSAudioRawFrame

from tests.harness import Harness
from tests.support.waiting import until

#: 40 ms of 24 kHz mono s16le, distinguishable by their first byte so a test can tell which
#: utterance a chunk at the fake session came from without private state.
AUDIO_1 = b"\x01\x00" * 960
AUDIO_2 = b"\x02\x00" * 960
AUDIO_3 = b"\x03\x00" * 960
SAMPLE_RATE = 24000


def tts_audio(context_id: str | None = None, audio: bytes = AUDIO_1) -> TTSAudioRawFrame:
    """A `TTSAudioRawFrame` at the pipeline's default output rate."""
    return TTSAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1, context_id=context_id)


def records_at(caplog: pytest.LogCaptureFixture, level: int, keyword: str) -> list[logging.LogRecord]:
    """Captured records at exactly `level` whose message contains `keyword`."""
    return [r for r in caplog.records if r.levelno == level and keyword in r.getMessage()]


async def delivered(task: asyncio.Task | None) -> None:
    """Await a callback the fake emitted; None means the fake had no callbacks installed yet."""
    assert task is not None, "the fake had no callbacks installed"
    await task


async def processed(h: Harness, marker: str) -> None:
    """Every frame queued before this one has been processed: frames are handled in order."""
    await h.queue(TextFrame(marker))
    await until(
        lambda: any(isinstance(f, TextFrame) and f.text == marker for f in h.down),
        what=f"the {marker!r} marker downstream",
    )
