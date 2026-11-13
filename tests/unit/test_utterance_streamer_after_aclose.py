"""`UtteranceStreamer` after `aclose()`: a later `write()` opens nothing; after `end()` or `abort()` it does.

`aclose()` closes the streamer, while `end()` and `abort()` only end an utterance. A write that reaches the
streamer after `aclose()`, for instance one queued on its lock behind the close, must not open a byte stream
that nothing would close (*one-open-stream rule*). The session's own liveness check stops such a write
before it gets here; this is the streamer's own guarantee, so it is pinned on the streamer directly.
"""

from __future__ import annotations

import pytest
from loguru import logger

from pipecat_synthesia.livekit.utterance_streamer import UtteranceStreamer
from tests.fakes.fake_rtc import FakeLocalParticipant

CHUNK = b"\x01\x02" * 480  # 20 ms of 24 kHz mono s16le


def make_streamer(participant: FakeLocalParticipant) -> UtteranceStreamer:
    return UtteranceStreamer(
        participant,
        destination_identity="synthesia-avatar-agent",
        idle_close_secs=10.0,
        log=logger.bind(component="test"),
    )


@pytest.mark.unit
async def test_a_write_after_aclose_opens_no_byte_stream() -> None:
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    await streamer.write(CHUNK, 24000, 1)
    await streamer.aclose()

    await streamer.write(CHUNK, 24000, 1)

    assert len(participant.streams) == 1, "a byte stream was opened after aclose()"
    assert participant.unclosed_streams() == []
    await streamer.aclose()  # idempotent


@pytest.mark.unit
@pytest.mark.parametrize("ending", ["end", "abort"])
async def test_a_write_after_an_utterance_ends_opens_the_next_byte_stream(ending: str) -> None:
    participant = FakeLocalParticipant()
    streamer = make_streamer(participant)
    try:
        await streamer.write(CHUNK, 24000, 1)
        await getattr(streamer, ending)()

        await streamer.write(CHUNK, 24000, 1)

        assert len(participant.streams) == 2, f"no byte stream was opened after {ending}()"
    finally:
        await streamer.aclose()
    assert participant.unclosed_streams() == []
