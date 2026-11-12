"""`SendQueue` driven as a *pre-launch interruption* drives it, checked against a reference model.

The pre-launch boundary rule (`SynthesiaVideoService._keep_prelaunch_audio`) relies on two properties of the
queue: an overflow drops the oldest audio but never an utterance end, so the end it remembers as the
boundary survives; and `truncate_after` finds that end by identity and drops exactly what follows it. This
test drives a real `SendQueue` through seeded random sequences of audio, utterance ends and interruptions
under a small cap, so overflow is frequent, and replays each step on a plain list (the reference model) that
implements the same rules independently.

The driver applies the rule as the service does: an interruption with no audio queued sets no boundary; the
first one with audio queued adds an utterance end if the last item is audio and remembers the last item as
the boundary; every later one truncates after it. After every step the queue's public `last`, `bytes` and
`empty` must match the model, along with what `put` and `truncate_after` return. The boundary must still be
queued; no audio queued when it was set may be lost except to overflow; nothing may sit after it right after
a trim. At the end of each sequence `take_all()` must return the model's items, by identity.

The service-level behaviour is pinned in `test_a_greeting_queued_before_the_session_is_active.py`; this
module locks the queue semantics underneath it. A seeded run, so a failure names its sequence and reproduces
exactly.
"""

from __future__ import annotations

import random

import pytest

from pipecat_synthesia._send_queue import AudioChunk, SendQueue, UtteranceEnd

SEED = 20261030
SEQUENCES = 400
CAP = 600
CHUNK_SIZES = (50, 120, 200, 310, 450)


def _model_put(
    model: list[AudioChunk | UtteranceEnd], item: AudioChunk | UtteranceEnd
) -> tuple[int, set[int]]:
    """Append `item`; while the audio exceeds the cap, drop the oldest audio, moving the utterance ends passed
    on the way to the front in their order. Return the bytes dropped and the ids of the dropped chunks."""
    model.append(item)
    total = _model_bytes(model)
    if total <= CAP:
        return 0, set()
    ends: list[AudioChunk | UtteranceEnd] = []
    dropped_ids: set[int] = set()
    dropped = 0
    i = 0
    while i < len(model) and total > CAP:
        x = model[i]
        if isinstance(x, AudioChunk):
            total -= len(x.audio)
            dropped += len(x.audio)
            dropped_ids.add(id(x))
        else:
            ends.append(x)
        i += 1
    model[:] = ends + model[i:]
    return dropped, dropped_ids


def _model_bytes(model: list[AudioChunk | UtteranceEnd]) -> int:
    return sum(len(x.audio) for x in model if isinstance(x, AudioChunk))


def _run_sequence(rng: random.Random, seq: int, counts: dict[str, int]) -> None:
    q = SendQueue(max_bytes=CAP)
    model: list[AudioChunk | UtteranceEnd] = []
    boundary: AudioChunk | UtteranceEnd | None = None
    kept_audio: set[int] = set()  # ids of the audio queued when the boundary was set
    overflowed: set[int] = set()
    alive: list[AudioChunk | UtteranceEnd] = []  # every item stays referenced, so no id is reused

    for _ in range(rng.randint(1, 25)):
        r = rng.random()
        if r < 0.55:
            chunk = AudioChunk(bytes([rng.randrange(256)]) * rng.choice(CHUNK_SIZES), 24000, 1, None)
            alive.append(chunk)
            want, ids = _model_put(model, chunk)
            overflowed |= ids
            counts["overflows"] += bool(want)
            assert q.put(chunk) == want, (seq, "bytes dropped by overflow")
        elif r < 0.75:
            end = UtteranceEnd(None)
            alive.append(end)
            _model_put(model, end)
            assert q.put(end) == 0, (seq, "an utterance end dropped audio")
        elif boundary is not None:
            idx = next(i for i, x in enumerate(model) if x is boundary)
            want = _model_bytes(model[idx + 1 :])
            del model[idx + 1 :]
            assert q.truncate_after(boundary) == want, (seq, "bytes dropped by the trim")
            assert q.last is boundary, (seq, "something after the boundary survived the trim")
            counts["trims"] += 1
        elif not q.bytes:
            assert _model_bytes(model) == 0, (seq, "the queue reports no audio, the model has some")
            counts["no_audio"] += 1
        else:
            if isinstance(q.last, AudioChunk):
                end = UtteranceEnd(None)
                alive.append(end)
                _model_put(model, end)
                q.put(end)
            boundary = q.last
            assert isinstance(boundary, UtteranceEnd), (seq, "the boundary is not an utterance end")
            kept_audio = {id(x) for x in model if isinstance(x, AudioChunk)}
            counts["boundaries"] += 1

        assert q.last is (model[-1] if model else None), (seq, "last item")
        assert q.bytes == _model_bytes(model), (seq, "bytes")
        assert q.empty == (not model), (seq, "empty")
        if boundary is not None:
            assert any(x is boundary for x in model), (seq, "the boundary was dropped")
            present = {id(x) for x in model}
            assert all(a in present or a in overflowed for a in kept_audio), (
                seq,
                "kept audio lost to a trim",
            )

    assert [id(x) for x in q.take_all()] == [id(x) for x in model], (seq, "contents")


@pytest.mark.unit
def test_send_queue_keeps_the_pre_launch_boundary_through_overflow_and_trims() -> None:
    rng = random.Random(SEED)
    counts = {"overflows": 0, "trims": 0, "no_audio": 0, "boundaries": 0}
    for seq in range(SEQUENCES):
        _run_sequence(rng, seq, counts)
    # the generator must keep exercising every path, or the test would pass vacuously
    assert counts["overflows"] > 100 and counts["trims"] > 100, counts
    assert counts["no_audio"] > 20 and counts["boundaries"] > 100, counts
