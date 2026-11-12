"""Filters and assertions over a recorded frame list: filter it instead of comparing exact sequences.

Pipecat's ``run_test`` asserts an exact length plus an ordered ``isinstance`` walk, which makes every
strict test fail the day we emit a new frame type, and makes any test racy whose frames arrive from a
task (``Bot*SpeakingFrame``, ``ErrorFrame``). The helpers here ask the questions a test actually means:
*is there no ``TTSAudioRawFrame`` downstream*, *exactly one ``BotStartedSpeakingFrame``*, *does
``TTSStartedFrame`` precede ``TTSStoppedFrame``*.

Every function is pure and synchronous: pass it ``h.down`` / ``h.up`` (or any ``Sequence[Frame]``)
after the harness has finished. ``isinstance`` is the matcher throughout, so a listed type also
matches its subclasses — pass the narrowest type you mean (``TTSAudioRawFrame``, not
``AudioRawFrame``) unless you want the family.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

from pipecat.frames.frames import Frame

FrameT = TypeVar("FrameT", bound=Frame)

__all__ = ["assert_absent", "assert_ordered", "count", "first", "frames_of"]


def _names(seq: Sequence[Frame]) -> str:
    """The frame list as a readable type sequence, for failure messages."""
    return "[" + ", ".join(type(frame).__name__ for frame in seq) + "]"


def frames_of(seq: Sequence[Frame], *types: type[Frame]) -> list[Frame]:
    """Every frame in ``seq`` that is an instance of any of ``types``, in arrival order.

    With no types given, returns a plain list copy — handy for snapshotting ``h.down`` before letting
    the pipeline run on.
    """
    if not types:
        return list(seq)
    return [frame for frame in seq if isinstance(frame, types)]


def first(seq: Sequence[Frame], frame_type: type[FrameT]) -> FrameT:
    """The first ``frame_type`` in ``seq``; ``AssertionError`` naming the type if there is none."""
    for frame in seq:
        if isinstance(frame, frame_type):
            return frame
    raise AssertionError(f"no {frame_type.__name__} in {_names(seq)}")


def count(seq: Sequence[Frame], frame_type: type[Frame]) -> int:
    """How many frames in ``seq`` are instances of ``frame_type``.

    Use for the "exactly one" assertions (one ``BotStartedSpeakingFrame`` per utterance) and for
    ">= 1" on anything periodic (``BotSpeakingFrame`` is never asserted as a count).
    """
    return sum(1 for frame in seq if isinstance(frame, frame_type))


def assert_absent(seq: Sequence[Frame], frame_type: type[Frame], *, where: str = "") -> None:
    """Assert no frame in ``seq`` is a ``frame_type``, naming the offender if one is.

    Args:
        seq: The recorded frames.
        frame_type: The type that must not appear.
        where: Phrase naming the place being checked ("downstream of the service in ENDED"), used in
            the failure message.
    """
    for index, frame in enumerate(seq):
        if isinstance(frame, frame_type):
            location = f" {where}" if where else ""
            raise AssertionError(
                f"unexpected {type(frame).__name__} at index {index}{location}: "
                f"{frame!r}\n  full sequence: {_names(seq)}"
            )


def assert_ordered(seq: Sequence[Frame], types: Sequence[type[Frame]]) -> None:
    """Assert each type in ``types`` appears in ``seq`` and their *first* occurrences are in that order.

    A subsequence check: frames of other types interleaved anywhere never fail it, so this survives
    us emitting new frame types and arrival races between unrelated frames. What it does
    catch is a genuinely inverted pair — a ``TTSStoppedFrame`` before any ``TTSStartedFrame``.

    Ordering is by first occurrence, not by a greedy match, so a stray early frame of a later type is
    a failure rather than something a subsequent occurrence papers over. That also means a type may
    not be listed twice (the first occurrence of a type is a single index); assert repeats with
    :func:`count`, or split the sequence and call this once per utterance.
    """
    repeated = sorted(name.__name__ for name, n in Counter(types).items() if n > 1)
    if repeated:
        raise ValueError(
            f"assert_ordered takes each type at most once (first-occurrence ordering), got "
            f"{repeated} twice; use count() for repeats"
        )

    positions: dict[type[Frame], int] = {}
    for frame_type in types:
        index = next((i for i, frame in enumerate(seq) if isinstance(frame, frame_type)), None)
        if index is None:
            raise AssertionError(
                f"expected order {[t.__name__ for t in types]} but there is no "
                f"{frame_type.__name__} in {_names(seq)}"
            )
        positions[frame_type] = index

    ordered = list(types)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if positions[earlier] >= positions[later]:
            raise AssertionError(
                f"expected {earlier.__name__} before {later.__name__} but the first "
                f"{later.__name__} is at index {positions[later]} and the first "
                f"{earlier.__name__} at index {positions[earlier]}\n"
                f"  expected order: {[t.__name__ for t in ordered]}\n"
                f"  full sequence:  {_names(seq)}"
            )
