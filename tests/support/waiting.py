"""Polling helpers: the sanctioned way to wait for anything asynchronous.

"Durations are inputs, never assertions" only works if there is a waiting primitive that does not encode
a duration in the assertion. That is :func:`until`: it polls a predicate and asserts the *eventual
outcome*, so a CI runner that stalls for 300 ms makes the test slower rather than red.

``await asyncio.sleep(0.1)`` followed by an assertion is the mistake this module exists to remove.
``h.sleep()`` in the harness is for *modelling* a delay (letting a hysteresis window elapse with a 10x
margin), never for synchronising with a background task.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from pipecat.utils.asyncio.task_manager import BaseTaskManager

T = TypeVar("T")

__all__ = ["assert_no_dangling", "eventually_equal", "until"]

DEFAULT_TIMEOUT = 5.0
DEFAULT_INTERVAL = 0.01


async def until(
    pred: Callable[[], bool],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    interval: float = DEFAULT_INTERVAL,
    what: str = "",
) -> None:
    """Poll ``pred()`` until it is truthy; raise ``AssertionError`` if it is still false at ``timeout``.

    Never returns ``False`` — the only two outcomes are "the predicate came true" and a failed
    assertion, so a caller can never accidentally ignore a timeout. ``pred`` is evaluated once before
    the first sleep, so a condition that is already true costs no wall-clock time.

    ``timeout`` is deliberately generous by default: it is a backstop against a hung test, not a
    performance budget, and shrinking it is how a suite starts flaking on a loaded runner. Pass a
    short ``timeout`` only when the timeout *is* the assertion (proving a predicate stays false).

    Args:
        pred: Synchronous, cheap, side-effect free; called repeatedly.
        timeout: Seconds to keep polling before failing.
        interval: Seconds between polls.
        what: Phrase naming the condition, used in the failure message. Always pass it — ``repr`` of a
            lambda names a file and a line number, which is a poor failure message.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if pred():
            return
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what or pred}")
        await asyncio.sleep(interval)


async def eventually_equal(
    get: Callable[[], T],
    expected: T,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    interval: float = DEFAULT_INTERVAL,
    what: str = "",
) -> None:
    """Poll ``get()`` until it equals ``expected``; raise ``AssertionError`` naming the last value seen.

    :func:`until` with an equality predicate loses the observed value, which is the one thing you want
    in the failure message ("expected 2, last saw 1" beats "timed out").
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        actual = get()
        if actual == expected:
            return
        if loop.time() >= deadline:
            raise AssertionError(
                f"timed out after {timeout}s waiting for {what or get} to equal {expected!r}; "
                f"last value was {actual!r}"
            )
        await asyncio.sleep(interval)


def assert_no_dangling(task_manager: BaseTaskManager) -> None:
    """Assert every task the manager created has finished, so a test leaves nothing running.

    ``TaskManager.current_tasks()`` is one of the two sanctioned exceptions to asserting only through
    public interfaces: hangs and leaked tasks are the historical Pipecat failure mode (#4547), and no
    public interface shows them. Call it after the harness/fixture has torn everything
    down; a task that is merely *slow* to cancel fails here, which is the point.
    """
    pending = [task for task in task_manager.current_tasks() if not task.done()]
    assert not pending, "tasks still running after teardown: " + ", ".join(
        task.get_name() for task in pending
    )
