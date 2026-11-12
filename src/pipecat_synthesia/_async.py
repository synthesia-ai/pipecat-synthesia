"""Shared asyncio helpers: *cancel-proof cleanup*, a bound around it, and callback tasks.

The rules are described in docs/architecture.md (*cancel-proof cleanup*, *bounded waits*,
*state-before-await*).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import loguru

__all__ = ["CallbackTasks", "await_cancel_proof", "wait_bounded"]


async def await_cancel_proof(task: asyncio.Future[Any]) -> bool:
    """Wait until ``task`` has finished, even if the caller is cancelled meanwhile.

    Returns whether the caller was cancelled during the wait, so it can re-raise ``CancelledError`` once
    its cleanup is complete. Every cancellation is absorbed rather than passed to ``task`` or allowed to end
    the wait early, including the second one Pipecat's ``cancel_task`` sends when its own timeout expires.
    If the wait is not interrupted, an exception raised by ``task`` propagates.
    """
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
    if not task.cancelled():
        # Mark the outcome retrieved: a task that finished while we were absorbing a cancel would otherwise
        # have its exception reported again by the event loop at garbage collection.
        task.exception()
    return interrupted


async def wait_bounded(task: asyncio.Future[Any], timeout: float) -> bool:  # noqa: ASYNC109
    """Wait up to ``timeout`` seconds for ``task`` to finish, and return whether it did.

    This is how a bound around *cancel-proof cleanup* is written. ``asyncio.wait_for`` cannot bound it:
    when its timeout expires it cancels the cleanup, which absorbs the cancel and returns only once it
    has finished, so the wait lasts as long as the cleanup does. ``task`` is never cancelled here: past
    the timeout, or when the caller is cancelled, it carries on detached, and the caller must hold a
    reference to it until it finishes. If it finished in time, its exception propagates.
    """
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        return False
    task.result()
    return True


class CallbackTasks:
    """Runs each callback delivery in a task of its own, as *state-before-await* requires.

    A failing callback is logged, never propagated. Every task is held by a strong reference until it
    finishes: the event loop keeps only a weak one, so an unreferenced task can be collected mid-flight.
    """

    def __init__(self, what: str) -> None:
        self._what = what
        self._tasks: set[asyncio.Task[Any]] = set()

    def schedule(self, awaitable: Awaitable[None], *, log: loguru.Logger) -> asyncio.Task[None]:
        """Run ``awaitable`` in a guarded task, and return the task.

        Called only on the running loop: rtc delivers room events from a task on it, and runs each RPC
        handler as a task on it.
        """
        loop = asyncio.get_running_loop()
        return self.track(loop.create_task(self.guard(awaitable, log=log)))

    def track(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        """Hold a strong reference to ``task`` until it finishes, and return it."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def guard(self, awaitable: Awaitable[None], *, log: loguru.Logger) -> None:
        """Await ``awaitable``, logging an exception instead of raising it. Cancellation propagates."""
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f"{self._what} failed")
