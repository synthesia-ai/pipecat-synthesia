"""The asyncio helpers in `_async.py`: *cancel-proof cleanup*, the bound around it, and callback tasks."""

from __future__ import annotations

import asyncio
import logging

import pytest

from pipecat_synthesia._async import CallbackTasks, await_cancel_proof, wait_bounded
from pipecat_synthesia._log import get_logger
from tests.support.waiting import until
from tests.unit._service_helpers import records_at


@pytest.mark.unit
async def test_cleanup_runs_to_the_end_through_repeated_cancels() -> None:
    """A caller cancelled twice mid-wait still waits for the cleanup, and is told it was cancelled."""
    release = asyncio.Event()
    finished = asyncio.Event()

    async def cleanup() -> None:
        await release.wait()
        finished.set()

    async def caller() -> bool:
        return await await_cancel_proof(asyncio.ensure_future(cleanup()))

    task = asyncio.create_task(caller())
    for _ in range(2):  # the second cancel is what Pipecat's cancel_task sends when its timeout expires
        await asyncio.sleep(0)
        task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "the wait must not end while the cleanup is still running"

    release.set()
    assert await task is True
    assert finished.is_set()


@pytest.mark.unit
async def test_uninterrupted_cleanup_reports_so_and_raises_its_own_error() -> None:
    async def succeeds() -> None:
        return None

    async def fails() -> None:
        raise ValueError("cleanup failed")

    assert await await_cancel_proof(asyncio.ensure_future(succeeds())) is False
    with pytest.raises(ValueError, match="cleanup failed"):
        await await_cancel_proof(asyncio.ensure_future(fails()))


@pytest.mark.unit
async def test_wait_bounded_returns_at_the_bound_and_leaves_cancel_proof_cleanup_running() -> None:
    """A bound around *cancel-proof cleanup* returns once the bound expires, where `wait_for` would wait
    for the whole cleanup, and it never cancels the cleanup. The 0.05 s bound is an input: what is
    asserted is the order, that the wait returned before the cleanup finished."""
    release = asyncio.Event()
    finished = asyncio.Event()

    async def inner() -> None:
        await release.wait()
        finished.set()

    async def cancel_proof_cleanup() -> None:
        if await await_cancel_proof(asyncio.ensure_future(inner())):
            raise asyncio.CancelledError

    cleanup = asyncio.ensure_future(cancel_proof_cleanup())
    waiter = asyncio.ensure_future(wait_bounded(cleanup, 0.05))
    try:
        await until(waiter.done, what="the bounded wait to return")
        assert waiter.result() is False
        assert not finished.is_set(), "the wait returned before the cleanup finished"
        assert not cleanup.done()
    finally:
        release.set()
    await until(cleanup.done, what="the cleanup to finish on its own")
    assert finished.is_set()
    assert not cleanup.cancelled()


@pytest.mark.unit
async def test_wait_bounded_reports_a_finished_task_and_raises_its_error() -> None:
    async def succeeds() -> None:
        return None

    async def fails() -> None:
        raise ValueError("close failed")

    assert await wait_bounded(asyncio.ensure_future(succeeds()), 5.0) is True
    with pytest.raises(ValueError, match="close failed"):
        await wait_bounded(asyncio.ensure_future(fails()), 5.0)


@pytest.mark.unit
async def test_failing_callback_is_logged_not_raised(caplog_loguru: pytest.LogCaptureFixture) -> None:
    async def fails() -> None:
        raise RuntimeError("callback broke")

    task = CallbackTasks("test callback").schedule(fails(), log=get_logger())
    await task  # the guard absorbed the failure, so the task itself completes normally

    assert records_at(caplog_loguru, logging.ERROR, "test callback failed")
