"""Ctrl-C and SIGTERM under Pipecat's development runner: end every call in order, then exit.

`pipecat.runner.run.main` runs a bot in one of two ways, and says which through the runner arguments:

- **The bot is the whole process** (`-t daily -d`): `runner_args.handle_sigint` is True. The bot's
  `WorkerRunner` handles SIGINT and SIGTERM itself, ending its one call in order, and the process
  exits when `bot()` returns. Nothing extra is needed.
- **A web server runs the bots** (`-t webrtc`, the default; Daily through `/start`):
  `runner_args.handle_sigint` is False. Each call is a `bot()` task inside uvicorn. A `WorkerRunner`
  that installed its own handlers would take them from uvicorn for the life of the process, so after
  the first call Ctrl-C would stop nothing. One with no handlers leaves the call to uvicorn's shutdown,
  which does not end it in order: SIGTERM exits with no CancelFrame reaching the avatar, and the
  session keeps billing until the avatar worker notices the bot has gone; SIGINT waits for the call
  until a second Ctrl-C.

So in the second case this module owns the two signals, for the life of the process. The first one
cancels every running call (and no call starts after it), waits for them to finish (at most
`SHUTDOWN_SECS`), and then re-raises the signal with its default disposition, so the process exits as
that signal (exit status 130 or 143) without uvicorn's own graceful shutdown. Repeats while it waits
are ignored: `uv run` forwards a Ctrl-C the terminal already delivered, so one keypress can arrive
twice. As PID 1 in a container the kernel drops the re-raised signal, so the process then exits with
`128 + signal` itself.

Where the event loop cannot take signal handlers (Windows), each call's `WorkerRunner` owns the
signals instead, as it did before this module existed.
"""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

from loguru import logger
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner

__all__ = ["SHUTDOWN_SECS", "run_under_signals"]

#: How long a signal waits for the running calls to end before the process exits anyway: Docker's
#: default stop grace, after which it would be SIGKILLed. An ordered end normally takes one to three
#: seconds, and if this cuts one short, the platform finishes the teardown itself.
#: `docs/deployment.md` in the pipecat-synthesia repository has the detail, and when to raise it.
SHUTDOWN_SECS = 10.0

_SIGNALS = (signal.SIGINT, signal.SIGTERM)

#: The calls running under the web server: each one's runner and its `run()` task.
_calls: dict[WorkerRunner, asyncio.Task[None]] = {}
#: The event loop `_install_handlers` last ran on, and whether it could take the handlers there.
_handlers: tuple[asyncio.AbstractEventLoop, bool] | None = None
_shutdown: asyncio.Task[None] | None = None
_repeat_logged = False


async def run_under_signals(worker: BaseWorker, runner_args: Any) -> None:
    """Run `worker` to its end, ending it in order on SIGINT or SIGTERM however the runner runs the bot."""
    owns_process = runner_args.handle_sigint
    if not owns_process and not _install_handlers():
        # The runner owns the signals, as before; after the first call Ctrl-C may stop nothing.
        runner = WorkerRunner(handle_sigint=True, handle_sigterm=True)
        await runner.add_workers(worker)
        await runner.run()
        return

    runner = WorkerRunner(handle_sigint=owns_process, handle_sigterm=owns_process)
    await runner.add_workers(worker)
    if owns_process:
        await runner.run()
        return

    # `_shutdown` is set synchronously in the signal handler, before the shutdown task is even scheduled,
    # and nothing between this check and the call's registration awaits. So a call is either registered
    # before the signal, and the shutdown ends it and waits for it, or sees the flag here and never starts.
    # Keep it that way: no await may go between this check and `_calls[runner] = call`.
    if _shutdown is not None:
        logger.info("shutting down: not starting this call")
        return
    call = asyncio.create_task(runner.run(), name="bot-call")
    _calls[runner] = call
    try:
        await call
    finally:
        _calls.pop(runner, None)


def _install_handlers() -> bool:
    """Take SIGINT and SIGTERM once per event loop; False where the loop cannot (Windows)."""
    global _handlers
    loop = asyncio.get_running_loop()
    if _handlers is not None and _handlers[0] is loop:
        return _handlers[1]
    installed = True
    try:
        for sig in _SIGNALS:
            loop.add_signal_handler(sig, _on_signal, sig)
    except NotImplementedError:
        logger.warning("this event loop cannot take signal handlers; each call's runner will handle Ctrl-C")
        installed = False
    _handlers = (loop, installed)
    return installed


def _on_signal(sig: signal.Signals) -> None:
    global _shutdown, _repeat_logged
    if _shutdown is not None:
        if not _repeat_logged:
            _repeat_logged = True
            logger.info(
                f"{sig.name} again: still ending the calls in order; forcing exit after {SHUTDOWN_SECS:g} s"
            )
        return
    logger.info(
        f"{sig.name}: ending {len(_calls)} call(s) in order, then exiting (at most {SHUTDOWN_SECS:g} s)"
    )
    _shutdown = asyncio.get_running_loop().create_task(_end_calls_then_exit(sig), name="bot-shutdown")


async def _end_calls_then_exit(sig: signal.Signals) -> None:
    # Every call there will be: none starts once `_shutdown` is set (see run_under_signals).
    # This relies on Pipecat's `WorkerRunner.run()` clearing its shutdown event in its first step, and
    # `cancel()` only setting it: a call registered just before the signal has its first step run
    # before this task's (tasks start in the order they were created), so the cancel below is not
    # cleared. If a Pipecat upgrade changes that order, a call can run on to the `SHUTDOWN_SECS` bound.
    # `test_pipecats_runner_ends_on_a_cancel_from_a_task_made_just_after_its_run`, in the
    # pipecat-synthesia repository's `tests/unit/test_examples_end_calls_in_order_on_a_signal.py`,
    # pins that order with the real runner.
    calls = dict(_calls)
    for runner in calls:
        await runner.cancel(reason=f"{sig.name} received")
    if calls:
        _, pending = await asyncio.wait(calls.values(), timeout=SHUTDOWN_SECS)
        if pending:
            logger.warning(f"{len(pending)} call(s) still ending after {SHUTDOWN_SECS:g} s; exiting anyway")
    _exit_by(sig)


def _exit_by(sig: signal.Signals) -> None:
    """Exit as `sig` would have, had nothing handled it."""
    signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)
    # Still here: as PID 1 the kernel drops a default-action signal the process sends itself.
    os._exit(128 + sig)
