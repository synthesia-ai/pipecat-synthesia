"""Ctrl-C or SIGTERM under the development runner's web server ends every call in order, then exits.

`_shutdown.run_under_signals` (one copy in `examples/`, one beside the quickstart) owns SIGINT and
SIGTERM once a bot runs as a call inside the web server (`runner_args.handle_sigint` False). The first
signal cancels every running call, and any call that starts afterwards, waits for them at most
`SHUTDOWN_SECS`, and re-raises the signal once, so the process exits as that signal; if it is still
running after that (PID 1), it exits with `128 + signal`. Repeats while it waits are ignored. A bot the
runner hands the whole process keeps its `WorkerRunner`'s own handlers, for both signals, and so does
every bot where the event loop cannot take signal handlers (Windows).

The tests load `examples/_shutdown.py`; the quickstart's copy is checked to be byte-identical to it.
The runner is a stand-in, so the calls' ending is an input (`ending_secs`), and `os.kill` and
`os._exit` are replaced, so the re-raised signal and the fallback exit are recorded instead of ending
the test run. Signals are real ones, delivered to this process with `signal.raise_signal` only after
checking that the calls installed a handler for them. One test uses Pipecat's real `WorkerRunner`, to
pin the order of its `run()` and `cancel()` that the stand-in takes for granted.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import signal
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner

from tests.support.waiting import until

REPO_ROOT = Path(__file__).resolve().parents[2]
COPIES = ["examples/_shutdown.py", "quickstarts/pipecat/_shutdown.py"]

#: The web server runs the bot: the runner leaves signals to it.
IN_THE_SERVER = SimpleNamespace(handle_sigint=False, handle_sigterm=False)
#: The bot is the whole process (`-t daily -d`): the runner hands it SIGINT.
OWNS_THE_PROCESS = SimpleNamespace(handle_sigint=True, handle_sigterm=False)

#: How long a stand-in call takes to end once cancelled: an input, short.
ENDING_SECS = 0.05


class Recorder:
    """Everything the coordinator did, in order: ("cancel" | "finished", runner) and ("kill", signal)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.runners: list[StandInRunner] = []
        #: How long the next runners' calls take to end once cancelled; None for never.
        self.ending_secs: float | None = ENDING_SECS
        #: Each signal's disposition before any call started.
        self.before: dict[signal.Signals, Any] = {}

    def kills(self) -> list[Any]:
        return [what for kind, what in self.events if kind == "kill"]

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


class StandInRunner:
    """A `WorkerRunner` stand-in: its call runs until cancelled, then ends after `ending_secs`, or never."""

    def __init__(self, recorder: Recorder, *, handle_sigint: bool, handle_sigterm: bool) -> None:
        self.recorder = recorder
        self.handle_sigint = handle_sigint
        self.handle_sigterm = handle_sigterm
        self.ending_secs = recorder.ending_secs
        self.started = False
        self.finished = False
        self.cancel_reason: str | None = None
        self._cancelled = asyncio.Event()
        recorder.runners.append(self)

    async def add_workers(self, *workers: Any) -> None:
        pass

    async def run(self) -> None:
        self.started = True
        await self._cancelled.wait()
        if self.ending_secs is None:
            await asyncio.Event().wait()
        # Models the ordered end taking a moment; never used to synchronise the test.
        await asyncio.sleep(self.ending_secs)
        self.finished = True
        self.recorder.events.append(("finished", self))

    async def cancel(self, reason: str | None = None) -> None:
        self.cancel_reason = reason
        self.recorder.events.append(("cancel", self))
        self._cancelled.set()


@pytest.fixture
def coordinator(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ModuleType, Recorder]]:
    """A fresh `examples/_shutdown.py` module with a stand-in runner and a recording `os.kill`."""
    spec = importlib.util.spec_from_file_location("_shutdown", REPO_ROOT / COPIES[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recorder = Recorder()
    monkeypatch.setattr(module, "WorkerRunner", lambda **flags: StandInRunner(recorder, **flags))

    def kill(pid: int, sig: int) -> None:
        assert pid == os.getpid()
        recorder.events.append(("kill", sig))

    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "_exit", lambda code: recorder.events.append(("exit", code)))
    try:
        yield module, recorder
    finally:
        # The coordinator sets the default disposition before re-raising; give this process its own back.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def deliver(recorder: Recorder, sig: signal.Signals) -> None:
    """Send `sig` to this process, refusing unless the calls installed a handler for it.

    Whatever handled it before (the default, or the test loop's own SIGINT handler) would end the run.
    """
    assert signal.getsignal(sig) is not recorder.before[sig], f"no handler was installed for {sig.name}"
    signal.raise_signal(sig)


async def start_calls(module: ModuleType, recorder: Recorder, count: int) -> list[asyncio.Task[None]]:
    recorder.before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    calls = [asyncio.create_task(module.run_under_signals(object(), IN_THE_SERVER)) for _ in range(count)]
    await until(
        lambda: len(recorder.runners) == count and all(r.started for r in recorder.runners),
        what="every call to be running",
    )
    return calls


@pytest.mark.unit
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
async def test_a_signal_ends_every_call_then_exits_as_that_signal(
    coordinator: tuple[ModuleType, Recorder], sig: signal.Signals
) -> None:
    module, recorder = coordinator
    calls = await start_calls(module, recorder, 2)
    assert all(not r.handle_sigint and not r.handle_sigterm for r in recorder.runners)

    deliver(recorder, sig)
    await until(lambda: bool(recorder.kills()), what="the signal to be re-raised")
    await asyncio.gather(*calls)

    kinds = recorder.kinds()
    assert kinds.count("cancel") == 2 and kinds.count("finished") == 2, recorder.events
    assert max(i for i, k in enumerate(kinds) if k == "finished") < kinds.index("kill"), (
        f"the process exited before every call had ended: {recorder.events}"
    )
    assert recorder.kills() == [sig]
    assert all(sig.name in (r.cancel_reason or "") for r in recorder.runners)


@pytest.mark.unit
async def test_the_wait_for_the_calls_to_end_is_bounded(
    coordinator: tuple[ModuleType, Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, recorder = coordinator
    monkeypatch.setattr(module, "SHUTDOWN_SECS", ENDING_SECS)
    recorder.ending_secs = None
    (call,) = await start_calls(module, recorder, 1)

    deliver(recorder, signal.SIGTERM)
    await until(lambda: bool(recorder.kills()), what="the signal to be re-raised")

    assert recorder.kills() == [signal.SIGTERM]
    assert not recorder.runners[0].finished
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)


@pytest.mark.unit
async def test_a_repeated_signal_does_not_cut_the_ordered_end_short(
    coordinator: tuple[ModuleType, Recorder], caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """`uv run` forwards a Ctrl-C the terminal already delivered: one keypress, two SIGINTs."""
    module, recorder = coordinator
    recorder.ending_secs = 4 * ENDING_SECS
    calls = await start_calls(module, recorder, 1)

    deliver(recorder, signal.SIGINT)
    await until(lambda: recorder.runners[0].cancel_reason is not None, what="the call to be cancelled")
    deliver(recorder, signal.SIGINT)
    deliver(recorder, signal.SIGTERM)
    await until(lambda: bool(recorder.kills()), what="the signal to be re-raised")
    await asyncio.gather(*calls)

    assert recorder.kinds().index("finished") < recorder.kinds().index("kill"), recorder.events
    assert recorder.kills() == [signal.SIGINT]
    # One line for the signal, one for the repeats however many there are.
    assert len([r for r in caplog_loguru.records if r.name == "_shutdown" and r.levelno == logging.INFO]) == 2


@pytest.mark.unit
async def test_every_call_that_entered_before_the_signal_is_waited_for(
    coordinator: tuple[ModuleType, Recorder],
) -> None:
    """However long each takes: the exit waits for the slowest, not the first to end."""
    module, recorder = coordinator
    recorder.before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    calls = []
    for count, secs in enumerate((ENDING_SECS, 8 * ENDING_SECS), start=1):
        recorder.ending_secs = secs
        calls.append(asyncio.create_task(module.run_under_signals(object(), IN_THE_SERVER)))
        await until(lambda n=count: len(recorder.runners) == n, what="the call's runner")
    await until(
        lambda: len(recorder.runners) == 2 and all(r.started for r in recorder.runners), what="both calls"
    )

    deliver(recorder, signal.SIGTERM)
    await until(lambda: bool(recorder.kills()), what="the signal to be re-raised")
    await asyncio.gather(*calls)

    assert [r.ending_secs for r in recorder.runners] == [ENDING_SECS, 8 * ENDING_SECS]
    assert all(r.finished for r in recorder.runners)
    assert max(i for i, k in enumerate(recorder.kinds()) if k == "finished") < recorder.kinds().index("kill")


@pytest.mark.unit
async def test_a_call_entering_in_the_same_turn_as_the_signal_never_starts(
    coordinator: tuple[ModuleType, Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shutdown is decided in the signal handler itself, before its task is even scheduled.

    The entry is created just before the handler runs, so it is checked before the shutdown task's first
    step: only a flag the handler sets synchronously keeps it from starting.
    """
    module, recorder = coordinator
    loop = asyncio.get_running_loop()
    real_add = loop.add_signal_handler
    entries: list[asyncio.Task[None]] = []

    def add_signal_handler(sig: int, callback: Any, *args: Any) -> None:
        def entry_then_handler(*handler_args: Any) -> None:
            if not entries:
                entries.append(asyncio.ensure_future(module.run_under_signals(object(), IN_THE_SERVER)))
            callback(*handler_args)

        real_add(sig, entry_then_handler, *args)

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    calls = await start_calls(module, recorder, 1)

    deliver(recorder, signal.SIGTERM)
    await until(lambda: bool(recorder.kills()), what="the signal to be re-raised")
    await asyncio.gather(*calls, *entries)

    assert len(entries) == 1
    assert not any(r.started for r in recorder.runners[1:]), "a call started after the signal"


@pytest.mark.unit
async def test_a_process_the_signal_does_not_end_exits_itself(
    coordinator: tuple[ModuleType, Recorder],
) -> None:
    """As PID 1 the kernel drops a default-action signal a process sends itself; `os.kill` returns."""
    module, recorder = coordinator
    calls = await start_calls(module, recorder, 1)

    deliver(recorder, signal.SIGTERM)
    await until(lambda: ("exit", 128 + signal.SIGTERM) in recorder.events, what="the fallback exit")
    await asyncio.gather(*calls)

    assert recorder.events[-2:] == [("kill", signal.SIGTERM), ("exit", 128 + signal.SIGTERM)]


@pytest.mark.unit
async def test_where_the_loop_takes_no_signal_handlers_the_runner_owns_the_signals(
    coordinator: tuple[ModuleType, Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows' event loop raises NotImplementedError for `add_signal_handler`: every call, not just
    the first, then runs as before, with its runner owning the signals."""
    module, recorder = coordinator

    def unsupported(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", unsupported)
    for count in (1, 2):
        call = asyncio.create_task(module.run_under_signals(object(), IN_THE_SERVER))
        await until(
            lambda n=count: len(recorder.runners) == n and recorder.runners[-1].started,
            what=f"call {count} to be running",
        )
        runner = recorder.runners[-1]
        assert runner.handle_sigint and runner.handle_sigterm, f"call {count} has no signal handling"
        await runner.cancel()
        await call


@pytest.mark.unit
async def test_a_call_that_has_ended_is_not_ended_again(coordinator: tuple[ModuleType, Recorder]) -> None:
    """A finished call leaves the shutdown's list: a later signal neither cancels nor waits for it."""
    module, recorder = coordinator
    (call,) = await start_calls(module, recorder, 1)
    await recorder.runners[0].cancel(reason="the call ended")
    await call

    deliver(recorder, signal.SIGTERM)
    await until(lambda: bool(recorder.kills()), what="the signal to be re-raised")

    assert recorder.kinds().count("cancel") == 1, recorder.events


@pytest.mark.unit
async def test_pipecats_runner_ends_on_a_cancel_from_a_task_made_just_after_its_run() -> None:
    """The order `_end_calls_then_exit` relies on, with Pipecat's real `WorkerRunner`.

    A call can enter just before a signal: its `run()` task is created, then the shutdown task, in the
    same event-loop turn. `run()` clears the runner's shutdown event in its first step and `cancel()`
    only sets it, and the call's first step runs before the shutdown's, so the cancel is not cleared
    and `run()` returns. If Pipecat cleared the event any later, the cancel would be lost and the call
    would run on to `SHUTDOWN_SECS`.
    """
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(BaseWorker("idle"))
    call = asyncio.create_task(runner.run(), name="call")
    shutdown = asyncio.create_task(runner.cancel(reason="SIGTERM received"), name="shutdown")
    try:
        await asyncio.wait_for(asyncio.shield(call), timeout=5)
    except TimeoutError:
        pytest.fail("the runner's run() did not return after a cancel made just after it started")
    finally:
        call.cancel()
        await asyncio.gather(call, shutdown, return_exceptions=True)


@pytest.mark.unit
def test_the_quickstarts_copy_is_the_examples_copy() -> None:
    """The quickstart is copied whole, so it carries its own copy; the two must not drift."""
    examples, quickstart = (REPO_ROOT / path for path in COPIES)
    assert examples.read_bytes() == quickstart.read_bytes()


@pytest.mark.unit
async def test_a_bot_that_owns_its_process_keeps_its_runners_own_handlers(
    coordinator: tuple[ModuleType, Recorder],
) -> None:
    module, recorder = coordinator
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    call = asyncio.create_task(module.run_under_signals(object(), OWNS_THE_PROCESS))
    await until(lambda: bool(recorder.runners) and recorder.runners[0].started, what="the call to be running")

    (runner,) = recorder.runners
    assert runner.handle_sigint and runner.handle_sigterm
    assert {sig: signal.getsignal(sig) for sig in before} == before
    await runner.cancel()
    await call
    assert recorder.kills() == []
