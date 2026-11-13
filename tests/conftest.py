"""Fixtures shared by every suite."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator

import pytest
from loguru import logger

from pipecat_synthesia import _once
from tests.support.waiting import assert_no_dangling

# `assert_no_dangling` is re-exported so the leak detector is reachable from here as well as from
# `tests.support.waiting`; a test with no harness can use the `no_dangling_tasks` fixture below.
__all__ = [
    "assert_no_dangling",
    "caplog_loguru",
    "no_dangling_tasks",
    "reset_once_flags",
    "untested_pipecat_warning_off",
]


class PropagateHandler(logging.Handler):
    """Re-emit a loguru record into the stdlib logging tree so pytest's caplog handler receives it.

    ``logging.Logger.handle()`` deliberately skips the logger's own level check and goes straight to
    ``callHandlers``, which filters on *handler* level only. pytest's capture handler sits on the root
    logger at level 0, so every loguru record lands in ``caplog`` no matter how the stdlib loggers are
    configured. This is the handler from loguru's own "propagate to standard logging" recipe.
    """

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def caplog_loguru(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture this package's loguru output in pytest's ``caplog`` for the duration of one test.

    Every "no secrets in logs" test depends on this fixture, so it is deliberately total and
    deliberately self-cleaning:

    * ``level=0`` — per-frame logs go through ``logger.trace()`` (``_log.py``) and loguru's TRACE
      level (5) is below ``logger.add()``'s ``DEBUG`` default; without this a secret leaked at TRACE
      would be invisible to the assertion that is supposed to catch it.
    * ``format="{message}"`` — the captured text is exactly what the code logged, with none of
      loguru's timestamp/level/location decoration to match against by accident.
    * the sink is removed on teardown, so no state leaks into the next test and loguru's default
      stderr sink is left untouched.

    Yields the standard ``caplog`` fixture, so ``.text``, ``.records``, ``.messages`` and
    ``.set_level()`` all behave normally. Note that loguru levels stdlib does not know keep their
    loguru numbers on the captured record (TRACE=5, SUCCESS=25).

    Usage::

        def test_token_is_never_logged(caplog_loguru):
            ...
            assert "eyJhbGciOi" not in caplog_loguru.text
    """
    handler_id = logger.add(PropagateHandler(), format="{message}", level=0)
    try:
        yield caplog
    finally:
        try:
            logger.remove(handler_id)
        except ValueError:
            # The test removed every loguru sink itself; ours is already gone.
            pass


@pytest.fixture(autouse=True)
def reset_once_flags() -> Iterator[None]:
    """Clear `_once`'s per-process state around every test.

    "Warn once per process" state is, by construction, shared between tests in the same interpreter:
    without this fixture `test_advisories_logged_once` passes or fails depending on whether some
    earlier test already tripped the same key, and the failure looks like a product bug. Autouse
    because opting in is exactly the thing test authors forget.

    Cleared before *and* after each test: before, so the test under way sees a clean process however
    it was reached; after, so a test that deliberately trips a key does not hand that key to whatever
    runs next, including a test in another file.
    """
    _once.reset_for_tests()
    yield
    _once.reset_for_tests()


@pytest.fixture(autouse=True)
def untested_pipecat_warning_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never log the "newer than any tested Pipecat" WARNING, except in its own test file.

    The warning is exercised only in tests/unit/test_untested_pipecat_warning.py, which turns it back on and
    sets the installed version itself. Everywhere else, whether it fires would depend on the installed
    pipecat-ai, not on the test: the nightly CI cell that installs Pipecat's newest release runs every tier
    on a version that may be newer than any tested, and there every service construction would add that
    WARNING, failing each test that counts warnings (25 in test_service_config.py, for one) for a reason that
    has nothing to do with what it checks.
    """
    monkeypatch.setattr("pipecat_synthesia.service.untested_newer_pipecat", lambda: None)


@pytest.fixture
def no_dangling_tasks() -> Callable[[object], None]:
    """The leaked-task detector, for a test that owns a `TaskManager` directly.

    Returns `tests.support.waiting.assert_no_dangling`: call it with the task manager once everything
    is torn down. The harness calls the same function from its own context manager on exit, so both
    paths report a leak identically.
    """
    return assert_no_dangling


# The LiveKit SDK frees its native handles through garbage collection while the interpreter shuts
# down (livekit/python-sdks#524). On macOS that prints `FfiHandle.__del__` noise; on Linux the SDK's
# Rust runtime can panic there and abort, so a run whose tests all passed exits 134 (seen in CI).
# Once pytest has its result, exit with it and skip that shutdown.
_session: pytest.Session | None = None


def pytest_sessionfinish(session: pytest.Session) -> None:
    global _session
    _session = session


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    # Only when the SDK's native library was loaded, and never under coverage.py, which saves its
    # data at interpreter exit.
    if _session is None or "livekit.rtc._ffi_client" not in sys.modules or "coverage" in sys.modules:
        return
    config.get_terminal_writer().flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(_session.exitstatus))
