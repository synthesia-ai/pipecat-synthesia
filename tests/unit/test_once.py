"""`_once`, and the autouse reset that makes "logged once" testable.

Other tests assert "this line is logged exactly once", and that verdict must not depend on which test
ran first. `test_once_a` and `test_once_b` are deliberately identical and both expect the *first* call
to log, which holds only if the autouse `reset_once_flags` fixture in tests/conftest.py clears the
state between them, in either order.
"""

from __future__ import annotations

import logging

import pytest

from pipecat_synthesia import _once
from pipecat_synthesia._log import get_logger

KEY = "test:once:key"
MESSAGE = "the once-guarded warning"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelname == "WARNING"]


@pytest.mark.unit
def test_once_is_true_then_false() -> None:
    """`once(key)` is True the first time and False for every later call; keys are independent."""
    assert _once.once(KEY) is True
    assert _once.once(KEY) is False
    assert _once.once(KEY) is False
    assert _once.once("test:once:other") is True


@pytest.mark.unit
def test_warn_once_logs_one_warning_and_reset_makes_it_log_again(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """Two `warn_once` calls log one WARNING; after `reset_for_tests()` the next call logs again."""
    log = get_logger()

    assert _once.warn_once(log, KEY, MESSAGE) is True
    assert _once.warn_once(log, KEY, MESSAGE) is False
    logged = [record for record in _warnings(caplog_loguru) if MESSAGE in record.message]
    assert len(logged) == 1, f"expected one WARNING, got {[r.message for r in _warnings(caplog_loguru)]}"

    _once.reset_for_tests()

    assert _once.warn_once(log, KEY, MESSAGE) is True
    logged = [record for record in _warnings(caplog_loguru) if MESSAGE in record.message]
    assert len(logged) == 2


@pytest.mark.unit
def test_debug_once_logs_at_debug(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """`debug_once` is the same guard at DEBUG, and shares the keyspace with `warn_once`."""
    log = get_logger()

    assert _once.debug_once(log, KEY, MESSAGE) is True
    assert _once.warn_once(log, KEY, MESSAGE) is False, "warn_once must see debug_once's key"

    levels = {record.levelname for record in caplog_loguru.records if MESSAGE in record.message}
    assert levels == {"DEBUG"}, levels


# --- the pair that proves the autouse reset works in both orders ---------------------------------

# Same key in both, so if the fixture did not reset, whichever ran second would see False. pytest
# runs them in file order by default; `-p no:randomly` is not needed because the assertion holds
# either way, which is the point.
SHARED_KEY = "test:once:shared-between-two-tests"


@pytest.mark.unit
def test_once_a() -> None:
    """The first `once(SHARED_KEY)` in this test is True, whatever ran before it."""
    assert _once.once(SHARED_KEY) is True


@pytest.mark.unit
def test_once_b() -> None:
    """Identical to `test_once_a`: both pass only because `reset_once_flags` runs between them."""
    assert _once.once(SHARED_KEY) is True
