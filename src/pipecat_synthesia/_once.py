"""Once-per-process log lines, with a reset for tests.

This is the only home for "already logged this" state, so the autouse ``reset_once_flags`` fixture in
``tests/conftest.py`` can clear it between tests. A module-level flag anywhere else would make "logged
exactly once" assertions depend on test order.

Keys follow ``f"{module}:{what}:{value}"`` (``"api:advisory:Deprecation:true"``, ``"token:ws-insecure"``,
``"service:prelaunch-full"``), so call sites in different modules never collide, and a key that varies by
value (a header name, an avatar id) logs once per distinct value. A flag that belongs to one session,
such as the streamer's format-change line, stays on that object instead.

This module imports nothing else from the package, so every module may import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import loguru

__all__ = ["debug_once", "once", "reset_for_tests", "warn_once"]

_seen: set[str] = set()


def once(key: str) -> bool:
    """Return ``True`` the first time ``key`` is seen in this process, ``False`` afterwards.

    Use directly only when the once-guarded work is not a log line (a metric, a lazy probe); prefer
    :func:`warn_once` / :func:`debug_once` so the key and the message stay in one place.
    """
    if key in _seen:
        return False
    _seen.add(key)
    return True


def warn_once(log: loguru.Logger, key: str, message: str, **kwargs: Any) -> bool:
    """Log ``message`` at WARNING the first time ``key`` is seen; return whether it logged.

    The return value is for callers that pair the line with other first-time-only work (bumping a
    counter, stamping a metric); ignoring it is normal.
    """
    if not once(key):
        return False
    log.warning(message, **kwargs)
    return True


def debug_once(log: loguru.Logger, key: str, message: str, **kwargs: Any) -> bool:
    """Log ``message`` at DEBUG the first time ``key`` is seen; return whether it logged."""
    if not once(key):
        return False
    log.debug(message, **kwargs)
    return True


def reset_for_tests() -> None:
    """Clear all state.

    Called by the autouse ``reset_once_flags`` fixture in ``tests/conftest.py``, before and after every
    test so a case is unaffected by whichever case ran first. Never called from ``src/``.
    """
    _seen.clear()
