"""Package-wide contracts that need the complete package: the public exports, the livekit import guard,
every module importing on its own, and the LiveKit time bounds. The contracts on the core helpers are in
`test_core_helpers.py`."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "pipecat_synthesia"


def _package_files() -> list[Path]:
    """Every .py file shipped under src/pipecat_synthesia, sorted, build artefacts excluded."""
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path) -> str:
    """livekit/_rtc.py -> pipecat_synthesia.livekit._rtc; __init__.py -> the package it belongs to."""
    parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# --- public exports ------------------------------------------------------------------------------

# LiveKit classes are deliberately absent, so the core package imports without the [livekit] extra
# (*LiveKit-free core*).
PUBLIC_EXPORTS = (
    "SynthesiaVideoService",
    "SynthesiaVideoSettings",
    "SynthesiaVideoUpdateSettingsFrame",
    "ServiceState",
    "SynthesiaError",
    "ErrorType",
    "AvatarConnected",
    "AvatarReady",
    "AvatarRoom",
    "AvatarSession",
    "JoinMode",
    "SessionMetrics",
    "__version__",
)


@pytest.mark.unit
def test_public_exports() -> None:
    """`pipecat_synthesia.__all__` is exactly `PUBLIC_EXPORTS`: same members, no duplicates, all defined."""
    import pipecat_synthesia

    exported = list(pipecat_synthesia.__all__)
    assert set(exported) == set(PUBLIC_EXPORTS)
    assert len(exported) == len(PUBLIC_EXPORTS), f"__all__ has duplicates: {exported}"
    missing = [name for name in PUBLIC_EXPORTS if not hasattr(pipecat_synthesia, name)]
    assert not missing, f"names in __all__ that the package does not actually define: {missing}"


# --- the livekit import guard --------------------------------------------------------------------


@pytest.mark.unit
def test_livekit_import_guard_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the livekit subpackage without `livekit` installed raises an ImportError naming the extra."""
    prefix = "pipecat_synthesia.livekit"
    evicted = {
        name: mod for name, mod in sys.modules.items() if name == prefix or name.startswith(prefix + ".")
    }
    for name in evicted:
        del sys.modules[name]
    # `None` in sys.modules makes `import livekit` raise ModuleNotFoundError, which is what a missing
    # extra looks like; monkeypatch restores the entry (or its absence) on teardown.
    monkeypatch.setitem(sys.modules, "livekit", None)
    try:
        importlib.invalidate_caches()
        with pytest.raises(ImportError) as excinfo:
            importlib.import_module(prefix)
        message = str(excinfo.value)
        # The command that installs the extra from PyPI, as the README gives it.
        assert 'pip install "pipecat-synthesia[livekit]"' in message, message
    finally:
        # A failed import can leave partially-initialised entries behind: drop everything under the
        # subpackage, then put the modules the session already had back exactly as they were.
        for name in [n for n in list(sys.modules) if n == prefix or n.startswith(prefix + ".")]:
            del sys.modules[name]
        sys.modules.update(evicted)


# --- modules import alone ------------------------------------------------------------------------


@pytest.mark.unit
# A hang backstop, not a bound: one fresh interpreter per module, run one after another, took about
# 16 s of the 30 s default on a loaded machine, and timed out under a heavier load.
@pytest.mark.timeout(180)
def test_each_module_imports_alone() -> None:
    """Every module imports on its own in a fresh interpreter (catches order-dependent cycles)."""
    modules = [_module_name(path) for path in _package_files()]
    assert modules, f"no modules discovered under {PACKAGE_ROOT}"
    for module in modules:
        try:
            subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            pytest.fail(f"`import {module}` fails in a fresh interpreter:\n{exc.stderr}")


# --- time bounds ---------------------------------------------------------------------------------

# livekit/_types.py is the single home for these bounds (*bounded waits*). Enumerated rather than
# discovered, so the test documents the list and a constant moved elsewhere fails here.
TIME_BOUND_CONSTANTS = (
    "TASK_CANCEL_SECS",
    "STREAM_CLOSE_SECS",
    "CLEAR_BUFFER_SECS",
    "ROOM_CONNECT_SECS",
    "ROOM_CREATE_SECS",
    "ROOM_DISCONNECT_SECS",
    "ROOM_DELETE_SECS",
    "CLOSE_HOOK_SECS",
    "POST_CREATE_JOIN_WAIT_SECS",
    "REPLACEMENT_GRACE_SECS",
)


@pytest.mark.unit
def test_time_bound_constants_positive() -> None:
    """Every teardown/connect time bound lives in livekit/_types.py and is a positive number."""
    from pipecat_synthesia.livekit import _types

    for name in TIME_BOUND_CONSTANTS:
        value = getattr(_types, name)
        assert isinstance(value, int | float) and not isinstance(value, bool), f"{name} is {value!r}"
        assert value > 0, f"{name} must be positive, got {value!r}"
