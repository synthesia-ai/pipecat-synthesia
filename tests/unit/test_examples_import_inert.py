"""Importing an example loads no credentials.

python-dotenv searches upward from the importing file's directory, so a module-scope `load_dotenv()` in
an example loads the credentials file at the root of whichever checkout imports it, with
`override=True`, into the importing process. In a test process that replaces whatever the test had set,
and a test running an example against the local fakes could then reach the real LiveKit project and the
real Synthesia API. Running a script loads the file from its entry point instead.

The spy is what makes this fail in any checkout: with no credentials file anywhere above `examples/`
(a worktree, CI), a module-scope call would leave `os.environ` unchanged and only the spy would see it.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dotenv
import dotenv.main
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

#: Every Python file in `examples/`, so a new example is covered without anyone remembering to add it.
EXAMPLE_MODULES = sorted(path.stem for path in EXAMPLES.glob("*.py") if not path.name.startswith("__"))


@pytest.fixture
def fresh_examples() -> Iterator[None]:
    """`examples/` on sys.path as a script run puts it, with no example already imported.

    An example imported earlier in the session would be returned from `sys.modules` without running its
    module scope again, which is the code under test.
    """
    for name in EXAMPLE_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(EXAMPLES))
    try:
        yield
    finally:
        if str(EXAMPLES) in sys.path:
            sys.path.remove(str(EXAMPLES))
        # Imported here with the spy bound in place of `load_dotenv`; never leave that for a later test.
        for name in EXAMPLE_MODULES:
            sys.modules.pop(name, None)


@pytest.mark.unit
def test_the_scan_covers_the_examples_that_loaded_dotenv() -> None:
    """A sabotage check: if the glob matched nothing, the test below would pass vacuously."""
    assert {"bot_direct_join", "mint_tokens"} <= set(EXAMPLE_MODULES), EXAMPLE_MODULES


@pytest.mark.unit
@pytest.mark.parametrize("name", EXAMPLE_MODULES)
def test_importing_an_example_loads_no_credentials(
    name: str, fresh_examples: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy(*args: Any, **kwargs: Any) -> bool:
        calls.append((args, kwargs))
        return False

    # Both names: the examples bind `from dotenv import load_dotenv` at import time, and `dotenv`
    # re-exports `dotenv.main.load_dotenv`.
    monkeypatch.setattr(dotenv, "load_dotenv", spy)
    monkeypatch.setattr(dotenv.main, "load_dotenv", spy)
    before = dict(os.environ)

    importlib.import_module(name)

    assert calls == [], f"importing {name} called load_dotenv{calls[0]}; call it from the entry point"
    assert dict(os.environ) == before, f"importing {name} changed the environment"
