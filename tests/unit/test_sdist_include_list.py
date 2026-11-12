"""The source distribution ships the package and its user docs, not the repository.

Without an sdist section in `pyproject.toml`, hatch puts every tracked file in the sdist: the tests (with a
fake avatar worker), the maintenance scripts, `tools/` and `.github/`, which are for maintaining the
repository. The wheel is unaffected either way. This reads `pyproject.toml` rather than building an
sdist, so it stays a unit test; building one and listing it is the release check's job.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: Top-level paths that stay in the repository and never ship in an sdist.
NEVER_SHIPPED = frozenset({"tests", "scripts", "tools", ".github", "docs", "examples", "quickstarts"})


def _sdist_config() -> dict:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return config.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("sdist", {})


def _top_level(pattern: str) -> str:
    """`/tests`, `tests/**` and `tests` all name `tests`."""
    return pattern.strip().lstrip("/").split("/")[0]


@pytest.mark.unit
def test_the_sdist_has_an_explicit_include_list_with_the_package() -> None:
    include = _sdist_config().get("include")
    assert include, (
        "pyproject.toml needs [tool.hatch.build.targets.sdist] include = [...]; "
        "without it the sdist ships every tracked file"
    )
    assert "src" in {_top_level(entry) for entry in include}, include


@pytest.mark.unit
def test_the_sdist_includes_no_tests_scripts_or_repository_tooling() -> None:
    config = _sdist_config()
    # force-include maps a source path to a path in the archive; either side can name a top-level dir.
    forced = config.get("force-include", {})
    entries = [*config.get("include", []), *forced.keys(), *forced.values()]
    shipped = sorted(entry for entry in entries if _top_level(entry) in NEVER_SHIPPED)
    assert not shipped, f"the sdist must not ship {shipped}"
