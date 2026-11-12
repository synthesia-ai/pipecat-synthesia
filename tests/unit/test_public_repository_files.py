"""The files a public repository of Synthesia's carries: a licence the package ships, one place for the
security contact, and code owners on the release path.

Synthesia's security review set these rules: `SECURITY.md` is the only file that names the security address,
so a reader is never given two routes; `LICENSE` is in the wheel's metadata (through `license-files`), so
the licence travels with the package; and `@synthesia-ai/security` owns the files a release depends on.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SECURITY_ADDRESS = re.compile(r"security(-reports)?@synthesia\.io", re.I)


def _tracked_text_files() -> list[Path]:
    names = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [REPO_ROOT / name for name in names if name != "uv.lock" and (REPO_ROOT / name).is_file()]


@pytest.mark.unit
def test_only_security_md_names_the_security_address() -> None:
    assert _SECURITY_ADDRESS.search((REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8"))
    elsewhere = []
    for path in _tracked_text_files():
        if path.name == "SECURITY.md" or "test_public_repository_files" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _SECURITY_ADDRESS.search(text):
            elsewhere.append(path.relative_to(REPO_ROOT).as_posix())
    assert elsewhere == []


@pytest.mark.unit
def test_the_licence_is_declared_and_shipped() -> None:
    assert (REPO_ROOT / "LICENSE").is_file()
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]


@pytest.mark.unit
def test_the_security_team_owns_the_release_path() -> None:
    owners = {}
    for line in (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            pattern, *teams = line.split()
            owners[pattern] = teams
    assert owners["*"] == ["@synthesia-ai/interactive-avatars"]
    release_path = ("/.github/", "/pyproject.toml", "/uv.lock", "/RELEASING.md", "/SECURITY.md")
    secret_scanning = ("/.gitleaks.toml", "/.pre-commit-config.yaml")
    for path in release_path + secret_scanning:
        assert "@synthesia-ai/security" in owners.get(path, []), path
