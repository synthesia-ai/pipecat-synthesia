"""The README's links to files in the repository are absolute, because the README is also the PyPI page.

`readme = "README.md"` makes the README the project description on PyPI, where a relative link such as
`docs/deployment.md` resolves against pypi.org and leads nowhere. Links to repository files therefore name
the public repository's `main` branch in full.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: An inline link's target (`[text](target "title")`, `<target>` too) or a reference definition's.
_LINK = re.compile(r"\]\(<?([^)\s>]+)|^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.M)
_REPO = "https://github.com/synthesia-ai/pipecat-synthesia/"


def _links() -> list[str]:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return [inline or reference for inline, reference in _LINK.findall(text)]


@pytest.mark.unit
def test_every_readme_link_is_absolute_or_an_anchor() -> None:
    links = _links()
    assert links, "no links found; the pattern no longer matches the README"
    relative = [link for link in links if not re.match(r"https?://|mailto:|#", link)]
    assert relative == []


@pytest.mark.unit
def test_every_link_into_the_repository_names_a_file_that_exists() -> None:
    links = _links()
    missing = []
    for link in links:
        if link.startswith(_REPO):
            path = re.sub(r"^(blob|tree)/main/", "", link.removeprefix(_REPO)).split("#")[0]
            if not (REPO_ROOT / path).exists():
                missing.append(link)
    assert missing == []
