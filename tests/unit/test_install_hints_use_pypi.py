"""Every install hint installs the plugin from PyPI, with its `[livekit]` extra.

The plugin is on PyPI, so a hint that installs it from a URL or a wheel file is left over from the private
beta and points at a file the reader cannot get. This scans the text a developer copies from and fails on a
requirement that names a URL (`pipecat-synthesia @ <URL>`), on a wheel file of the package, and on `uv add
--raw-sources`, a package index or find-links option, or a 0.x pin for it, and on a `[tool.uv.sources]`
entry for it. It also fails on a requirement that leaves out `[livekit]`, which both join modes need: in an
install command, in a quoted string in TOML or a code block, as a requirements line in a code block, or as
inline code with a version. The name is matched case-insensitively with `-`, `_` and `.` folded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_NAME = r"(?<![\w.-])pipecat[-_.]synthesia(?![-_.]?[a-z])"
_ANY = re.compile(_NAME, re.I)
#: A requirement on a URL: the name, optional extras, then `@`.
_DIRECT = re.compile(_NAME + r"\s*(\[[^\]]*\])?\s*@", re.I)
#: A wheel file of the package, such as `pipecat_synthesia-1.0.0-py3-none-any.whl`.
_WHEEL = re.compile(r"pipecat[-_.]synthesia[^\s\"'`]*\.whl", re.I)
_WITH_EXTRA = re.compile(_NAME + r"\[[^\]]*\blivekit\b[^\]]*\]", re.I)
#: The name as a requirement at the start of a string: then the end, extras, a version, a marker or `@`.
_REQUIREMENT = re.compile(_NAME + r"(?=$|[\s\[~=<>!;@,])", re.I)
#: Inline code that starts with the name pinned: extras or a version right after it.
_PINNED = re.compile(_NAME + r"\s*[\[~=<>!]", re.I)
#: A pin to a version below 1.0: only the private beta's 0.1.0 pre-releases had one, and none is on PyPI.
_BELOW_1 = re.compile(_NAME + r"\s*(\[[^\]]*\])?\s*(==|~=|<=|<|>=|>)\s*0\.", re.I)
#: A `[tool.uv.sources]` entry, which sends uv to a URL, a path, git or another index for the package.
_UV_SOURCE = re.compile(r"^\s*" + _NAME + r"\s*=\s*\{", re.I)
#: Install options that take the package from somewhere other than PyPI.
_ELSEWHERE = re.compile(r"(^|\s)(-f|-i|--find-links|--index-url|--extra-index-url|--index|--default-index)\b")
_COMMAND = re.compile(r"\b(uv add|pip3? install)\b([^`\n]*)")
_QUOTED = re.compile(r"\"([^\"]*)\"|'([^']*)'")
_INLINE = re.compile(r"`([^`]+)`")


def _scanned_files() -> list[Path]:
    files = [REPO_ROOT / name for name in ("README.md", "SUPPORT.md", "SECURITY.md", "RELEASING.md")]
    files += sorted((REPO_ROOT / "docs").rglob("*.md"))
    for pattern in ("*.md", "*.toml", "*.py"):
        files += sorted((REPO_ROOT / "quickstarts").rglob(pattern))
    for pattern in ("*.md", "*.py"):
        files += sorted((REPO_ROOT / "examples").rglob(pattern))
    return files


def _flagged(text: str, *, relative: str, suffix: str) -> list[str]:
    """The lines of ``text`` that install the package from anywhere but PyPI, or without ``[livekit]``."""
    found, in_block = [], False
    for number, line in enumerate(text.splitlines(), start=1):
        if suffix == ".md" and line.lstrip().startswith(("```", "~~~")):
            in_block = not in_block
            continue
        bad = any(pattern.search(line) for pattern in (_DIRECT, _WHEEL, _BELOW_1, _UV_SOURCE))
        for _, args in _COMMAND.findall(line):
            if _ANY.search(args):
                bad |= "--raw-sources" in args or _ELSEWHERE.search(args) is not None
                bad |= _WITH_EXTRA.search(args) is None
        code = [a or b for a, b in _QUOTED.findall(line)] if suffix == ".toml" or in_block else []
        for text_ in code + ([line.strip(" -")] if in_block else []):
            text_ = text_.strip()
            bad |= _REQUIREMENT.match(text_) is not None and _WITH_EXTRA.match(text_) is None
        for span in _INLINE.findall(line):
            span = span.strip()
            bad |= _PINNED.match(span) is not None and _WITH_EXTRA.match(span) is None
        if bad:
            found.append(f"{relative}:{number}: {line.strip()}")
    return found


@pytest.mark.unit
def test_the_scan_covers_what_a_developer_copies_from() -> None:
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _scanned_files()}
    expected = {
        "README.md",
        "SUPPORT.md",
        "RELEASING.md",
        "docs/deployment.md",
        "examples/bot_relayed_join.py",
    }
    expected |= {f"quickstarts/pipecat/{name}" for name in ("README.md", "pyproject.toml", "bot.py")}
    assert expected <= scanned, sorted(scanned)


@pytest.mark.unit
def test_every_install_hint_uses_pypi_with_the_livekit_extra() -> None:
    flagged = []
    for path in _scanned_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        flagged += _flagged(path.read_text(encoding="utf-8"), relative=relative, suffix=path.suffix)
    assert flagged == []


_MUST_FLAG = [
    (".md", 'uv add --raw-sources "pipecat-synthesia[livekit] @ https://x/p.whl#sha256=ab12"'),
    (".md", 'pip install "pipecat-synthesia[livekit] @ file:///<path>.whl"'),
    (".md", 'uv add --raw-sources "pipecat-synthesia[livekit]"'),
    (".md", "pip install pipecat_synthesia"),
    (".md", "pip install Pipecat-Synthesia"),
    (".md", "pip install pipecat-synthesia[silero]"),
    (".toml", 'dependencies = ["pipecat-synthesia @ https://x/y.whl"]'),
    (".toml", 'dependencies = ["pipecat-synthesia>=1.0,<2"]'),
    (".md", "```\npipecat-synthesia>=1.0\n```"),
    (".md", "Download pipecat_synthesia-1.0.0-py3-none-any.whl first."),
    (".md", "pin it with `pipecat-synthesia~=1.0`"),
    (".toml", 'pipecat-synthesia = { url = "https://x/p.whl" }'),
    (".toml", 'pipecat-synthesia = { index = "synthesia" }'),
    (".md", 'pip install -f https://x/simple/ "pipecat-synthesia[livekit]"'),
    (".md", 'pip install --index-url https://x/simple "pipecat-synthesia[livekit]"'),
    (".md", 'uv add --index https://x/simple "pipecat-synthesia[livekit]"'),
    (".md", "pip install 'pipecat-synthesia[livekit]==0.1.0b4'"),
    (".toml", 'dependencies = ["pipecat-synthesia[livekit]~=0.1.0b1"]'),
    (".md", "~~~\npipecat-synthesia>=1.0\n~~~"),
]
_MUST_PASS = [
    (".md", 'uv add "pipecat-synthesia[livekit]"'),
    (".md", 'pip install "pipecat-synthesia[livekit]>=1.0,<2"'),
    (".toml", 'dependencies = ["pipecat-synthesia[livekit]>=1.0,<2"]'),
    (".md", "```\npipecat-synthesia[livekit]>=1.0,<2\n```"),
    (".md", "pin it with `pipecat-synthesia[livekit]>=1.0,<2`"),
    (".md", "`pipecat-synthesia` is on PyPI"),
    (".md", '```\npython -c "import pipecat_synthesia; print(pipecat_synthesia.__version__)"\n```'),
    (".md", 'pip install "pipecat-ai[runner]"'),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("suffix", "text", "unsafe"), [(*c, True) for c in _MUST_FLAG] + [(*c, False) for c in _MUST_PASS]
)
def test_what_the_scan_flags(suffix: str, text: str, unsafe: bool) -> None:
    assert bool(_flagged(text, relative="case", suffix=suffix)) is unsafe
