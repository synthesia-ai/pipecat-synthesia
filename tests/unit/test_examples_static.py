"""The examples, scripts and quickstarts put `vad_analyzer=` in the one place where it works.

At Pipecat 1.10, VAD belongs on the user aggregator. `TransportParams`, `LiveKitParams` and
`DailyParams` have no `vad_analyzer` field and, as pydantic models with the default `extra` behaviour,
they accept the keyword and **silently discard it**. `test_transport_params_really_swallow_vad_analyzer`
below pins that, so this suite fails loudly if a future Pipecat starts raising instead and these guards
become theatre.

The failure a misplaced `vad_analyzer=` produces is a bot that never detects speech and never
interrupts, with nothing in any log to say why. Nothing at runtime catches it, which is why it is
caught here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directories whose Python is developer-facing sample code.
SCANNED_DIRS = ("examples", "scripts", "quickstarts")

#: The only callee allowed to take `vad_analyzer=`.
ALLOWED_CALLEE = "LLMUserAggregatorParams"

KEYWORD = "vad_analyzer"


def _python_files() -> list[Path]:
    found: list[Path] = []
    for name in SCANNED_DIRS:
        directory = REPO_ROOT / name
        if directory.is_dir():
            found.extend(p for p in sorted(directory.rglob("*.py")) if "__pycache__" not in p.parts)
    return found


def _callee_name(node: ast.Call) -> str:
    """`Foo(...)` -> "Foo"; `mod.Foo(...)` -> "Foo"; anything else -> ""."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _calls_with_vad_keyword(tree: ast.AST) -> list[tuple[str, int]]:
    """Every call that passes `vad_analyzer=` explicitly, as (callee name, line number)."""
    return [
        (_callee_name(node), node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == KEYWORD
    ]


@pytest.mark.unit
def test_scan_covers_something() -> None:
    """A sabotage check: if the glob silently matched nothing, every test below would pass vacuously."""
    files = _python_files()
    assert files, f"no Python found under {SCANNED_DIRS} -- the other assertions would be vacuous"
    assert (REPO_ROOT / "examples").is_dir(), "examples/ is the directory these guards exist for"


@pytest.mark.unit
def test_vad_analyzer_never_on_a_params_object() -> None:
    """No `*Params(vad_analyzer=...)` other than `LLMUserAggregatorParams`."""
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders.extend(
            f"{path.relative_to(REPO_ROOT)}:{line}: {callee}({KEYWORD}=...)"
            for callee, line in _calls_with_vad_keyword(tree)
            if callee.endswith("Params") and callee != ALLOWED_CALLEE
        )
    assert not offenders, (
        f"{KEYWORD}= on a transport params object is silently dropped by pydantic, so the bot never "
        f"detects speech and never interrupts. Move it into {ALLOWED_CALLEE}(...):\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_vad_analyzer_is_actually_configured_somewhere() -> None:
    """The positive half: at least one `LLMUserAggregatorParams(vad_analyzer=...)` exists.

    Without this, deleting every `vad_analyzer=` in the repo would pass the negative test above
    while leaving examples that cannot be interrupted.
    """
    good = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in _python_files()
        for callee, line in _calls_with_vad_keyword(ast.parse(path.read_text(), filename=str(path)))
        if callee == ALLOWED_CALLEE
    ]
    assert good, (
        f"no {ALLOWED_CALLEE}({KEYWORD}=...) anywhere in {SCANNED_DIRS}: the examples would have no "
        "voice activity detection, so the user could never interrupt the avatar"
    )


@pytest.mark.unit
def test_docs_mention_vad_analyzer_only_in_the_right_call() -> None:
    """Prose is copy-pasted, so the same rule applies to README.md and docs/."""
    targets = [REPO_ROOT / "README.md"]
    docs = REPO_ROOT / "docs"
    if docs.is_dir():
        # .mdx as well as .md: a page written for *Pipecat's* own documentation site, which is the
        # worst possible place to publish the broken form, is .mdx.
        # A `plan` directory is skipped: planning notes are not developer-facing guidance.
        targets.extend(
            p
            for pattern in ("*.md", "*.mdx")
            for p in sorted(docs.rglob(pattern))
            if "plan" not in p.relative_to(docs).parts
        )
    quickstarts = REPO_ROOT / "quickstarts"
    if quickstarts.is_dir():
        targets.extend(p for pattern in ("*.md", "*.mdx") for p in sorted(quickstarts.rglob(pattern)))

    offenders: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if f"{KEYWORD}=" in line and f"{ALLOWED_CALLEE}({KEYWORD}=" not in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        f"docs show {KEYWORD}= outside {ALLOWED_CALLEE}(...), which readers will copy:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_transport_params_really_swallow_vad_analyzer() -> None:
    """The premise of every guard above, pinned against the installed Pipecat.

    If a future Pipecat adds a real `vad_analyzer` field, or starts rejecting unknown keywords,
    this fails and the guards should be revisited rather than left as cargo cult.
    """
    from pipecat.transports.base_transport import TransportParams

    assert "vad_analyzer" not in TransportParams.model_fields

    params = TransportParams(audio_in_enabled=True, vad_analyzer="not-an-analyzer")
    assert not hasattr(params, "vad_analyzer"), (
        "TransportParams now keeps vad_analyzer -- Pipecat's API changed and these guards need review"
    )
