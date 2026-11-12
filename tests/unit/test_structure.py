"""Static checks on source and CI config. They read files, not behaviour, so they never flake; the one that
runs a workflow's shell script runs it against a scratch git repository."""

from __future__ import annotations

import ast
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from pipecat_synthesia._compat import TESTED_PIPECAT_VERSIONS

SERVICE_PATH = Path(__file__).resolve().parents[2] / "src" / "pipecat_synthesia" / "service.py"
CI_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
REPO_ROOT = Path(__file__).resolve().parents[2]


class _CallSites(ast.NodeVisitor):
    """Collect `<expr>.push_error(...)` calls with the name of the function each one sits in."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.hits: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node)

    def _enter(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "push_error":
            self.hits.append((self.stack[-1] if self.stack else "<module>", node.lineno))
        self.generic_visit(node)


@pytest.mark.unit
def test_push_error_single_call_site() -> None:
    """`push_error` is called exactly once in service.py, inside `_push_synthesia_error`
    (*permanent-error rule*)."""
    tree = ast.parse(SERVICE_PATH.read_text(encoding="utf-8"), filename=str(SERVICE_PATH))
    visitor = _CallSites()
    visitor.visit(tree)
    assert visitor.hits == [("_push_synthesia_error", visitor.hits[0][1])] if visitor.hits else False, (
        f"push_error call sites in service.py: {visitor.hits}"
    )


@pytest.mark.unit
def test_service_has_no_module_level_once_state() -> None:
    """Every "warn once" line goes through `_once`: service.py declares no module-level set or dict that
    could hold already-warned state, which the per-test reset of `_once` would not clear."""
    tree = ast.parse(SERVICE_PATH.read_text(encoding="utf-8"), filename=str(SERVICE_PATH))
    offenders = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and isinstance(getattr(node, "value", None), ast.Call | ast.Set | ast.Dict | ast.SetComp)
        and not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
        )
    ]
    assert offenders == [], f"module-level mutable state in service.py at lines {offenders}"


@pytest.mark.unit
@pytest.mark.parametrize("job", ["unit", "integration"])
def test_ci_pipecat_matrix_matches_tested_versions(job: str) -> None:
    """The CI cells above the lock equal the tested versions `_compat.py` declares, so neither can drift,
    and each cell installs its version and runs pytest without re-syncing to the lock."""
    workflow = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    cells = [str(cell) for cell in workflow["jobs"][job]["strategy"]["matrix"]["pipecat"]]
    assert cells == ["lock", *TESTED_PIPECAT_VERSIONS[1:]], f"{job}: {cells} vs {TESTED_PIPECAT_VERSIONS}"
    steps = workflow["jobs"][job]["steps"]
    install = next((s for s in steps if s.get("if") == "matrix.pipecat != 'lock'"), {})
    version = (install.get("env") or {}).get("PIPECAT_VERSION")
    assert version == "${{ matrix.pipecat }}", f"{job}: install is not given the cell's version: {version!r}"
    command = install.get("run", "")
    assert "==$PIPECAT_VERSION" in command, f"{job}: install ignores the cell's version: {command!r}"
    tests = next((s["run"] for s in steps if "pytest" in s.get("run", "")), "")
    assert tests.startswith("uv run --no-sync pytest"), f"{job}: pytest re-syncs to the lock: {tests!r}"


@pytest.mark.unit
def test_the_lock_holds_the_oldest_tested_pipecat() -> None:
    """The `lock` CI cells test uv.lock, so uv.lock's pipecat-ai must be the floor:
    TESTED_PIPECAT_VERSIONS[0], the oldest version pyproject.toml allows. An upgrade of the lock in passing
    would move the floor cell."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package.get("version") for package in lock["package"]}
    assert locked["pipecat-ai"] == TESTED_PIPECAT_VERSIONS[0], locked["pipecat-ai"]
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = TESTED_PIPECAT_VERSIONS[0].rsplit(".", 1)[0]
    project = pyproject["project"]
    requirements = [
        *project["dependencies"],
        *(r for extra in project["optional-dependencies"].values() for r in extra),
        *(r for group in pyproject["dependency-groups"].values() for r in group),
    ]
    pipecat = [r for r in requirements if r.startswith("pipecat-ai")]
    assert len(pipecat) >= 4, pipecat
    wrong = [r for r in pipecat if not r.endswith(f">={floor},<2")]
    assert wrong == [], f"pipecat-ai requirements not at the floor {floor}: {wrong}"


@pytest.mark.unit
def test_lint_checks_the_lock_matches_pyproject() -> None:
    """Every CI job syncs from uv.lock without `--locked`, so `lint` runs `uv lock --check` first."""
    steps = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))["jobs"]["lint"]["steps"]
    runs = [step.get("run") for step in steps]
    assert "uv lock --check" in runs
    assert runs.index("uv lock --check") < runs.index("uv sync --extra livekit --group dev")


#: Every job runs on GitHub-hosted ubuntu-24.04. See the comment at the top of ci.yml.
RUNS_ON = "ubuntu-24.04"
#: The aggregate job branch protection requires, after the eight that do the work.
CI_JOBS = 9


@pytest.mark.unit
def test_every_ci_job_runs_on_github_hosted_ubuntu_24_04() -> None:
    """Every ci.yml job runs on GitHub-hosted ubuntu-24.04, written out: workflows have no anchors."""
    workflow = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    runs_on = {name: job.get("runs-on") for name, job in workflow["jobs"].items()}
    assert len(runs_on) == CI_JOBS, f"ci.yml has {len(runs_on)} jobs, expected {CI_JOBS}: {sorted(runs_on)}"
    wrong = {name: value for name, value in runs_on.items() if value != RUNS_ON}
    assert wrong == {}, f"jobs whose runs-on is not {RUNS_ON!r}: {wrong}"


HARDEN_RUNNER = {
    "name": "Harden the runner",
    "uses": "step-security/harden-runner@ccd8616d44fd3846e67624a50d5aad6d37bf2d25",
    "with": {"egress-policy": "audit"},
}


@pytest.mark.unit
def test_every_ci_job_hardens_its_runner_first() -> None:
    """Every ci.yml job's first step is the pinned harden-runner, with no condition."""
    workflow = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    first_steps = {name: job["steps"][0] for name, job in workflow["jobs"].items()}
    assert len(first_steps) == CI_JOBS, f"ci.yml has {len(first_steps)} jobs, expected {CI_JOBS}"
    wrong = {name: step for name, step in first_steps.items() if step != HARDEN_RUNNER}
    assert wrong == {}, f"jobs whose first step is not {HARDEN_RUNNER}: {wrong}"


@pytest.mark.unit
def test_the_required_job_fails_unless_every_gating_job_succeeds() -> None:
    """`required` is the one check branch protection names: it needs every job that gates a pull request
    (all but the advisory ones, which may fail), runs even when one of them failed (`always()`), and only
    on pull requests, where `changelog` runs."""
    jobs = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))["jobs"]
    job = jobs["required"]
    on_pull_requests = {
        name
        for name, other in jobs.items()
        if name != "required"
        and not other.get("continue-on-error")
        and "schedule' ||" not in str(other.get("if", ""))
        and "integration-extended" not in str(other.get("if", ""))
    }
    assert set(job["needs"]) == on_pull_requests, sorted(on_pull_requests)
    assert job["needs"] == ["lint", "unit", "install-smoke", "integration", "changelog"]
    assert job["if"] == "${{ always() && github.event_name == 'pull_request' }}"
    check = job["steps"][-1]
    assert check["env"] == {"RESULTS": "${{ toJSON(needs) }}"}
    assert 'all(.value.result == "success")' in check["run"] and "exit 1" in check["run"]


def _docker_run_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("docker run ")]
    assert len(lines) == 1, f"expected one `docker run` line, found {lines}"
    return lines[0]


@pytest.mark.unit
def test_livekit_dev_server_is_started_the_same_way_everywhere() -> None:
    """The integration fixture, the CI action and `just livekit-up` start the same livekit-server, so a local
    run and a CI run see the same server, and it publishes the media ports as well as 7880."""
    conftest = ast.parse((REPO_ROOT / "tests" / "integration" / "conftest.py").read_text(encoding="utf-8"))
    fixture = next(
        node.value.value
        for node in conftest.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DOCKER_RUN" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    )
    action = yaml.safe_load(
        (REPO_ROOT / ".github" / "actions" / "livekit-dev" / "action.yml").read_text(encoding="utf-8")
    )
    steps = action["runs"]["steps"]
    start = next(step["run"] for step in steps if step.get("name") == "Start livekit-server")
    commands = {
        "tests/integration/conftest.py": fixture,
        ".github/actions/livekit-dev/action.yml": _docker_run_line(start),
        "justfile": _docker_run_line((REPO_ROOT / "justfile").read_text(encoding="utf-8")),
    }
    assert len(set(commands.values())) == 1, f"the livekit-server commands differ: {commands}"
    docker_args, image, server_args = fixture.partition(" livekit/livekit-server:v1.9.1 ")
    assert image, f"the image is not livekit/livekit-server:v1.9.1: {fixture}"
    for published in ("-p 7880:7880", "-p 7881:7881/tcp", "-p 7882:7882/udp"):
        assert f" {published} " in f" {docker_args} ", f"`{published}` is not before the image: {fixture}"
    # After the image, so docker passes them to livekit-server instead of rejecting them. `--node-ip`
    # puts the published loopback ports in the ICE candidates instead of the container's own address.
    for server_arg in ("--dev", "--bind 0.0.0.0", "--node-ip 127.0.0.1"):
        assert f" {server_arg} " in f" {server_args} ", f"`{server_arg}` is not after the image: {fixture}"
        flag = server_arg.split()[0]
        assert flag not in docker_args.split(), f"`{flag}` is before the image: {fixture}"


RELEASE_PATH = CI_PATH.with_name("release.yml")


def _release_workflow() -> dict:
    return yaml.safe_load(RELEASE_PATH.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_release_only_the_publish_job_can_mint_a_pypi_token() -> None:
    """Only `publish` can mint the OIDC token PyPI accepts, it waits for the `pypi` environment's reviewer,
    and it runs nothing from the repository: no checkout and no shell step. The code a release runs
    (dependencies, tests, the build backend) runs in `build`, which cannot publish."""
    workflow = _release_workflow()
    assert workflow["permissions"] == {"contents": "read"}, f"workflow default: {workflow['permissions']}"
    jobs = workflow["jobs"]
    # A mapping, never a string: `write-all` grants id-token too, and `"id-token" in "write-all"` is False.
    shorthand = {
        name: job["permissions"]
        for name, job in jobs.items()
        if not isinstance(job.get("permissions", {}), dict)
    }
    assert shorthand == {}, f"jobs with permissions that are not a mapping: {shorthand}"
    minting = sorted(name for name, job in jobs.items() if "id-token" in job.get("permissions", {}))
    assert minting == ["publish"], f"jobs that can mint an OIDC token: {minting}"
    publish = jobs["publish"]
    # id-token alone: publish reads nothing from the repository, and its own run's artifact needs no token.
    assert publish["permissions"] == {"id-token": "write"}, f"publish: {publish['permissions']}"
    assert publish["environment"] == "pypi"
    assert publish["needs"] == "build"
    steps = publish["steps"]
    assert [s for s in steps if "run" in s] == [], "publish runs a shell step"
    assert [s for s in steps if s.get("uses", "").startswith("actions/checkout@")] == [], "publish checks out"


# A scratch repository that ignores the user's and the system's git config: no hooks, signing or templates.
_GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=_GIT_ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.unit
def test_release_refuses_a_tag_on_a_commit_that_is_not_on_main(tmp_path: Path) -> None:
    """The release's ancestry check, run as the workflow runs it: it comes before any shell step, passes for a
    tagged commit on main, and fails for one only on a branch or when the clone has no `origin/main`."""
    steps = _release_workflow()["jobs"]["build"]["steps"]
    check = next(i for i, s in enumerate(steps) if s.get("name") == "Assert the tagged commit is on main")
    script = steps[check]["run"]
    # It runs before any of the tagged commit's code: every step ahead of it is an action, none a shell step.
    assert [s for s in steps[:check] if "run" in s] == [], "a shell step runs before the ancestry check"
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "user.email", "t@example.com")

    def run_check() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", script], cwd=tmp_path, env=_GIT_ENV, capture_output=True, text=True
        )

    _git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "on main")
    on_main = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "--quiet", "-b", "side")
    _git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "only on a branch")
    only_on_branch = _git(tmp_path, "rev-parse", "HEAD")

    # Without origin/main the failure must say so, not blame the commit: the fix is in the clone, not the tag.
    no_main = run_check()
    assert no_main.returncode != 0, "passed with no origin/main in the clone"
    assert "is not on main" not in no_main.stdout, f"blamed the commit, not the clone: {no_main.stdout}"
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", on_main)
    _git(tmp_path, "checkout", "--quiet", "--detach", only_on_branch)
    assert run_check().returncode != 0, "passed for a commit that is not on main"
    _git(tmp_path, "checkout", "--quiet", "--detach", on_main)
    assert run_check().returncode == 0, "failed for a commit on main"
