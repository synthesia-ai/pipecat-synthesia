"""A newer Pipecat minor than any tested logs one WARNING per process, at service construction.

The package allows any `pipecat-ai` 1.x from 1.14, but is tested only with `TESTED_PIPECAT_VERSIONS`. On a
newer minor than all of them, which includes a pre-release or dev build of one, constructing a
`SynthesiaVideoService` logs one WARNING naming the installed version and the tested ones; the second service
in the same process logs nothing. A tested version and a patch of a tested minor log nothing.

The installed version is an input here: each test sets `pipecat_synthesia._compat.PIPECAT_VERSION`, which
the check reads at call time. The autouse `untested_pipecat_warning_off` fixture in tests/conftest.py turns
the warning off for every other test; these tests turn it back on.
"""

from __future__ import annotations

import pytest

from pipecat_synthesia import _compat
from pipecat_synthesia._compat import TESTED_PIPECAT_VERSIONS
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.version import __version__
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import make_service

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def untested_pipecat_warning_on(untested_pipecat_warning_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo tests/conftest.py's `untested_pipecat_warning_off` for this file; it takes that fixture so it runs
    after it."""
    monkeypatch.setattr("pipecat_synthesia.service.untested_newer_pipecat", _compat.untested_newer_pipecat)


def _construct_two_services(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, installed: str
) -> list[str]:
    """Construct two services with `installed` as the Pipecat version; return the WARNING lines about it."""
    monkeypatch.setattr("pipecat_synthesia._compat.PIPECAT_VERSION", installed)
    make_service(FakeAvatarRoom(JoinMode.DIRECT))
    make_service(FakeAvatarRoom(JoinMode.RELAYED))
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "pipecat-ai" in record.getMessage()
    ]


@pytest.mark.parametrize("installed", ["1.17.0", "2.0.0"])
def test_a_newer_minor_warns_once_across_two_services(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, installed: str
) -> None:
    """One WARNING for two services, naming the installed version, this release and every tested version."""
    warnings = _construct_two_services(monkeypatch, caplog_loguru, installed)

    assert len(warnings) == 1, warnings
    tested = ", ".join(TESTED_PIPECAT_VERSIONS)
    assert warnings[0].startswith(
        f"pipecat-ai {installed} is newer than any Pipecat this pipecat-synthesia ({__version__}) was tested "
        f"with ({tested}). It is allowed"
    ), warnings[0]
    assert "pin pipecat-ai in your app" in warnings[0]


@pytest.mark.parametrize("installed", ["1.17.0.dev3", "1.17.0rc1"])
def test_a_pre_release_of_a_newer_minor_warns(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, installed: str
) -> None:
    """A dev build or release candidate of a newer minor is that minor: it warns, once."""
    warnings = _construct_two_services(monkeypatch, caplog_loguru, installed)

    assert len(warnings) == 1, warnings
    assert warnings[0].startswith(f"pipecat-ai {installed} is newer than any Pipecat"), warnings[0]


@pytest.mark.parametrize(
    "installed",
    [
        *TESTED_PIPECAT_VERSIONS,
        "1.16.3",  # a patch of the newest tested minor
        "1.14.7",  # a patch of an older tested minor
        # Builds of Pipecat's main branch. Pipecat's scheme for them is not documented: `1.15.1.dev627` is
        # what a git install of main reported on 2026-11-09 (commit dea6543), and the `+g<hash>` form is an
        # unconfirmed guess at what a build with a local version would report. Either reads as the minor it
        # was built after, so a git build of the next minor is not detected (docs/architecture.md,
        # "Pipecat range").
        "1.15.1.dev627",
        "1.15.1.dev12+g0a1b2c3",
        "not-a-version",  # never raises: the constructor must not fail on a version it cannot read
    ],
)
def test_no_warning_up_to_the_newest_tested_minor(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture, installed: str
) -> None:
    """A tested version, a patch of a tested minor, a main-branch build, or no readable version: silent."""
    assert _construct_two_services(monkeypatch, caplog_loguru, installed) == []
