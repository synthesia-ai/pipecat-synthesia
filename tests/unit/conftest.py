"""Fixtures shared by tests/unit."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.fakes.fake_api import FakeSynthesiaApi, ScriptedResponse

__all__ = ["fake_api_factory"]


@pytest.fixture
def fake_api_factory() -> Callable[[list[ScriptedResponse]], FakeSynthesiaApi]:
    """Build a `FakeSynthesiaApi` for a script; use as `async with fake_api_factory(script) as api:`."""

    def factory(script: list[ScriptedResponse]) -> FakeSynthesiaApi:
        return FakeSynthesiaApi(script)

    return factory
