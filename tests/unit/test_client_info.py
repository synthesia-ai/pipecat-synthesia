"""`_client_info.py`: the eight client identification fields.

The six process-wide fields are filled in by `build_client_info()`; the join mode and the user transport
are each service's, set on its own copy. The header always carries all eight keys, a missing value as
`null`: a wire contract, so the tests use equality.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from collections.abc import Iterator

import pytest

from pipecat_synthesia._client_info import ClientInfo, build_client_info


@pytest.fixture
def uncached_client_info() -> Iterator[None]:
    """`build_client_info` is cached process-wide; clear it around the test so no other test sees ours."""
    build_client_info.cache_clear()
    yield
    build_client_info.cache_clear()


PROCESS_FIELDS = {
    "sdk": "pipecat-synthesia",
    "sdk_version": "1.2.3",
    "framework": "pipecat-ai",
    "framework_version": "1.10.0",
    "lang_version": "3.11.9",
    "platform": "Linux; x86_64",
}


@pytest.mark.unit
def test_the_process_wide_fields_are_populated_and_the_per_service_fields_are_not() -> None:
    info = build_client_info()
    for name in PROCESS_FIELDS:
        value = getattr(info, name)
        assert isinstance(value, str) and value, f"{name} is not a populated string: {value!r}"
    assert (info.join_mode, info.transport) == (None, None)


@pytest.mark.unit
@pytest.mark.parametrize("join_mode, transport", [("direct", "livekit"), (None, None)])
def test_as_header_json_is_compact_sorted_eight_keys(join_mode: str | None, transport: str | None) -> None:
    info = ClientInfo(**PROCESS_FIELDS, join_mode=join_mode, transport=transport)
    header = info.as_header_json()
    assert header == json.dumps(dataclasses.asdict(info), sort_keys=True, separators=(",", ":"))
    assert json.loads(header) == {**PROCESS_FIELDS, "join_mode": join_mode, "transport": transport}


@pytest.mark.unit
def test_token_client_attr() -> None:
    info = ClientInfo(**PROCESS_FIELDS, join_mode="relayed", transport="daily")
    assert info.token_client_attr == "pipecat-synthesia/1.2.3"


@pytest.mark.unit
def test_client_info_is_immutable() -> None:
    """One cached instance is shared by every caller, so an assignment would change what all of them see."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        build_client_info().sdk_version = "replaced"  # type: ignore[misc]


@pytest.mark.unit
def test_framework_version_is_unknown_without_pipecat_metadata(
    monkeypatch: pytest.MonkeyPatch, uncached_client_info: None
) -> None:
    """A vendored or editable pipecat-ai with no distribution metadata must never fail a launch."""

    def not_installed(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)

    info = build_client_info()

    assert isinstance(info, ClientInfo)
    assert info.framework == "pipecat-ai"
    assert info.framework_version == "unknown"
    assert "pipecat-ai/unknown" in info.user_agent
