"""Client identification: what the package tells the Synthesia API and LiveKit about itself.

It ships in every release so that Synthesia can tell which installed versions are still calling
(*installed-version channels*, defined in docs/architecture.md).
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass

from pipecat_synthesia.version import __version__


@dataclass(frozen=True)
class ClientInfo:
    """The fields sent in the X-Synthesia-Client header, the first six also in the User-Agent, and some on
    the avatar token.

    The first six are the process's, the same for every service. ``join_mode`` and ``transport`` are one
    service's: ``build_client_info()`` leaves them None, and each service sets them on its own copy.
    """

    sdk: str
    sdk_version: str
    framework: str
    framework_version: str
    lang_version: str
    platform: str
    join_mode: str | None = None  # `JoinMode.value`
    transport: str | None = None  # from a closed set (see `_compat.transport_name`), never a class name

    @property
    def user_agent(self) -> str:
        """``f"{sdk}/{sdk_version} {framework}/{framework_version} python/{lang_version} ({platform})"``."""
        return (
            f"{self.sdk}/{self.sdk_version} {self.framework}/{self.framework_version} "
            f"python/{self.lang_version} ({self.platform})"
        )

    @property
    def token_client_attr(self) -> str:
        """``f"{sdk}/{sdk_version}"``; the value of the synthesia client token attribute."""
        return f"{self.sdk}/{self.sdk_version}"

    def as_header_json(self) -> str:
        """``json.dumps`` of every field, ``sort_keys=True``, compact separators; a field that is None is
        ``null``, so the Synthesia API always sees the same eight keys."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@functools.cache
def build_client_info() -> ClientInfo:
    """Return the process-wide ClientInfo; cached, so two calls return the SAME object.

    ``framework_version`` is ``importlib.metadata.version("pipecat-ai")``, or ``"unknown"`` when
    pipecat-ai is not installed as a distribution.
    """
    try:
        framework_version = importlib.metadata.version("pipecat-ai")
    except importlib.metadata.PackageNotFoundError:
        framework_version = "unknown"
    return ClientInfo(
        sdk="pipecat-synthesia",
        sdk_version=__version__,
        framework="pipecat-ai",
        framework_version=framework_version,
        lang_version=platform.python_version(),
        platform=f"{platform.system()}; {platform.machine()}",
    )
