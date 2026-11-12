"""Render a Synthesia interactive avatar in a Pipecat pipeline.

docs/architecture.md describes how the package fits together. Comments here cite its design rules by name,
in *italics*; each is defined there.
"""

from __future__ import annotations

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.events import AvatarConnected, AvatarReady
from pipecat_synthesia.frames import SynthesiaVideoSettings, SynthesiaVideoUpdateSettingsFrame
from pipecat_synthesia.metrics import SessionMetrics
from pipecat_synthesia.room import AvatarRoom, AvatarSession, JoinMode
from pipecat_synthesia.service import ServiceState, SynthesiaVideoService
from pipecat_synthesia.version import __version__

__all__ = [
    "SynthesiaVideoService",
    "SynthesiaVideoSettings",
    "SynthesiaVideoUpdateSettingsFrame",
    "ServiceState",
    "SynthesiaError",
    "ErrorType",
    "AvatarConnected",
    "AvatarReady",
    "AvatarRoom",
    "AvatarSession",
    "JoinMode",
    "SessionMetrics",
    "__version__",
]
