"""The service's runtime settings, and the frame that updates them."""

from __future__ import annotations

from dataclasses import dataclass, field

from pipecat.frames.frames import ServiceUpdateSettingsFrame
from pipecat.services.settings import ServiceSettings
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class SynthesiaVideoSettings(ServiceSettings):
    # an avatar id, or "default" for the first of avatar_ids
    avatar_id: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class SynthesiaVideoUpdateSettingsFrame(ServiceUpdateSettingsFrame[SynthesiaVideoSettings]):
    pass
