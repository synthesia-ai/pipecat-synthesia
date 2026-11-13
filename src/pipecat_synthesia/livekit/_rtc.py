"""The LiveKit SDK import guard: one clear ImportError when the [livekit] extra is missing."""

from __future__ import annotations

try:
    from livekit import api as lk_api  # noqa: F401
    from livekit import rtc as lk_rtc  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError(
        "pipecat-synthesia's LiveKit support requires the [livekit] extra: "
        'pip install "pipecat-synthesia[livekit]"'
    ) from e
