"""The LiveKit room adapters: `LiveKitTransportRoom` (direct join) and `PrivateLiveKitRoom` (relayed join)."""

from __future__ import annotations

# The SDK import guard runs first, so a missing [livekit] extra is one clear ImportError naming it.
from pipecat_synthesia.livekit import _rtc as _rtc  # noqa: F401
from pipecat_synthesia.livekit._types import (
    LiveKitCredentials,
    MintedAvatarToken,
    ProvidedAvatarToken,
)
from pipecat_synthesia.livekit.audio_filter import AvatarAudioFilter
from pipecat_synthesia.livekit.private_room import PrivateLiveKitRoom
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom

__all__ = [
    "LiveKitTransportRoom",
    "PrivateLiveKitRoom",
    "LiveKitCredentials",
    "MintedAvatarToken",
    "ProvidedAvatarToken",
    "AvatarAudioFilter",
]
