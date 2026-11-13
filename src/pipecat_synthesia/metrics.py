"""Per-session launch and playback metrics."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# Every field not listed here is a monotonic timestamp, reported as seconds since launch_started. The
# timestamp fields are found with dataclasses.fields() rather than listed, because one of them
# (`avatar_ready`) is also a wire name, which only protocol.py may spell out.
_NON_DELTA_FIELDS = frozenset(
    {
        "session_id",
        "request_id",
        "launch_started",
        "end_reason",
        "utterance_ttfb_seconds",
        "interruption_latency_seconds",
        "playback_finished_count",
        "prelaunch_dropped_bytes",
    }
)


@dataclass
class SessionMetrics:
    """All timestamps are ``time.monotonic()`` seconds on one clock; None until observed."""

    launch_started: float | None = None
    room_connected: float | None = None
    session_requested: float | None = None
    session_created: float | None = None
    avatar_joined: float | None = None
    avatar_ready: float | None = None
    # Can be wrong if a previous avatar with the same identity is still in the room when a launch succeeds
    # (a restart within the avatar worker's 5 s grace).
    avatar_video_published: float | None = None
    first_video_frame: float | None = None
    transport_ready: float | None = None
    ended: float | None = None
    end_reason: str | None = None
    # Not recorded in this release (docs/protocol-compatibility.md, section 10); kept for code that reads it.
    utterance_ttfb_seconds: list[float] = field(default_factory=list)
    # Not recorded in this release (docs/protocol-compatibility.md, section 10); kept for code that reads it.
    interruption_latency_seconds: list[float] = field(default_factory=list)
    playback_finished_count: int = 0
    prelaunch_dropped_bytes: int = 0
    # from the Synthesia API's 201, so set even when the launch fails after it
    session_id: str | None = None
    request_id: str | None = None

    def summary(self) -> dict[str, Any]:
        """Deltas from ``launch_started`` in seconds (ms precision). Fields that are None are skipped."""
        out: dict[str, Any] = {}
        start = self.launch_started
        if start is not None:
            for spec in dataclasses.fields(self):
                if spec.name in _NON_DELTA_FIELDS:
                    continue
                value = getattr(self, spec.name)
                if value is not None:
                    out[spec.name] = round(value - start, 3)
        if self.end_reason is not None:
            out["end_reason"] = self.end_reason
        out["playback_finished_count"] = self.playback_finished_count
        out["prelaunch_dropped_bytes"] = self.prelaunch_dropped_bytes
        if self.utterance_ttfb_seconds:
            out["utterance_ttfb_seconds"] = [round(v, 3) for v in self.utterance_ttfb_seconds]
        if self.interruption_latency_seconds:
            out["interruption_latency_seconds"] = [round(v, 3) for v in self.interruption_latency_seconds]
        if self.session_id is not None:
            out["session_id"] = self.session_id
        if self.request_id is not None:
            out["request_id"] = self.request_id
        return out
