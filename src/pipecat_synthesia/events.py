"""The service's event names and payloads."""

from __future__ import annotations

from dataclasses import dataclass

EVENT_AVATAR_CONNECTED = "on_avatar_connected"
EVENT_AVATAR_READY = "on_avatar_ready"
EVENT_AVATAR_DISCONNECTED = "on_avatar_disconnected"
EVENT_SESSION_ENDED = "on_session_ended"
EVENT_METRICS = "on_metrics"
REGISTERED_EVENTS = (
    EVENT_AVATAR_CONNECTED,
    EVENT_AVATAR_READY,
    EVENT_AVATAR_DISCONNECTED,
    EVENT_SESSION_ENDED,
    EVENT_METRICS,
)
# `on_error` is not registered here: Pipecat's FrameProcessor already registers it and fires it from
# push_error with (processor, ErrorFrame). `ErrorFrame.exception` is the SynthesiaError.


@dataclass(frozen=True)
class AvatarConnected:
    session_id: str
    avatar_identity: str
    active_avatar_id: str
    room_name: str | None


@dataclass(frozen=True)
class AvatarReady:
    session_id: str
    status: dict[str, str]


# Handler signatures (the first argument is the service):
#   on_avatar_connected(service, info: AvatarConnected)   once, when the session becomes active
#   on_avatar_ready(service, info: AvatarReady)           once, when every session avatar is ready and the
#                                                         session is active (never before on_avatar_connected)
#   on_avatar_disconnected(service, reason: str)          at most once; reason is a leave reason
#   on_session_ended(service, reason: str)                exactly once; reason is an end reason (below), and
#                                                         equals metrics.end_reason
#   on_metrics(service, metrics: SessionMetrics)          exactly once, right after on_session_ended
SESSION_END_REASONS = (
    "pipeline_ended",
    "pipeline_cancelled",
    "pipeline_cleanup",
    "no_humans_left",
    "avatar_left",
    "launch_failed",
    "degraded",
    "launch_timeout",
)
