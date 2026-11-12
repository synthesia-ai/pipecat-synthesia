"""Every wire value the plugin shares with the avatar worker and the Synthesia API, as literals.

The fake avatar worker imports the same constants, so a renamed value would move both in lockstep and pass
every other test. The session end reasons and the event names are public contracts too.
"""

from __future__ import annotations

import pytest

from pipecat_synthesia import events, protocol

# Copied from protocol.py, never derived from it: changing a value is a protocol change.
WIRE_VALUES: dict[str, object] = {
    # livekit-agents avatar data-stream protocol
    "AUDIO_STREAM_TOPIC": "lk.audio_stream",
    "AUDIO_STREAM_NAME_PREFIX": "AUDIO_",
    "RPC_CLEAR_BUFFER": "lk.clear_buffer",
    "RPC_PLAYBACK_STARTED": "lk.playback_started",
    "RPC_PLAYBACK_FINISHED": "lk.playback_finished",
    "STREAM_ATTR_SAMPLE_RATE": "sample_rate",
    "STREAM_ATTR_NUM_CHANNELS": "num_channels",
    "RPC_HANDLER_RESPONSE": "ok",
    "RPC_HANDLER_REJECT": "reject",
    "TOKEN_ATTR_PUBLISH_ON_BEHALF": "lk.publish_on_behalf",
    # Synthesia worker protocol
    "RPC_SWAP_AVATAR": "swapAvatar",
    "SWAP_DEFAULT": "default",
    "SWAP_RESPONSE_STATUS_OK": "ok",
    "SWAP_ERROR_CODE_AVATAR_FAILED": "AVATAR_FAILED",
    "SWAP_ERROR_CODES": ("AVATAR_INCOMPLETE", "AVATAR_FAILED", "AVATAR_PREPARING"),
    "ATTR_AVATAR_READY": "avatar_ready",
    "ATTR_AVATAR_STATUS": "avatar_status",
    "AVATAR_STATE_PREPARING": "preparing",
    "AVATAR_STATE_READY": "ready",
    "AVATAR_STATE_FAILED": "failed",
    "WORKER_AUDIO_TRACK_NAME": "avatar_audio",
    "WORKER_VIDEO_TRACK_NAME": "avatar_video",
    # Synthesia-owned token attributes
    "TOKEN_ATTR_PROTOCOL": "synthesia.protocol",
    "TOKEN_ATTR_CLIENT": "synthesia.client",
    "TOKEN_ATTR_JOIN_MODE": "synthesia.join_mode",
    "TOKEN_ATTR_LAUNCH": "synthesia.launch",
    "PROTOCOL_VERSION": "1",
    "JOIN_MODE_DIRECT": "direct",
    "JOIN_MODE_RELAYED": "relayed",
    # client identification: the user transport
    "TRANSPORT_PACKAGES": ("livekit", "daily", "smallwebrtc", "websocket", "local"),
    "TRANSPORT_OTHER": "other",
    "TRANSPORT_UNKNOWN": "unknown",
    # Worker media format
    "WORKER_VIDEO_WIDTH": 512,
    "WORKER_VIDEO_HEIGHT": 512,
    "WORKER_VIDEO_FPS": 25,
    "WORKER_AUDIO_SAMPLE_RATE": 24000,
    "WORKER_AUDIO_CHANNELS": 1,
    # Synthesia public API
    "DEFAULT_API_URL": "https://developers.synthesia.io",
    "SESSIONS_PATH": "/api/interactive-avatars/sessions",
    "REQUEST_ID_HEADER": "Request-Id",
    "MAX_AVATAR_IDS": 5,
    "AVATAR_ID_PREFIX": "av_",
    "AVATAR_ID_PATTERN": r"^av_[A-Za-z0-9_-]{1,64}$",
    "ERROR_CODE_CLIENT_UPGRADE_REQUIRED": "client_upgrade_required",
    "ERROR_CODE_SERVICE_UNAVAILABLE": "service_unavailable",
}


def _public_constants(module: object) -> set[str]:
    return {name for name in vars(module) if name.isupper() and not name.startswith("_")}


@pytest.mark.unit
def test_the_table_covers_every_protocol_constant() -> None:
    """A constant added to `protocol.py` without a row here is a wire value nobody pinned."""
    assert _public_constants(protocol) == set(WIRE_VALUES)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(WIRE_VALUES))
def test_protocol_constant_has_its_wire_value(name: str) -> None:
    assert getattr(protocol, name) == WIRE_VALUES[name]


@pytest.mark.unit
def test_session_end_reasons_are_the_reasons_on_session_ended_reports() -> None:
    """The set of reasons is the contract; their order in the tuple is not."""
    expected = {
        "pipeline_ended",
        "pipeline_cancelled",
        "pipeline_cleanup",
        "no_humans_left",
        "avatar_left",
        "launch_failed",
        "degraded",
        "launch_timeout",
    }
    assert set(events.SESSION_END_REASONS) == expected
    assert len(events.SESSION_END_REASONS) == len(expected), "no reason is listed twice"


@pytest.mark.unit
def test_service_event_names() -> None:
    """Each name is public API a developer registers a handler under; `REGISTERED_EVENTS` may grow."""
    names = [
        (events.EVENT_AVATAR_CONNECTED, "on_avatar_connected"),
        (events.EVENT_AVATAR_READY, "on_avatar_ready"),
        (events.EVENT_AVATAR_DISCONNECTED, "on_avatar_disconnected"),
        (events.EVENT_SESSION_ENDED, "on_session_ended"),
        (events.EVENT_METRICS, "on_metrics"),
    ]
    assert [constant for constant, _ in names] == [name for _, name in names]
    assert {name for _, name in names} <= set(events.REGISTERED_EVENTS)
