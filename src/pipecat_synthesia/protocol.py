"""Every wire name and wire fact shared with the avatar worker and the Synthesia API.

Additive only: never rename or reuse a value, because deployed workers and clients depend on it (see
docs/protocol-compatibility.md). No other module may spell these strings out.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

# --- livekit-agents avatar data-stream protocol (owned by LiveKit; do not extend) ---
AUDIO_STREAM_TOPIC = "lk.audio_stream"
AUDIO_STREAM_NAME_PREFIX = "AUDIO_"  # name = AUDIO_ + uuid4().hex[:12]
RPC_CLEAR_BUFFER = "lk.clear_buffer"  # bot -> worker, payload ""
RPC_PLAYBACK_STARTED = "lk.playback_started"  # worker -> bot, payload ""
# worker -> bot, payload JSON {"playback_position": float, "interrupted": bool, ...}
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
# str(int); REQUIRED by the worker (missing => worker session dies)
STREAM_ATTR_SAMPLE_RATE = "sample_rate"
STREAM_ATTR_NUM_CHANNELS = "num_channels"  # str(int); REQUIRED
# the receiver ignores the response; it must be a str and we must never raise
RPC_HANDLER_RESPONSE = "ok"
RPC_HANDLER_REJECT = "reject"  # returned to callers other than the avatar identity
TOKEN_ATTR_PUBLISH_ON_BEHALF = "lk.publish_on_behalf"

# --- Synthesia worker protocol (owned by Synthesia) ---
RPC_SWAP_AVATAR = "swapAvatar"  # payload JSON {"avatar_id": "<unprefixed avatar id>" | "default"}
SWAP_DEFAULT = "default"
SWAP_RESPONSE_STATUS_OK = "ok"  # success: {"status": "ok", "avatar_id": "<unprefixed id>"}
# failures: {"error": str, "code"?: str, "avatar_id"?: str}
# named so readiness.py can stamp the worker's own code on a failure it detects locally
SWAP_ERROR_CODE_AVATAR_FAILED = "AVATAR_FAILED"
SWAP_ERROR_CODES = ("AVATAR_INCOMPLETE", SWAP_ERROR_CODE_AVATAR_FAILED, "AVATAR_PREPARING")
# participant attribute; "true" once EVERY avatar is ready; ABSENT otherwise (never "false")
ATTR_AVATAR_READY = "avatar_ready"
# participant attribute; compact JSON
# {"<unprefixed id>": {"state": ..., "incomplete"?, "missing"?, "reason"?}}
ATTR_AVATAR_STATUS = "avatar_status"
AVATAR_STATE_PREPARING = "preparing"
AVATAR_STATE_READY = "ready"
AVATAR_STATE_FAILED = "failed"
WORKER_AUDIO_TRACK_NAME = "avatar_audio"  # SOURCE_MICROPHONE
WORKER_VIDEO_TRACK_NAME = "avatar_video"  # SOURCE_CAMERA

# --- Synthesia-owned additions (namespace synthesia.*), put on every token we mint ---
# client identification for the avatar worker (*installed-version channels*)
TOKEN_ATTR_PROTOCOL = "synthesia.protocol"
TOKEN_ATTR_CLIENT = "synthesia.client"
TOKEN_ATTR_JOIN_MODE = "synthesia.join_mode"  # only on a minted avatar token
# only on a minted avatar token: a random id per launch, so the session can tell its own avatar from a
# stale one with the same identity before the 201 (*early start*)
TOKEN_ATTR_LAUNCH = "synthesia.launch"
PROTOCOL_VERSION = "1"  # absent attribute on an old client == "1"
# the join mode, on that token attribute and in the X-Synthesia-Client header (`JoinMode`'s values)
JOIN_MODE_DIRECT = "direct"
JOIN_MODE_RELAYED = "relayed"

# --- Client identification: the user transport in the X-Synthesia-Client header ---
# The Pipecat transport packages named as themselves; never a class name (see `_compat.transport_name`)
TRANSPORT_PACKAGES = ("livekit", "daily", "smallwebrtc", "websocket", "local")
TRANSPORT_OTHER = "other"  # any other output transport
TRANSPORT_UNKNOWN = "unknown"  # no output transport downstream of the service

# --- Worker media facts (documentation and README defaults only; never used to resample) ---
WORKER_VIDEO_WIDTH = 512
WORKER_VIDEO_HEIGHT = 512
WORKER_VIDEO_FPS = 25
WORKER_AUDIO_SAMPLE_RATE = 24000
WORKER_AUDIO_CHANNELS = 1

# --- Synthesia public API ---
DEFAULT_API_URL = "https://developers.synthesia.io"
SESSIONS_PATH = "/api/interactive-avatars/sessions"
REQUEST_ID_HEADER = "Request-Id"
MAX_AVATAR_IDS = 5
AVATAR_ID_PREFIX = "av_"
AVATAR_ID_PATTERN = r"^av_[A-Za-z0-9_-]{1,64}$"
# reserved for a future API version gate (*installed-version channels*)
ERROR_CODE_CLIENT_UPGRADE_REQUIRED = "client_upgrade_required"
# 503: also what a direct join gets when the avatar worker cannot join the room (see livekit/session.py)
ERROR_CODE_SERVICE_UNAVAILABLE = "service_unavailable"
