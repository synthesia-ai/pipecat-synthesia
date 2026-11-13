"""The interface between the service and a room adapter: `AvatarRoom`, `AvatarSession`, and what passes
between them. The service sees only these types, never LiveKit directly.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat_synthesia._client_info import ClientInfo
from pipecat_synthesia.protocol import JOIN_MODE_DIRECT, JOIN_MODE_RELAYED

if TYPE_CHECKING:
    import loguru

    from pipecat_synthesia.api import SessionApiClient


class JoinMode(enum.Enum):
    """Where the avatar joins, which decides whether the service emits the avatar's media."""

    # direct join: the avatar publishes straight to the user; the service emits no media frames
    DIRECT = JOIN_MODE_DIRECT
    # relayed join: the avatar's video and audio come through the pipeline as frames
    RELAYED = JOIN_MODE_RELAYED


def _ignore_session_created(session_id: str, request_id: str | None, created_at: float) -> None:
    """The default `on_session_created`, so callbacks built without one keep working."""


@dataclass
class AvatarCallbacks:
    """How a session reports to the service. Delivery follows *state-before-await*.

    Callbacks run concurrently: the playback, status and leave callbacks each get a callback task of their
    own, and the frame callbacks are awaited by one consumer task per track. They can therefore interleave
    at any `await`, so the service's callbacks make their state changes before their first `await` and
    re-check state after each one. A failing asynchronous callback is logged, never allowed to break the
    session. Implementations must not serialise callbacks with a lock or a queue.

    `on_session_created` is the one synchronous callback, and the one exception: called directly, at most
    once per launch, as soon as the Synthesia API's 201 arrives, which in a *connect overlap* can be before
    the bot has connected. It is not guarded, so an implementation must not block or raise.

    Guarantees: `on_avatar_left` is delivered at most once, and no playback or status callback starts after
    it; one already in flight may finish. In a relayed join the frame callbacks can still arrive after it,
    until the session closes: track capture stops a track's consumer when that track ends, and the rest
    only at the close. The service drops them outside SESSION_ACTIVE. A slow frame callback slows only its
    own track, whose bounded SDK buffer drops the oldest frames.
    """

    on_playback_started: Callable[[], Awaitable[None]]
    # (playback_position_seconds, interrupted)
    on_playback_finished: Callable[[float, bool], Awaitable[None]]
    # the avatar worker went away (it disconnected into the *replacement grace*), so no playback_finished
    # will come for what it was playing
    on_playback_lost: Callable[[], Awaitable[None]]
    # ({avatar id: state}, all_ready), on every change; may fire before launch() returns
    on_avatar_status: Callable[[dict[str, str], bool], Awaitable[None]]
    # (reason, humans_present). reason is a leave reason: "avatar_left_room", "avatar_video_stopped" or
    # "room_connection_lost". humans_present: in a direct join, whether a human participant remained when the
    # avatar left, where False means the no-humans-left end is the expected cause, and True is settled by
    # AvatarSession.wait_no_humans_left() (*no-humans-left window*); in a relayed join, None.
    on_avatar_left: Callable[[str, bool | None], Awaitable[None]]
    # "video" | "audio"; metrics only
    on_track_published: Callable[[str], Awaitable[None]]
    # (rgb24_bytes, width, height)  relayed join only
    on_video_frame: Callable[[bytes, int, int], Awaitable[None]] | None
    # (pcm_s16le, sample_rate, num_channels)  relayed join only
    on_audio_frame: Callable[[bytes, int, int], Awaitable[None]] | None
    # (session_id, request_id, created_at); created_at is time.monotonic(), the clock of LaunchTimings
    on_session_created: Callable[[str, str | None, float], None] = _ignore_session_created


@dataclass(frozen=True)
class LaunchContext:
    """What the service hands an adapter's `launch()`."""

    api: SessionApiClient
    # the session avatars, normalised; the first is active
    avatar_ids: tuple[str, ...]
    # seconds for the whole launch (*launch timeout*); the service enforces it around launch(), and
    # adapters cap their own waits by what is left of it
    launch_timeout: float
    # from FrameProcessorSetup.audio_out_sample_rate; relayed join captures avatar audio at this rate
    audio_out_sample_rate: int
    # always 1
    audio_out_channels: int
    client: ClientInfo
    callbacks: AvatarCallbacks
    # already bound with component/avatar fields
    log: loguru.Logger


@dataclass(frozen=True)
class LaunchTimings:
    """All values are time.monotonic() seconds (same clock as SessionMetrics.launch_started); fixed once
    launch() returns, except a `session_created` the 201 fills in after an *early start*."""

    # RoomHandle.connected_at (when the rtc room connect was observed)
    room_connected: float | None
    # just before create_session() was called; its retries happen inside that call, so they never move it
    session_requested: float
    # when create_session() returned the 201; None when launch() returned before it (*early start*), in
    # which case on_session_created reports it
    session_created: float | None
    # when the avatar participant joined: the latest join of its identity seen before launch() returns, or
    # when the bot first saw it if it was there first; a worker that replaces it later (after a retried
    # session request) does not move it
    avatar_joined: float


class AvatarSession(ABC):
    # the 201's session id; after an *early start*, not there until the 201 lands (wait_session_created())
    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @property
    @abstractmethod
    def avatar_identity(self) -> str: ...

    @property
    @abstractmethod
    def room_name(self) -> str | None: ...

    @property
    @abstractmethod
    def active_avatar_id(self) -> str: ...

    # {avatar id: state}; {} until the first attribute update
    @property
    @abstractmethod
    def avatar_status(self) -> dict[str, str]: ...

    @property
    @abstractmethod
    def all_ready(self) -> bool: ...

    # False once the avatar has left, which is before on_avatar_left is delivered, or once aclose() has
    # begun; never True again
    @property
    @abstractmethod
    def is_alive(self) -> bool: ...

    @property
    @abstractmethod
    def timings(self) -> LaunchTimings: ...

    @abstractmethod
    async def send_audio(self, pcm_s16le: bytes, sample_rate: int, num_channels: int) -> None:
        """Append a chunk to the current utterance, opening a byte stream if none is open.

        A chunk whose sample rate or channel count differs ends the open stream and starts a new one. Once
        the session is no longer alive, audio is dropped, with one WARNING.
        """

    # end the current utterance by closing its byte stream; no-op if none
    @abstractmethod
    async def end_utterance(self) -> None: ...

    @abstractmethod
    async def interrupt(self) -> None:
        """Send clear buffer (bounded, never raises), then end the current utterance."""

    @abstractmethod
    async def swap_avatar(self, avatar_id: str, *, timeout: float) -> str:  # noqa: ASYNC109
        """Make ``avatar_id`` (or ``"default"``) the active avatar, and return its id.

        If that avatar is known to be still preparing, waits for it first, within ``timeout``. Raises
        ``SynthesiaError`` (``UNKNOWN_AVATAR``, ``AVATAR_SWAP_FAILED``, ``TIMEOUT`` or ``CONNECTION``), always
        carrying the session id.
        """

    async def wait_session_created(self) -> None:
        """Return once the Synthesia API's 201 has arrived, so ``session_id`` is set.

        ``launch()`` may return before the 201 (*early start*), once this launch's avatar is publishing:
        audio can then be sent while the 201 is still on its way. Raises ``SynthesiaError`` if the session
        request fails after all. The default is for a session whose launch always waits for the 201.
        """
        return None

    # False while the session holds the bot's audio back: after an *early start*, from a retry of the
    # session request until the request settles, because the retry's avatar worker may replace the one in
    # the room.
    # The service's send loop leaves queued audio queued meanwhile. The default never holds.
    @property
    def sending(self) -> bool:
        return True

    async def wait_sending(self) -> None:
        """Return once `sending` is True again, or the session has begun closing."""
        return None

    async def wait_no_humans_left(self) -> bool:
        """After a leave reported with humans present, wait for the last human participant to leave, for at
        most the *no-humans-left window*, and return whether none remains.

        The service calls it once the session has begun closing, so it must not rely on the session's own
        room listeners. The default, for a session that cannot tell, is False at once.
        """
        return False

    # run by aclose(), each bounded by CLOSE_HOOK_SECS
    @abstractmethod
    def add_close_hook(self, hook: Callable[[], Awaitable[None]]) -> None: ...

    # *single flight*, every wait bounded; never disconnects a borrowed room (*borrowed-room rule*)
    @abstractmethod
    async def aclose(self) -> None: ...


class AvatarRoom(ABC):
    # class attribute on each subclass
    join_mode: JoinMode

    # known at construction
    @property
    @abstractmethod
    def avatar_identity(self) -> str: ...

    @abstractmethod
    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        """Launch an avatar session in this room, and return it once the avatar has joined.

        Raises ``SynthesiaError`` after rolling back everything it did. Cancellation, including by the
        service's launch timeout, rolls back the same way and then re-raises.
        """
