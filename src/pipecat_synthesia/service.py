"""`SynthesiaVideoService`: the Pipecat processor that drives a Synthesia avatar.

It sees only the `AvatarRoom` / `AvatarSession` interface, and owns four things:

* the *frame contract*: ``TTSAudioRawFrame`` goes into the send queue, and a send loop the service owns
  writes it to the session; every other frame is forwarded unchanged;
* the service states, ``INIT -> LAUNCHING -> SESSION_ACTIVE -> ENDED``, with ``DEGRADED`` / ``FAILED`` for
  a launch that failed, before the session went active or, after an *early start*, once it had; in
  ``DEGRADED`` and ``ENDED`` TTS audio passes through, so the bot continues voice-only, and in ``FAILED`` it
  is dropped, because the service is ending the pipeline;
* the speaking state, derived from the avatar worker's playback events with a stop hysteresis, in both
  join modes (*speaking-state ownership*);
* events, metrics and error reporting, each through one helper: ``_end_session`` for the session end, and
  ``_push_synthesia_error`` for errors (*permanent-error rule*).

Concurrency: Pipecat calls ``process_frame`` from its input task (system frames) and its process task (data
and control frames); the room delivers each callback on a callback task of its own; and the send loop, the
launch and the resume after an interruption are tasks created here. So every callback follows
*state-before-await*, and nothing here takes a lock.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import inspect
from collections import deque
from collections.abc import Sequence
from time import monotonic
from typing import TYPE_CHECKING, Any

import aiohttp
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    EndWorkerFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.types import is_given

from pipecat_synthesia._async import wait_bounded
from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia._compat import (
    TESTED_PIPECAT_VERSIONS,
    output_transport_params,
    transport_name,
    untested_newer_pipecat,
)
from pipecat_synthesia._log import get_logger
from pipecat_synthesia._once import warn_once
from pipecat_synthesia._redact import describe_exception
from pipecat_synthesia._send_queue import AudioChunk, SendQueue, UtteranceEnd
from pipecat_synthesia.api import SessionApiClient
from pipecat_synthesia.config import ServiceConfig, api_url_outside_synthesia
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.events import (
    EVENT_AVATAR_CONNECTED,
    EVENT_AVATAR_DISCONNECTED,
    EVENT_AVATAR_READY,
    EVENT_METRICS,
    EVENT_SESSION_ENDED,
    REGISTERED_EVENTS,
    AvatarConnected,
    AvatarReady,
)
from pipecat_synthesia.frames import SynthesiaVideoSettings, SynthesiaVideoUpdateSettingsFrame
from pipecat_synthesia.metrics import SessionMetrics
from pipecat_synthesia.protocol import WORKER_VIDEO_FPS, WORKER_VIDEO_HEIGHT, WORKER_VIDEO_WIDTH
from pipecat_synthesia.room import AvatarCallbacks, AvatarRoom, AvatarSession, JoinMode, LaunchContext
from pipecat_synthesia.version import __version__

if TYPE_CHECKING:
    import loguru


class ServiceState(enum.Enum):
    INIT = "init"
    LAUNCHING = "launching"
    SESSION_ACTIVE = "session_active"
    DEGRADED = "degraded"
    FAILED = "failed"
    ENDED = "ended"


# In a terminal state the send queue takes no more audio: TTS audio passes through untouched (voice-only),
# except in FAILED, where it is dropped (*frame contract*).
_TERMINAL_STATES = frozenset({ServiceState.DEGRADED, ServiceState.FAILED, ServiceState.ENDED})
# How long the service waits for session.aclose(); its steps are bounded by livekit/_types constants.
# Past it, the close finishes detached (see _close_session).
SESSION_ACLOSE_BACKSTOP_SECS = 20.0
# Timeout for cancelling the launch task. A relayed join's rollback never cancels the bot's connect, so it
# can wait for it: its connect_timeout (15 s by default), + 1 s, + the room's disconnect (5 s), alongside the
# room's delete (5 s). This is above that with margin. Pipecat's cancel_task cancels, then waits; past this
# timeout it logs a warning but still waits for a launch that absorbs the cancel.
LAUNCH_CANCEL_SECS = 30.0
# How long the teardown lets an interrupt already under way run before cancelling it: clear buffer's own
# bound (livekit/_types.CLEAR_BUFFER_SECS, 2 s, plus a 0.5 s margin; the core cannot import it, see
# *LiveKit-free core*), with room to spare. Cutting the utterance close short is harmless: the session close
# ends the utterance too.
INTERRUPTION_WAIT_SECS = 3.0
# How long the teardown waits, before it cancels the launch or fires any end event, for an end already under
# way to finish dispatching its error: the voice-only handover before the push, and the app's on_error
# handlers, which Pipecat runs inline, one after another, in the task that pushes it. Within it, every
# handler sees the error before on_session_ended. Past it, the end events fire anyway: handlers still to run
# may run after them, or never, if the end cancels the task running them (the launch's), and an error not
# yet pushed is dropped. Not longer: it is the app's code, and the teardown must stay bounded.
ON_ERROR_WAIT_SECS = 1.0
# Pipecat's own cancel_task default. Equals livekit/_types.TASK_CANCEL_SECS, which the core cannot import
# (*LiveKit-free core*).
_TASK_CANCEL_SECS = 1.0
# How many dropped media frames between TRACE lines while the transport is not ready.
_MEDIA_DROP_TRACE_EVERY = 100


def _pipecat_keywords() -> frozenset[str]:
    """The keywords Pipecat's own constructor chain declares, read from its signatures.

    Read rather than listed, so a keyword a later Pipecat adds is accepted without a release of this
    package. At Pipecat 1.14 to 1.16 they are ``name``, ``enable_direct_mode``, ``metrics`` and
    ``task_manager``, plus ``settings``, which the service takes itself.
    """
    names: set[str] = set()
    for cls in AIService.__mro__:
        init = cls.__dict__.get("__init__")
        if init is None or cls is object:
            continue
        try:
            params = inspect.signature(init).parameters.values()
        except (TypeError, ValueError):  # no signature to read: nothing to accept from it
            continue
        keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        names.update(p.name for p in params if p.kind in keyword_kinds and p.name != "self")
    return frozenset(names)


def _reject_unknown_keywords(kwargs: dict[str, Any]) -> None:
    """Raise ``TypeError`` for a keyword neither the service nor Pipecat's constructor chain declares.

    Pipecat's ``BaseObject`` accepts any keyword and drops it, so without this a misspelt argument, or one
    from the LiveKit plugin, would do nothing and say nothing. A ``TypeError``, as Python raises for any
    unexpected keyword. The message names the keywords, never their values, which may be secrets
    (*redaction rule*).
    """
    if not kwargs:
        return
    accepted = _pipecat_keywords()  # read per call: a service is built once per pipeline
    unknown = [name for name in kwargs if name not in accepted]
    if not unknown:
        return
    names = ", ".join(repr(name) for name in unknown)
    if len(unknown) == 1:
        message = f"SynthesiaVideoService() got an unexpected keyword argument {names}"
    else:
        message = f"SynthesiaVideoService() got unexpected keyword arguments {names}"
    # the keywords a developer is likely to try, from the LiveKit plugin or by guessing
    known_mistakes = {
        "join_timeout": "instead of join_timeout, use launch_timeout, which bounds the whole launch",
        "avatar_id": "instead of avatar_id, use avatar_ids",
    }
    hints = [known_mistakes[name] for name in unknown if name in known_mistakes]
    raise TypeError("; ".join([message, *hints]))


def _check_room(room: object) -> None:
    """Raise ``TypeError`` unless ``room`` is an ``AvatarRoom``, naming the two the package provides.

    Named in the message only: importing them would make the core depend on ``livekit`` (*LiveKit-free
    core*).
    """
    if isinstance(room, AvatarRoom):
        return
    got = "None" if room is None else type(room).__name__
    raise TypeError(
        "room must be a LiveKitTransportRoom (direct join) or a PrivateLiveKitRoom (relayed join), both "
        f"from pipecat_synthesia.livekit; got {got}"
    )


class SynthesiaVideoService(AIService):
    """Render a Synthesia Interactive Avatar for a Pipecat pipeline.

    Place it right after the TTS service. It consumes ``TTSAudioRawFrame`` s (the avatar speaks them) and
    forwards everything else; how the avatar's own audio and video reach the user is decided by the
    ``room`` object: a direct join publishes it straight to the user, and a relayed join relays it
    through the pipeline as frames.
    """

    Settings = SynthesiaVideoSettings
    _settings: SynthesiaVideoSettings

    def __init__(
        self,
        *,
        avatar_ids: Sequence[str],
        room: AvatarRoom,
        api_key: str | None = None,
        api_url: str | None = None,
        allow_any_api_host: bool | None = None,
        launch_timeout: float = 90.0,
        required: bool = True,
        session: aiohttp.ClientSession | None = None,
        settings: SynthesiaVideoSettings | None = None,
        **kwargs,
    ) -> None:
        # Checked before anything else, so a mistake in the call is reported as itself (*misuse vs session
        # errors*); kwargs is passed on to Pipecat only once it holds nothing Pipecat would drop.
        _reject_unknown_keywords(kwargs)
        _check_room(room)
        # launch_timeout bounds the whole launch: the room connect, the session request with its retries,
        # and the join wait (*launch timeout*). The LiveKit plugin's join_timeout bounds only the last two.
        self._config = ServiceConfig.from_kwargs(
            avatar_ids=avatar_ids,
            api_key=api_key,
            api_url=api_url,
            launch_timeout=launch_timeout,
            required=required,
            allow_any_api_host=allow_any_api_host,
        )
        # Pipecat's settings store holds no NOT_GIVEN values; model=None means "unsupported".
        store = self.Settings(model=None, avatar_id=self._config.avatar_ids[0])
        if settings is not None:
            store.apply_update(settings)
        super().__init__(settings=store, **kwargs)
        for event_name in REGISTERED_EVENTS:
            self._register_event_handler(event_name)

        self._room = room
        # `_session_metrics`, not `_metrics`: FrameProcessor.__init__ already owns `self._metrics`
        # (its FrameProcessorMetrics) and setup()/cleanup() call methods on it.
        self._session_metrics = SessionMetrics()
        # The root of every logger the service, its room adapter and its session use; one patcher prints the
        # session id on each of their lines (see `_print_session_id`).
        self._log = get_logger(service=self.name, avatar=room.avatar_identity).patch(self._print_session_id)
        if api_url_outside_synthesia(self._config.api_url):
            # Without the URL, in case a secret was pasted into the wrong setting (*redaction rule*).
            self._log.warning(
                "the Synthesia API URL points outside synthesia.io, allowed by allow_any_api_host or "
                "SYNTHESIA_ALLOW_ANY_API_HOST: the API key and a LiveKit token for each session are sent "
                "to it"
            )
        # Any pipecat-ai 1.x is allowed (pyproject.toml), so say once when this one is a newer minor than
        # every tested one.
        if (version := untested_newer_pipecat()) is not None:
            warn_once(
                self._log,
                "service:untested-pipecat",
                f"pipecat-ai {version} is newer than any Pipecat this pipecat-synthesia ({__version__}) was "
                f"tested with ({', '.join(TESTED_PIPECAT_VERSIONS)}). It is allowed, and the nightly "
                "compatibility job tests Pipecat's newest release, but a Pipecat minor can change pipeline "
                "startup and transport behaviour (1.13 did); if the avatar misbehaves, report the Pipecat "
                "version, or pin pipecat-ai in your app.",
            )

        # Everything below is initialised here so _teardown() is safe on an object that was never set up:
        # Pipecat may call cleanup() on a half-built processor. No create_task here.
        self._state = ServiceState.INIT
        self._generation = 0
        self._stale_contexts: deque[str] = deque(maxlen=16)
        # the end of the audio a pre-launch interruption kept; None until one finds audio queued (see
        # _keep_prelaunch_audio)
        self._prelaunch_kept: AudioChunk | UtteranceEnd | None = None
        self._current_context: str | None = None
        self._transport_ready = False
        self._speaking = False
        self._stop_timer: asyncio.TimerHandle | None = None
        self._speaking_timer_task: asyncio.Task | None = None
        self._active_avatar_id: str = self._config.avatar_ids[0]
        # A constructor `settings.avatar_id` other than the first id is a pending swap, applied once the
        # session is active, like an avatar_id settings change during the launch.
        self._pending_swap: str | None = None
        #: One swap at a time (see `swap_avatar`).
        self._swap_lock = asyncio.Lock()
        if is_given(store.avatar_id) and store.avatar_id not in (None, self._active_avatar_id):
            self._pending_swap = store.avatar_id
        self._session: AvatarSession | None = None
        self._api: SessionApiClient | None = None
        self._launch_task: asyncio.Task | None = None
        # The session room.launch() returned, from that moment until _launch takes it over or closes it, so
        # that a cancel landing in between cannot drop it unclosed (see _launch_and_hold).
        self._launched_session: AvatarSession | None = None
        self._send_task: asyncio.Task | None = None
        # The chunk the send loop has taken off the send queue and is sending, so that retiring the loop can
        # put it back for the voice-only handover (see _retire_send_loop). None between sends.
        self._in_flight: AudioChunk | None = None
        self._interrupt_task: asyncio.Task | None = None
        # a leave's wait for the last human to leave (*no-humans-left window*); the teardown cuts it short
        self._no_humans_left_window_task: asyncio.Task[bool] | None = None
        self._teardown_task: asyncio.Task | None = None
        self._send_queue: SendQueue | None = None
        self._teardown_future: asyncio.Future[None] | None = None
        # the close of a session whose request failed after an *early start*, run in a task so the voice-only
        # bot does not wait for it; the teardown does
        self._failed_session_close: asyncio.Task | None = None
        # Session closes that carry on without a waiter, each with the moment its backstop runs out, held
        # until they finish (the loop keeps only weak refs); the teardown waits for them until then.
        self._detached_closes: dict[asyncio.Future[None], float] = {}
        self._session_active = asyncio.Event()
        self._drained = asyncio.Event()
        self._playback_idle = asyncio.Event()
        self._playback_idle.set()
        self._finished = False
        # The end's error while it is still to be pushed; set once it no longer will be (pushed, dropped, or
        # none); and whether each end event has fired: each happens at most once, and a later _end_session
        # completes an end that was cut short (see _end_session).
        self._end_error_due: SynthesiaError | None = None
        self._end_error_dispatched = asyncio.Event()
        self._session_ended_fired = False
        self._metrics_fired = False
        self._stopping = False
        self._ready_emitted = False
        # on_avatar_connected has fired; SESSION_ACTIVE can precede it by the 201 (*early start*)
        self._connected_emitted = False
        self._avatar_disconnected_emitted = False
        self._left_during_launch: str | None = None
        # TTS audio that arrived in a terminal state while a voice-only handover was pending, with its
        # direction, in arrival order; pushed right after the handover. None when no handover is pending.
        self._held_for_handover: list[tuple[TTSAudioRawFrame, FrameDirection]] | None = None
        self._interrupt_started: float | None = None
        # whether the send loop has sent a chunk since the last interruption (see _on_playback_finished)
        self._sent_since_interruption = False
        self._audio_out_sample_rate: int | None = None
        self._media_dropped = 0
        self._owns_http_session = False
        self._http_session: aiohttp.ClientSession | None = session

    # ------------------------------------------------------------------ public, read-only

    @property
    def session_id(self) -> str | None:
        # from the 201 (on_session_created): None until it arrives, which after an *early start* is just
        # after the session is active
        return self._session_metrics.session_id if self._session is not None else None

    @property
    def avatar_identity(self) -> str:
        return self._room.avatar_identity

    @property
    def active_avatar_id(self) -> str:
        return self._active_avatar_id

    @property
    def metrics(self) -> SessionMetrics:
        return self._session_metrics

    @property
    def state(self) -> ServiceState:
        return self._state

    async def swap_avatar(self, avatar_id: str, *, timeout: float | None = None) -> str:  # noqa: ASYNC109
        """Swap the active avatar, and return its avatar id.

        ``timeout`` bounds the swap, in seconds, including any wait for the avatar to become ready; ``None``
        means the service's default (15 s). It must be positive: ``0``, a negative number or NaN raises
        ``ValueError``, a mistake in the call rather than a failed swap (*misuse vs session errors*). The
        swap can take up to about one second longer than ``timeout``: the swap request to the avatar worker
        always gets at least 0.5 s, even when the readiness wait used up the budget, and the bound around
        that request is 0.5 s longer than the response timeout it gives LiveKit, because neither of
        LiveKit's RPC timeouts bounds the whole call. To answer, the avatar worker gets what is left of
        ``timeout`` after any readiness wait, minus two seconds, at least half of that and at least one
        second. A swap it answers later than that fails with ``TIMEOUT``, and an acknowledgement slower than
        two seconds also fails the swap with ``TIMEOUT`` (LiveKit's connection-timeout code 1501 rather than
        1502), which the app may retry.

        Swaps run one at a time, in the order they are asked for, so that a pending swap still waiting for
        its avatar to be ready cannot be applied after a later swap and undo it. The cost: a swap waits
        behind an earlier one, including that one's wait for its avatar to be ready. ``timeout`` bounds this
        wait too; a swap that is still waiting when it runs out fails with a ``TIMEOUT`` error, and one that
        gets its turn has what is left of ``timeout``.

        The only place a swap failure is reported: a ``SynthesiaError`` from the session, or that ``TIMEOUT``,
        is pushed once as a non-permanent ``ErrorFrame`` (the true category stays on
        ``ErrorFrame.exception``) and re-raised. Called while no session is active, it raises without
        pushing.
        """
        if timeout is not None and not timeout > 0:  # not `<= 0`: NaN fails every comparison
            raise ValueError("timeout must be positive, or None for the default")
        # not before on_avatar_connected, as documented, even though *early start* makes the session active
        # up to the 201 sooner
        if (
            self._state is not ServiceState.SESSION_ACTIVE
            or self._session is None
            or not self._connected_emitted
        ):
            raise self._with_ids(SynthesiaError("avatar session is not active", type=ErrorType.CONNECTION))
        if timeout is None:
            timeout = self._config.swap_timeout
        queued_at = monotonic() if self._swap_lock.locked() else None
        try:
            async with asyncio.timeout(timeout):
                await self._swap_lock.acquire()
        except TimeoutError:
            err = SynthesiaError(
                f"avatar swap to {avatar_id} timed out after {timeout}s waiting for an earlier swap",
                type=ErrorType.TIMEOUT,
                session_id=self.session_id,
            )
            await self._push_synthesia_error(
                err, unusable=False, message=f"avatar swap to {avatar_id} failed: {err}"
            )
            raise err from None
        try:
            if queued_at is not None:
                timeout -= monotonic() - queued_at  # what the wait left of the budget
            return await self._swap_in_turn(avatar_id, timeout)
        finally:
            self._swap_lock.release()

    async def _swap_in_turn(self, avatar_id: str, timeout: float) -> str:  # noqa: ASYNC109
        """`swap_avatar` once it is this swap's turn; the session may have ended while it waited."""
        if self._state is not ServiceState.SESSION_ACTIVE or self._session is None:
            raise self._with_ids(SynthesiaError("avatar session is not active", type=ErrorType.CONNECTION))
        try:
            new = await self._session.swap_avatar(avatar_id, timeout=timeout)
        except SynthesiaError as err:
            await self._push_synthesia_error(
                err, unusable=False, message=f"avatar swap to {avatar_id} failed: {err}"
            )
            raise
        self._active_avatar_id = new
        self._settings.avatar_id = new  # direct write, not via _update_settings (avoid recursion)
        self._log.info(f"active avatar is now {new}")
        return new

    # ------------------------------------------------------------------ the two sole-owner helpers

    def _with_ids(self, err: SynthesiaError) -> SynthesiaError:
        """Stamp the session and request ids of the Synthesia API's 201 on ``err`` where it has none."""
        return err.with_session(self._session_metrics.session_id).with_request(
            self._session_metrics.request_id
        )

    async def _push_synthesia_error(
        self, err: SynthesiaError, *, unusable: bool, message: str | None = None
    ) -> None:
        """The only place ``push_error`` is called (*permanent-error rule*).

        Pipecat marks a processor unusable for any permanent category, whatever ``force_treat_as_permanent``
        says (Pipecat 1.14 ``frame_processor.py:1007-1008``), and only a failed launch with ``required=True``
        may do that. Every other error is pushed with a permanent category downgraded to ``APPLICATION``;
        the true type and category stay on ``ErrorFrame.exception``. Never call ``set_usable(True)`` after a
        push: it races Pipecat's own handling of the error.

        Once the 201 has arrived, every error pushed carries both of its ids, whichever layer raised it.
        """
        self._with_ids(err)
        true_category = err.category or ErrorCategory.UNKNOWN
        if unusable or not true_category.is_permanent:
            category = true_category
        else:
            category = ErrorCategory.APPLICATION
        await self.push_error(
            message or str(err), exception=err, category=category, force_treat_as_permanent=unusable
        )

    async def _end_session(
        self, reason: str, *, error: SynthesiaError | None = None, unusable: bool = False
    ) -> None:
        """Record the end of the session, once: the first caller wins.

        If the pipeline continues after an error, first make the voice-only handover (see
        :meth:`_hand_over`), so the bot keeps talking. Then, with ``error``, push it (see
        :meth:`_push_synthesia_error`). An ``unusable`` error is a failed launch with ``required=True``,
        which ends the pipeline, so its queued audio is dropped. Without ``error``, this is a normal end,
        and leftover audio is dropped. Then ``on_session_ended(reason)``, ``on_metrics``, and one summary
        log line.

        The first caller records the end before its first await, but may not get to fire the events.
        Pipecat runs the app's ``on_error`` handlers inline, one after another, in the task that pushes the
        error, so the pipeline can end while one of them awaits; when that task is the launch, the teardown
        cancels it. So the teardown first waits, up to ``ON_ERROR_WAIT_SECS``, for the error to be dispatched
        (``_end_error_dispatched``: the push has returned, or there is none), so that every handler sees it
        before the end events. An error not pushed by then is dropped when the wait runs out (see
        :meth:`_drop_end_error`), so that the end's own task, if it keeps running (a leave's callback
        task, which nothing cancels), cannot push it later, behind ``on_session_ended``. Then the
        teardown's own call completes the end: it fires each event not yet fired, each at most once, with
        the reason recorded first. It never pushes the error, which would run the app's ``on_error`` inside
        the teardown; it drops one still to be pushed, as a backstop. Nor is delivery cancel-proof: holding
        back a cancel for an app handler the service does not control would unbound the teardown.
        """
        if self._finished:
            self._drop_end_error()
            await self._fire_end_events()
            return
        self._finished = True
        if self._session_metrics.session_id is None:
            # Before the Synthesia API's 201, the avatar publishing in the room may be a stale one with the
            # same identity, so the video it published is not this launch's. Cleared here, where every end
            # passes, and not only where a launch fails: a pipeline that ends during the launch cancels it,
            # `_launch` re-raises the cancellation, and `_launch_failed` never runs. An *early start* tells
            # this launch's avatar by a per-launch token attribute, but does not move this boundary: no 201,
            # no metric, even when the video was this launch's.
            self._session_metrics.avatar_video_published = None
        self._end_error_due = error
        if error is None:
            self._end_error_dispatched.set()
        self._session_metrics.end_reason = reason
        self._session_metrics.ended = monotonic()
        if error is not None and not unusable:
            # Hold later TTS audio from now, before the first await: the state is already terminal, so that
            # audio would otherwise pass straight through, ahead of the older audio being handed over. An
            # unexpected leave started the hold earlier still.
            if self._held_for_handover is None:
                self._held_for_handover = []
            await self._hand_over()
        else:
            # Leftover audio is dropped, including what an unexpected leave that met the pipeline's end held
            # for a handover it left to this end (see _on_avatar_left).
            self._held_for_handover = None
            if self._send_queue is not None:
                self._send_queue.clear()
        if error is not None and self._end_error_due is error:
            self._end_error_due = None
            try:
                # returns once every on_error handler has run: Pipecat runs them inline
                await self._push_synthesia_error(error, unusable=unusable)
            finally:
                self._end_error_dispatched.set()
        await self._fire_end_events()

    def _drop_end_error(self) -> None:
        """Drop the end's error if it is still to be pushed, with one INFO line, and mark it dispatched.

        The app has already ended the pipeline, so the error has lost its purpose (telling the app to react),
        and on_session_ended carries the fact. SessionMetrics records no error, only the end reason, so
        nothing else is owed it. Logged at INFO so a reader sees what did not arrive.
        """
        dropped, self._end_error_due = self._end_error_due, None
        self._end_error_dispatched.set()
        if dropped is not None:
            self._log.info(
                f"session ended ({self._session_metrics.end_reason}) as the pipeline ended; its error is "
                f"not pushed, on_session_ended carries the reason: {describe_exception(dropped)}"
            )

    async def _fire_end_events(self) -> None:
        """``on_session_ended`` and ``on_metrics`` with the summary line, each unless it has already fired.

        Each is marked fired before its first await, so it fires at most once, however many callers get here
        and whichever of them is cancelled (see :meth:`_end_session`).
        """
        if not self._session_ended_fired:
            self._session_ended_fired = True
            await self._call_event_handler(EVENT_SESSION_ENDED, self._session_metrics.end_reason)
        if not self._metrics_fired:
            self._metrics_fired = True
            await self._call_event_handler(EVENT_METRICS, self._session_metrics)
            # the values in the message, which loguru's default format prints, and bound for a structured sink
            summary = self._session_metrics.summary()
            self._log.info("synthesia session summary {}", summary, **summary)

    async def _hand_over(self) -> None:
        """The voice-only handover: the send queue's audio, then the TTS audio held behind it, in order.

        Queued audio goes to the transport as ``TTSAudioRawFrame`` s, at the front of the held frames, which
        go unchanged; one frame is pushed at a time, so an interruption can drop the rest (see
        ``process_frame``). Each push can yield, so this runs until both are empty, and then lifts the hold
        with no await after the last check: later TTS audio passes straight through again, never ahead of
        older audio.
        """
        try:
            while True:
                held = self._held_for_handover
                if held is None:
                    return
                queued = self._send_queue.take_all() if self._send_queue is not None else []
                held[0:0] = [
                    (
                        TTSAudioRawFrame(
                            audio=item.audio,
                            sample_rate=item.sample_rate,
                            num_channels=item.num_channels,
                            context_id=item.context_id,
                        ),
                        FrameDirection.DOWNSTREAM,
                    )
                    for item in queued
                    if isinstance(item, AudioChunk)
                ]
                if not held:
                    return
                frame, direction = held.pop(0)
                await self.push_frame(frame, direction)
        finally:
            self._held_for_handover = None

    # ------------------------------------------------------------------ Pipecat lifecycle

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Create the send queue, HTTP session and launch context, then start the launch in a task.

        Pipecat gives all processors' ``setup()`` a shared 20 s, so nothing here waits on the network: the
        launch runs in ``_launch``, bounded by ``launch_timeout``.
        """
        await super().setup(setup)
        cfg = self._config
        self._audio_out_sample_rate = setup.audio_out_sample_rate
        channels = 1
        self._send_queue = SendQueue(
            max_bytes=int(cfg.prelaunch_max_seconds * setup.audio_out_sample_rate * channels * 2)
        )
        self._owns_http_session = self._http_session is None
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        # this service's client identification; a pipeline links its processors before any setup()
        client = dataclasses.replace(
            build_client_info(), join_mode=self._room.join_mode.value, transport=transport_name(self)
        )
        self._api = SessionApiClient(
            api_key=cfg.api_key,
            api_url=cfg.api_url,
            client=client,
            http_session=self._http_session,
            log=self._log,
        )
        relayed = self._room.join_mode is JoinMode.RELAYED
        callbacks = AvatarCallbacks(
            on_playback_started=self._on_playback_started,
            on_playback_finished=self._on_playback_finished,
            on_playback_lost=self._on_playback_lost,
            on_avatar_status=self._on_avatar_status,
            on_avatar_left=self._on_avatar_left,
            on_track_published=self._on_track_published,
            on_video_frame=self._on_video_frame if relayed else None,
            on_audio_frame=self._on_audio_frame if relayed else None,
            on_session_created=self._on_session_created,
        )
        ctx = LaunchContext(
            api=self._api,
            avatar_ids=cfg.avatar_ids,
            launch_timeout=cfg.launch_timeout,
            audio_out_sample_rate=setup.audio_out_sample_rate,
            audio_out_channels=channels,
            client=client,
            callbacks=callbacks,
            log=self._log,
        )
        self._session_metrics.launch_started = monotonic()
        self._state = ServiceState.LAUNCHING
        self._launch_task = self.create_task(self._launch(ctx), name="synthesia-launch")
        self._send_task = self.create_task(self._send_loop(self._generation), name="synthesia-send")

    async def start(self, frame: StartFrame) -> None:
        """One-shot configuration checks only; never waits on the room."""
        await super().start(frame)
        # (a) direct join: the developer must place the avatar audio filter, because the service never sees
        # transport.input() frames (*self-hearing guard*).
        if self._room.join_mode is JoinMode.DIRECT and getattr(self._room, "filter_installed", True) is False:
            warn_once(
                self._log,
                "service:filter-missing",
                "direct join without AvatarAudioFilter: place room.input_filter() right after "
                "transport.input() or the bot will transcribe the avatar",
            )
        # (b) relayed join: the output transport must be configured for the worker's video. Read through
        # output_transport_params (*Pipecat-internals rule*), which never raises; never walk self.next or
        # read a private here.
        if self._room.join_mode is JoinMode.RELAYED:
            params = output_transport_params(self)
            if params is not None:
                size = (params.video_out_width, params.video_out_height)
                ok = (
                    params.video_out_enabled
                    and params.video_out_is_live
                    and size == (WORKER_VIDEO_WIDTH, WORKER_VIDEO_HEIGHT)
                    and params.video_out_framerate >= WORKER_VIDEO_FPS
                )
                if not ok:
                    warn_once(
                        self._log,
                        "service:video-out-params",
                        "output transport is not configured for the Synthesia avatar video (got "
                        f"enabled={params.video_out_enabled}, live={params.video_out_is_live}, "
                        f"size={size}, fps={params.video_out_framerate}); set TransportParams("
                        f"video_out_enabled=True, video_out_is_live=True, "
                        f"video_out_width={WORKER_VIDEO_WIDTH}, video_out_height={WORKER_VIDEO_HEIGHT}, "
                        f"video_out_framerate={WORKER_VIDEO_FPS})",
                    )

    async def stop(self, frame: EndFrame) -> None:
        """Graceful end: let the last utterance drain and play (bounded), then tear down."""
        self._stopping = True  # first, before any await: _on_avatar_left relies on it
        await super().stop(frame)
        if (
            self._state is ServiceState.SESSION_ACTIVE
            and self._session is not None
            and self._session.is_alive  # else the send loop has stopped sending: nothing would drain
            and self._send_queue is not None
        ):
            deadline = monotonic() + self._config.stop_drain_timeout
            self._drained.clear()
            self._send_queue.put(UtteranceEnd(None, final=True))
            try:
                await asyncio.wait_for(self._drained.wait(), max(0.0, deadline - monotonic()))
                await asyncio.wait_for(self._playback_idle.wait(), max(0.0, deadline - monotonic()))
            except TimeoutError:
                self._log.debug(
                    f"stop drain timed out after {self._config.stop_drain_timeout:.1f}s; tearing down"
                )
        await self._teardown("pipeline_ended")

    async def cancel(self, frame: CancelFrame) -> None:
        """Immediate end: stop the avatar now; teardown continues in a task so the CancelFrame is not held."""
        self._stopping = True  # first, before any await: _on_avatar_left relies on it
        await super().cancel(frame)
        if self._state is ServiceState.SESSION_ACTIVE:
            await self._handle_interruption()
        if self._state not in _TERMINAL_STATES:
            self._state = ServiceState.ENDED
        if self._teardown_future is None:
            # Reserve the single-flight slot now: a stop()/cleanup() that arrives before this task gets its
            # first turn must join this teardown, not start its own with a different reason.
            future = self._reserve_teardown()
            self._teardown_task = self.create_task(
                self._run_teardown("pipeline_cancelled", future), name="synthesia-teardown"
            )

    async def cleanup(self) -> None:
        """Release everything; joins a teardown already running (single flight)."""
        await self._teardown("pipeline_cleanup")
        await super().cleanup()

    # ------------------------------------------------------------------ *frame contract*

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # AIService dispatches start/stop/cancel first
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputTransportReadyFrame):
            # travels upstream; the only transport-ready signal (there is no fallback)
            self._mark_transport_ready()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            self._on_tts_started(frame.context_id)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            if self._state is ServiceState.FAILED:
                # the pipeline is ending; audio already in flight must not play before the EndFrame
                self._log.trace("dropping TTS audio: the avatar failed to start")
            elif self._state in _TERMINAL_STATES:
                if self._held_for_handover is not None:
                    self._held_for_handover.append((frame, direction))  # behind the voice-only handover
                else:
                    await self.push_frame(frame, direction)  # voice-only: the transport publishes it
            elif self._is_stale(frame.context_id):
                self._log.trace(f"dropping stale TTS audio (context {frame.context_id})")
            else:
                self._enqueue_audio(frame)  # consumed, not forwarded; never awaits the network
        elif isinstance(frame, TTSStoppedFrame):
            # a stale context's late stop would end the next utterance mid-stream
            if self._state not in _TERMINAL_STATES and not self._is_stale(frame.context_id):
                self._enqueue_end_utterance(frame.context_id)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            if self._state not in _TERMINAL_STATES:
                await self._handle_interruption()
            elif self._held_for_handover is not None:
                # Drop the audio still waiting for the voice-only handover, as passing it through would have
                # let the interruption clear it downstream.
                self._held_for_handover.clear()
                if self._send_queue is not None:
                    self._send_queue.clear()
            await self.push_frame(frame, direction)
        elif isinstance(frame, SynthesiaVideoUpdateSettingsFrame):
            # Handled as Pipecat's TTSService does: forward it if it targets another service; otherwise apply
            # the delta (or the deprecated dict) and consume it.
            if frame.service is not None and frame.service is not self:
                await self.push_frame(frame, direction)
            elif frame.delta is not None:
                await self._update_settings(frame.delta)
            elif frame.settings:
                delta = type(self._settings).from_mapping(frame.settings)
                await self._update_settings(delta)
        else:
            await self.push_frame(frame, direction)  # EndFrame, CancelFrame, everything else

    async def _update_settings(self, delta: SynthesiaVideoSettings) -> dict[str, Any]:
        if not self.is_usable:
            # AIService._update_settings revives the processor (set_usable(True)) on any change, which is
            # wrong after a required launch failure (*permanent-error rule*): the avatar will never render.
            # Record the change without the revive, and keep avatar_id truthful.
            changed = self._settings.apply_update(delta)
            if "model" in changed:
                self._sync_model_name_to_metrics()
            if "avatar_id" in changed:
                self._settings.avatar_id = self._active_avatar_id
            return changed
        changed = await super()._update_settings(delta)
        if "avatar_id" in changed:
            new = self._settings.avatar_id
            if is_given(new) and new is not None:
                if self._state in (ServiceState.INIT, ServiceState.LAUNCHING) or (
                    self._state is ServiceState.SESSION_ACTIVE and not self._connected_emitted
                ):
                    # a pending swap: _launch applies it once the session is active and connected
                    self._pending_swap = new
                elif self._state is ServiceState.SESSION_ACTIVE:
                    try:
                        await self.swap_avatar(new)
                    except SynthesiaError:
                        # already reported inside swap_avatar; do not push again; do not re-raise (a raise
                        # from process_frame would produce a second UNKNOWN ErrorFrame)
                        self._settings.avatar_id = self._active_avatar_id
                else:
                    # terminal: ignore; keep the settings truthful about the avatar that was active
                    self._settings.avatar_id = self._active_avatar_id
        return changed

    # ------------------------------------------------------------------ frame-path helpers

    def _on_tts_started(self, context_id: str | None) -> None:
        if self._is_stale(context_id):
            # A late start of a stale context begins no utterance that will play: the context streaming now
            # stays current, so the next interruption still marks it stale.
            return
        self._current_context = context_id

    def _is_stale(self, context_id: str | None) -> bool:
        return context_id is not None and context_id in self._stale_contexts

    def _enqueue_audio(self, frame: TTSAudioRawFrame) -> None:
        if self._send_queue is None:
            return
        dropped = self._send_queue.put(
            AudioChunk(frame.audio, frame.sample_rate, frame.num_channels, frame.context_id)
        )
        if dropped:
            self._session_metrics.prelaunch_dropped_bytes += dropped
            warn_once(
                self._log,
                "service:prelaunch-full",
                "pre-launch buffer full; dropping the oldest audio while the avatar launches "
                "or audio is held",
            )

    def _enqueue_end_utterance(self, context_id: str | None) -> None:
        if self._send_queue is not None:
            self._send_queue.put(UtteranceEnd(context_id))

    def _mark_transport_ready(self) -> None:
        if not self._transport_ready:
            self._transport_ready = True
            self._session_metrics.transport_ready = monotonic()

    async def _handle_interruption(self) -> None:
        """Runs on Pipecat's input task; keep it short. Never creates a send loop here."""
        if self._current_context is not None:
            self._stale_contexts.append(self._current_context)
        self._current_context = None
        if self._send_queue is not None:
            if self._state in (ServiceState.INIT, ServiceState.LAUNCHING):
                self._keep_prelaunch_audio(self._send_queue)
            else:
                self._send_queue.clear()
        # The chunk being sent is dropped with the queue: a leave while this yields must not put it back.
        self._in_flight = None
        # the worker may emit no playback_finished if nothing had played
        self._playback_idle.set()
        if self._state is ServiceState.SESSION_ACTIVE and self._session is not None:
            # Retire the old send loop (bump the generation), and record the interruption, before the first
            # await (*state-before-await*): a playback event can arrive at any await. A chunk of the next
            # utterance queued while this yields must be sent by the loop started after clear buffer, not by
            # the old loop before it, where clear buffer would wipe it.
            self._generation += 1
            self._interrupt_started = monotonic()
            self._sent_since_interruption = False
            old_send_task, self._send_task = self._send_task, None
            if old_send_task is not None:
                await self.cancel_task(old_send_task, timeout=_TASK_CANCEL_SECS)
            await self._stop_speaking_now()
            if self._interrupt_task is not None and not self._interrupt_task.done():
                await self.cancel_task(self._interrupt_task, timeout=_TASK_CANCEL_SECS)
            self._interrupt_task = self.create_task(
                self._interrupt_and_resume(self._generation), name="synthesia-interrupt"
            )
        # INIT/LAUNCHING: the send loop is parked on _session_active.wait(); nothing has reached the worker.

    def _keep_prelaunch_audio(self, queue: SendQueue) -> None:
        """A *pre-launch interruption*: nothing has played, so the first audio held when the user spoke
        stays, to play once the session is active. The rest of the interrupted utterance is stale, and its own
        late end is ignored, so the part already queued is ended here, or the next utterance would join its
        byte stream. A later pre-launch interruption drops what was queued since: it answered a turn this one
        supersedes, as an interruption would drop it once the session is active.

        The plugin cannot tell a greeting from a reply, so the kept audio is whatever was queued first: an
        interruption that finds no audio queued keeps nothing and sets no boundary; the next one tries again.
        """
        if self._prelaunch_kept is not None:
            dropped = queue.truncate_after(self._prelaunch_kept)
            if dropped:
                self._log.debug(f"user turn superseded {dropped} bytes of pre-launch audio; dropping them")
            return
        if not queue.bytes:
            # no audio held, only utterance ends at most (a TTS request that timed out queues a lone end)
            return
        if isinstance(queue.last, AudioChunk):
            queue.put(UtteranceEnd(None))
        self._prelaunch_kept = queue.last  # an UtteranceEnd, which the queue never drops
        self._log.info(
            f"user turn started before the avatar session is active; keeping {queue.bytes} bytes of "
            "pre-launch audio, to play once it is"
        )

    async def _interrupt_and_resume(self, gen: int) -> None:
        session = self._session
        if session is None:
            return
        # never raises: clear buffer and the utterance close each swallow their own failures
        await session.interrupt()
        if gen == self._generation and self._state is ServiceState.SESSION_ACTIVE:
            self._send_task = self.create_task(self._send_loop(gen), name="synthesia-send")

    # ------------------------------------------------------------------ launch

    async def _launch(self, ctx: LaunchContext) -> None:
        """Launch the session and make it active, unless the pipeline is already ending.

        The pipeline can end in the same loop turn as the launch's outcome. A ``cancel()`` only starts the
        teardown, which is what cancels this task, so the launch can still finish first. Then:

        * a session the launch produced is closed, never made active, whichever way the end arrives;
        * a launch that failed is not reported as a failed launch: the end is the pipeline's, with its own
          reason, and pushes no error and no ``EndWorkerFrame``.
        """
        deadline = monotonic() + self._config.launch_timeout
        try:
            # the *launch timeout*: wait_for cancels launch(), which rolls back after itself
            session = await asyncio.wait_for(self._launch_and_hold(ctx), self._config.launch_timeout)
        except asyncio.CancelledError:
            # A cancel that raced the launch's return. With Pipecat's wait_for2 (below Python 3.12) it
            # carries the session instead of returning it; _launch_and_hold kept it, so close it here,
            # whoever cancelled. Bounded and cancel-proof: a second cancel leaves the close detached.
            launched, self._launched_session = self._launched_session, None
            if launched is not None:
                await self._close_session(launched, what="closing the session a cancelled launch produced")
            raise
        except Exception as exc:
            await self._launch_failed(self._as_launch_error(exc))
            return

        self._launched_session = None  # from here on, this task owns it
        if self._stopping:
            # closed here, bounded: this session is never stored, so teardown could not close it later
            await self._close_session(session, what="closing the session that launched as the pipeline ended")
            return

        if self._left_during_launch is not None or not session.is_alive:
            # The 201's session id, if it has arrived: after an *early start* the session has none to give.
            session_id = self._session_metrics.session_id
            if session_id is None and session.timings.session_created is not None:
                session_id = session.session_id  # an adapter that never reported its 201 (on_session_created)
            # closed here, bounded: this session is never stored, so teardown could not close it later
            await self._close_session(session, what="closing the session that left during launch")
            await self._launch_failed(
                SynthesiaError(
                    f"avatar left during launch ({self._left_during_launch})",
                    type=ErrorType.CONNECTION,
                    category=ErrorCategory.CONNECTIVITY,
                    session_id=session_id,
                )
            )
            return

        self._session = session
        self._active_avatar_id = session.active_avatar_id
        self._state = ServiceState.SESSION_ACTIVE
        # the pre-launch interruption's boundary means nothing from here on (see _keep_prelaunch_audio)
        self._prelaunch_kept = None
        # the join, recorded now: it needs no session id, and the session may end before the 201
        timings = session.timings
        metrics = self._session_metrics
        metrics.room_connected = timings.room_connected
        metrics.session_requested = timings.session_requested
        metrics.avatar_joined = timings.avatar_joined
        # Audio flows from here. After an *early start* the 201 is still on its way: what needs its session
        # id (the metrics' ids, the log lines, on_avatar_connected and on_avatar_ready) waits for it.
        self._session_active.set()
        try:
            # The *launch timeout* bounds the session request too, so what is left of it bounds the wait for a
            # 201 still on its way. asyncio.timeout, not wait_for: a 201 already in returns without yielding,
            # so it is never timed out, even with nothing left.
            async with asyncio.timeout(max(0.0, deadline - monotonic())):
                await session.wait_session_created()
        except Exception as exc:
            await self._failed_after_early_start(session, self._as_launch_error(exc))
            return
        if self._state is not ServiceState.SESSION_ACTIVE or self._stopping:
            # Ended while the 201 was on its way, or ending: that end stands, and there is no
            # on_avatar_connected, as for a launch that finishes once the pipeline is ending.
            return
        if (created_at := session.timings.session_created) is not None:
            metrics.session_created = created_at
        if metrics.session_id is None:  # an adapter that never reported its 201 (on_session_created)
            metrics.session_id = session.session_id
        # Already bound at the 201 (`_on_session_created`); this covers an adapter that never reports one.
        self._log = self._log.bind(session_id=session.session_id)
        self._connected_emitted = True
        await self._call_event_handler(
            EVENT_AVATAR_CONNECTED,
            AvatarConnected(
                session_id=session.session_id,
                avatar_identity=session.avatar_identity,
                active_avatar_id=session.active_avatar_id,
                room_name=session.room_name,
            ),
        )
        if self._state is ServiceState.SESSION_ACTIVE and session.all_ready:
            await self._emit_ready_once(session.avatar_status)

        # a pending swap from the launch is applied now
        pending, self._pending_swap = self._pending_swap, None
        if (
            pending is not None
            and pending != self._active_avatar_id
            and self._state is ServiceState.SESSION_ACTIVE
        ):
            try:
                await self.swap_avatar(pending)
            except SynthesiaError:
                self._settings.avatar_id = self._active_avatar_id

    async def _failed_after_early_start(self, session: AvatarSession, err: SynthesiaError) -> None:
        """The session request failed after an *early start*: the launch failed after all, and ``required``
        decides the end, as for any failed launch. But the session was active, so audio has reached the
        avatar, which may be speaking: the end runs as an unexpected leave's does (see `_on_avatar_left`), and
        the avatar is stopped first, or it would go on over the voice-only bot."""
        if self._state is not ServiceState.SESSION_ACTIVE or self._stopping:
            self._log.debug(f"the session request failed as the session ended: {describe_exception(err)}")
            return
        required = self._config.required
        # Terminal before the first await (*state-before-await*): nothing more goes to the session; an
        # interruption from here drops the queued audio, as it did while the session was active, rather than
        # keeping it as *pre-launch audio*; and later TTS audio is dropped (required) or waits behind the
        # voice-only handover _end_session makes.
        self._state = ServiceState.FAILED if required else ServiceState.DEGRADED
        self._retire_send_loop()
        if not required:
            self._held_for_handover = []
        self._session = None
        try:
            try:
                await self._stop_speaking_now()
                await session.interrupt()  # clear buffer; bounded, never raises
            finally:
                # closed in a task of its own, as after a leave: the voice-only bot does not wait for it
                self._failed_session_close = self.create_task(
                    self._close_session(
                        session, what="closing the session whose request failed after an early start"
                    ),
                    name="synthesia-session-close",
                )
            await self._launch_failed(err)
        finally:
            if self._held_for_handover is not None:
                # The handover never ran (something above raised, or the pipeline is ending): never leave the
                # bot's voice held. Even when the pipeline is ending: a graceful end plays what was queued,
                # here voice-only ahead of the EndFrame, and after a cancel Pipecat drops what is pushed. This
                # path can hold whole replies behind a clear-buffer round trip, so it must not drop them.
                await self._hand_over()

    async def _launch_and_hold(self, ctx: LaunchContext) -> AvatarSession:
        """``room.launch()``, holding its session before anything can cancel the launch again.

        No ``await`` separates the return from the hold, and whichever ``wait_for`` is in use, the hold
        happens inside the waited-for coroutine. So once a session exists, ``_launched_session`` names it
        until ``_launch`` takes it over or closes it, and the teardown closes it if neither happened.
        """
        session = await self._room.launch(ctx)
        self._launched_session = session
        return session

    def _as_launch_error(self, exc: Exception) -> SynthesiaError:
        """The ``SynthesiaError`` a failed launch reports for ``exc``."""
        if isinstance(exc, TimeoutError):
            return SynthesiaError(
                f"avatar launch did not finish within {self._config.launch_timeout}s",
                type=ErrorType.TIMEOUT,
                category=ErrorCategory.CONNECTIVITY,
            )
        if isinstance(exc, SynthesiaError):
            return exc
        return SynthesiaError(
            f"avatar launch failed: {describe_exception(exc)}",
            type=ErrorType.UNKNOWN,
            category=ErrorCategory.APPLICATION,
        )

    async def _launch_failed(self, err: SynthesiaError) -> None:
        # Once the pipeline is ending, the end is its own (see `_launch`), whatever failed while it waited.
        if self._stopping:
            self._log.debug(f"the launch failed as the pipeline ended: {describe_exception(err)}")
            return
        required = self._config.required
        if required:
            reason = "launch_timeout" if err.type is ErrorType.TIMEOUT else "launch_failed"
            self._state = ServiceState.FAILED
        else:
            reason = "degraded"
            self._state = ServiceState.DEGRADED
            self._log.warning(
                f"Synthesia avatar unavailable ({err.type.value}); continuing voice-only: {err}"
            )
        # The Synthesia API may have started a session anyway: one it accepted before the launch was
        # cancelled, or one whose avatar had not joined by the deadline. The avatar worker ends it once the
        # bot is not in its room, which covers most failures: a relayed join's rollback deletes the private
        # room, a direct join whose bot never connected was never in the transport room, and required=True
        # ends the pipeline, so the bot leaves. Not a direct join with required=False: the bot stays in the
        # transport room, so the avatar can still join it late and stay there, idle and billing, until the
        # worker ends the session itself, when the bot or the last human leaves. A known limitation: changing
        # it needs a design.
        await self._end_session(reason, error=err, unusable=required)
        if required:
            # required=True ends the pipeline itself, whatever its processor_unusable_policy. The error stays
            # permanent, so is_usable stays truthful and a CANCEL policy is honoured; under END or CANCEL the
            # pipeline still ends once. Upstream: the pipeline worker turns it into an EndFrame at the source.
            await self.push_frame(
                EndWorkerFrame(reason=f"Synthesia avatar failed to start ({reason})"),
                FrameDirection.UPSTREAM,
            )

    # ------------------------------------------------------------------ send loop

    async def _send_loop(self, gen: int) -> None:
        await self._session_active.wait()
        if gen != self._generation:
            return
        assert self._send_queue is not None  # created in setup(), before this task exists
        while True:
            item = await self._send_queue.get()
            if gen != self._generation:
                # a zombie loop after a cancel_task timeout exits instead of becoming a second writer
                warn_once(
                    self._log,
                    "service:stale-send-loop",
                    "stale send loop exited after cancel timeout; dropped one item",
                )
                return
            session = self._session
            if session is None:
                continue
            if not session.is_alive:
                # The avatar has left, and the session is closing its byte stream before it reports the
                # leave; or the session is closing. Either way nothing more can be sent, so this item and
                # everything behind it stay queued, for the voice-only handover or for the end to drop.
                self._send_queue.put_back(item)
                self._drained.set()  # a graceful stop's drain has nothing left to wait for
                return
            if isinstance(item, AudioChunk) and not session.sending:
                # Audio held after an *early start* whose session request was retried, until the request
                # settles (its 201, or its failure) or the session closes; during a graceful stop, the drain
                # timeout bounds the wait. Only audio: an utterance end, a stop's final one included, still
                # passes. The chunk stays queued, so the voice-only handover or an interruption still finds
                # it, and it counts as sent only once it is. A failure that ends the launch can wake this loop
                # just before the failure path retires it: at most one chunk then reaches the abandoned
                # avatar, which the clear buffer stops, and the retire puts back if its send was cut short.
                self._send_queue.put_back(item)
                await session.wait_sending()
                continue
            try:
                if isinstance(item, AudioChunk):
                    self._playback_idle.clear()
                    self._sent_since_interruption = True
                    self._in_flight = item
                    try:
                        await session.send_audio(item.audio, item.sample_rate, item.num_channels)
                    finally:
                        # only this loop's own record: a stale loop unwinding late must not clear a newer one
                        if self._in_flight is item:
                            self._in_flight = None
                    if not session.is_alive:
                        # The session ended during this send (the avatar left, or stopped publishing video):
                        # a session may finish a send in flight before it reports that (the LiveKit session
                        # does), so the voice-only bot must not skip this chunk. It goes back to the front, as
                        # on the path above; at worst the user hears what the avatar had begun of it twice.
                        # Not from a loop retired meanwhile: the retire has put the chunk back itself, or an
                        # interruption has dropped it.
                        if gen == self._generation:
                            self._send_queue.put_back(item)
                        self._drained.set()
                        return
                else:
                    try:
                        await session.end_utterance()
                    finally:
                        if item.final:
                            self._drained.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The last-resort guard, so the loop survives. A failed send never gets here: the session
                # drops the utterance and warns once itself. Anything that does get here is unexpected, and
                # logged once per exception type, so a bug that raises on every chunk cannot flood the log.
                warn_once(
                    self._log,
                    f"service:send-failed:{type(exc).__name__}",
                    f"audio send failed unexpectedly: {describe_exception(exc)}",
                )

    # ------------------------------------------------------------------ room callbacks (*state-before-await*)

    def _cancel_stop_timer(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    def _fire_stop_timer(self) -> None:
        self._stop_timer = None
        self.create_task(self._stop_speaking_now(), name="synthesia-bot-stopped")

    async def _speaking_timer_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.bot_speaking_interval_secs)
            if self._state is not ServiceState.SESSION_ACTIVE or not self._speaking:
                return
            await self.broadcast_frame(BotSpeakingFrame)

    async def _on_playback_started(self) -> None:
        if self._state is not ServiceState.SESSION_ACTIVE:
            return
        self._cancel_stop_timer()
        if self._speaking:
            return
        self._speaking = True
        self._playback_idle.clear()
        self._speaking_timer_task = self.create_task(
            self._speaking_timer_loop(), name="synthesia-bot-speaking"
        )
        # mutations done; awaits last
        await self.broadcast_frame(BotStartedSpeakingFrame)

    async def _on_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        if self._state is not ServiceState.SESSION_ACTIVE:
            return
        self._session_metrics.playback_finished_count += 1
        # Interruption latency is not recorded in this release (docs/protocol-compatibility.md, section 10):
        # a finish does not say which clear buffer it answers, and a chunk written before an interruption
        # may be read by the worker after its clear buffer, so no rule can tell which interruption a finish
        # measures.
        # The avatar worker sends an interrupted finish from two places. One answers our clear buffer. The
        # other answers audio it held before binding the bot and then discarded unplayed (too old, cleared
        # before the bind, sent by a bot the worker did not bind, or the room closing first): sent at the
        # bind or the close, at position 0.0, before any of our audio has played. That one needs no
        # expiry. With no interruption pending it only arms the stop timer, which ends nothing while
        # nothing is speaking, and the next playback_started cancels it. With one pending (a clear buffer
        # sent after the session became active but before the bind, which the worker answers with no
        # finish of its own), it is that clear buffer's answer, as below.
        answers_interruption = interrupted and self._interrupt_started is not None
        if interrupted:
            # only an interrupted finish answers an interruption: a normal finish of a later utterance must
            # not consume it, or the interrupted finish arriving after that would arm the stop timer
            self._interrupt_started = None
        if answers_interruption and self._sent_since_interruption:
            # That interruption already ended the speaking state, and the next utterance may be playing by
            # now (the avatar worker can send this after the next playback_started): no stop timer. With
            # nothing sent since, any playback_started since was the interrupted utterance's own, arriving
            # late, and this finish ends it like any other.
            return
        self._arm_stop_timer()

    async def _on_playback_lost(self) -> None:
        # The avatar worker left while it may have been playing, and its playback_finished will never come.
        # Armed as for a finish, not stopped at once: a replacement worker that starts playing within the
        # hysteresis keeps the bot speaking, rather than stopping and starting again.
        if self._state is not ServiceState.SESSION_ACTIVE:
            return
        self._arm_stop_timer()

    def _arm_stop_timer(self) -> None:
        self._cancel_stop_timer()
        loop = asyncio.get_running_loop()
        self._stop_timer = loop.call_later(self._config.bot_stopped_hysteresis_secs, self._fire_stop_timer)

    async def _stop_speaking_now(self) -> None:
        emit = False
        if self._speaking:
            self._speaking = False
            if self._speaking_timer_task is not None:
                self._speaking_timer_task.cancel()
                self._speaking_timer_task = None
            self._cancel_stop_timer()
            emit = True
        self._playback_idle.set()  # always, even when not speaking: stop() waits on it
        if emit:
            await self.broadcast_frame(BotStoppedSpeakingFrame)

    async def _emit_ready_once(self, status: dict[str, str]) -> None:
        if (
            self._ready_emitted
            or not self._connected_emitted  # _launch emits it after on_avatar_connected
            or self._state is not ServiceState.SESSION_ACTIVE
            or self._session is None
        ):
            return
        self._ready_emitted = True
        self._session_metrics.avatar_ready = monotonic()
        await self._call_event_handler(
            EVENT_AVATAR_READY, AvatarReady(session_id=self._session.session_id, status=dict(status))
        )

    async def _on_avatar_status(self, status: dict[str, str], all_ready: bool) -> None:
        # before SESSION_ACTIVE this returns; _launch covers the pre-SESSION_ACTIVE case
        if all_ready:
            await self._emit_ready_once(status)

    async def _on_avatar_left(self, reason: str, humans_present: bool | None) -> None:
        if self._state is ServiceState.LAUNCHING:
            self._left_during_launch = reason
            return
        if self._state is not ServiceState.SESSION_ACTIVE:
            return
        stopping = self._stopping
        if not stopping:
            self._state = ServiceState.ENDED
            # Nothing more goes to the session: the send loop must not take queued audio to it while this
            # yields. The queue is the voice-only handover's, or dropped at a no-humans-left end.
            self._retire_send_loop()
            if humans_present is not False:
                # an unexpected leave: later TTS audio waits behind the handover _end_session makes
                self._held_for_handover = []
        disconnected_event = not self._avatar_disconnected_emitted
        self._avatar_disconnected_emitted = True
        session = self._session
        shutting_down = stopping
        # mutations done; awaits from here on
        try:
            await self._stop_speaking_now()
            if session is not None:
                # release owned rooms now rather than at teardown
                self.create_task(session.aclose(), name="synthesia-session-close")
            if disconnected_event:
                await self._call_event_handler(EVENT_AVATAR_DISCONNECTED, reason)
            if humans_present and session is not None and not self._stopping:
                # A hang-up can look like an unexpected leave: the avatar worker leaves as the last human
                # does, and its leave can arrive first. Only the error and the end reason wait for the
                # human's own leave (*no-humans-left window*); the state and the hold above did not.
                humans_present = not await self._wait_no_humans_left(session)
            # Read again: the pipeline may have started ending while this awaited, after the teardown's one
            # check for an end in progress. stop(), cancel() and the teardown each set _stopping before their
            # first await, and nothing awaits from here to _end_session marking the session finished, so the
            # teardown either finds this end under way or ends the session itself.
            shutting_down = stopping or self._stopping
            if shutting_down:
                self._log.info(f"avatar left during shutdown ({reason})")
                # The teardown finishes with its own reason; its _end_session drops any audio held above.
                # Dropped, not handed over, deliberately, although a graceful end otherwise plays what was
                # queued. A leave gets here when the pipeline is already stopping while the session is still
                # active, which covers a graceful stop's whole drain (up to stop_drain_timeout; 0.1.0b1 drops
                # it the same way there), or through the second read above if an app's on_before_push_frame
                # or on_after_push_frame handler held the BotStoppedSpeakingFrame. It needs a leave in the
                # last seconds of a call, and what dropping costs over a handover is only audio not yet sent
                # to the avatar, usually little since sending runs ahead of playback (what the avatar was
                # sent leaves with it either way). A path that holds whole replies would hand them over;
                # this one keeps the drop.
                return
            if humans_present is False:
                self._log.info(f"Synthesia worker ended the session: no human participants left ({reason})")
                await self._end_session("no_humans_left")
                return
            await self._end_session(
                "avatar_left",
                error=SynthesiaError(
                    f"Synthesia avatar left the room ({reason}); continuing voice-only",
                    type=ErrorType.CONNECTION,
                    category=ErrorCategory.CONNECTIVITY,
                    session_id=self.session_id,  # None if the avatar left before the 201 (*early start*)
                ),
                unusable=False,
            )
        finally:
            if self._held_for_handover is not None and not shutting_down:
                # the handover never ran (something above raised): never leave the bot's voice held
                await self._hand_over()

    async def _wait_no_humans_left(self, session: AvatarSession) -> bool:
        """The *no-humans-left window*, in a task the teardown cancels, so a pipeline that ends during it ends
        it too. True only if the session says no human participant remains; a cancelled or failed wait is
        False, which leaves the leave unexpected (Pipecat's task wrapper logs a failure and returns None)."""
        window = self._no_humans_left_window_task = self.create_task(
            session.wait_no_humans_left(), name="synthesia-no-humans-left-window"
        )
        try:
            # asyncio.wait, not await: the teardown's cancel of the window must not cancel this leave
            await asyncio.wait({window})
        finally:
            if self._no_humans_left_window_task is window:
                self._no_humans_left_window_task = None
        if window.cancelled():
            return False
        return window.exception() is None and window.result() is True

    def _retire_send_loop(self) -> None:
        """Stop the send loop now, without waiting: cancelled, and its generation retired, so a loop that
        outlived the cancel exits rather than sending. The teardown still waits for the task.

        A chunk whose send this cancels mid-flight goes back to the front of the send queue first, so the
        voice-only handover starts with it rather than losing it."""
        self._generation += 1
        in_flight, self._in_flight = self._in_flight, None
        if in_flight is not None and self._send_queue is not None:
            # The loop is retired only once the avatar is gone or about to be stopped: it has left the room or
            # stopped publishing video (see _on_avatar_left), or the session request failed after an *early
            # start* and a clear buffer follows (see _failed_after_early_start). At worst the user hears part
            # of this chunk twice, from the avatar and then voice-only, which beats a gap in the speech.
            self._send_queue.put_back(in_flight)
        if self._send_task is not None:
            self._send_task.cancel()

    def _media_gate_open(self) -> bool:
        if self._state is ServiceState.SESSION_ACTIVE and self._transport_ready:
            return True
        self._media_dropped += 1
        if self._media_dropped % _MEDIA_DROP_TRACE_EVERY == 1:
            self._log.trace(
                f"dropping avatar media before the transport is ready ({self._media_dropped} so far)"
            )
        return False

    async def _on_video_frame(self, rgb: bytes, width: int, height: int) -> None:
        if not self._media_gate_open():
            return
        if self._session_metrics.first_video_frame is None:
            self._session_metrics.first_video_frame = monotonic()
        await self.push_frame(OutputImageRawFrame(image=rgb, size=(width, height), format="RGB"))

    async def _on_audio_frame(self, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        if not self._media_gate_open():
            return
        # Plain OutputAudioRawFrame, not a speech frame: the output transport would otherwise emit a second
        # set of bot speaking frames (*speaking-state ownership*).
        frame = OutputAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=num_channels)
        await self.push_frame(frame)

    def _print_session_id(self, record: loguru.Record) -> None:
        """Start the message with ``[<session id>]`` once the Synthesia API's 201 has set it.

        Loguru's default format prints no bound field, so the id must be in the message. It is read from the
        metrics, not from ``extra``, because the loggers derived before the 201 are never rebound. Loguru
        formats the message before patchers run, and derived loggers keep the patcher.
        """
        if (session_id := self._session_metrics.session_id) is not None:
            record["message"] = f"[{session_id}] {record['message']}"

    def _on_session_created(self, session_id: str, request_id: str | None, created_at: float) -> None:
        """The Synthesia API's 201, during the launch. The ids go on the metrics now, not at session
        active, so a launch that fails after it still reports them. The `session_id` property shows them
        once the session is active, which after an *early start* comes before them. The service's own
        log lines carry the session id from here on too."""
        metrics = self._session_metrics
        metrics.session_id, metrics.request_id, metrics.session_created = session_id, request_id, created_at
        self._log = self._log.bind(session_id=session_id)

    async def _on_track_published(self, kind: str) -> None:
        if kind == "video" and self._session_metrics.avatar_video_published is None:
            self._session_metrics.avatar_video_published = monotonic()

    # ------------------------------------------------------------------ teardown

    async def _teardown(self, reason: str) -> None:
        """Single flight: the first caller runs it, later callers wait for it; every step is bounded.

        A run cancelled part-way frees the slot as it releases its waiters, so a waiter runs the rest itself.
        """
        while (future := self._teardown_future) is not None:
            await asyncio.shield(future)
            if self._teardown_future is future:
                return
        await self._run_teardown(reason, self._reserve_teardown())

    async def _close_session(self, session: AvatarSession, *, what: str) -> None:
        """Close ``session``, waiting at most ``SESSION_ACLOSE_BACKSTOP_SECS`` (*bounded waits*). Never raises
        except ``CancelledError``.

        The close is *cancel-proof cleanup*, so it cannot be cut short: past the backstop, or if the caller
        is cancelled, it finishes detached, held in ``_detached_closes`` until it does. A cancelled caller's
        close has the rest of its backstop left, and the teardown waits for it that long.
        """
        deadline = monotonic() + SESSION_ACLOSE_BACKSTOP_SECS
        closing = asyncio.ensure_future(session.aclose())
        try:
            finished = await wait_bounded(closing, SESSION_ACLOSE_BACKSTOP_SECS)
        except asyncio.CancelledError:
            self._run_detached(closing, what=what, deadline=deadline)
            raise
        except Exception as exc:
            self._log.warning(f"{what} failed: {describe_exception(exc)}")
            return
        if not finished:
            self._run_detached(closing, what=what, deadline=deadline)
            self._log.warning(
                f"{what} is taking over {SESSION_ACLOSE_BACKSTOP_SECS}s; not waiting for it to finish"
            )

    def _run_detached(self, closing: asyncio.Future[None], *, what: str, deadline: float) -> None:
        """Keep a close alive without a waiter: hold a reference, and log its failure when it lands.

        ``deadline`` is the ``monotonic()`` time its backstop runs out: the teardown waits for it until then.
        """
        self._detached_closes[closing] = deadline

        def _done(task: asyncio.Future[None]) -> None:
            self._detached_closes.pop(task, None)
            if not task.cancelled() and (exc := task.exception()) is not None:
                self._log.warning(
                    f"{what} failed after the service stopped waiting: {describe_exception(exc)}"
                )

        closing.add_done_callback(_done)

    def _reserve_teardown(self) -> asyncio.Future[None]:
        """Claim the single teardown slot synchronously; the returned future completes when it has run."""
        loop = asyncio.get_running_loop()
        self._teardown_future = future = loop.create_future()
        return future

    async def _run_teardown(self, reason: str, future: asyncio.Future[None]) -> None:
        try:
            self._stopping = True  # first, before any await: _on_avatar_left relies on it
            if self._state not in _TERMINAL_STATES:
                self._state = ServiceState.ENDED
            if self._finished and not self._end_error_dispatched.is_set():
                # The teardown gives an end already in progress up to ON_ERROR_WAIT_SECS to push its error,
                # whether it is still handing audio over or already inside the app's handlers; past that the
                # error is dropped at INFO, here, if it was not yet pushed: dropped now, not later, because
                # the end's own task may keep running (a leave's callback task, which nothing cancels) and
                # would push it behind the end events. One already pushed is not dropped: the end events fire
                # ahead of the handlers still to run, which may run after them, or never if cancelling the
                # launch stops them. Until then nothing is cancelled and no end event fires (*bounded waits*).
                try:
                    async with asyncio.timeout(ON_ERROR_WAIT_SECS):
                        await self._end_error_dispatched.wait()
                except TimeoutError:
                    if self._end_error_dispatched.is_set():
                        pass  # dispatched after the timeout fired, before this task resumed: none held back
                    elif self._end_error_due is not None:
                        self._drop_end_error()
                    else:
                        # The end's own task may still fire the end events first, once its handlers return.
                        self._log.info(
                            f"the session's end error was not dispatched within {ON_ERROR_WAIT_SECS}s; the "
                            "end events no longer wait for it, and on_error handlers still to run may run "
                            "after them or not at all"
                        )
            if self._launch_task is not None:
                try:
                    # not the 1 s default: a second cancel must not abort the launch's rollback
                    await self.cancel_task(self._launch_task, timeout=LAUNCH_CANCEL_SECS)
                except Exception as exc:
                    self._log.warning(f"teardown: cancelling the launch failed: {describe_exception(exc)}")
                self._launch_task = None
            if self._interrupt_task is not None and not self._interrupt_task.done():
                # An interrupt under way (a cancel has always just started one): let clear buffer reach the
                # avatar worker before the task is cancelled below, or the avatar goes on speaking its
                # buffered audio. Bounded, so a hung one cannot hold the teardown (*bounded waits*).
                await asyncio.wait({self._interrupt_task}, timeout=INTERRUPTION_WAIT_SECS)
            for attr in (
                "_send_task",
                "_interrupt_task",
                "_speaking_timer_task",
                "_no_humans_left_window_task",
            ):
                task: asyncio.Task | None = getattr(self, attr)
                if task is not None:
                    try:
                        await self.cancel_task(task, timeout=_TASK_CANCEL_SECS)
                    except Exception as exc:
                        self._log.warning(f"teardown: cancelling {attr} failed: {describe_exception(exc)}")
                    setattr(self, attr, None)
            self._cancel_stop_timer()
            try:
                await self._stop_speaking_now()
            except Exception as exc:
                self._log.warning(f"teardown: stopping bot-speaking failed: {describe_exception(exc)}")
            launched, self._launched_session = self._launched_session, None
            if launched is not None:
                # A backstop: _launch takes over or closes every session it holds, unless it ended some other
                # way, such as a BaseException that is not a cancel.
                await self._close_session(launched, what="teardown: closing the launch's session")
            if self._session is not None:
                await self._close_session(self._session, what="teardown: session close")
            if self._failed_session_close is not None:
                # bounded by _close_session's own backstop; a cancel here leaves the close running detached
                await asyncio.wait({self._failed_session_close})
            # The pipeline must not finish while a session is still closing. Cancelling the launch above
            # detaches any close it had under way: the session that left during the launch, or the one that
            # launched as the pipeline ended, neither of which the teardown holds. A relayed join's close
            # deletes the private room, and a bot process that exits once its pipeline ends would cut that
            # short. So wait for every detached close, each until its own backstop runs out (*bounded waits*).
            # asyncio.wait, not wait_for: past the bound a close carries on, held in _detached_closes.
            for closing, deadline in sorted(self._detached_closes.items(), key=lambda item: item[1]):
                await asyncio.wait({closing}, timeout=max(0.0, deadline - monotonic()))
            if self._owns_http_session and self._http_session is not None:
                try:
                    await self._http_session.close()
                except Exception as exc:
                    self._log.warning(f"teardown: closing the HTTP session failed: {describe_exception(exc)}")
            try:
                await self._end_session(reason)
            except Exception as exc:
                self._log.warning(f"teardown: session-end bookkeeping failed: {describe_exception(exc)}")
        except asyncio.CancelledError:
            # Torn down only part-way: release the joiners and free the slot, so that a joiner, or a later
            # caller, runs the remaining (idempotent) steps instead of recording this attempt as complete.
            self._log.warning("teardown interrupted by cancellation; a waiting or later caller finishes it")
            if self._teardown_future is future:
                self._teardown_future = None
            raise
        finally:
            if not future.done():
                future.set_result(None)
