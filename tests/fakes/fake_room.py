"""Fakes for the `AvatarRoom` / `AvatarSession` interface. Read this docstring before using them.

`FakeAvatarRoom` is an `AvatarRoom` whose `launch()` returns a pre-built session after an optional delay,
or fails, or blocks on a gate. `FakeAvatarSession` is an `AvatarSession` that records every call and lets
a test *emit* the callbacks a real session would deliver. Together they let the whole of
`SynthesiaVideoService` run with no LiveKit and no HTTP: `FakeAvatarRoom.launch()` never touches
`ctx.api`, so `SessionApiClient.create_session` is never reached.

**These fakes record; they do not decide.** The only behaviour modelled is what `AvatarSession` promises:

* each callback is delivered on a task of its own, one per emit, as the real session delivers them
  (*state-before-await*). `emit_*` therefore returns the task; `await` it when the test needs the
  callback to have run;
* `swap_avatar` errors are stamped with the session id;
* `is_alive` turns False on `emit_avatar_left` and on `aclose()`.

**Recording.** Every method appends a tuple to `calls` (`("send_audio", pcm, rate, channels)`,
`("end_utterance",)`, `("interrupt",)`, `("swap_avatar", id, timeout)`, `("aclose",)`,
`("add_close_hook", hook)`, `("wait_no_humans_left",)`) and the typed accessors (`send_audio_calls`,
`interrupt_count`, `end_utterance_count`, `aclose_count`, `swap_calls`) are sugar over it, so tests never
index tuple positions. `interrupt` is recorded when it **completes**, with `interrupt_entered` counting
entries: that is the one place the difference matters (a double interruption cancels the first
`interrupt()` mid-flight, and the test wants to know exactly one reached the worker). Every other call is
instantaneous, so entry and completion coincide.

**Knobs** (all optional; every one is an input the test names in its docstring):

* `FakeAvatarRoom(launch_delay=, fail_with=, cancel_cleanup_delay=, hold_launch=, during_launch=)`;
* `FakeAvatarSession(hold_send=, hold_interrupt=, hold_aclose=, hold_created=, swap_error=, send_error=,
  created_error=, humans_window=, on_call=)`
  -- the `hold_*` events make one call block until the test sets the event (modelling a slow worker
  without a duration; `hold_created` holds `wait_session_created()`, a 201 still on its way after an
  *early start*, and until it is set, or with `created_error`, `session_id` raises and
  `timings.session_created` is None, as with the real session); `*_error` makes
  that call raise after it is recorded (`send_error` is a plain
  attribute, so a test can set it back to None to stop the failures); `humans_window` is the
  *no-humans-left window*: `wait_no_humans_left()` waits for the future and returns its result (True: the
  last human left inside it), and without one returns False at once, as the interface's default does;
  `on_call(call)` runs after each call is recorded, so a test can react to "the service just called
  `end_utterance`" without polling.

**Delivery tasks are not in the harness's `TaskManager`.** `emit_*` schedules with `asyncio.ensure_future`
because the real session's callback tasks are plain asyncio tasks too, so a callback still running at
harness exit is invisible to `assert_no_dangling`. Await the task `emit_*` returns, or
`await session.wait_emitted()`, whenever a test needs "the callback has run".
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from pipecat_synthesia.errors import SynthesiaError
from pipecat_synthesia.room import (
    AvatarCallbacks,
    AvatarRoom,
    AvatarSession,
    JoinMode,
    LaunchContext,
    LaunchTimings,
)

__all__ = ["FakeAvatarRoom", "FakeAvatarSession"]

Call = tuple[Any, ...]


class FakeAvatarSession(AvatarSession):
    """An `AvatarSession` that records calls and emits callbacks on demand."""

    def __init__(
        self,
        *,
        session_id: str = "ses_fake",
        avatar_identity: str = "synthesia-avatar-agent",
        room_name: str | None = "fake-room",
        active_avatar_id: str = "av_a",
        avatar_status: dict[str, str] | None = None,
        all_ready: bool = False,
        timings: LaunchTimings | None = None,
        hold_send: asyncio.Event | None = None,
        hold_interrupt: asyncio.Event | None = None,
        hold_aclose: asyncio.Event | None = None,
        hold_created: asyncio.Event | None = None,
        swap_error: SynthesiaError | None = None,
        send_error: Exception | None = None,
        created_error: SynthesiaError | None = None,
        humans_window: asyncio.Future[bool] | None = None,
        on_call: Callable[[Call], Any] | None = None,
    ) -> None:
        self._session_id = session_id
        self._avatar_identity = avatar_identity
        self._room_name = room_name
        self._active_avatar_id = active_avatar_id
        self._avatar_status: dict[str, str] = dict(avatar_status or {})
        self._all_ready = all_ready
        self._alive = True
        self._timings_given = timings is not None
        self._timings = timings or self._now_timings()
        self.hold_send = hold_send
        self.hold_interrupt = hold_interrupt
        self.hold_aclose = hold_aclose
        self.hold_created = hold_created
        self.created_error = created_error
        self.swap_error = swap_error
        self.send_error = send_error
        self.humans_window = humans_window
        self.on_call = on_call

        #: Set by `FakeAvatarRoom.launch()` from `ctx.callbacks`; None until then (and in a direct join
        #: the media callbacks inside it are None, so `emit_video`/`emit_audio` return None).
        self.callbacks: AvatarCallbacks | None = None
        #: Every call, in order, as tuples; see the module docstring.
        self.calls: list[Call] = []
        #: Entries into `interrupt()`, including ones cancelled before they completed.
        self.interrupt_entered = 0
        #: True once `aclose()` has run to completion (past `hold_aclose`).
        self.aclose_completed = False
        #: Close hooks registered by the service/adapters (recorded, never run by the fake).
        self.close_hooks: list[Callable[[], Awaitable[None]]] = []
        #: Tasks created by `emit_*`, so a test can await them all.
        self.emitted: list[asyncio.Task] = []

    @staticmethod
    def _now_timings() -> LaunchTimings:
        now = monotonic()
        return LaunchTimings(
            room_connected=now, session_requested=now, session_created=now, avatar_joined=now
        )

    def stamp_timings(self) -> None:
        """Re-stamp default timings at `launch()` time so they sit after `metrics.launch_started`.

        Called by `FakeAvatarRoom.launch()`; a test that passed explicit `timings` keeps them.
        """
        if not self._timings_given:
            self._timings = self._now_timings()

    # -- recording -----------------------------------------------------------------------------

    def _record(self, call: Call) -> None:
        self.calls.append(call)
        if self.on_call is not None:
            self.on_call(call)

    @property
    def send_audio_calls(self) -> list[Call]:
        return [c for c in self.calls if c[0] == "send_audio"]

    @property
    def swap_calls(self) -> list[Call]:
        return [c for c in self.calls if c[0] == "swap_avatar"]

    @property
    def interrupt_count(self) -> int:
        return sum(1 for c in self.calls if c[0] == "interrupt")

    @property
    def end_utterance_count(self) -> int:
        return sum(1 for c in self.calls if c[0] == "end_utterance")

    @property
    def aclose_count(self) -> int:
        return sum(1 for c in self.calls if c[0] == "aclose")

    # -- AvatarSession -------------------------------------------------------------------------

    def _no_201(self) -> bool:
        """No 201 yet, or never: still on its way after an *early start* (`hold_created` not yet set), or a
        request that failed (`created_error`). As with the LiveKit session, `session_id` then raises and
        `timings.session_created` is None."""
        return (
            self.hold_created is not None and not self.hold_created.is_set()
        ) or self.created_error is not None

    @property
    def session_id(self) -> str:
        if self._no_201():
            raise RuntimeError("avatar session is not connected")
        return self._session_id

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def room_name(self) -> str | None:
        return self._room_name

    @property
    def active_avatar_id(self) -> str:
        return self._active_avatar_id

    @property
    def avatar_status(self) -> dict[str, str]:
        return dict(self._avatar_status)

    @property
    def all_ready(self) -> bool:
        return self._all_ready

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def timings(self) -> LaunchTimings:
        if self._no_201():
            return dataclasses.replace(self._timings, session_created=None)
        return self._timings

    async def send_audio(self, pcm_s16le: bytes, sample_rate: int, num_channels: int) -> None:
        self._record(("send_audio", pcm_s16le, sample_rate, num_channels))
        if self.hold_send is not None:
            await self.hold_send.wait()
        if self.send_error is not None:
            raise self.send_error

    async def end_utterance(self) -> None:
        self._record(("end_utterance",))

    async def interrupt(self) -> None:
        self.interrupt_entered += 1
        if self.hold_interrupt is not None:
            await self.hold_interrupt.wait()
        self._record(("interrupt",))

    async def swap_avatar(self, avatar_id: str, *, timeout: float) -> str:  # noqa: ASYNC109
        self._record(("swap_avatar", avatar_id, timeout))
        if self.swap_error is not None:
            # the real session always stamps swap errors with its session id
            raise self.swap_error.with_session(self._session_id)
        self._active_avatar_id = avatar_id
        return avatar_id

    async def wait_session_created(self) -> None:
        if self.hold_created is not None:
            await self.hold_created.wait()
        if self.created_error is not None:
            raise self.created_error

    async def wait_no_humans_left(self) -> bool:
        self._record(("wait_no_humans_left",))
        if self.humans_window is None:
            return False
        return await asyncio.shield(self.humans_window)

    def add_close_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        self._record(("add_close_hook", hook))
        self.close_hooks.append(hook)

    async def aclose(self) -> None:
        self._record(("aclose",))
        self._alive = False
        if self.hold_aclose is not None:
            await self.hold_aclose.wait()
        self.aclose_completed = True

    # -- emitters: deliver a callback the way the real session does (own task) --------------------

    def _schedule(self, coro: Awaitable[None]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self.emitted.append(task)
        return task

    def emit_playback_started(self) -> asyncio.Task | None:
        if self.callbacks is None:
            return None
        return self._schedule(self.callbacks.on_playback_started())

    def emit_playback_finished(self, playback_position: float = 0.0, interrupted: bool = False):
        if self.callbacks is None:
            return None
        return self._schedule(self.callbacks.on_playback_finished(playback_position, interrupted))

    def set_status(self, status: dict[str, str], all_ready: bool) -> None:
        """Update what `avatar_status` / `all_ready` report without delivering a callback."""
        self._avatar_status = dict(status)
        self._all_ready = all_ready

    def emit_status(self, status: dict[str, str], all_ready: bool) -> asyncio.Task | None:
        self.set_status(status, all_ready)
        if self.callbacks is None:
            return None
        return self._schedule(self.callbacks.on_avatar_status(dict(status), all_ready))

    def emit_avatar_left(self, reason: str, humans_present: bool | None = None) -> asyncio.Task | None:
        self._alive = False
        if self.callbacks is None:
            return None
        return self._schedule(self.callbacks.on_avatar_left(reason, humans_present))

    def emit_track_published(self, kind: str) -> asyncio.Task | None:
        if self.callbacks is None:
            return None
        return self._schedule(self.callbacks.on_track_published(kind))

    def emit_video(self, width: int, height: int) -> asyncio.Task | None:
        if self.callbacks is None or self.callbacks.on_video_frame is None:
            return None
        return self._schedule(self.callbacks.on_video_frame(bytes(width * height * 3), width, height))

    def emit_audio(self, pcm: bytes, sample_rate: int, num_channels: int) -> asyncio.Task | None:
        if self.callbacks is None or self.callbacks.on_audio_frame is None:
            return None
        return self._schedule(self.callbacks.on_audio_frame(pcm, sample_rate, num_channels))

    async def wait_emitted(self) -> None:
        """Await every task `emit_*` created so far (a test's "the callbacks have run" barrier)."""
        pending = [t for t in self.emitted if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class FakeAvatarRoom(AvatarRoom):
    """An `AvatarRoom` whose `launch()` is fully scripted: delay, gate, failure, cancellation cleanup."""

    def __init__(
        self,
        join_mode: JoinMode,
        *,
        launch_delay: float = 0.0,
        fail_with: SynthesiaError | None = None,
        cancel_cleanup_delay: float = 0.0,
        avatar_identity: str = "synthesia-avatar-agent",
        filter_installed: bool = True,
        hold_launch: asyncio.Event | None = None,
        during_launch: Callable[[FakeAvatarRoom, LaunchContext], Awaitable[None]] | None = None,
        session: FakeAvatarSession | None = None,
    ) -> None:
        self.join_mode = join_mode
        self.launch_delay = launch_delay
        self.fail_with = fail_with
        self.cancel_cleanup_delay = cancel_cleanup_delay
        self._avatar_identity = avatar_identity
        #: Read by `SynthesiaVideoService.start()` in a direct join (*self-hearing guard*).
        self.filter_installed = filter_installed
        #: When set, `launch()` waits for it (after `launch_delay`) before failing or returning.
        self.hold_launch = hold_launch
        #: Awaited inside `launch()` after the callbacks are installed and before it returns: how a test
        #: makes a status or leave callback fire before `launch()` returns.
        self.during_launch = during_launch
        #: Created here so tests can close over it before the harness runs.
        self.session = session or FakeAvatarSession(avatar_identity=avatar_identity)
        self.captured_ctx: LaunchContext | None = None
        self.launch_calls = 0
        #: Set when `launch()` observed `CancelledError` (the service's launch_timeout or a teardown).
        self.cancelled = False
        #: Set once the cancellation cleanup (`cancel_cleanup_delay`) has run to completion.
        self.cancel_cleanup_completed = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        self.captured_ctx = ctx
        self.launch_calls += 1
        try:
            if self.launch_delay:
                await asyncio.sleep(self.launch_delay)
            if self.hold_launch is not None:
                await self.hold_launch.wait()
            if self.fail_with is not None:
                raise self.fail_with
            self.session.callbacks = ctx.callbacks
            self.session.stamp_timings()
            if self.during_launch is not None:
                await self.during_launch(self, ctx)
            return self.session
        except asyncio.CancelledError:
            self.cancelled = True
            # A real adapter rolls back what it created before re-raising; the delay models that rollback
            # taking time, which is what LAUNCH_CANCEL_SECS exists for.
            if self.cancel_cleanup_delay:
                await asyncio.sleep(self.cancel_cleanup_delay)
            self.cancel_cleanup_completed = True
            raise
