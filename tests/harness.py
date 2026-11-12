"""The bidirectional pipeline harness.

Keep it importable standalone (only Pipecat, `pipecat_synthesia` and `tests/support`).

`pipecat.tests.utils.run_test` sends every frame in ONE direction and asserts an exact ordered frame
list. The service needs both: TTS frames travel DOWNSTREAM while the transport's
`OutputTransportReadyFrame` travels UPSTREAM, and the frames the service emits from tasks (`Bot*Speaking`,
`ErrorFrame`) arrive at racy positions. So this module copies the body of `run_test` once -- a real
`Pipeline([source, processor, sink])` under a real `PipelineWorker(cancel_on_idle_timeout=False)` -- and
exposes it as a context manager the test drives step by step::

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(OutputTransportReadyFrame(), FrameDirection.UPSTREAM)
        frame = await h.frame(BotStartedSpeakingFrame)
        await h.finish()
    assert_absent(h.down, TTSAudioRawFrame)

`h.down` / `h.up` are live lists of what reached the sink / the source (StartFrame excluded, as in
`run_test`); assert on them with `tests.support.frames`, never by strict enumeration, except in
`test_passthrough_all_other_frames`, which uses `run_bidirectional(..., expected_*=[...])`.

On exit the harness ends the pipeline if the test did not, awaits the runner, and calls
`assert_no_dangling(h.task_manager)`: every task any processor created through the shared
`TaskManager` must be done.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import EndFrame, Frame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.runner import WorkerRunner

from pipecat_synthesia.room import AvatarRoom
from pipecat_synthesia.service import SynthesiaVideoService
from tests.support.secrets import API_KEY
from tests.support.waiting import DEFAULT_TIMEOUT, assert_no_dangling, until

__all__ = [
    "Harness",
    "Step",
    "assert_strict_frames",
    "bidirectional_pipeline",
    "make_service",
    "make_setup",
    "run_bidirectional",
]

FrameT = TypeVar("FrameT", bound=Frame)

#: One step of `run_bidirectional`: a frame with its direction, a sleep in seconds, or a callable.
Step = tuple[FrameDirection, Frame] | float | Callable[[], Awaitable[None]]

# Well over the ~10 ms a fake-room pipeline takes to start, far under Pipecat's shared 20 s for setup; a slow
# `setup()` in the processor under test fails here with a clear message rather than hanging the test.
START_TIMEOUT = 5.0


class _Capture(FrameProcessor):
    """Direct-mode end of the pipeline: records frames travelling in one direction, forwards everything."""

    def __init__(self, *, sink: list[Frame], direction: FrameDirection, **kwargs) -> None:
        super().__init__(enable_direct_mode=True, **kwargs)
        self._sink = sink
        self._direction = direction

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == self._direction and not isinstance(frame, StartFrame):
            self._sink.append(frame)
        await self.push_frame(frame, direction)


class Harness:
    """What `bidirectional_pipeline` yields."""

    def __init__(
        self, *, processor: FrameProcessor, worker: PipelineWorker, task_manager: TaskManager
    ) -> None:
        self.processor = processor
        self.worker = worker
        self.task_manager = task_manager
        #: Frames that reached the sink (DOWNSTREAM), in arrival order; `StartFrame` excluded.
        self.down: list[Frame] = []
        #: Frames that reached the source (UPSTREAM), in arrival order; `StartFrame` excluded.
        self.up: list[Frame] = []
        self._run_task: asyncio.Task | None = None
        self._finished = False

    async def queue(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        """Inject a frame: DOWNSTREAM enters at the source, UPSTREAM at the sink (`worker.queue_frame`)."""
        await self.worker.queue_frame(frame, direction)

    async def sleep(self, secs: float) -> None:
        """Model a delay (let a hysteresis window elapse). Never use it to wait for a task: use `until`."""
        await asyncio.sleep(secs)

    async def call(self, coro_fn: Callable[[], Awaitable[Any]]) -> Any:
        """Run a coroutine function to completion before the next step; returns its result."""
        return await coro_fn()

    def _seq(self, direction: FrameDirection) -> list[Frame]:
        return self.down if direction == FrameDirection.DOWNSTREAM else self.up

    async def frame(
        self,
        frame_type: type[FrameT],
        *,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> FrameT:
        """Wait for and return the first frame of `frame_type` recorded in `direction`."""
        seq = self._seq(direction)
        await until(
            lambda: any(isinstance(f, frame_type) for f in seq),
            timeout=timeout,
            what=f"a {frame_type.__name__} {direction.name.lower()} "
            f"(seen: {[type(f).__name__ for f in seq]})",
        )
        return next(f for f in seq if isinstance(f, frame_type))

    async def finish(self, send_end_frame: bool = True) -> None:
        """End the pipeline (an `EndFrame` from the source unless told otherwise) and await the runner."""
        if self._finished:
            return
        self._finished = True
        if send_end_frame:
            await self.worker.queue_frame(EndFrame())
        if self._run_task is not None:
            await self._run_task


@asynccontextmanager
async def bidirectional_pipeline(
    processor: FrameProcessor,
    *,
    pipeline_params: PipelineParams | None = None,
    start_timeout: float = START_TIMEOUT,
    may_never_start: bool = False,
    **worker_kwargs: Any,
) -> AsyncIterator[Harness]:
    """Run `processor` between a recording source and sink; yield once the `StartFrame` reached the sink.

    Extra keyword arguments go to `PipelineWorker` (e.g. `processor_unusable_policy=...`). The worker
    always runs with `cancel_on_idle_timeout=False` and `enable_rtvi=False`, as `run_test` does.
    `may_never_start=True` also yields once the runner has returned without the pipeline starting, for
    the one test of a pipeline cancelled while it is still being set up.
    """
    task_manager = TaskManager()
    h_down: list[Frame] = []
    h_up: list[Frame] = []
    source = _Capture(sink=h_up, direction=FrameDirection.UPSTREAM, name="harness-source")
    sink = _Capture(sink=h_down, direction=FrameDirection.DOWNSTREAM, name="harness-sink")
    pipeline = Pipeline([source, processor, sink])
    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
        params=pipeline_params or PipelineParams(),
        task_manager=task_manager,
        **worker_kwargs,
    )
    harness = Harness(processor=processor, worker=worker, task_manager=task_manager)
    harness.down = h_down
    harness.up = h_up

    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(_worker: PipelineWorker, _frame: Frame) -> None:
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    run_task = asyncio.create_task(runner.run(), name="harness-runner")
    harness._run_task = run_task
    try:
        # Processors drop frames that arrive before StartFrame, and upstream frames enter at the sink,
        # which StartFrame reaches last.
        if may_never_start:
            # Pipecat 1.13+ tears down a pipeline cancelled while it is still being set up without ever
            # starting it: no StartFrame reaches the sink and `on_pipeline_finished` does not fire, the
            # runner just returns. The test is let in either way, and its `finish()` finds the runner done.
            await until(
                lambda: started.is_set() or run_task.done(),
                timeout=start_timeout,
                what="the pipeline to start or its runner to return",
            )
        else:
            try:
                await asyncio.wait_for(started.wait(), timeout=start_timeout)
            except TimeoutError:
                raise AssertionError(
                    f"pipeline did not start within {start_timeout}s: a processor blocked in setup()/start() "
                    "or swallowed the StartFrame"
                ) from None
        yield harness
        await harness.finish()
    finally:
        if not run_task.done():
            # the test body raised, or finish() itself failed: cancel and wait so nothing leaks
            await worker.cancel(reason="harness exit")
            try:
                await asyncio.wait_for(run_task, timeout=30.0)
            except Exception:
                run_task.cancel()
        assert_no_dangling(task_manager)


async def run_bidirectional(
    processor: FrameProcessor,
    steps: Sequence[Step],
    *,
    expected_down_frames: Sequence[type[Frame]] | None = None,
    expected_up_frames: Sequence[type[Frame]] | None = None,
    pipeline_params: PipelineParams | None = None,
    send_end_frame: bool = True,
    **worker_kwargs: Any,
) -> tuple[list[Frame], list[Frame]]:
    """Drive `steps` through `bidirectional_pipeline` and return `(down, up)`.

    `expected_*` are strict (length + ordered `isinstance`) exactly like `run_test`; pass them ONLY in a
    test where nothing asynchronous can emit. The `EndFrame` sent by `send_end_frame`
    is part of `down` (it proves pass-through), so list it when asserting strictly.
    """
    async with bidirectional_pipeline(processor, pipeline_params=pipeline_params, **worker_kwargs) as h:
        for step in steps:
            if isinstance(step, tuple):
                direction, frame = step
                await h.queue(frame, direction)
            elif isinstance(step, int | float):
                await h.sleep(float(step))
            else:
                await h.call(step)
        await h.finish(send_end_frame=send_end_frame)

    if expected_down_frames is not None:
        assert_strict_frames(h.down, expected_down_frames, "DOWN")
    if expected_up_frames is not None:
        assert_strict_frames(h.up, expected_up_frames, "UP")
    return h.down, h.up


def assert_strict_frames(
    received: Sequence[Frame], expected: Sequence[type[Frame]], label: str = "frames"
) -> None:
    """Strict length + ordered `isinstance` check, as `run_test` does; for the one pass-through test only."""
    got = [type(f).__name__ for f in received]
    want = [t.__name__ for t in expected]
    assert len(received) == len(expected), f"{label}: received {got}, expected {want}"
    for real, cls in zip(received, expected, strict=True):
        assert isinstance(real, cls), f"{label}: received {got}, expected {want}"


def make_setup(
    *,
    audio_out_sample_rate: int = 24000,
    enable_metrics: bool = False,
    worker: PipelineWorker | None = None,
) -> tuple[FrameProcessorSetup, TaskManager]:
    """A `FrameProcessorSetup` for driving a processor's lifecycle by hand; call inside a running loop.

    `TaskManager()` binds to the running loop (never call the deprecated `TaskManager.setup()`).
    """
    task_manager = TaskManager()
    pipeline_worker = worker or PipelineWorker(
        Pipeline([]), cancel_on_idle_timeout=False, enable_rtvi=False, task_manager=task_manager
    )
    setup = FrameProcessorSetup(
        clock=SystemClock(),
        task_manager=task_manager,
        pipeline_worker=pipeline_worker,
        audio_out_sample_rate=audio_out_sample_rate,
        enable_metrics=enable_metrics,
    )
    return setup, task_manager


# Constructor kwargs `make_service` passes straight through instead of treating as config overrides.
_CTOR_KWARGS = frozenset(
    {"api_url", "allow_any_api_host", "session", "settings", "name", "required", "launch_timeout"}
)


def make_service(
    room: AvatarRoom,
    *,
    avatar_ids: Sequence[str] = ("av_a",),
    api_key: str = API_KEY,
    **overrides: Any,
) -> SynthesiaVideoService:
    """Construct a service and apply `ServiceConfig` overrides.

    `api_key` defaults to the shared secret sentinel so every service test is also a no-secrets-in-logs
    test for free. Constructor kwargs (`required`, `launch_timeout`, `session`, `settings`, `api_url`,
    `allow_any_api_host`, `name`) go to the constructor; every other keyword is a `ServiceConfig` field
    replaced with `dataclasses.replace` (a `TypeError` names an unknown field). Overrides must be applied
    BEFORE `setup()`, which is why this helper is the only sanctioned way to touch `service._config`.
    """
    ctor = {k: overrides.pop(k) for k in list(overrides) if k in _CTOR_KWARGS}
    service = SynthesiaVideoService(avatar_ids=avatar_ids, room=room, api_key=api_key, **ctor)
    if overrides:
        service._config = dataclasses.replace(service._config, **overrides)
    return service
