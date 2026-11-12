"""Recorder for the service's event handlers.

Pipecat delivers a non-sync event handler in its own ``asyncio.create_task``, so
"the event fired" is never true on the line after the call that causes it. :class:`EventRecorder`
registers one handler per event up front and stores arrivals in order, so a test asserts on the
recorded list (``count``/``payload``) or waits for an arrival (``wait``) instead of sleeping.

Handler exceptions are logged and swallowed by ``_run_handler``, never propagated, so a recorder that
raised would fail silently; these handlers only append.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from pipecat_synthesia.events import REGISTERED_EVENTS
from tests.support.waiting import DEFAULT_TIMEOUT, until

__all__ = ["EXTRA_EVENTS", "RECORDED_EVENTS", "EventRecorder"]

# Registered by ``FrameProcessor`` itself, not by us: ``on_error`` (sync, fired from ``push_error`` with
# the ``ErrorFrame``) and ``on_usable_changed`` (async). Both show the *permanent-error rule* at work, so
# every recorder sees them.
EXTRA_EVENTS = ("on_error", "on_usable_changed")
RECORDED_EVENTS = (*REGISTERED_EVENTS, *EXTRA_EVENTS)


class EventRecorder:
    """Records every event a processor fires, in arrival order.

    Attach it right after constructing the processor and before anything can fire::

        service = make_service(room)
        rec = EventRecorder(service)
        ...
        info = await rec.wait("on_avatar_connected")

    Attributes:
        processor: The processor being watched.
        events: ``(name, args)`` in arrival order. ``args`` excludes the processor that Pipecat passes
            as the first handler argument, so ``args[0]`` is the payload the contract names.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.events: list[tuple[str, tuple]] = []
        for name in RECORDED_EVENTS:
            processor.add_event_handler(name, self._handler(name))

    def _handler(self, name: str) -> Callable[..., Coroutine[Any, Any, None]]:
        async def record(_processor: Any, *args: Any) -> None:
            self.events.append((name, args))

        return record

    def names(self) -> list[str]:
        """The event names recorded so far, in order — the whole ordering assertion in one line."""
        return [name for name, _ in self.events]

    def count(self, name: str) -> int:
        """How many times ``name`` has been recorded."""
        return sum(1 for recorded, _ in self.events if recorded == name)

    def payload(self, name: str, index: int = 0) -> tuple:
        """The args tuple of the ``index``-th ``name``; ``AssertionError`` if there is no such arrival."""
        hits = [args for recorded, args in self.events if recorded == name]
        assert len(hits) > index, (
            f"{name} fired {len(hits)} time(s), wanted at least {index + 1}; recorded: {self.names()}"
        )
        return hits[index]

    async def wait(self, name: str, *, timeout: float = DEFAULT_TIMEOUT, count: int = 1) -> tuple:
        """Wait for the ``count``-th ``name`` and return its args tuple.

        Safe to start before the event is caused (it polls), so the usual pattern is to kick off the
        action and then await this — no ordering constraint between the two.
        """
        await until(
            lambda: self.count(name) >= count,
            timeout=timeout,
            what=f"{count}x {name} (recorded so far: {self.names()})",
        )
        return self.payload(name, count - 1)
