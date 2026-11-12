"""The examples greet the first human and end the call when the last one has gone.

Marker: ``integration_extended``. It runs the real ``bot_direct_join.run("fake")``, and the relayed-join
example's ``bot()`` over SmallWebRTC, against the local livekit-server, with the scripted brain, the fake
Synthesia API and ``FakeAvatarWorker``.

**Why this exists when ``_presence`` has unit tests.** The helper decides; this tests the *wiring*: which
transport event the example listens to, what argument type that event passes, and that a human already in
the room before the bot connected is reported at all. That wiring is what has broken before, and nothing
else runs an example's event handlers.

**How it watches the example without changing it.** ``run()`` builds its own transport, service and
pipeline worker, so the test replaces those three names in the example's module with factories that build
the real object and attach recorders to it: an ``EventRecorder`` on the service, a recorder for the
transport's ``on_participant_connected`` and ``on_participant_disconnected``, and a ``reached_downstream``
filter on the worker. ``_presence.NO_HUMANS_GRACE_SECS``, which ``Presence`` reads when it is made, is set to
``SHORT_GRACE``, so a test can wait past the grace and see an end that should not have happened, and a
grace the example passed at its call site would show. The example code runs unmodified.

**The end reason is one of two published values.** The example ends its pipeline once nobody has been in
the room for its grace, a second longer than the avatar worker's own default, so a real worker normally
ends the session first, with ``no_humans_left``; ``FakeAvatarWorker`` does not, so here it is
``pipeline_ended``. Either is a normal end. A call ended with ``worker.end()`` reports
``pipeline_cancelled`` instead (Pipecat 1.10's runner cancels a worker that is still draining).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription
from pipecat.frames.frames import DataFrame, Frame
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from tests.integration._join_modes import AVATAR_IDS, WAIT, direct_join_bot_token
from tests.integration._launcher import Launcher
from tests.support.events import EventRecorder
from tests.support.livekit_dev import LiveKitDev, connect_room, mint_dev_token
from tests.support.secrets import API_KEY
from tests.support.waiting import until

pytestmark = pytest.mark.integration_extended

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_MODULES = ("_brain", "_fake_brain", "_presence", "_shutdown", "bot_direct_join", "bot_relayed_join")

HUMAN_A = "human-a"
HUMAN_B = "human-b"

#: A backstop for the session to end, not a budget. Longer than ``WAIT``: the example waits out its
#: no-humans grace, then a graceful end lets the greeting finish playing, for up to the service's 5 s
#: stop drain.
END_WAIT = 30.0

#: The direct join's grace in these tests: an input, no longer than the real 6 s so the tests can wait past
#: it. Its margin covers two things: the ~3 s tick on which livekit-server tells the bot about other
#: participants' joins and leaves, and the rejoining human's own connect, which starts only once the bot
#: has seen the leave. 4 s left under a second for that connect; don't trim it. It replaces the constant
#: Presence reads, so a grace the example passed at its call site would show.
SHORT_GRACE = 5.0

#: How soon a SmallWebRTC hang-up must start ending the relayed-join call: well under the 6 s grace a
#: Daily call gets. The end itself then lets the greeting finish, which can take longer.
HANG_UP_ENDS_WITHIN = 3.0


class StillRunning(DataFrame):
    """Queued behind a leave: it reaches the end of the pipeline only if no ``EndFrame`` went ahead of it."""


@dataclass
class Built:
    """What ``run()`` built, and what the recorders attached to it saw."""

    service: EventRecorder | None = None
    worker: Any = None
    #: Identities the bot's transport reported leaving, in order.
    left: list[str] = field(default_factory=list)
    #: Identities the bot's transport reported joining, in order.
    joined: list[str] = field(default_factory=list)
    #: ``StillRunning`` frames that reached the end of the pipeline.
    reached_end: list[Frame] = field(default_factory=list)


def _fresh(name: str) -> Iterator[ModuleType]:
    """An example, imported freshly with ``examples/`` on sys.path, as a script run has it."""
    for module in EXAMPLE_MODULES:
        sys.modules.pop(module, None)
    sys.path.insert(0, str(EXAMPLES))
    try:
        yield importlib.import_module(name)
    finally:
        if str(EXAMPLES) in sys.path:
            sys.path.remove(str(EXAMPLES))
        for module in EXAMPLE_MODULES:
            sys.modules.pop(module, None)


@pytest.fixture
def example() -> Iterator[ModuleType]:
    """``bot_direct_join``."""
    yield from _fresh("bot_direct_join")


@pytest.fixture
def relayed_example() -> Iterator[ModuleType]:
    """``bot_relayed_join``."""
    yield from _fresh("bot_relayed_join")


def _watch(example: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Built:
    """Replace the three constructors ``run()`` calls with ones that attach recorders."""
    built = Built()
    real_transport = example.LiveKitTransport
    real_service = example.SynthesiaVideoService
    real_worker = example.PipelineWorker

    async def record_left(_transport: Any, participant_id: str) -> None:
        built.left.append(participant_id)

    async def record_joined(_transport: Any, participant_id: str) -> None:
        built.joined.append(participant_id)

    async def record_reached(_worker: Any, frame: Frame) -> None:
        built.reached_end.append(frame)

    def transport(*args: Any, **kwargs: Any) -> Any:
        instance = real_transport(*args, **kwargs)
        instance.add_event_handler("on_participant_disconnected", record_left)
        instance.add_event_handler("on_participant_connected", record_joined)
        return instance

    def service(*args: Any, **kwargs: Any) -> Any:
        instance = real_service(*args, **kwargs)
        built.service = EventRecorder(instance)
        return instance

    def worker(*args: Any, **kwargs: Any) -> Any:
        instance = real_worker(*args, **kwargs)
        instance.add_reached_downstream_filter((StillRunning,))
        instance.add_event_handler("on_frame_reached_downstream", record_reached)
        built.worker = instance
        return instance

    monkeypatch.setattr(example, "LiveKitTransport", transport)
    monkeypatch.setattr(example, "SynthesiaVideoService", service)
    monkeypatch.setattr(example, "PipelineWorker", worker)
    monkeypatch.setattr(sys.modules[example.Presence.__module__], "NO_HUMANS_GRACE_SECS", SHORT_GRACE)
    return built


def _configure(monkeypatch: pytest.MonkeyPatch, dev: LiveKitDev, room_name: str, api_url: str) -> None:
    """Every variable ``run()`` reads, pointed at the local server and the fakes."""
    env = {
        "LIVEKIT_URL": dev.url,
        "LIVEKIT_ROOM": room_name,
        "BOT_TOKEN": direct_join_bot_token(dev, room_name=room_name),
        "LIVEKIT_API_KEY": dev.api_key,
        "LIVEKIT_API_SECRET": dev.api_secret,
        "SYNTHESIA_API_KEY": API_KEY,
        "SYNTHESIA_API_URL": api_url,
        "SYNTHESIA_AVATAR_ID": AVATAR_IDS[0],
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _human(dev: LiveKitDev, room_name: str, identity: str) -> contextlib.AbstractAsyncContextManager[Any]:
    """A human participant: a STANDARD kind, which is what the avatar worker counts too."""
    return connect_room(dev, mint_dev_token(dev, identity=identity, room=room_name, kind=None))


async def _greeted(launcher: Launcher) -> None:
    """The greeting is an LLM run, which the fake brain answers with speech the avatar receives."""
    await until(
        lambda: bool(launcher.fakes) and any(s["bytes"] for s in launcher.fakes[0].streams),
        timeout=WAIT,
        what="the greeting's audio to reach the fake avatar worker",
    )


@contextlib.asynccontextmanager
async def _running(example: ModuleType):
    """``run("fake")`` in a task, cancelled on the way out if the test did not see it finish."""
    task = asyncio.create_task(example.run("fake"), name="bot_direct_join.run")
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait([task], timeout=END_WAIT)


async def _ended_normally(built: Built, task: asyncio.Task[None]) -> None:
    assert built.service is not None
    (reason,) = await built.service.wait("on_session_ended", timeout=END_WAIT)
    assert reason in {"pipeline_ended", "no_humans_left"}, built.service.names()
    await until(task.done, timeout=END_WAIT, what="run() to return once the pipeline ended")
    task.result()
    assert built.service.count("on_session_ended") == 1, built.service.names()


async def test_the_call_ends_when_the_last_human_leaves_not_the_first(
    example: ModuleType,
    livekit_server: LiveKitDev,
    room_name: str,
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two humans join, one leaves: still running. The other leaves: the call ends, as a normal end."""
    api = await launcher.api()
    _configure(monkeypatch, livekit_server, room_name, api.base_url)
    built = _watch(example, monkeypatch)

    async with _running(example) as task, contextlib.AsyncExitStack() as humans:
        await until(lambda: built.service is not None, timeout=WAIT, what="run() to build the service")
        assert built.service is not None
        await built.service.wait("on_avatar_connected", timeout=WAIT)

        human_a = contextlib.AsyncExitStack()
        humans.push_async_callback(human_a.aclose)
        await human_a.enter_async_context(_human(livekit_server, room_name, HUMAN_A))
        await humans.enter_async_context(_human(livekit_server, room_name, HUMAN_B))
        await _greeted(launcher)

        await human_a.aclose()
        # The recorder's handler task is created alongside the example's, so once it has run, the
        # example has already decided; an end it queued would be ahead of this frame.
        await until(lambda: HUMAN_A in built.left, timeout=WAIT, what="the bot to see human A leave")
        # Past the grace, so an end it started would have happened.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2 * SHORT_GRACE
        await until(lambda: loop.time() >= deadline, timeout=3 * SHORT_GRACE, what="the grace to run out")
        await built.worker.queue_frame(StillRunning())
        await until(
            lambda: bool(built.reached_end),
            timeout=WAIT,
            what="a frame queued after human A left to reach the end of the pipeline",
        )
        assert built.service.count("on_session_ended") == 0, built.service.names()

        await humans.aclose()
        await _ended_normally(built, task)


async def test_a_human_who_rejoins_within_the_grace_keeps_the_call(
    example: ModuleType,
    livekit_server: LiveKitDev,
    room_name: str,
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last human leaves and comes back within the grace, as after a dropped connection: it goes on."""
    api = await launcher.api()
    _configure(monkeypatch, livekit_server, room_name, api.base_url)
    built = _watch(example, monkeypatch)

    async with _running(example) as task:
        await until(lambda: built.service is not None, timeout=WAIT, what="run() to build the service")
        assert built.service is not None
        await built.service.wait("on_avatar_connected", timeout=WAIT)

        async with _human(livekit_server, room_name, HUMAN_A):
            await _greeted(launcher)
        await until(lambda: HUMAN_A in built.left, timeout=WAIT, what="the bot to see the human leave")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2 * SHORT_GRACE

        async with _human(livekit_server, room_name, HUMAN_A):
            await until(
                lambda: built.joined.count(HUMAN_A) == 2, timeout=WAIT, what="the bot to see the human rejoin"
            )
            # Past the grace, so an end it started would have happened.
            await until(lambda: loop.time() >= deadline, timeout=3 * SHORT_GRACE, what="the grace to run out")
            await built.worker.queue_frame(StillRunning())
            await until(
                lambda: bool(built.reached_end),
                timeout=WAIT,
                what="a frame queued after the grace to reach the end of the pipeline",
            )
            assert built.service.count("on_session_ended") == 0, built.service.names()

        await _ended_normally(built, task)


async def test_a_human_already_in_the_room_is_greeted_and_their_leave_ends_the_call(
    example: ModuleType,
    livekit_server: LiveKitDev,
    room_name: str,
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The usual LiveKit order: the human is in the room before the bot is dispatched into it."""
    api = await launcher.api()
    _configure(monkeypatch, livekit_server, room_name, api.base_url)
    built = _watch(example, monkeypatch)

    async with contextlib.AsyncExitStack() as humans:
        await humans.enter_async_context(_human(livekit_server, room_name, HUMAN_A))

        async with _running(example) as task:
            await until(lambda: built.service is not None, timeout=WAIT, what="run() to build the service")
            assert built.service is not None
            await built.service.wait("on_avatar_connected", timeout=WAIT)
            await _greeted(launcher)

            await humans.aclose()
            await _ended_normally(built, task)


async def test_a_smallwebrtc_hang_up_ends_the_relayed_call_without_a_grace(
    relayed_example: ModuleType,
    livekit_server: LiveKitDev,
    room_name: str,
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over SmallWebRTC a reconnect is a new call, so a grace could only add billing: a hang-up ends it.

    This pins the relayed example's wiring and its grace selector together, with the call made the way
    the development runner makes it: an offer answered by a ``SmallWebRTCConnection``, then ``bot()``.
    """
    api = await launcher.api()
    _configure(monkeypatch, livekit_server, room_name, api.base_url)
    monkeypatch.setattr(relayed_example, "OPTS", argparse.Namespace(fake_brain=True, s2s=False))
    built = Built()
    real_service = relayed_example.SynthesiaVideoService

    def service(*args: Any, **kwargs: Any) -> Any:
        instance = real_service(*args, **kwargs)
        built.service = EventRecorder(instance)
        return instance

    monkeypatch.setattr(relayed_example, "SynthesiaVideoService", service)
    real_worker = relayed_example.PipelineWorker
    ends: list[str] = []

    def worker(*args: Any, **kwargs: Any) -> Any:
        instance = real_worker(*args, **kwargs)
        stop_when_done = instance.stop_when_done

        async def recorded_stop_when_done() -> None:
            ends.append("stop_when_done")
            await stop_when_done()

        instance.stop_when_done = recorded_stop_when_done
        return instance

    monkeypatch.setattr(relayed_example, "PipelineWorker", worker)

    client = RTCPeerConnection()
    client.addTransceiver("audio", direction="sendrecv")
    client.addTransceiver("video", direction="sendrecv")
    connection = SmallWebRTCConnection()
    task: asyncio.Task[None] | None = None
    try:
        await client.setLocalDescription(await client.createOffer())
        await connection.initialize(sdp=client.localDescription.sdp, type=client.localDescription.type)
        answer = connection.get_answer()
        assert answer is not None
        task = asyncio.create_task(
            relayed_example.bot(SmallWebRTCRunnerArguments(webrtc_connection=connection))
        )
        await client.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        await until(lambda: built.service is not None, timeout=WAIT, what="bot() to build the service")
        assert built.service is not None
        await built.service.wait("on_avatar_connected", timeout=WAIT)
        await _greeted(launcher)

        await client.close()
        await until(
            lambda: bool(ends),
            timeout=HANG_UP_ENDS_WITHIN,
            what=f"the call to start ending within {HANG_UP_ENDS_WITHIN:g} s of the hang-up",
        )
        await until(task.done, timeout=END_WAIT, what="bot() to return once the call ended")
        task.result()
        (reason,) = built.service.payload("on_session_ended")
        assert reason == "pipeline_ended", built.service.names()
    finally:
        await client.close()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait([task], timeout=END_WAIT)
