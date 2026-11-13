"""The examples' presence helper: greet the first human, end the pipeline once the last one has gone.

`examples/_presence.py` is what decides, in both examples, when a call ends. Ending on the first leave
takes the avatar away from everyone still in the call; never ending keeps billing. Ending the moment
nobody is present ends the call under someone rejoining with the same identity, which is reported as a
leave and a join in the same instant, so the end waits out a grace that any human joining cancels.
These tests drive it through its public methods only, with the ids an example would pass in, and with
the grace as an input.

The relayed join with several people (Daily) is covered only here: no multi-party transport runs
locally. The direct join's real event wiring is covered by
`tests/integration/test_examples_presence.py`.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import logging
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments

from tests.support.waiting import until

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

AVATAR = "synthesia-avatar-agent"

#: The grace these tests run with: an input, short so the suite stays fast.
GRACE = 0.05
#: The grace for a join that arrives inside it: long enough that the join, a loop tick or so after the
#: leave, is well inside it even on a slow runner.
LONG_GRACE = 0.5


@pytest.fixture(scope="module")
def presence_module() -> Iterator[ModuleType]:
    """`_presence`, imported the way the examples import it: with `examples/` on sys.path, as a sibling."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        yield importlib.import_module("_presence")
    finally:
        if str(EXAMPLES) in sys.path:
            sys.path.remove(str(EXAMPLES))
        sys.modules.pop("_presence", None)


@pytest.fixture(scope="module")
def direct_join_module(presence_module: ModuleType) -> Iterator[ModuleType]:
    """`bot_direct_join`, imported beside the `_presence` above (it imports that same module)."""
    try:
        yield importlib.import_module("bot_direct_join")
    finally:
        for name in ("bot_direct_join", "_brain"):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def relayed_join_module(presence_module: ModuleType) -> Iterator[ModuleType]:
    """`bot_relayed_join`, imported beside the `_presence` above (it imports that same module)."""
    try:
        yield importlib.import_module("bot_relayed_join")
    finally:
        for name in ("bot_relayed_join", "_brain"):
            sys.modules.pop(name, None)


class StubWorker:
    """Stands in for the example's pipeline worker: counts graceful stops."""

    def __init__(self) -> None:
        self.stops = 0

    async def stop_when_done(self) -> None:
        self.stops += 1


def others() -> set[asyncio.Task[Any]]:
    """Every task on the loop but the running test's own."""
    return {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}


class Call:
    """A `Presence` whose `on_end` records the loop time of each end."""

    def __init__(self, module: ModuleType, *, grace_secs: float = GRACE, **kwargs: Any) -> None:
        self.loop = asyncio.get_running_loop()
        self.grace = grace_secs
        self.ends: list[float] = []

        async def on_end() -> None:
            self.ends.append(self.loop.time())

        self.presence = module.Presence(on_end=on_end, grace_secs=grace_secs, **kwargs)

    async def ended(self) -> None:
        await until(lambda: bool(self.ends), what="the call to end")

    async def past_the_grace(self) -> None:
        """Let the loop run until twice the grace has passed: long enough for any pending end to fire."""
        deadline = self.loop.time() + 2 * self.grace
        await until(lambda: self.loop.time() >= deadline, what="the grace to run out")


@pytest.mark.unit
async def test_one_of_two_humans_leaving_does_not_end_the_call(presence_module: ModuleType) -> None:
    call = Call(presence_module)
    call.presence.joined("alice")
    call.presence.joined("bob")

    call.presence.left("alice")
    await call.past_the_grace()

    assert call.ends == []
    assert len(call.presence) == 1


@pytest.mark.unit
async def test_the_last_human_leaving_ends_the_call_exactly_once(presence_module: ModuleType) -> None:
    call = Call(presence_module)
    call.presence.joined("alice")
    call.presence.joined("bob")
    call.presence.left("alice")
    call.presence.left("bob")
    await call.ended()

    # Someone arriving during the end and leaving again must not queue a second end.
    call.presence.joined("carol")
    call.presence.left("carol")
    await call.past_the_grace()
    assert len(call.ends) == 1


@pytest.mark.unit
async def test_the_greeting_fires_once(presence_module: ModuleType) -> None:
    Action = presence_module.Action
    call = Call(presence_module)

    assert call.presence.joined("alice") is Action.GREET
    assert call.presence.joined("bob") is Action.NONE
    call.presence.left("alice")
    call.presence.left("bob")
    assert call.presence.joined("carol") is Action.NONE


@pytest.mark.unit
async def test_the_avatar_is_not_a_human_joining_or_leaving(presence_module: ModuleType) -> None:
    Action = presence_module.Action
    call = Call(presence_module, ignore={AVATAR})

    # The avatar normally arrives first: greeting it would play to an empty room.
    assert call.presence.joined(AVATAR) is Action.NONE
    assert len(call.presence) == 0
    assert call.presence.joined("alice") is Action.GREET
    # The avatar leaving while a human is present is not the call ending.
    call.presence.left(AVATAR)
    assert len(call.presence) == 1
    call.presence.left("alice")
    await call.ended()


@pytest.mark.unit
async def test_an_unknown_leave_and_a_repeated_leave_change_nothing(presence_module: ModuleType) -> None:
    call = Call(presence_module)
    call.presence.joined("alice")
    call.presence.joined("bob")

    call.presence.left("nobody")
    call.presence.left("alice")
    call.presence.left("alice")
    assert len(call.presence) == 1
    call.presence.left("bob")
    await call.ended()
    call.presence.left("bob")
    await call.past_the_grace()
    assert len(call.ends) == 1


@pytest.mark.unit
async def test_a_leave_before_anyone_joined_does_not_end_the_call(presence_module: ModuleType) -> None:
    Action = presence_module.Action
    call = Call(presence_module)

    call.presence.left("nobody")
    await call.past_the_grace()
    assert call.ends == []
    assert call.presence.joined("alice") is Action.GREET


@pytest.mark.unit
async def test_a_seeded_human_is_greeted_and_their_leave_ends_the_call(presence_module: ModuleType) -> None:
    Action = presence_module.Action
    call = Call(presence_module, ignore={AVATAR})

    assert call.presence.seed(["alice", AVATAR]) is Action.GREET
    assert len(call.presence) == 1
    call.presence.left("alice")
    await call.ended()


@pytest.mark.unit
async def test_seeding_nobody_neither_greets_nor_ends(presence_module: ModuleType) -> None:
    Action = presence_module.Action
    call = Call(presence_module, ignore={AVATAR})

    assert call.presence.seed([]) is Action.NONE
    assert call.presence.seed([AVATAR]) is Action.NONE
    assert call.presence.joined("alice") is Action.GREET
    assert call.ends == []


@pytest.mark.unit
@pytest.mark.parametrize("join_first", [False, True], ids=["seed-then-join", "join-then-seed"])
async def test_a_seeded_id_that_also_arrives_as_a_join_is_counted_once(
    presence_module: ModuleType, join_first: bool
) -> None:
    """Whether a transport also reports people already present as joins is unknowable; it must not matter."""
    Action = presence_module.Action
    call = Call(presence_module)

    if join_first:
        actions = [call.presence.joined("alice"), call.presence.seed(["alice"])]
    else:
        actions = [call.presence.seed(["alice"]), call.presence.joined("alice")]

    assert actions.count(Action.GREET) == 1
    assert len(call.presence) == 1
    call.presence.left("alice")
    await call.ended()


@pytest.mark.unit
async def test_one_to_one_connect_greets_and_disconnect_ends(presence_module: ModuleType) -> None:
    """SmallWebRTC: no seeding, one connect, one disconnect."""
    Action = presence_module.Action
    call = Call(presence_module)

    assert call.presence.joined("pc-1") is Action.GREET
    call.presence.left("pc-1")
    await call.ended()


@pytest.mark.unit
async def test_nobody_present_ends_the_call_at_the_grace_not_before(presence_module: ModuleType) -> None:
    call = Call(presence_module)
    call.presence.joined("alice")

    left_at = call.loop.time()
    call.presence.left("alice")
    await call.ended()

    # The loop may run a timer up to one clock tick early.
    tick = time.get_clock_info("monotonic").resolution
    assert call.ends[0] - left_at >= GRACE - tick, (left_at, call.ends)
    assert len(call.ends) == 1


@pytest.mark.unit
@pytest.mark.parametrize("who", ["alice", "bob"], ids=["the-same-identity-rejoining", "another-human"])
async def test_a_human_joining_inside_the_grace_keeps_the_call(presence_module: ModuleType, who: str) -> None:
    """A dropped connection reconnecting is a leave and a join under the same identity, moments apart.

    The join comes after the loop has run, as it does from a transport's separate handler task, so a
    grace that had already run out would have ended the call before it.
    """
    call = Call(presence_module, ignore={AVATAR}, grace_secs=LONG_GRACE)
    call.presence.joined("alice")

    left_at = call.loop.time()
    call.presence.left("alice")
    await until(lambda: call.loop.time() >= left_at + GRACE / 5, what="a moment to pass")
    call.presence.joined(who)
    await call.past_the_grace()

    assert call.ends == []
    assert len(call.presence) == 1


@pytest.mark.unit
async def test_the_avatar_joining_inside_the_grace_does_not_keep_the_call(
    presence_module: ModuleType,
) -> None:
    call = Call(presence_module, ignore={AVATAR})
    call.presence.joined("alice")

    call.presence.left("alice")
    call.presence.joined(AVATAR)
    await call.ended()


@pytest.mark.unit
def test_the_default_grace_is_one_second_past_the_avatar_workers(presence_module: ModuleType) -> None:
    """The worker stops a session 5 s (its default) after the last human leaves; the example waits 1 s more.

    In a direct join that keeps the end reason the worker's, rather than a race between the two.
    """
    assert presence_module.WORKER_NO_HUMANS_GRACE_SECS == 5.0
    assert presence_module.NO_HUMANS_GRACE_SECS == 6.0


@pytest.mark.unit
def test_the_relayed_join_waits_out_the_grace_over_daily_and_not_over_smallwebrtc(
    presence_module: ModuleType, relayed_join_module: ModuleType
) -> None:
    """A SmallWebRTC reconnect is a new call, so a grace there only bills; a Daily participant can rejoin."""
    grace = relayed_join_module.no_humans_grace_secs
    assert grace(SmallWebRTCRunnerArguments(webrtc_connection=None)) == 0.0
    assert (
        grace(DailyRunnerArguments(room_url="https://example.daily.co/room"))
        == presence_module.NO_HUMANS_GRACE_SECS
    )


@pytest.mark.unit
async def test_the_worker_ending_the_session_with_no_humans_left_ends_the_call_during_the_grace(
    presence_module: ModuleType, direct_join_module: ModuleType
) -> None:
    """In a direct join the worker stops first; a join after that must not leave the bot with no avatar."""
    call = Call(presence_module, grace_secs=LONG_GRACE)
    worker = StubWorker()
    call.presence.joined("alice")
    call.presence.left("alice")

    await direct_join_module.end_with_the_session("no_humans_left", call.presence, worker)
    assert worker.stops == 1

    # The human who joins now cannot keep a call the worker has already left.
    call.presence.joined("bob")
    call.presence.left("bob")
    await until(lambda: not others(), what="no grace left running")
    assert call.ends == []


@pytest.mark.unit
@pytest.mark.parametrize("reason", ["avatar_left", "pipeline_ended", "launch_failed"])
async def test_any_other_session_end_leaves_the_call_running(
    presence_module: ModuleType, direct_join_module: ModuleType, reason: str
) -> None:
    """A cap or an unexpected leave continues voice-only; the call still ends when the humans have gone."""
    call = Call(presence_module)
    worker = StubWorker()
    call.presence.joined("alice")
    call.presence.left("alice")

    await direct_join_module.end_with_the_session(reason, call.presence, worker)
    assert worker.stops == 0
    await call.ended()


@pytest.mark.unit
async def test_close_during_the_grace_leaves_no_timer_and_no_late_end(presence_module: ModuleType) -> None:
    call = Call(presence_module)
    call.presence.joined("alice")
    call.presence.left("alice")

    call.presence.close()
    await until(lambda: not others(), what="the grace to be cancelled")
    call.presence.joined("bob")
    call.presence.left("bob")
    await call.past_the_grace()

    assert call.ends == []
    assert not others()


@pytest.mark.unit
async def test_an_end_that_raises_is_logged(
    presence_module: ModuleType, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    async def on_end() -> None:
        raise RuntimeError("the pipeline would not stop")

    presence = presence_module.Presence(on_end=on_end, grace_secs=0)
    presence.joined("alice")
    presence.left("alice")

    # Presence's own line, not asyncio's "Task exception was never retrieved" for a task left to the GC.
    await until(
        lambda: any(
            r.name == "_presence" and r.levelno >= logging.ERROR and r.exc_info for r in caplog_loguru.records
        ),
        what="the failed end to be logged with its exception",
    )
    await until(lambda: not others(), what="the end task to finish")


@pytest.mark.unit
@pytest.mark.parametrize("seed", [20261026, 1, 2, 3, 4], ids=lambda seed: f"seed-{seed}")
async def test_random_joins_and_leaves_end_the_call_exactly_once_when_everyone_has_gone(
    presence_module: ModuleType, seed: int
) -> None:
    """2,000 joins and leaves, the loop running between some of them so graces start and cancel, then
    everyone leaves: one end, with nobody present, and nothing left running."""
    rng = random.Random(seed)
    ends: list[int] = []

    async def on_end() -> None:
        ends.append(len(presence))

    presence = presence_module.Presence(on_end=on_end, grace_secs=GRACE, ignore={AVATAR})
    for _ in range(2000):
        who = rng.choice(["alice", "bob", "carol", AVATAR])
        if rng.random() < 0.5:
            presence.joined(who)
        else:
            presence.left(who)
        if rng.random() < 0.2:
            # Lets a cancelled grace finish between events, as a transport's handler tasks would.
            await asyncio.sleep(0)

    for who in ("alice", "bob", "carol", AVATAR):
        presence.left(who)
    await until(lambda: bool(ends), what=f"the call to end once everyone had gone (seed {seed})")
    presence.close()
    await until(lambda: not others(), what=f"nothing left running (seed {seed})")

    assert ends == [0], f"seed {seed}: {ends}"


@pytest.mark.unit
async def test_a_leave_a_join_and_a_leave_in_one_turn_still_end_the_call(presence_module: ModuleType) -> None:
    """The join cancels the first grace; the second leave must start another, or the call never ends."""
    call = Call(presence_module)
    call.presence.joined("alice")

    call.presence.left("alice")
    call.presence.joined("alice")
    call.presence.left("alice")

    await call.ended()
    assert len(call.ends) == 1


@pytest.mark.unit
def test_the_relayed_join_takes_its_grace_from_the_transport() -> None:
    """Guards the wiring, not the behaviour: no Daily call runs in the test tiers, so nothing else would
    notice the relayed example passing a fixed grace (0 included) for every transport."""
    tree = ast.parse((EXAMPLES / "bot_relayed_join.py").read_text())
    graces = [
        ast.unparse(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Presence"
        for keyword in node.keywords
        if keyword.arg == "grace_secs"
    ]
    assert graces == ["no_humans_grace_secs(runner_args)"], graces


@pytest.mark.unit
def test_the_direct_joins_session_end_handler_is_wired_to_end_with_the_session() -> None:
    """Guards the wiring, not the behaviour: the fake avatar worker cannot end a session with
    `no_humans_left`, so no test runs `end_with_the_session` from a real session end."""
    tree = ast.parse((EXAMPLES / "bot_direct_join.py").read_text())
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "event_handler"
        and [arg.value for arg in decorator.args if isinstance(arg, ast.Constant)] == ["on_session_ended"]
    ]
    assert len(handlers) == 1, "bot_direct_join.py should register one on_session_ended handler"
    calls = {
        node.func.id
        for node in ast.walk(handlers[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "end_with_the_session" in calls


@pytest.mark.unit
async def test_a_join_or_close_while_the_call_is_ending_does_not_stop_the_end(
    presence_module: ModuleType,
) -> None:
    """Once the grace has run out the call is ending: a late join, or closing Presence, cannot undo it."""
    ending = asyncio.Event()
    release = asyncio.Event()
    ended: list[str] = []

    async def on_end() -> None:
        ending.set()
        await release.wait()
        ended.append("ended")

    presence = presence_module.Presence(on_end=on_end, grace_secs=GRACE)
    presence.joined("alice")
    presence.left("alice")
    await until(ending.is_set, what="the end to start")

    presence.joined("bob")
    presence.close()
    release.set()
    await until(lambda: not others(), what="the end to finish")

    assert ended == ["ended"]
