"""``Launcher``: a scripted Synthesia API whose 201 launches a fake worker into a real room.

The ``launcher`` fixture that yields one lives in ``tests/integration/conftest.py``; the class lives here
so a test module can import it for a type annotation (and ``SESSION_BODY`` for an assertion) without
importing a conftest.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any

from pipecat_synthesia._redact import redact_mapping
from tests.fakes.fake_api import FakeSynthesiaApi, RecordedRequest, ScriptedResponse
from tests.integration.fake_worker import FakeAvatarWorker
from tests.support.livekit_dev import LiveKitDev, delete_room

__all__ = ["LAUNCH_DRAIN_SECS", "SESSION_BODY", "Launcher"]

#: The 201 body a launch gets unless a test scripts its own. Only `id` is load-bearing (`api.py` reads
#: it as the session id); the other two are there so the body looks like the real one.
SESSION_BODY: Mapping[str, Any] = {
    "id": "ses_test",
    "status": "active",
    "createdAt": "2026-01-01T00:00:00Z",
}

#: How long teardown lets an in-flight `FakeAvatarWorker.start()` finish before giving up on it. Longer than
#: the fake's own `CONNECT_TIMEOUT_SECS` (20 s) plus the longest `join_delay` any test uses, because the one
#: thing teardown must not do is cancel a task that is awaiting `room.connect()`: a half-connected room
#: then blocks every later call on it for its full timeout. Past this bound a straggler IS cancelled (a
#: hung teardown helps nobody), but it is reported, not swallowed.
LAUNCH_DRAIN_SECS = 30.0


class Launcher:
    """Turns "the Synthesia API answered 201" into "a fake worker has published both its tracks".

    This is the piece that makes the integration tier test the *product's* launch path rather than a
    test's imitation of it. The fake API's ``before`` hook runs **inside** the request handler, so a
    hook that awaits ``FakeAvatarWorker.start()`` holds the 201 open until the worker's tracks are
    published, which is exactly what the real Synthesia API guarantees. Every assertion about
    ``lk.publish_on_behalf`` downstream of that is therefore made against the very token the product
    minted and handed over, not one the test wrote down twice.

    Only ``2xx`` entries are hooked. A ``[503, 201]`` script must launch exactly **one** worker: a retry
    re-sends the same token for one session, and that would be untestable if the fixture launched a
    worker per HTTP attempt. An entry that already carries a ``before`` is left alone, so a test that
    needs a non-2xx entry hooked (the same-identity replacement test, where the timed-out first attempt
    is the slow worker) attaches :meth:`hook` to it by hand.

    Attributes:
        dev: The dev server every launched fake connects to.
        fakes: Every ``FakeAvatarWorker`` launched, in launch order.
    """

    def __init__(self, dev: LiveKitDev) -> None:
        self.dev = dev
        self.fakes: list[FakeAvatarWorker] = []
        self._apis: list[FakeSynthesiaApi] = []
        # Strong references: the loop keeps only weak ones, and a deferred `start()` that is garbage
        # collected mid-connect is a fake that silently never joins.
        self._launches: set[asyncio.Task[None]] = set()

    async def api(
        self, script: list[ScriptedResponse] | None = None, **fake_worker_options: Any
    ) -> FakeSynthesiaApi:
        """A started ``FakeSynthesiaApi`` whose successful responses launch a fake worker.

        Args:
            script: The scripted responses, or ``None`` for a single 201. Every ``2xx`` entry without a
                ``before`` of its own gets the launch hook.
            **fake_worker_options: Passed to every ``FakeAvatarWorker`` this script launches (``join_delay``,
                ``never_join``, ``leave_after``, ``status_delay``, ``tone_hz``, ``avatar_ids``). An
                unknown name raises inside the hook, which the fake API turns into a 500 and records in
                ``hook_errors`` — loud, as a broken fixture should be.

        Returns:
            The running server; closed by the fixture's teardown. Read ``api.base_url``,
            ``api.requests``.
        """
        entries = [
            dataclasses.replace(entry, before=self.hook(**fake_worker_options))
            if 200 <= entry.status < 300 and entry.before is None
            else entry
            for entry in (script if script is not None else [ScriptedResponse(201, dict(SESSION_BODY))])
        ]
        api = FakeSynthesiaApi(entries)
        await api.__aenter__()
        self._apis.append(api)
        return api

    def hook(self, **fake_worker_options: Any) -> Callable[[RecordedRequest], Awaitable[None]]:
        """A ``ScriptedResponse.before`` that launches one fake worker with these options."""
        return partial(self._launch, dict(fake_worker_options))

    async def _launch(self, options: Mapping[str, Any], req: RecordedRequest) -> None:
        """Launch one fake worker from the request body, the way the Synthesia API launches the real one.

        The worker learns its avatar ids from the launch request (``avatarIds``) rather than from the
        test, unless the test said otherwise: that is how the real one learns them, and it keeps a
        readiness or swap test from stating the same ids in two places and getting away with them
        disagreeing.

        Whether this **awaits** the join is the whole point of the fixture: a worker with no modelled
        delay is joined and publishing before the 201 goes out, and one with ``join_delay`` or
        ``never_join`` is deliberately not, which is what the join-timeout and same-identity
        replacement paths are made of.
        """
        body = req.json if isinstance(req.json, Mapping) else {}
        launch_options = dict(options)
        launch_options.setdefault("avatar_ids", list(body.get("avatarIds") or ()))
        fake = FakeAvatarWorker(body["livekitUrl"], body["livekitToken"], **launch_options)
        self.fakes.append(fake)
        if fake.join_delay == 0 and not fake.never_join:
            await fake.start()
            return
        task = asyncio.ensure_future(fake.start())
        task.set_name(f"launcher-start-{len(self.fakes)}")
        self._launches.add(task)
        task.add_done_callback(self._launches.discard)

    def debug_dump(self) -> str:
        """Every fake's recorders and every request the fake API saw: the message for a failing assertion.

        Secret-free by construction: ``livekitToken`` goes through ``redact_mapping``, and the request
        headers (which carry the ``Authorization`` bearer) are not printed at all. A CI log that quotes a
        credential to explain a failure is still a credential in a CI log.
        """
        lines = [f"Launcher: {len(self.fakes)} fake worker(s), {len(self._apis)} fake API server(s)"]
        for index, fake in enumerate(self.fakes):
            lines.append(f"--- fake worker {index} ---")
            lines.append(fake.debug_dump())
        for api in self._apis:
            lines.append(f"--- fake API {api.base_url} ---")
            for number, request in enumerate(api.requests, start=1):
                body = redact_mapping(request.json) if isinstance(request.json, Mapping) else request.json
                lines.append(f"  {number}. {request.method} {request.path} {body}")
            if api.hook_errors:
                lines.append(f"  hook_errors={[repr(err) for err in api.hook_errors]}")
        return "\n".join(lines)

    async def aclose(self) -> None:
        """Stop everything this launcher started, in the one order that cannot wedge a room.

        Deferred ``start()`` tasks are **joined**, not cancelled (see ``LAUNCH_DRAIN_SECS``), then the
        fakes are stopped, then the API servers close, and only then are the rooms deleted: deleting a
        room out from under a still-connected fake would make its teardown report failures that say
        nothing about the test.

        ``asyncio.wait``, not ``wait_for``: a ``wait_for`` timeout cancels what it was waiting on, which
        is the one thing this must not do to a task mid-``room.connect()``. A straggler past the drain is
        cancelled explicitly below, as a last resort and with its own message, rather than silently as a
        side effect of the timeout.
        """
        launch_failures: list[BaseException] = []
        if self._launches:
            done, pending = await asyncio.wait(self._launches, timeout=LAUNCH_DRAIN_SECS)
            launch_failures = [err for task in done if (err := task.exception()) is not None]
            for task in pending:
                task.cancel()
            if pending:
                launch_failures.append(
                    TimeoutError(
                        f"{len(pending)} FakeAvatarWorker.start() did not finish in {LAUNCH_DRAIN_SECS}s"
                    )
                )

        for fake in self.fakes:
            with contextlib.suppress(Exception):
                await fake.stop()
        for api in self._apis:
            with contextlib.suppress(Exception):
                await api.__aexit__(None, None, None)

        # Every room a launch ever named, deduplicated and in launch order. The room name comes from the
        # token the product minted, so this covers a room whose name no fixture knows (a relayed join can
        # generate its own) as well as the one `room_name` already deletes.
        for name in dict.fromkeys(fake.room_name for fake in self.fakes if fake.room_name):
            await delete_room(self.dev, name)

        # Last, and never skipped: a launch that failed must not pass for "the worker never joined".
        # An AWAITED launch surfaces through the fake API (the hook raises -> 500 -> `hook_errors`); a
        # DEFERRED one has nowhere to surface but here, and the two paths are indistinguishable to the
        # product, so both are asserted together. In the fixture's `finally`, so pytest reports this as a
        # teardown error *alongside* any test failure rather than in place of it.
        problems = [repr(err) for err in launch_failures]
        problems += [repr(err) for api in self._apis for err in api.hook_errors]
        assert not problems, (
            "a fake worker launch failed; the test's verdict cannot be trusted:\n  " + "\n  ".join(problems)
        )
