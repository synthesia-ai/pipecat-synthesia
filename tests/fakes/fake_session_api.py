"""A recording stand-in for `SessionApiClient.create_session`, for the layers above the HTTP client.

`tests/fakes/fake_api.py` is the HTTP server the real client is tested against; this is the *client*
side, for code that takes a `LaunchContext.api` and only ever calls `create_session`. It records every
call, answers with a `SessionCreated`, and can be told to delay, block, raise, or run a hook. The hook is
how a session test launches a `FakeAvatarWorker` with the very token the session minted.

It records; it does not decide. There is no retry logic here because there is nothing to retry: that
behaviour belongs to `SessionApiClient` and is tested in `test_api.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pipecat_synthesia.api import SessionCreated

__all__ = ["FakeSessionApiClient"]

#: `on_create(avatar_ids, livekit_url, livekit_token)`; a returned `SessionCreated` replaces the default.
CreateHook = Callable[[Sequence[str], str, str], Awaitable[SessionCreated | None]]


class FakeSessionApiClient:
    """Duck-types `SessionApiClient` for `create_session` only.

    Args:
        session_id: The `id` of the fabricated 201 body.
        request_id: The `Request-Id` the fabricated response carries.
        delay: Seconds to sleep inside `create_session` before answering (models a slow API).
        gate: When set, `create_session` waits on it before answering — a hold for ordering tests.
        error: Raised by `create_session` instead of answering, after `delay`/`gate`.
        on_create: Awaited after the call is recorded; may return a `SessionCreated` to answer with.
        retries: How many retries to report through the caller's `on_retry`, as `SessionApiClient` does
            before each retry's wait; each waits for `retry_gate` first when one is given.
        retry_gate: When set, each reported retry waits on it, so a test can place the retry after the
            avatar's video (an *early start*) or before it.

    Attributes:
        calls: One dict per call, with the three request keys the real client sends.
        returned: How many calls have answered with a 201.
    """

    def __init__(
        self,
        *,
        session_id: str = "ses_fake",
        request_id: str | None = "req_fake",
        delay: float = 0.0,
        gate: asyncio.Event | None = None,
        error: BaseException | None = None,
        on_create: CreateHook | None = None,
        retries: int = 0,
        retry_gate: asyncio.Event | None = None,
    ) -> None:
        self.session_id = session_id
        self.request_id = request_id
        self.delay = delay
        self.gate = gate
        self.error = error
        self.on_create = on_create
        self.retries = retries
        self.retry_gate = retry_gate
        #: How many retries have been reported through `on_retry` so far.
        self.retries_reported = 0
        self.calls: list[dict[str, Any]] = []
        self.returned = 0

    async def create_session(
        self,
        *,
        avatar_ids: Sequence[str],
        livekit_url: str,
        livekit_token: str,
        on_retry: Callable[[], None] | None = None,
    ) -> SessionCreated:
        self.calls.append(
            {"avatar_ids": tuple(avatar_ids), "livekit_url": livekit_url, "livekit_token": livekit_token}
        )
        for _ in range(self.retries):
            if self.retry_gate is not None:
                await self.retry_gate.wait()
            if on_retry is not None:
                on_retry()
            self.retries_reported += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        created = SessionCreated(session_id=self.session_id, request_id=self.request_id)
        if self.on_create is not None:
            replacement = await self.on_create(avatar_ids, livekit_url, livekit_token)
            if replacement is not None:
                created = replacement
        self.returned += 1
        return created

    def __repr__(self) -> str:
        return f"FakeSessionApiClient(calls={len(self.calls)}, returned={self.returned})"
