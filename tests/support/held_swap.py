"""`HeldSwapSession`: a `FakeAvatarSession` whose first swap is held, for tests of swaps queued behind it."""

from __future__ import annotations

import asyncio
from typing import Any

from tests.fakes.fake_room import FakeAvatarSession

__all__ = ["HeldSwapSession"]


class HeldSwapSession(FakeAvatarSession):
    """Its first swap is recorded, then waits for `release` before it returns, as a swap to a preparing
    avatar waits for it to be ready. Other keywords go to `FakeAvatarSession`."""

    def __init__(self, release: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._release = release

    async def swap_avatar(self, avatar_id: str, *, timeout: float) -> str:  # noqa: ASYNC109
        swapped = await super().swap_avatar(avatar_id, timeout=timeout)
        if len(self.swap_calls) == 1:
            await self._release.wait()
        return swapped
