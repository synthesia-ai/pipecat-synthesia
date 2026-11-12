"""Who is in the call: greet the first human, end the pipeline once the last one has gone.

Both examples need the same two moments, so the bookkeeping lives here once, the way the brain
lives in `_brain.py`. It works in participant ids only: each example turns its transport's handler
argument into an id (a LiveKit identity, Daily's `participant["id"]`, SmallWebRTC's `pc_id`) and
passes that in, and acts on the answer.

**Why "the last human", not "a human".** Ending the pipeline takes the bot out of the room, and the
avatar session with it. Ending on the first leave would take the avatar away from everyone still in
the call. Never ending is no better: the bot leaving once nobody is left is what stops billing
promptly.

**Why the end waits out a grace.** In a direct join, and over Daily, a human can leave and come
straight back under the same identity: when a second connection takes the identity over, the leave
and the join come in the same instant (seen once by hand against a local livekit-server, not in a
test), and when a dropped connection reconnects, moments apart.
Ending on the leave would end the call under the person rejoining, so there the call ends only once
nobody has been present for `NO_HUMANS_GRACE_SECS`, and any human joining in the meantime cancels
it. In a direct join that cannot help when the last human's client said it was leaving (LiveKit
reports that as `CLIENT_INITIATED`, as when the app calls `disconnect()`): the avatar worker then
ends the session at once, and the call ends with it. Whether a browser reload or a closed tab is
such a client-initiated leave is inferred, not checked. Over SmallWebRTC a reconnect is a new call,
so nothing can cancel a grace there and the relayed-join example uses none.

**Why every method is synchronous.** Pipecat runs each event handler in its own task, so a join and
a leave that arrive close together can interleave at any `await`. Each method updates the set and
returns its decision before the caller awaits anything, so events apply in the order they arrived.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Awaitable, Callable, Iterable

from loguru import logger

__all__ = ["NO_HUMANS_GRACE_SECS", "WORKER_NO_HUMANS_GRACE_SECS", "Action", "Presence"]

#: The avatar worker's own grace before it stops a session for lack of humans, 5.0 s by default.
#: Copied, because the examples cannot import the worker. It applies to every way the last human can
#: leave except one: when their client said it was leaving (LiveKit's disconnect reason
#: `CLIENT_INITIATED`), the worker stops at once, cancelling a grace already running.
WORKER_NO_HUMANS_GRACE_SECS = 5.0

#: How long nobody must be present before the call ends: one second more than the worker's grace. In a
#: direct join the worker sees the humans, so while it uses its default its stop comes first, and the
#: session normally ends with `no_humans_left`; that assumes the avatar has left within the second,
#: which has not been measured. In a relayed join over Daily the worker cannot see the
#: humans, so this grace is the only one, and the session ends with `pipeline_ended`.
NO_HUMANS_GRACE_SECS = WORKER_NO_HUMANS_GRACE_SECS + 1.0


class Action(enum.Enum):
    """What the example should do after a presence change."""

    NONE = "none"
    GREET = "greet"
    END = "end"


class Presence:
    """The humans in the call, and when to greet and when to end.

    - Greets once, when the first human is present, including one who was there before the bot.
    - Ends once, by awaiting ``on_end``, when the set has stayed empty for ``grace_secs`` after
      having been non-empty. A human joining during the grace cancels it.
    - Ids in ``ignore`` (the avatar, in a direct join) never count, joining or leaving.
    - A leave for an id that is not present changes nothing.
    - After ``close()``, it never ends the call.

    Args:
        on_end: Awaited once, to end the call.
        ignore: Ids that are participants but not humans.
        grace_secs: How long the set must stay empty before the call ends; by default
            `NO_HUMANS_GRACE_SECS`, read when the Presence is made.
    """

    def __init__(
        self,
        *,
        on_end: Callable[[], Awaitable[None]],
        ignore: Iterable[str] = (),
        grace_secs: float | None = None,
    ) -> None:
        self._on_end = on_end
        self._ignore = frozenset(ignore)
        self._grace_secs = NO_HUMANS_GRACE_SECS if grace_secs is None else grace_secs
        self._humans: set[str] = set()
        self._greeted = False
        self._ended = False
        self._grace: asyncio.Task[None] | None = None

    def __len__(self) -> int:
        """How many humans are present."""
        return len(self._humans)

    def seed(self, ids: Iterable[str]) -> Action:
        """Add the humans already present when the bot connected. Overlapping a join is harmless."""
        for participant_id in ids:
            self._add(participant_id)
        return self._after_arrival()

    def joined(self, participant_id: str) -> Action:
        """A participant joined."""
        self._add(participant_id)
        return self._after_arrival()

    def left(self, participant_id: str) -> None:
        """A participant left. If nobody is left, the grace starts; ``on_end`` follows if it runs out."""
        if participant_id not in self._humans:
            return
        self._humans.remove(participant_id)
        if self._humans or self._ended:
            return
        self._grace = asyncio.get_running_loop().create_task(self._end_after_grace(), name="presence-grace")

    def close(self) -> None:
        """The call is over some other way: cancel a grace that is running, and never end it from here."""
        self._cancel_grace()
        self._ended = True

    async def _end_after_grace(self) -> None:
        try:
            await asyncio.sleep(self._grace_secs)
            self._ended = True
            await self._on_end()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ending the call failed")
        finally:
            # A grace cancelled by a join may finish only after a later leave started the next one.
            if self._grace is asyncio.current_task():
                self._grace = None

    def _cancel_grace(self) -> None:
        """Cancel the grace while it is still waiting; never `on_end` once it has started."""
        if self._grace is not None and not self._ended:
            self._grace.cancel()
            self._grace = None

    def _add(self, participant_id: str) -> None:
        if participant_id in self._ignore:
            return
        self._humans.add(participant_id)
        self._cancel_grace()

    def _after_arrival(self) -> Action:
        if not self._humans or self._greeted:
            return Action.NONE
        self._greeted = True
        return Action.GREET
