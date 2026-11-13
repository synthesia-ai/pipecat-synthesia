"""A slow byte-stream open, write and close, each in the order rtc does it.

Not a test module (no `test_` prefix). Three holds, used by the tests about a byte stream that is still
opening, writing or closing when something else happens:

* `GatedOpenParticipant`: rtc sends the open request before it waits for the writer, so the open is
  recorded at once (its header may be on the wire) and only the writer is held.
* `HeldWriteParticipant`: the write of one chosen chunk, on a stream already open, waits for a gate before
  it is recorded.
* `hold_after_the_close_request`: rtc 1.1.19's `ByteStreamWriter.aclose()` sends the close request in its
  first step, with no `await` before it, and then waits for the acknowledgement. So the fake writer is
  closed at once and only the acknowledgement is held. Holding the whole `aclose()` instead, as
  `test_utterance_streamer.py`'s `hold_the_close` does, models a close whose request has not gone out,
  which rtc 1.1.19 cannot produce once `aclose()` has started.
"""

from __future__ import annotations

import asyncio

from tests.fakes.fake_rtc import FakeByteStreamWriter, FakeLocalParticipant
from tests.unit._session_rig import BOT

__all__ = ["GatedOpenParticipant", "HeldWriteParticipant", "hold_after_the_close_request"]


class GatedOpenParticipant(FakeLocalParticipant):
    """While `open_gate` is set, an open is recorded but its writer is held until the gate opens."""

    def __init__(self) -> None:
        super().__init__(BOT)
        self.open_gate: asyncio.Event | None = None

    async def stream_bytes(
        self,
        name: str,
        *,
        topic: str,
        destination_identities: list[str],
        attributes: dict[str, str],
        open_delay: float | None = None,
    ) -> FakeByteStreamWriter:
        gate = self.open_gate
        writer = await super().stream_bytes(
            name,
            topic=topic,
            destination_identities=destination_identities,
            attributes=attributes,
            open_delay=open_delay,
        )
        if gate is not None:
            await gate.wait()
        return writer


class HeldWriteParticipant(FakeLocalParticipant):
    """Every byte stream's write of `hold_chunk` waits for `gate`; `entered` is set when that write starts."""

    def __init__(self, hold_chunk: bytes) -> None:
        super().__init__(BOT)
        self.hold_chunk = hold_chunk
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()

    async def stream_bytes(
        self,
        name: str,
        *,
        topic: str,
        destination_identities: list[str],
        attributes: dict[str, str],
        open_delay: float | None = None,
    ) -> FakeByteStreamWriter:
        writer = await super().stream_bytes(
            name,
            topic=topic,
            destination_identities=destination_identities,
            attributes=attributes,
            open_delay=open_delay,
        )
        write = writer.write

        async def held_write(data: bytes) -> None:
            if data == self.hold_chunk:
                self.entered.set()
                await self.gate.wait()
            await write(data)

        writer.write = held_write  # type: ignore[method-assign]
        return writer


def hold_after_the_close_request(writer: FakeByteStreamWriter) -> asyncio.Event:
    """Close `writer` at once when `aclose()` starts, then hold the acknowledgement; returns the release."""
    release = asyncio.Event()
    request_close = writer.aclose

    async def aclose() -> None:
        await request_close()
        await release.wait()

    writer.aclose = aclose  # type: ignore[method-assign]
    return release
