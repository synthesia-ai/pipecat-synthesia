"""Two minimal raw-``rtc`` senders: one that obeys the avatar protocol, one that breaks it on request.

**These deliberately import nothing from ``pipecat_synthesia``, not even ``protocol.py``.** The wire
strings are written out below by hand. That is the point: ``GoodSender`` is an independent restatement of
what livekit-agents' ``DataStreamAudioOutput`` puts on the wire, so it is a second opinion about the
contract rather than an echo of ours. If somebody renames a constant in ``protocol.py``, the product and
``FakeAvatarWorker`` move together, and the self-tests that use these senders stay put and notice.

``GoodSender`` is also the reference for what ``UtteranceStreamer`` should do to a fake worker: the same
thing, in thirty lines with no state machine.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from livekit import rtc

__all__ = ["BadSender", "GoodSender"]

# Hand-written on purpose; see the module docstring. The values livekit-agents' avatar data stream uses.
AUDIO_STREAM_TOPIC = "lk.audio_stream"
RPC_PLAYBACK_STARTED = "lk.playback_started"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
RPC_CLEAR_BUFFER = "lk.clear_buffer"


class GoodSender:
    """Everything the worker requires and nothing else: two handlers, one stream, correct attributes."""

    def __init__(
        self, room: rtc.Room, *, avatar_identity: str, sample_rate: int = 24000, num_channels: int = 1
    ) -> None:
        self.room = room
        self.avatar_identity = avatar_identity
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        #: Payloads of every ``lk.playback_started`` / ``lk.playback_finished`` the worker sent us.
        self.started_calls: list[str] = []
        self.finished_calls: list[dict[str, Any]] = []

    @property
    def attributes(self) -> dict[str, str]:
        """The two required stream attributes, as decimal strings: the receiver parses them with ``int()``."""
        return {"sample_rate": str(self.sample_rate), "num_channels": str(self.num_channels)}

    def register_handlers(self) -> None:
        """Register both playback RPCs before the first byte: the only ordering the worker survives."""
        participant = self.room.local_participant
        participant.register_rpc_method(RPC_PLAYBACK_STARTED, self._on_started)
        participant.register_rpc_method(RPC_PLAYBACK_FINISHED, self._on_finished)

    def _on_started(self, data: rtc.RpcInvocationData) -> str:
        self.started_calls.append(data.payload)
        return "ok"

    def _on_finished(self, data: rtc.RpcInvocationData) -> str:
        self.finished_calls.append(json.loads(data.payload or "{}"))
        return "ok"

    async def open(self, attributes: dict[str, str] | None = None) -> rtc.ByteStreamWriter:
        """Open one ``lk.audio_stream`` byte stream addressed to the avatar."""
        return await self.room.local_participant.stream_bytes(
            name=f"AUDIO_{uuid.uuid4().hex[:12]}",
            topic=AUDIO_STREAM_TOPIC,
            destination_identities=[self.avatar_identity],
            attributes=self.attributes if attributes is None else attributes,
        )

    async def send(self, payload: bytes, *, attributes: dict[str, str] | None = None) -> None:
        """One utterance: open, write, close. Closing the stream *is* the end-of-utterance marker."""
        writer = await self.open(attributes)
        await writer.write(payload)
        await writer.aclose()

    async def clear_buffer(self) -> str:
        """Barge-in: ``lk.clear_buffer`` with an empty payload."""
        return await self.room.local_participant.perform_rpc(
            destination_identity=self.avatar_identity,
            method=RPC_CLEAR_BUFFER,
            payload="",
            response_timeout=5.0,
        )


class BadSender(GoodSender):
    """One method per protocol violation, each independently switchable.

    A subclass of ``GoodSender`` so that every violation differs from the correct behaviour in exactly
    one respect: if the bad path had its own copy of the stream-opening code, a test could pass because
    the two copies drifted rather than because the fake detected anything.
    """

    async def send_without_playback_finished(self, payload: bytes) -> None:
        """Stream while registering only ``lk.playback_started``: fatal, the worker ends its session."""
        self.room.local_participant.register_rpc_method(RPC_PLAYBACK_STARTED, self._on_started)
        await self.send(payload)

    async def send_with_raising_playback_started(self, payload: bytes) -> None:
        """Register both handlers, but let ``lk.playback_started`` raise: a playback handler must never raise.

        rtc turns a handler exception into ``APPLICATION_ERROR`` at the caller. That is not the
        ``UNSUPPORTED_METHOD`` the receiver tolerates for this method, so the receiver retries three times
        and then raises, ending the worker's session.
        """

        def _raise(_data: rtc.RpcInvocationData) -> str:
            raise RuntimeError("this handler is broken")

        participant = self.room.local_participant
        participant.register_rpc_method(RPC_PLAYBACK_STARTED, _raise)
        participant.register_rpc_method(RPC_PLAYBACK_FINISHED, self._on_finished)
        await self.send(payload)

    async def send_without_sample_rate(self, payload: bytes) -> None:
        """Stream with ``num_channels`` but no ``sample_rate``: fatal, the receiver raises."""
        await self.send(payload, attributes={"num_channels": str(self.num_channels)})

    async def send_overlapping(self, payload: bytes) -> None:
        """Open a second stream before closing the first, breaking the *one-open-stream rule*."""
        first = await self.open()
        second = await self.open()
        await first.write(payload)
        await second.write(payload)
        await first.aclose()
        await second.aclose()
