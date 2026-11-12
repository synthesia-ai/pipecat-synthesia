"""``FakeAvatarWorker``: a stand-in for the Synthesia avatar worker, faithful at the wire and nowhere else.

It joins a real livekit-server, publishes the two tracks the real worker publishes, and applies exactly the
rules the worker's receiver (livekit-agents' ``DataStreamAudioReceiver``, in the version the worker runs)
applies to what the bot sends it. It is the oracle for the worker protocol: both playback handlers, the two
stream attributes, one open stream at a time, clear buffer, and the avatar token's kind and
``lk.publish_on_behalf``. **The fake records; the only decisions it makes are the ones the real receiver
makes.** Every rule it enforces has a comment naming the receiver behaviour it mirrors, and a rule without
one does not belong here: an oracle that invents a constraint fails honest senders, and one that drops a
constraint makes the integration tier decorative.

**Where it knowingly differs from the real receiver**, and why a bot cannot tell:

* The receiver sends its outgoing RPCs from a separate ``_send_task``; this fake sends them inline from the
  reader loop. The bot sees the same methods in the same order, but not the same back-pressure: a bot whose
  ``lk.playback_started`` handler *hangs* stalls this fake's reading for up to
  ``RPC_MAX_ATTEMPTS * RPC_RESPONSE_TIMEOUT_SECS``, where the real receiver would keep reading. Accepted,
  because the tests assert on ordering, and a second task is a second concurrency seam in an oracle that
  must be obviously correct.
* The real ``lk.playback_started`` fires on the first non-silent *rendered* frame, and
  ``playback_position`` counts audio actually handed to LiveKit. Nothing renders here, so the fake fires on
  the first received chunk and counts received bytes. Assert on ``interrupted`` and on counts, never on the
  value of ``playback_position``.
* RPC handlers are registered right **after** ``connect()``, because ``room.local_participant`` raises until
  the room is connected. The real worker does the same, with no await in between, so the handlers are in
  place before any participant can call them.
* A stream idle timeout and supersede logic are **not** modelled: the worker's receiver does not have
  them, and modelling them would make the fake more forgiving than the worker.
  Streams are read strictly one at a time (``_recv_loop``), so an unclosed stream blocks every stream behind
  it, as it wedges the real receiver; ``open_streams`` and ``overlapping_streams`` make that visible.

**livekit behaviour found while building it** (livekit rtc 1.1.19, livekit-server 1.9.1):

1. ``set_attributes`` does nothing without the ``can_update_own_metadata`` grant, and the server discards
   the write **silently**, so ``avatar_ready`` / ``avatar_status`` would never arrive and nothing would say
   why. ``mint_dev_token`` and the product's avatar token minter both set it.
2. ``LocalParticipant.set_attributes`` does not return until the FFI acknowledges, which takes a flat 5 s;
   see ``ATTRIBUTE_ACK_TIMEOUT_SECS``.
3. Cancelling a task that is awaiting ``room.disconnect()`` leaves the room neither connected nor closed,
   and every later call on it then blocks for its full timeout. Hence the single, uncancellable
   ``_leave_soon()`` task; see ``stop()``.
4. ``LocalParticipant.unpublish_track`` waits for a server acknowledgement, so it **never returns** once the
   room is gone server-side (a relayed join deletes its private room at teardown) or the server ejected the
   participant, and ``room.isconnected()`` still says ``True`` for a while. ``room.disconnect()`` is a
   local teardown and drops the publications anyway, so ``stop()`` does not unpublish.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from livekit import rtc

from pipecat_synthesia.protocol import (
    ATTR_AVATAR_READY,
    ATTR_AVATAR_STATUS,
    AUDIO_STREAM_TOPIC,
    AVATAR_ID_PREFIX,
    AVATAR_STATE_PREPARING,
    AVATAR_STATE_READY,
    PROTOCOL_VERSION,
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_REJECT,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    RPC_SWAP_AVATAR,
    STREAM_ATTR_NUM_CHANNELS,
    STREAM_ATTR_SAMPLE_RATE,
    SWAP_DEFAULT,
    SWAP_RESPONSE_STATUS_OK,
    TOKEN_ATTR_PROTOCOL,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
    WORKER_AUDIO_CHANNELS,
    WORKER_AUDIO_SAMPLE_RATE,
    WORKER_AUDIO_TRACK_NAME,
    WORKER_VIDEO_FPS,
    WORKER_VIDEO_HEIGHT,
    WORKER_VIDEO_TRACK_NAME,
    WORKER_VIDEO_WIDTH,
)
from tests.support.livekit_dev import LiveKitDev, mint_dev_token

__all__ = ["DEFAULT_AVATAR_IDENTITY", "FakeAvatarWorker"]

DEFAULT_AVATAR_IDENTITY = "synthesia-avatar-agent"

#: The worker's own code for "that avatar is not ready yet". Defined here rather than imported because
#: `protocol.SWAP_ERROR_CODES` is the *client's* list of codes it recognises; the fake is the counterparty
#: that emits them, and an oracle that took the string from the code under test could never catch that code
#: misreading it. The self-tests assert the two agree.
SWAP_CODE_AVATAR_PREPARING = "AVATAR_PREPARING"

#: The receiver's retry policy: three attempts, 0.1 s apart.
RPC_MAX_ATTEMPTS = 3
RPC_RETRY_GAP_SECS = 0.1
#: Not the receiver's value (it passes none, so the FFI default applies); bounded here so a bot whose
#: handler never answers fails this fake sooner.
RPC_RESPONSE_TIMEOUT_SECS = 5.0

CONNECT_TIMEOUT_SECS = 20.0
DISCONNECT_TIMEOUT_SECS = 5.0
#: Gap between "every avatar preparing" and "every avatar ready".
READY_STEP_SECS = 0.2
#: `LocalParticipant.set_attributes` does not return until the FFI acknowledges the write, which takes a
#: flat 5 s (livekit rtc 1.1.19, livekit-server 1.9.1), while the new attributes reach remote participants
#: in about 10 ms. Waiting for the ack would put 10 s of dead time into every readiness sequence and every
#: test that waits for one. The write itself is what matters, so the ack gets a short window and is then
#: abandoned.
ATTRIBUTE_ACK_TIMEOUT_SECS = 1.0
#: 20 ms of audio per capture, the frame size the worker's generator produces.
AUDIO_FRAME_MS = 20
#: s16le: two bytes per sample per channel. The divisor in `playback_position`.
BYTES_PER_SAMPLE = 2


@dataclass
class _Segment:
    """One utterance: the bytes of a single ``lk.audio_stream`` byte stream, and what we told the bot."""

    sample_rate: int
    num_channels: int
    bytes_received: int = 0
    cleared: bool = False
    started_sent: bool = False

    def position_secs(self) -> float:
        """Seconds of audio received so far — the fake's stand-in for the worker's playback position."""
        divisor = BYTES_PER_SAMPLE * self.sample_rate * self.num_channels
        return self.bytes_received / divisor if divisor else 0.0


@dataclass
class _Token:
    """What the avatar's JWT says, read without verifying anything."""

    identity: str = ""
    kind: str = ""
    publish_on_behalf: str = ""
    room: str = ""
    violations: list[str] = field(default_factory=list)


def _decode_unsigned(token: str) -> _Token:
    """Read ``sub``, ``kind``, ``video.room`` and ``lk.publish_on_behalf`` out of a JWT. Never raises.

    Base64 + ``json``, not ``pyjwt``: this module is the *counterparty*, and a counterparty that can only
    read tokens its peer's library produced cannot detect a malformed one. Anything unreadable becomes a
    recorded violation, because a fake that crashed on a bad token would report a product bug as an error
    in the test harness.
    """
    result = _Token()
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError, binascii.Error, UnicodeDecodeError) as err:
        result.violations.append(f"token is not a readable JWT: {type(err).__name__}")
        return result
    if not isinstance(claims, dict):
        result.violations.append("token payload is not a JSON object")
        return result
    attributes = claims.get("attributes") or {}
    result.identity = str(claims.get("sub", ""))
    result.kind = str(claims.get("kind", ""))
    result.publish_on_behalf = str(attributes.get(TOKEN_ATTR_PUBLISH_ON_BEHALF, ""))
    result.room = str((claims.get("video") or {}).get("room", ""))
    # The real avatar joins as kind=agent. The worker counts standard participants as humans
    # (*no-humans-left rule*), so a standard avatar would count itself as one.
    if result.kind != "agent":
        result.violations.append(f"token kind is {result.kind or '<absent>'!r}, expected 'agent'")
    # The real worker rejects the launch when the token has no `lk.publish_on_behalf` attribute.
    if not result.publish_on_behalf:
        result.violations.append(f"token is missing the {TOKEN_ATTR_PUBLISH_ON_BEHALF} attribute")
    return result


class FakeAvatarWorker:
    """A fake avatar worker in a real LiveKit room.

    Construct it with the very token the product minted for the avatar: that is what makes
    ``lk.publish_on_behalf`` a real assertion rather than a restatement of the test's setup. Then
    ``await fake.start()``, which returns only once **both tracks are published**, so the ``Launcher`` keeps
    the real API's promise that a 201 means the avatar's tracks are published.

    Args:
        url: LiveKit WebSocket URL.
        token: The avatar's JWT. Read unsigned; never logged.
        join_delay: Seconds to wait before connecting — models a slow worker launch.
        never_join: Register nothing, connect nothing. Models a launch that silently never arrives.
        leave_after: Disconnect this many seconds after joining. Models the avatar leaving mid-session.
        status_delay: Seconds before the readiness attributes are published. A value longer than the test
            keeps every avatar ``preparing``, which is how the ``AVATAR_PREPARING`` swap path is reached.
        avatar_ids: Avatar ids for this session, prefixed or not; stored and reported unprefixed, because
            the worker keys its readiness attributes that way (*unprefixed ids on the wire*).
        tone_hz: Publish a sine at this frequency instead of digital silence, so a test can prove the bot
            does **not** hear the avatar.
        swap_delay: Seconds ``swapAvatar`` takes before it answers. Models the worker's handler, which can
            block for up to 180 s while the target avatar prepares.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        join_delay: float = 0.0,
        never_join: bool = False,
        leave_after: float | None = None,
        status_delay: float = 0.0,
        avatar_ids: list[str] | tuple[str, ...] | None = None,
        tone_hz: float | None = None,
        swap_delay: float = 0.0,
    ) -> None:
        self.url = url
        self._token = token
        self.join_delay = join_delay
        self.never_join = never_join
        self.leave_after = leave_after
        self.status_delay = status_delay
        self.tone_hz = tone_hz
        self.swap_delay = swap_delay

        claims = _decode_unsigned(token)
        self.identity = claims.identity
        self.publish_on_behalf = claims.publish_on_behalf
        self.room_name = claims.room

        #: Every avatar id this session declared, unprefixed and in declaration order. ``default``
        #: resolves to the first.
        self.avatar_ids: list[str] = [
            aid[len(AVATAR_ID_PREFIX) :] if aid.startswith(AVATAR_ID_PREFIX) else aid
            for aid in (avatar_ids or ("a",))
        ]
        # The worker declares every avatar preparing before preparation begins, so this starts at
        # construction, not when `status_delay` lets the attributes go out.
        self._states: dict[str, str] = dict.fromkeys(self.avatar_ids, AVATAR_STATE_PREPARING)

        # --- recorders: plain public attributes that tests read directly ---
        self.streams: list[dict[str, Any]] = []
        self.open_streams = 0
        self.overlapping_streams = 0
        self.rejected_stream_identities: list[str] = []
        self.attribute_violations: list[dict[str, str]] = []
        self.token_violations: list[str] = list(claims.violations)
        self.clear_buffer_calls = 0
        self.swap_calls: list[dict[str, Any]] = []
        self.fatal_unsupported_playback_finished = False
        self.connected = False

        self._room: rtc.Room | None = None
        #: The utterance being read right now, ``None`` between utterances. This stands for the worker's
        #: "audio playing" state: it is what makes an ``lk.clear_buffer`` that arrives *between* utterances
        #: a no-op here, as it is for the real worker.
        self._segment: _Segment | None = None
        #: Readers waiting their turn, and the event that wakes the one reader loop. The real receiver
        #: works the same way: the stream handler appends to a list, and
        #: ``_recv_task`` pops one reader at a time.
        self._pending: list[tuple[rtc.ByteStreamReader, str]] = []
        self._reader_arrived = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._audio_source: rtc.AudioSource | None = None
        self._video_source: rtc.VideoSource | None = None
        self._leaving: asyncio.Task[None] | None = None
        self._started = False
        self._stopped = False
        self._tone: list[int] = []

    # ------------------------------------------------------------------ construction

    @classmethod
    def for_bot(
        cls,
        dev: LiveKitDev,
        *,
        room_name: str,
        bot_identity: str,
        avatar_identity: str = DEFAULT_AVATAR_IDENTITY,
        **options: Any,
    ) -> FakeAvatarWorker:
        """An unstarted fake holding a valid avatar token it minted itself.

        For the tests with no Synthesia API in the loop (the wire conformance tests and the fake's
        self-tests): they need a correct counterparty, not a launch. The attributes are the ones the
        product's minter puts on a real avatar token, taken from ``protocol.py`` so a rename there reaches
        here.
        """
        token = mint_dev_token(
            dev,
            identity=avatar_identity,
            room=room_name,
            kind="agent",
            attributes={
                TOKEN_ATTR_PUBLISH_ON_BEHALF: bot_identity,
                TOKEN_ATTR_PROTOCOL: PROTOCOL_VERSION,
            },
        )
        return cls(dev.url, token, **options)

    # ------------------------------------------------------------------ lifecycle

    @property
    def room(self) -> rtc.Room:
        """The fake's own ``rtc.Room``. Raises if ``start()`` has not run."""
        if self._room is None:
            raise RuntimeError("FakeAvatarWorker.start() has not been called")
        return self._room

    async def start(self) -> None:
        """Join the room and publish both tracks; returns once they are published.

        Unless ``never_join``, in which case it returns having done nothing observable: no participant,
        no handler, no track. That models a launch the API acknowledged and the worker never honoured,
        which is what the join-timeout paths are about.
        """
        if self._started:
            return
        self._started = True

        room = rtc.Room()
        self._room = room
        room.on("disconnected", self._on_disconnected)
        # register_byte_stream_handler raises ValueError on a duplicate topic, so it happens exactly once
        # per Room, and before connect so that a stream sent the instant we appear cannot arrive unhandled.
        room.register_byte_stream_handler(AUDIO_STREAM_TOPIC, self._on_byte_stream)

        if self.never_join:
            return
        if self.join_delay:
            await asyncio.sleep(self.join_delay)

        await asyncio.wait_for(
            room.connect(self.url, self._token, options=rtc.RoomOptions(auto_subscribe=True)),
            CONNECT_TIMEOUT_SECS,
        )
        self.connected = True

        # No await between connect and registration: this is the worker's own order, and the only one real
        # rtc allows, because `room.local_participant` raises before the room is connected.
        participant = room.local_participant
        participant.register_rpc_method(RPC_CLEAR_BUFFER, self._on_clear_buffer)
        participant.register_rpc_method(RPC_SWAP_AVATAR, self._on_swap_avatar)

        await self._publish_tracks(participant)

        self._spawn(self._recv_loop(), "recv")
        self._spawn(self._readiness_loop(), "readiness")
        if self.leave_after is not None:
            self._spawn(self._leave_loop(), "leave")

    async def stop(self) -> None:
        """Stop every loop, unpublish, disconnect. Idempotent, and safe on a fake that never started."""
        if self._stopped:
            return
        self._stopped = True

        # Drain rather than snapshot-and-clear. rtc's listen task keeps delivering while we await the
        # gather below, and a handler that spawns during that window would otherwise be dropped from
        # `_tasks` by a blind `clear()` — never cancelled, never awaited, and still parked in an
        # outgoing `perform_rpc` after the room is gone. `_spawn` also refuses once `_stopped` is set,
        # so this loop normally runs exactly once.
        while self._tasks:
            batch = list(self._tasks)
            for task in batch:
                task.cancel()
            await asyncio.gather(*batch, return_exceptions=True)
            self._tasks.difference_update(batch)

        for source in (self._audio_source, self._video_source):
            if source is not None:
                with contextlib.suppress(Exception):
                    await source.aclose()
        self._audio_source = self._video_source = None

        # One disconnect, whoever asked for it, awaited under a shield: a `room.disconnect()` cancelled
        # half way wedges the room (finding 3 in the module docstring). A caller that cancels `stop()`
        # still gets its CancelledError (which is why only `Exception` is suppressed), and the disconnect
        # still runs to completion. No `unpublish_track` here, deliberately: see finding 4.
        with contextlib.suppress(Exception):
            await asyncio.shield(self._leave_soon())

    def _leave_soon(self) -> asyncio.Task[None]:
        """Start leaving the room, once, from whatever noticed we must — and return that one task.

        Kept out of ``self._tasks`` on purpose (see ``stop()``): this is the one piece of work that must
        be *joined* rather than cancelled. Callers inside a task ``stop()`` cancels must **not** await the
        returned task bare, for the same reason — cancelling a task cancels whatever future it awaits, so
        a ``_consume`` task that awaited this one would drag the disconnect down with it. Start it and
        move on, or await it under ``asyncio.shield`` as ``stop()`` does. ``connected`` is cleared
        synchronously here, so a test polling for the disconnect sees it without waiting for the round
        trip. Every caller funnels through here, so two disconnects can never run concurrently.
        """
        self.connected = False
        if self._leaving is None:
            self._leaving = asyncio.ensure_future(self._leave())
            self._leaving.set_name("fake-worker-leave")
        return self._leaving

    async def _leave(self) -> None:
        """Leave the room, bounded, swallowing whatever a half-dead connection raises on the way out."""
        room = self._room
        self.connected = False
        if room is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(room.disconnect(), DISCONNECT_TIMEOUT_SECS)

    def _on_disconnected(self, *_args: Any) -> None:
        """Room-side disconnect (kicked, room deleted, server gone). Sync: rtc rejects async handlers."""
        self.connected = False

    def _spawn(self, coro: Any, name: str) -> asyncio.Task[None] | None:
        """Run ``coro`` as a task this fake owns, keeping a strong reference until it finishes.

        Returns ``None`` and closes the coroutine once ``stop()`` has begun. rtc's listen task is still
        delivering events at that point, so without this guard a late RPC or stream handler could start
        work the teardown has already walked past.
        """
        if self._stopped:
            coro.close()
            return None
        task = asyncio.ensure_future(coro)
        task.set_name(f"fake-worker-{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ------------------------------------------------------------------ media

    async def _publish_tracks(self, participant: rtc.LocalParticipant) -> None:
        """Publish ``avatar_video`` and ``avatar_audio`` with the real names, sources and formats.

        Names and sources: ``avatar_audio`` / ``SOURCE_MICROPHONE`` and ``avatar_video`` / ``SOURCE_CAMERA``
        at 512x512 @ 25 fps, 24 kHz mono, as the worker publishes them. Both names come from
        ``protocol.py``: a direct-join test that checks "the avatar's audio track" was unsubscribed is
        matching on these strings.
        """
        self._video_source = rtc.VideoSource(WORKER_VIDEO_WIDTH, WORKER_VIDEO_HEIGHT)
        video_track = rtc.LocalVideoTrack.create_video_track(WORKER_VIDEO_TRACK_NAME, self._video_source)
        self._audio_source = rtc.AudioSource(WORKER_AUDIO_SAMPLE_RATE, WORKER_AUDIO_CHANNELS)
        audio_track = rtc.LocalAudioTrack.create_audio_track(WORKER_AUDIO_TRACK_NAME, self._audio_source)

        # The publications are not kept: nothing unpublishes them (finding 4 in the module docstring).
        await participant.publish_track(
            video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        )
        await participant.publish_track(
            audio_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        self._spawn(self._video_loop(), "video")
        self._spawn(self._audio_loop(), "audio")

    async def _video_loop(self) -> None:
        """A solid frame whose colour changes every frame, at 25 fps.

        Changing colour, not a static green: a capture test that asserts "we received video" against a
        source that only ever sends one frame cannot tell a live track from a stuck one.
        """
        period = 1.0 / WORKER_VIDEO_FPS
        pixels = WORKER_VIDEO_WIDTH * WORKER_VIDEO_HEIGHT
        counter = 0
        while True:
            colour = bytes(((counter * 7) % 256, (counter * 13) % 256, (counter * 29) % 256))
            frame = rtc.VideoFrame(
                WORKER_VIDEO_WIDTH, WORKER_VIDEO_HEIGHT, rtc.VideoBufferType.RGB24, colour * pixels
            )
            source = self._video_source
            if source is None:
                return
            source.capture_frame(frame)
            counter += 1
            await asyncio.sleep(period)

    async def _audio_loop(self) -> None:
        """24 kHz mono: digital silence, or a ``tone_hz`` sine when one was asked for.

        ``AudioSource.capture_frame`` paces itself against the source's queue, so this loop needs no sleep
        and drifts with neither.
        """
        samples = WORKER_AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000
        if self.tone_hz:
            self._tone = _sine_cycle(self.tone_hz, WORKER_AUDIO_SAMPLE_RATE)
        offset = 0
        while True:
            source = self._audio_source
            if source is None:
                return
            frame = rtc.AudioFrame.create(WORKER_AUDIO_SAMPLE_RATE, WORKER_AUDIO_CHANNELS, samples)
            if self._tone:
                view = frame.data
                cycle = self._tone
                for index in range(samples):
                    view[index] = cycle[(offset + index) % len(cycle)]
                offset = (offset + samples) % len(cycle)
            await source.capture_frame(frame)

    # ------------------------------------------------------------------ readiness

    async def _readiness_loop(self) -> None:
        """Publish ``avatar_status`` (all preparing), then ``avatar_status`` + ``avatar_ready`` (all ready).

        Two writes, in that order, because that is what the worker does: every avatar is declared
        preparing before preparation starts, then each is marked ready. A test that only ever saw the final
        state could not tell a worker that declares readiness from one that was born ready.
        """
        if self.status_delay:
            await asyncio.sleep(self.status_delay)
        await self._write_status()
        await asyncio.sleep(READY_STEP_SECS)
        self._states = dict.fromkeys(self.avatar_ids, AVATAR_STATE_READY)
        await self._write_status()

    async def _write_status(self) -> None:
        """Write the two readiness attributes, preserving every other attribute on the participant.

        The worker re-sends its whole current attribute map on every write, so the token attributes
        (``lk.publish_on_behalf`` and the ``synthesia.*`` ones) survive. ``avatar_ready`` is written
        **only** when every avatar is ready, and never as ``"false"``.
        """
        room = self._room
        if room is None or not room.isconnected():
            return
        status = json.dumps(
            {aid: {"state": state} for aid, state in self._states.items()},
            separators=(",", ":"),
            sort_keys=True,
        )
        attributes = {**room.local_participant.attributes, ATTR_AVATAR_STATUS: status}
        if all(state == AVATAR_STATE_READY for state in self._states.values()):
            attributes[ATTR_AVATAR_READY] = "true"
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                room.local_participant.set_attributes(attributes), ATTRIBUTE_ACK_TIMEOUT_SECS
            )

    async def _leave_loop(self) -> None:
        """Disconnect ``leave_after`` seconds after joining — the avatar that goes away mid-call."""
        await asyncio.sleep(self.leave_after or 0.0)
        self._leave_soon()

    # ------------------------------------------------------------------ incoming byte streams

    def _on_byte_stream(self, reader: rtc.ByteStreamReader, participant_identity: str) -> None:
        """Accept or reject a stream synchronously; the body is read later, by ``_recv_loop``.

        The gating decisions happen here rather than in the reader loop on purpose: two stream headers can
        arrive before the loop gets a slot, and an overlap counted after the fact is not counted at all
        (*one-open-stream rule*: the overlap is recorded at the moment it happens).
        """
        if participant_identity != self.publish_on_behalf:
            # The receiver drops streams from any identity but the bound sender silently, in the handler,
            # before the reader is queued. The reader must still be closed, or its FFI subscription leaks.
            self.rejected_stream_identities.append(participant_identity)
            reader.close()
            return

        self.open_streams += 1
        if self.open_streams > 1:
            # Readers are processed strictly one at a time, and the avatar worker's receiver has no
            # supersede and no idle timeout, so a second stream opened before the first closed queues
            # behind it forever. Counted now, while the overlap exists; later, the evidence is gone.
            self.overlapping_streams += 1

        # Everything else waits its turn. The real receiver's handler does only this, append and signal,
        # and parses the attributes and reads the body in `_recv_task` once the reader is popped.
        self._pending.append((reader, participant_identity))
        self._reader_arrived.set()

    async def _recv_loop(self) -> None:
        """Read queued streams strictly one at a time, in arrival order.

        This mirrors the receiver's ``_recv_task``, and the serialisation is the point: in the avatar
        worker's receiver there is no idle timeout and no supersede, so a stream the bot never closes
        blocks every later stream forever. A fake that read them concurrently would give a bot that overlaps
        streams a full set of healthy playback callbacks here and a wedged session in production, which is
        the regression the *one-open-stream rule* exists to prevent.
        """
        while True:
            await self._reader_arrived.wait()
            while self._pending:
                reader, identity = self._pending.pop(0)
                await self._consume(reader, identity)
            self._reader_arrived.clear()

    async def _consume(self, reader: rtc.ByteStreamReader, identity: str) -> None:
        """Read one stream to EOF, recording it and reporting playback to the bot."""
        attributes = dict(reader.info.attributes or {})
        parsed = self._parse_stream_attributes(attributes)
        if parsed is None:
            reader.close()
            self.open_streams -= 1
            self._leave_soon()
            return
        sample_rate, num_channels = parsed

        segment = _Segment(sample_rate=sample_rate, num_channels=num_channels)
        self._segment = segment
        body = bytearray()
        failed = False
        try:
            async for chunk in reader:
                if not segment.started_sent:
                    segment.started_sent = True
                    # The receiver sends an empty payload.
                    await self._perform_rpc(RPC_PLAYBACK_STARTED, "")
                body += chunk
                segment.bytes_received = len(body)
        except Exception:
            # The sender aborted, or the room went away mid-stream (rtc raises `StreamError`). Record what
            # arrived: a partial utterance is evidence, not an error in the fake.
            failed = True
        finally:
            self.open_streams -= 1
            self.streams.append({"identity": identity, "attributes": attributes, "bytes": bytes(body)})

        cleared = segment.cleared
        # Retire the utterance before the final report. The real receiver forgets its current reader at
        # the end of every utterance, and the worker sends no playback report once nothing is playing.
        # Leaving the slot populated would make an `lk.clear_buffer` arriving *between* utterances emit a
        # second, interrupted `lk.playback_finished` for an utterance that already reported: a report the
        # real worker never sends, and one the bot would read as a spurious stop.
        self._segment = None

        if failed or cleared:
            return
        await self._send_playback_finished(segment.position_secs(), interrupted=False)

    def _parse_stream_attributes(self, attributes: dict[str, str]) -> tuple[int, int] | None:
        """``(sample_rate, num_channels)``, or ``None`` after recording why the stream is unusable.

        The receiver requires both attributes and parses them with bare ``int()``; a missing key or an
        unparseable value raises ``ValueError`` out of ``_recv_task``, which propagates to the worker and
        ends the session. Fatal here too: a fake that noted the violation and carried on would let a
        missing attribute regress into a warning. This runs when the reader is *dequeued*, not when its
        header arrived, because that is where the real receiver does it: a stream
        queued behind a healthy one does not end the session until the healthy one has finished.
        """
        missing = [
            name for name in (STREAM_ATTR_SAMPLE_RATE, STREAM_ATTR_NUM_CHANNELS) if name not in attributes
        ]
        if not missing:
            try:
                return (
                    int(attributes[STREAM_ATTR_SAMPLE_RATE]),
                    int(attributes[STREAM_ATTR_NUM_CHANNELS]),
                )
            except ValueError:
                # Name the offending key: "one of them was not a number" is not a diagnosis.
                missing = [
                    name
                    for name in (STREAM_ATTR_SAMPLE_RATE, STREAM_ATTR_NUM_CHANNELS)
                    if not attributes[name].lstrip("-").isdigit()
                ]
        self.attribute_violations.append({"missing": ",".join(missing), **attributes})
        return None

    # ------------------------------------------------------------------ outgoing RPCs

    async def _send_playback_finished(self, position: float, *, interrupted: bool) -> None:
        """``lk.playback_finished`` with the receiver's exact three-key payload.

        The receiver sends ``json.dumps(asdict(PlaybackFinishedEvent))``: ``playback_position``,
        ``interrupted`` and ``synchronized_transcript: null``. The worker's receiver declares the
        third key too, so this is byte-for-byte the payload the avatar worker sends, and a bot handler
        doing ``PlaybackFinishedEvent(**payload)`` is exercised against the real thing rather than a
        subset of it.
        """
        payload = json.dumps(
            {
                "playback_position": position,
                "interrupted": interrupted,
                "synchronized_transcript": None,
            }
        )
        await self._perform_rpc(RPC_PLAYBACK_FINISHED, payload)

    async def _perform_rpc(self, method: str, payload: str) -> bool:
        """Call ``method`` on the bound sender with the receiver's retry policy. Returns success.

        Three attempts, 0.1 s apart. The receiver treats the two playback methods differently, and so
        does this:

        * ``lk.playback_started``: ``UNSUPPORTED_METHOD`` is logged and skipped, with no retry.
        * ``lk.playback_finished``: nothing is tolerated. After the third failure the receiver **raises**,
          which ends the worker's session. Here that is ``fatal_unsupported_playback_finished`` plus a
          disconnect, so a bot that forgot the handler fails a test instead of silently losing every
          session in production.
        """
        room = self._room
        if room is None or not room.isconnected():
            return False

        last: Exception | None = None
        for attempt in range(RPC_MAX_ATTEMPTS):
            try:
                await room.local_participant.perform_rpc(
                    destination_identity=self.publish_on_behalf,
                    method=method,
                    payload=payload,
                    response_timeout=RPC_RESPONSE_TIMEOUT_SECS,
                )
                return True
            except rtc.RpcError as err:
                last = err
                if method == RPC_PLAYBACK_STARTED and err.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD:
                    return False
            except Exception as err:
                last = err
            if attempt < RPC_MAX_ATTEMPTS - 1:
                await asyncio.sleep(RPC_RETRY_GAP_SECS)

        # Every exhausted retry is fatal, for *both* methods. The one tolerated case is the early return
        # above, `UNSUPPORTED_METHOD` on `lk.playback_started`. Anything else, including a
        # `lk.playback_started` whose handler raised (rtc reports that to the caller as
        # APPLICATION_ERROR), ends the worker's session: in the receiver's `_send_task`, the
        # `if method == RPC_PLAYBACK_STARTED` guard sits *inside* the `UNSUPPORTED_METHOD` branch.
        if isinstance(last, rtc.RpcError) and last.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD:
            # The recorder keeps its narrow meaning: the bot never registered `lk.playback_finished`. A
            # handler that registered and then misbehaved shows up as `connected` going False, which is
            # what the real session does.
            if method == RPC_PLAYBACK_FINISHED:
                self.fatal_unsupported_playback_finished = True
        self._leave_soon()
        return False

    # ------------------------------------------------------------------ served RPCs

    async def _on_clear_buffer(self, data: rtc.RpcInvocationData) -> str:
        """``lk.clear_buffer``: reject foreign callers, else truncate the current utterance.

        The receiver returns ``"reject"`` to any caller other than the bound sender and does nothing else.
        On accept it discards the rest of the current stream, and the worker sends
        ``playback_finished(interrupted=True, playback_position=<audio already submitted>)``, or nothing
        at all if nothing had played yet.

        The gate is ``started_sent``, which stands for the worker's "audio playing" state: the bot has been
        told playback began, so it is owed the matching report. ``bytes_received`` would leave a window,
        because ``_consume`` awaits the ``lk.playback_started`` round trip *before* it counts the first
        chunk; a ``lk.clear_buffer`` arriving then would get no report, and the bot would go on believing
        the avatar was speaking.
        """
        if data.caller_identity != self.publish_on_behalf:
            return RPC_HANDLER_REJECT
        self.clear_buffer_calls += 1
        segment = self._segment
        if segment is not None and not segment.cleared:
            segment.cleared = True
            if segment.started_sent:
                # Sent from a task, not awaited here: the worker sends it from its own loop, and an
                # outgoing RPC awaited inside an incoming handler is a needless round-trip nesting.
                self._spawn(
                    self._send_playback_finished(segment.position_secs(), interrupted=True),
                    "interrupted-finished",
                )
        return RPC_HANDLER_RESPONSE

    async def _on_swap_avatar(self, data: rtc.RpcInvocationData) -> str:
        """``swapAvatar``: always a JSON string, never a raise, as in the worker.

        The response bodies are the worker's, verbatim, including the distinction the product depends on:
        an id this session never declared is a plain ``error``, while one that is still preparing carries
        ``"code": "AVATAR_PREPARING"``, the only signal that retrying later could work.
        """
        try:
            payload = json.loads(data.payload)
        except ValueError:
            # Recorded before the rejection, so a product that sent malformed JSON is diagnosed as
            # "the payload was wrong" rather than "the RPC never arrived".
            self.swap_calls.append({"avatar_id": None, "payload": data.payload})
            return json.dumps({"error": "invalid payload"})
        if not isinstance(payload, dict) or "avatar_id" not in payload:
            self.swap_calls.append({"avatar_id": None, "payload": data.payload})
            return json.dumps({"error": "missing avatar_id"})

        raw = payload["avatar_id"]
        self.swap_calls.append({"avatar_id": raw})
        if self.swap_delay:
            await asyncio.sleep(self.swap_delay)

        requested = str(raw)
        if requested == SWAP_DEFAULT:
            # As in the worker, "default" resolves to the first avatar in avatarIds.
            requested = self.avatar_ids[0]
        elif requested.startswith(AVATAR_ID_PREFIX):
            requested = requested[len(AVATAR_ID_PREFIX) :]

        state = self._states.get(requested)
        if state is None:
            return json.dumps({"error": f"avatar {raw} not prepared"})
        if state == AVATAR_STATE_PREPARING:
            return json.dumps({"error": f"avatar {raw} still preparing", "code": SWAP_CODE_AVATAR_PREPARING})
        return json.dumps({"status": SWAP_RESPONSE_STATUS_OK, "avatar_id": raw})

    # ------------------------------------------------------------------ diagnostics

    def debug_dump(self) -> str:
        """Every recorder, one per line: the message for a failing assertion.

        Deliberately free of the token and of stream bodies: a failure message is read in a CI log, and
        neither a JWT nor 40 KB of PCM belongs there. Stream bodies are reported as byte counts.
        """
        streams = [
            {"identity": s["identity"], "attributes": s["attributes"], "bytes": len(s["bytes"])}
            for s in self.streams
        ]
        return "\n".join(
            [
                f"FakeAvatarWorker identity={self.identity!r} room={self.room_name!r} "
                f"publish_on_behalf={self.publish_on_behalf!r}",
                f"  connected={self.connected}",
                f"  avatar_ids={self.avatar_ids} states={self._states}",
                f"  streams={streams}",
                f"  open_streams={self.open_streams}",
                f"  overlapping_streams={self.overlapping_streams}",
                f"  rejected_stream_identities={self.rejected_stream_identities}",
                f"  attribute_violations={self.attribute_violations}",
                f"  token_violations={self.token_violations}",
                f"  clear_buffer_calls={self.clear_buffer_calls}",
                f"  swap_calls={self.swap_calls}",
                f"  fatal_unsupported_playback_finished={self.fatal_unsupported_playback_finished}",
            ]
        )


def _sine_cycle(frequency: float, sample_rate: int) -> list[int]:
    """One whole number of periods of a sine at ``frequency``, as s16 samples.

    A whole number of periods means the loop can wrap the buffer without a click at the seam, which
    matters for the one test that asks "did any avatar audio reach the bot" — a click is energy.
    """
    period = max(1, round(sample_rate / frequency))
    return [int(0.5 * 32767 * math.sin(2 * math.pi * index / period)) for index in range(period)]
