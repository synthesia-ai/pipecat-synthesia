"""Relayed join: the whole product against a real ``livekit-server`` and a fake worker.

Every test here runs a real ``PipelineWorker`` (``tests/harness.py``) around a real
``SynthesiaVideoService`` whose room is a real ``PrivateLiveKitRoom``, talking to a real
``livekit-server`` and to ``FakeAvatarWorker``. The only stand-in is the Synthesia API itself, and its 201
is what launches the fake worker (``tests/integration/_launcher.py``). This is the tier where the worker
protocol stops being an assumption about a third party's receiver, so **every protocol assertion is made
against the fake's recorders**, never against our own fakes or livekit-server log text, and every failing
assertion carries ``launcher.debug_dump()``.

Waits are ``until()`` / ``rec.wait()`` / ``h.frame()`` with a generous backstop; durations are inputs,
never assertions. Time constants stay at their real values, because in this tier they bound real network
operations and shrinking one turns runner jitter into a false failure; the 1 s ``DEFAULT_IDLE_CLOSE_SECS``
in the idle-close test is the case that matters. The tests that leave a stream open for another reason
raise it, and the same-identity replacement test raises ``REPLACEMENT_GRACE_SECS``, for the reasons their
docstrings give.
"""

from __future__ import annotations

import pytest
from livekit import api as lk_api
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.errors import ErrorCategory

from pipecat_synthesia.errors import SynthesiaError
from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_BOT_IDENTITY
from pipecat_synthesia.protocol import (
    SESSIONS_PATH,
    STREAM_ATTR_NUM_CHANNELS,
    STREAM_ATTR_SAMPLE_RATE,
    WORKER_VIDEO_HEIGHT,
    WORKER_VIDEO_WIDTH,
)
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_api import ScriptedResponse
from tests.harness import bidirectional_pipeline, make_service
from tests.integration._join_modes import (
    AGENT,
    AVATAR_IDS,
    CHUNK_A,
    CHUNK_B,
    STANDARD,
    WAIT,
    private_room,
    speak,
)
from tests.integration._launcher import SESSION_BODY, Launcher
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, frames_of
from tests.support.livekit_dev import LiveKitDev, list_participants, server_api
from tests.support.waiting import until

UP = FrameDirection.UPSTREAM
#: The two stream attributes the worker requires, byte-exact: decimal *strings* taken from the frame.
ATTRIBUTES = {STREAM_ATTR_SAMPLE_RATE: "24000", STREAM_ATTR_NUM_CHANNELS: "1"}
#: The *replacement grace* for the same-identity replacement test only, RAISED from the real 1 s; see that
#: test's docstring.
GRACE = 3.0
#: Utterance idle close for the tests that deliberately leave a stream open and are NOT about the idle
#: timer, raised from the real 1 s so it cannot close the stream out from under them.
LONG_IDLE_CLOSE = 10.0


def _plain_audio(frames) -> list[OutputAudioRawFrame]:
    """Exactly ``OutputAudioRawFrame``, not a subclass (*speaking-state ownership*).

    ``TTSAudioRawFrame`` is a subclass, and would make the output transport emit a second set of bot
    speaking frames.
    """
    return [frame for frame in frames if type(frame) is OutputAudioRawFrame]


def _service(dev: LiveKitDev, room_name: str, api_url: str, **overrides):
    """A service whose room is a private room on the dev server."""
    return make_service(
        private_room(dev, room_name=room_name), avatar_ids=AVATAR_IDS, api_url=api_url, **overrides
    )


# ======================================================================================= gate (every PR)


@pytest.mark.integration
async def test_session_created_and_avatar_joins(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """One POST launches one worker, and the token we minted is the one it is bound to.

    The token assertions are made through the fake's own recorders rather than by decoding the JWT
    here, so they are the *counterparty's* reading of the token the product actually handed over.
    ``token_violations`` covers the two the worker itself would die on (no ``kind=agent``, no
    ``lk.publish_on_behalf`` at all); the equality against ``DEFAULT_BOT_IDENTITY`` is what catches a
    ``lk.publish_on_behalf`` pointed at the *wrong* identity (this test's sabotage check), because a
    wrong-but-present value is not a violation the fake can recognise on its own. ``ListParticipants`` is
    the server's own view of each participant's kind: the avatar an agent, and the private room's bot
    deliberately STANDARD, which keeps the room open while the bot is in it (*agent-kind requirement*).
    ``ListRooms`` is the server's own view of the room's departure timeout.
    """
    dev = livekit_server
    api = await launcher.api()
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        (info,) = await rec.wait("on_avatar_connected", timeout=WAIT)

        # --- exactly one launch, with the three-key body and the URL we connected with -------------
        assert len(api.requests) == 1, launcher.debug_dump()
        request = api.requests[0]
        assert request.method == "POST"
        assert request.path == SESSIONS_PATH
        # Equality, not "contains": the Synthesia API rejects a body with extra keys.
        assert set(request.json) == {"avatarIds", "livekitUrl", "livekitToken"}, launcher.debug_dump()
        assert request.json["avatarIds"] == list(AVATAR_IDS)
        assert request.json["livekitUrl"] == dev.url.rstrip("/")
        assert request.json["livekitUrl"].startswith("ws://")

        # --- the worker accepted the token and is bound to our bot ---------------------------------
        assert len(launcher.fakes) == 1, launcher.debug_dump()
        fake = launcher.fakes[0]
        assert fake.token_violations == [], launcher.debug_dump()
        assert fake.publish_on_behalf == DEFAULT_BOT_IDENTITY, launcher.debug_dump()
        assert fake.room_name == room_name, launcher.debug_dump()

        # --- what the service reports, and what the server sees ------------------------------------
        assert info.session_id == SESSION_BODY["id"]
        assert info.avatar_identity == DEFAULT_AVATAR_IDENTITY
        assert info.active_avatar_id == AVATAR_IDS[0]
        assert info.room_name == room_name
        assert service.state is ServiceState.SESSION_ACTIVE

        kinds = {p.identity: p.kind for p in await list_participants(dev, room_name)}
        assert kinds == {DEFAULT_BOT_IDENTITY: STANDARD, DEFAULT_AVATAR_IDENTITY: AGENT}, (
            launcher.debug_dump()
        )

        # The timeout that closes the room once the bot is gone, as the server holds it.
        async with server_api(dev) as lkapi:
            listed = await lkapi.room.list_rooms(lk_api.ListRoomsRequest(names=[room_name]))
        [created] = listed.rooms
        assert created.departure_timeout == 20, launcher.debug_dump()

        await h.finish()

    assert rec.count("on_session_ended") == 1


@pytest.mark.integration
async def test_handlers_registered_before_first_byte(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """The first utterance of the session is delivered whole, with both attributes, and reported back.

    The frames are queued **before** the launch completes, so this is the earliest possible utterance:
    if ``lk.playback_started``/``lk.playback_finished`` were not registered by the time the session is
    active, when the first byte goes out, the worker's three retries would fail and it would end the
    session (``fatal_unsupported_playback_finished``): the single most damaging regression in the plugin,
    and invisible in every other tier. The attribute dict is compared byte-exactly because both values must
    be decimal *strings* taken from the frame; ``str(int(rate))`` → ``int(rate)`` fails here.
    """
    dev = livekit_server
    api = await launcher.api()
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await speak(h, context_id="c1")
        await rec.wait("on_avatar_connected", timeout=WAIT)
        fake = launcher.fakes[0]

        await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="one stream at the worker")
        assert fake.streams[0]["identity"] == DEFAULT_BOT_IDENTITY, launcher.debug_dump()
        assert fake.streams[0]["attributes"] == ATTRIBUTES, launcher.debug_dump()
        assert fake.streams[0]["bytes"] == CHUNK_A, launcher.debug_dump()

        # The worker reported playback both ways round, which is only possible if both handlers existed.
        await h.frame(BotStartedSpeakingFrame, timeout=WAIT)
        await until(
            lambda: service.metrics.playback_finished_count == 1,
            timeout=WAIT,
            what="lk.playback_finished at the service",
        )
        assert fake.fatal_unsupported_playback_finished is False, launcher.debug_dump()
        assert fake.attribute_violations == [], launcher.debug_dump()
        assert fake.rejected_stream_identities == [], launcher.debug_dump()
        assert fake.overlapping_streams == 0, launcher.debug_dump()
        await h.finish()

    assert_absent(h.down, TTSAudioRawFrame, where="downstream of the service in SESSION_ACTIVE")


@pytest.mark.integration
async def test_two_utterances_two_streams_no_overlap(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """Two utterances are two streams, one at a time, each carrying its own bytes.

    On the avatar worker's receiver a second stream opened before the first closed queues behind it
    forever (*one-open-stream rule*), so ``overlapping_streams`` is the difference between a working
    avatar and one that goes silent after the first sentence. The per-utterance byte comparison is what
    makes this more than a count: a streamer that reopened per chunk would also produce two streams.
    """
    dev = livekit_server
    api = await launcher.api()
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        fake = launcher.fakes[0]

        await speak(h, context_id="c1", audio=CHUNK_A)
        await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="the first utterance's stream")
        await speak(h, context_id="c2", audio=CHUNK_B)
        await until(lambda: len(fake.streams) == 2, timeout=WAIT, what="the second utterance's stream")

        # No `open_streams == 0` here: `_consume`'s `finally` decrements it and appends to `streams`
        # in the same synchronous block, so the `until` above already implies it.
        assert [stream["bytes"] for stream in fake.streams] == [CHUNK_A, CHUNK_B], launcher.debug_dump()
        assert fake.overlapping_streams == 0, launcher.debug_dump()
        await until(
            lambda: service.metrics.playback_finished_count == 2,
            timeout=WAIT,
            what="one lk.playback_finished per utterance",
        )
        await h.finish()


@pytest.mark.integration
async def test_idle_close_without_tts_stopped(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """A dropped ``TTSStoppedFrame`` still ends the utterance, by idle close, and the worker reports it.

    The stream is closed by ``utterance_streamer.DEFAULT_IDLE_CLOSE_SECS`` (1 s), left at its real value
    here on purpose: in this tier it bounds a real operation. The assertion is the outcome (the stream
    closed and ``lk.playback_finished`` came back), never how long it took.
    """
    dev = livekit_server
    api = await launcher.api()
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        fake = launcher.fakes[0]

        await speak(h, context_id="c1", end=False)

        await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="the idle timer to close the stream")
        assert fake.streams[0]["bytes"] == CHUNK_A, launcher.debug_dump()
        await until(
            lambda: service.metrics.playback_finished_count == 1,
            timeout=WAIT,
            what="lk.playback_finished for the idle-closed utterance",
        )
        await h.finish()


@pytest.mark.integration
async def test_interruption(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Barge-in is ``lk.clear_buffer`` then the close; the bot stops speaking, and can speak again after.

    ``BotStartedSpeakingFrame`` is the signal that the worker has bytes for this utterance, so waiting on
    it is how the interruption lands mid-utterance without a sleep. The sabotage check: remove
    ``perform_clear_buffer`` from ``interrupt()`` and ``clear_buffer_calls`` stays 0. The metric is
    asserted as a **count**, never a value.

    The idle close is raised to 10 s because this test is not about it: the utterance is deliberately
    left open, so the real 1 s would put a clock on "round trip to the worker, ``BotStartedSpeakingFrame``
    to the sink, then interrupt", and a loaded runner could close the stream before the barge-in lands.
    ``session.py`` from-imports the constant, so the patch goes on the name that module resolves.
    """
    monkeypatch.setattr(session_module, "DEFAULT_IDLE_CLOSE_SECS", LONG_IDLE_CLOSE)
    dev = livekit_server
    api = await launcher.api()
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        fake = launcher.fakes[0]

        await speak(h, context_id="c1", end=False)
        await h.frame(BotStartedSpeakingFrame, timeout=WAIT)

        finished_before = service.metrics.playback_finished_count
        await h.queue(InterruptionFrame())

        await until(lambda: fake.clear_buffer_calls == 1, timeout=WAIT, what="one lk.clear_buffer")
        await until(lambda: fake.open_streams == 0, timeout=WAIT, what="the stream to be closed")
        await h.frame(BotStoppedSpeakingFrame, timeout=WAIT)
        await until(
            lambda: service.metrics.playback_finished_count == finished_before + 1,
            timeout=WAIT,
            what="the fake worker's playback_finished answering the clear buffer",
        )

        # A new utterance still flows afterwards.
        await speak(h, context_id="c2", audio=CHUNK_B)
        await until(lambda: len(fake.streams) == 2, timeout=WAIT, what="the post-interruption utterance")
        assert fake.streams[1]["bytes"] == CHUNK_B, launcher.debug_dump()
        assert fake.overlapping_streams == 0, launcher.debug_dump()
        await h.finish()


@pytest.mark.integration
async def test_media_into_pipeline(
    livekit_server: LiveKitDev,
    room_name: str,
    launcher: Launcher,
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """The avatar's tracks reach the pipeline, and only after the transport said it is ready.

    The media gate and *speaking-state ownership*, through a real ``AvatarTrackCapture`` on real
    ``rtc.VideoStream``/``AudioStream``. The "nothing before" half is only worth anything if frames were
    actually arriving while the gate was shut, so the test first waits for the service's own drop TRACE
    (the one place where the log *is* the behaviour) and snapshots the sink at that moment.
    ``type(f) is OutputAudioRawFrame`` rather than ``isinstance``: a ``SpeechOutputAudioRawFrame`` would
    make the transport's silence detector a second ``Bot*SpeakingFrame`` emitter.
    """
    dev = livekit_server
    api = await launcher.api()
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)

        await until(
            lambda: any("dropping avatar media" in r.getMessage() for r in caplog_loguru.records),
            timeout=WAIT,
            what="avatar media to arrive while the transport gate is shut",
        )
        before = frames_of(h.down)
        assert_absent(before, OutputImageRawFrame, where="before OutputTransportReadyFrame")
        assert _plain_audio(before) == [], launcher.debug_dump()

        await h.queue(OutputTransportReadyFrame(), UP)
        await h.frame(OutputTransportReadyFrame, direction=UP, timeout=WAIT)

        image = await h.frame(OutputImageRawFrame, timeout=WAIT)
        assert image.format == "RGB"
        # Size and payload must agree, and the payload must be RGB24-sized for it — that is what a
        # passthrough bug breaks. The *value* is deliberately not asserted: the worker publishes 512x512
        # (`WORKER_VIDEO_*`) but what arrives is whatever spatial layer the SFU chose, and a loaded runner
        # really does deliver a smaller one (observed: 350x350). Asserting 512 would be asserting on
        # LiveKit's quality adaptation, and a test must not assert what the network picks.
        width, height = image.size
        assert width > 0 and height > 0
        assert len(image.image) == width * height * 3
        assert width <= WORKER_VIDEO_WIDTH and height <= WORKER_VIDEO_HEIGHT
        await until(
            lambda: _plain_audio(h.down), timeout=WAIT, what="a plain OutputAudioRawFrame at the sink"
        )
        audio = _plain_audio(h.down)[0]
        assert audio.num_channels == 1
        assert service.metrics.first_video_frame is not None
        await h.finish()


# ========================================================================= extended (nightly / PR label)


@pytest.mark.integration_extended
async def test_readiness_and_swap(livekit_server: LiveKitDev, room_name: str, launcher: Launcher) -> None:
    """Readiness arrives through participant attributes, and a swap crosses the wire unprefixed.

    ``status_delay=0.5`` (every modelled delay is at least 0.5 s) so "ready" is something that *happens*
    rather than something that was already true. Both halves of *unprefixed ids on the wire* are asserted
    on opposite sides of the wire at once: the value the developer gets back is ``av_``-prefixed, and the
    value the worker was asked for is the unprefixed avatar id.
    """
    dev = livekit_server
    api = await launcher.api(status_delay=0.5)
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        (ready,) = await rec.wait("on_avatar_ready", timeout=WAIT)
        fake = launcher.fakes[0]

        assert ready.session_id == SESSION_BODY["id"]
        assert ready.status == {"av_a": "ready", "av_b": "ready"}, launcher.debug_dump()

        assert await service.swap_avatar(AVATAR_IDS[1]) == AVATAR_IDS[1]
        assert service.active_avatar_id == AVATAR_IDS[1]
        assert fake.swap_calls == [{"avatar_id": "b"}], launcher.debug_dump()
        await h.finish()

    assert rec.count("on_avatar_ready") == 1, "on_avatar_ready fires once per session"


@pytest.mark.integration_extended
async def test_avatar_leaves(livekit_server: LiveKitDev, room_name: str, launcher: Launcher) -> None:
    """An avatar that leaves mid-call ends the session and leaves the bot voice-only.

    A relayed join has no humans in its private room (``owns_room`` ⇒ ``humans_present is None``), so
    the worker leaving is an *unexpected leave*: it reports ``on_avatar_disconnected``, ends the session
    and pushes a ``CONNECTIVITY`` ``ErrorFrame``, non-permanent so ``is_usable`` stays True and TTS keeps
    flowing to the transport (*permanent-error rule*). ``leave_after=2`` is a modelled delay, not an
    assertion.
    """
    dev = livekit_server
    api = await launcher.api(leave_after=2.0)
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)

        (reason,) = await rec.wait("on_avatar_disconnected", timeout=WAIT)
        assert reason == "avatar_left_room", launcher.debug_dump()
        (ended,) = await rec.wait("on_session_ended", timeout=WAIT)
        assert ended == "avatar_left"
        assert service.state is ServiceState.ENDED

        error = await h.frame(ErrorFrame, direction=UP, timeout=WAIT)
        assert error.category is ErrorCategory.CONNECTIVITY
        assert error.category.is_permanent is False
        assert isinstance(error.exception, SynthesiaError)
        assert error.exception.session_id == SESSION_BODY["id"]
        assert service.is_usable is True

        # Voice-only from here: TTS audio passes through to the transport untouched (*frame contract*).
        await speak(h, context_id="after", audio=CHUNK_B)
        passed = await h.frame(TTSAudioRawFrame, timeout=WAIT)
        assert passed.audio == CHUNK_B
        await h.finish()

    assert rec.count("on_avatar_disconnected") == 1, "reported at most once"


@pytest.mark.integration_extended
async def test_launch_timeout_required_false(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """A worker that never joins, with ``required=False``, degrades to voice-only and stays usable.

    Its error is non-permanent (*permanent-error rule*). ``never_join`` models a launch the API
    acknowledged and the worker never honoured, which no unit fake can produce faithfully: here the real
    ``asyncio.wait_for`` really does cancel a real wait for the join. ``launch_timeout=2`` is the timeout
    *being tested*, so shrinking it is allowed, and the verdict does not depend on its size: a runner slow
    enough to spend the whole timeout creating the room still reaches DEGRADED.
    """
    dev = livekit_server
    api = await launcher.api(never_join=True)
    service = _service(dev, room_name, api.base_url, required=False, launch_timeout=2.0)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        (ended,) = await rec.wait("on_session_ended", timeout=WAIT)
        assert ended == "degraded"
        assert service.state is ServiceState.DEGRADED

        error = await h.frame(ErrorFrame, direction=UP, timeout=WAIT)
        assert error.category is not None and error.category.is_permanent is False, (
            "a required=False failure must never cost the processor its usability (permanent-error rule)"
        )
        assert service.is_usable is True

        await speak(h, context_id="c1", audio=CHUNK_B)
        passed = await h.frame(TTSAudioRawFrame, timeout=WAIT)
        assert passed.audio == CHUNK_B
        assert service.is_usable is True, "nothing later flips usability back"
        await h.finish()

    assert rec.count("on_avatar_connected") == 0
    assert len(api.requests) == 1, launcher.debug_dump()


@pytest.mark.integration_extended
async def test_same_identity_replacement_not_reported_as_left(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried launch replacing a stalled worker is not "the avatar left" (*replacement grace*).

    How it happens: the first attempt times out at the API while its worker is still starting up, the
    retry's worker joins and serves the session, and then the *first* worker arrives on the same identity
    and livekit-server kicks the second with ``DUPLICATE_IDENTITY``. Treat that as a leave and every
    retried launch reports a spurious avatar loss to the developer.

    Extended tier because the verdict is inherently relative to ``REPLACEMENT_GRACE_SECS``, and the grace
    is **raised** to 3 s here rather than left at its real 1 s, because in this test it bounds a real
    operation: livekit-server accepts the join (kicking the old participant, which starts the grace) well
    before the joining client has returned from ``connect()`` and published its video, which is what
    ``_grace_expired`` looks for. On a loaded runner that gap exceeded 1 s and the session declared the
    avatar gone. Whether the product's 1 s default is generous enough for a real worker is a separate
    question; this test is about the mechanism. Promote to the gate only after two weeks of clean nightly
    runs.
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    dev = livekit_server
    # Both entries hooked: the timed-out attempt is the slow worker, the retry the fast one.
    api = await launcher.api(
        [
            ScriptedResponse(504, {"code": "timeout"}, before=launcher.hook(join_delay=3.0)),
            ScriptedResponse(201, dict(SESSION_BODY)),
        ]
    )
    service = _service(dev, room_name, api.base_url)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        assert len(api.requests) == 2, launcher.debug_dump()
        assert len(launcher.fakes) == 2, launcher.debug_dump()
        slow, fast = launcher.fakes

        await until(lambda: slow.connected, timeout=WAIT, what="the slow worker to arrive and replace")
        await until(lambda: fast.connected is False, timeout=WAIT, what="the second worker to be kicked")
        # The replacement is fully in; a grace armed at or before this moment has expired by now.
        await h.sleep(GRACE)

        assert rec.count("on_avatar_disconnected") == 0, launcher.debug_dump()
        assert rec.count("on_session_ended") == 0, launcher.debug_dump()
        assert service.state is ServiceState.SESSION_ACTIVE, launcher.debug_dump()

        # Still usable, and the utterance goes to the worker that is actually in the room.
        await speak(h, context_id="c1", audio=CHUNK_B)
        await until(
            lambda: len(slow.streams) == 1, timeout=WAIT, what="the utterance at the surviving worker"
        )
        assert slow.streams[0]["bytes"] == CHUNK_B, launcher.debug_dump()
        await h.finish()

    assert_absent(h.up, ErrorFrame, where="upstream of a session that only saw a replacement")
