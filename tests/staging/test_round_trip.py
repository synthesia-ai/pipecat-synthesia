"""One real round trip against the Synthesia API: avatar joins, speaks, is interrupted, tears down.

Marker: `staging`.

Run it:

    uv run --env-file .env pytest -m staging
    SYNTHESIA_STAGING_DIRECT_JOIN=1 uv run --env-file .env pytest -m staging   # both join modes

The tier reads its credentials from the environment only (the variables in `.env.example`), and
skips when any is missing; nothing in it loads a `.env` file. `--env-file` puts a file's values in
the environment, or export them yourself.

**What this is, and what it deliberately is not.** It is a smoke run: it proves the wire still works
end to end against a real worker after a change to the protocol, the token recipe or the API client.
It is *not* a regression suite and not a benchmark.

So it asserts the **presence and order of events**, never a latency. Every timing here is recorded
and printed for a human to read, and asserting any of them would produce a test that fails when a
staging region is busy -- the classic way a smoke run gets disabled and then deleted. The one thing
resembling a timing assertion is the launch timeout, and that is the product's own `launch_timeout`
doing its job, not a threshold this file invented.

**Cost and blast radius.** Each run creates one billed session and one LiveKit room. The room name
is unique per run and teardown deletes it by exact name; nothing here lists or touches a room it
did not create, because the LiveKit credentials are usually shared and cannot be scoped down.
"""

from __future__ import annotations

import math
import os
import struct

import pytest
from livekit import api as lk_api
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia.livekit import PrivateLiveKitRoom
from pipecat_synthesia.livekit._types import LiveKitCredentials
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until

pytestmark = pytest.mark.staging

#: The worker's rate. Also what the service advertises, so a tone built at this rate needs no resample.
SAMPLE_RATE = 24000

#: Generous backstops, never budgets. A real staging region under load is slow, not broken.
LAUNCH_WAIT = 120.0
MEDIA_WAIT = 60.0
PLAYBACK_WAIT = 30.0


def one_second_tone(hz: float = 220.0) -> bytes:
    """1 s of 24 kHz mono s16le. A tone rather than silence so the avatar's mouth actually moves."""
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)))
        for i in range(SAMPLE_RATE)
    )


async def room_exists(env, name: str) -> bool:
    """Whether the LiveKit room `name` still exists.

    Scoped to the exact name: `ListRoomsRequest(names=[...])` never enumerates the project, which
    matters because these credentials are shared with other people's live sessions.
    """
    api = lk_api.LiveKitAPI(
        url=env.livekit_url, api_key=env.livekit_api_key, api_secret=env.livekit_api_secret
    )
    try:
        listed = await api.room.list_rooms(lk_api.ListRoomsRequest(names=[name]))
        return any(room.name == name for room in listed.rooms)
    finally:
        await api.aclose()


def report(title: str, recorder: EventRecorder, service) -> None:
    """Print the event order and the metrics line, the timings this tier records but never asserts."""
    print(f"\n=== {title} ===")
    print(f"events: {' -> '.join(recorder.names())}")
    print(f"metrics: {service.metrics.summary()}")


async def _drive_round_trip(service, recorder: EventRecorder, *, expect_video: bool, processor=None) -> None:
    """The shared body: launch, one utterance, one interruption, clean end.

    `expect_video` is False for a direct join, where the avatar publishes into the LiveKit room itself
    and no video frame ever enters the pipeline -- asserting one there would be asserting the direct
    join is broken.

    `processor` is what the harness wraps, defaulting to the service alone. A direct join must pass a
    full pipeline instead: its `LiveKitTransport` only connects when `transport.input()` and
    `transport.output()` are set up by a pipeline, and `LiveKitTransportRoom.launch()` waits 30 s
    for that connection before failing. Handing the harness the bare service there produces a
    launch that can never succeed.
    """
    async with bidirectional_pipeline(
        processor if processor is not None else service,
        pipeline_params=PipelineParams(enable_metrics=True),
    ) as h:
        # The transport gates media until it is ready; without this the service holds everything back.
        await h.queue(OutputTransportReadyFrame(), FrameDirection.UPSTREAM)

        connected = await recorder.wait("on_avatar_connected", timeout=LAUNCH_WAIT)
        info = connected[0]
        assert info.session_id, "on_avatar_connected carried no session_id"
        print(f"\nsession {info.session_id} avatar {info.active_avatar_id} room {info.room_name}")

        if expect_video:
            await h.frame(OutputImageRawFrame, timeout=MEDIA_WAIT)

        # --- one utterance, exactly as a TTS service produces it -------------------------------
        await h.queue(TTSStartedFrame(context_id="utterance-1"))
        await h.queue(
            TTSAudioRawFrame(
                audio=one_second_tone(),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                context_id="utterance-1",
            )
        )
        await h.queue(TTSStoppedFrame(context_id="utterance-1"))

        # Bot*Speaking come from the worker's real playout reports, not from when we sent the audio:
        # seeing them at all is the round trip being proven.
        await h.frame(BotStartedSpeakingFrame, timeout=PLAYBACK_WAIT)

        # --- interrupt it mid-word --------------------------------------------------------------
        await h.queue(InterruptionFrame())
        await h.frame(BotStoppedSpeakingFrame, timeout=PLAYBACK_WAIT)

        # The interruption is acknowledged when the worker confirms the buffer was cleared with an
        # interrupted playback_finished: the service holds the interruption pending until one arrives (a
        # normal finish does not clear it).
        await until(
            lambda: service._interrupt_started is None,
            timeout=PLAYBACK_WAIT,
            what="the worker to acknowledge clear_buffer (an interrupted playback_finished)",
        )

        await h.finish()

    # After the pipeline ends: exactly one session end, and metrics right after it.
    assert recorder.count("on_session_ended") == 1, recorder.names()
    assert recorder.count("on_metrics") == 1, recorder.names()
    names = recorder.names()
    assert names.index("on_avatar_connected") < names.index("on_session_ended"), names
    assert names.index("on_session_ended") < names.index("on_metrics"), (
        f"on_metrics must follow on_session_ended: {names}"
    )
    assert service.metrics.playback_finished_count >= 1, (
        f"the worker never reported a finished playback: {service.metrics.summary()}"
    )


async def test_relayed_join_round_trip(staging_env, room_name) -> None:
    """Relayed join: the avatar joins a private room, and its media enters the pipeline as frames.

    Afterwards the private room must be gone: the plugin owns it and deletes it on the way out.
    """
    room = PrivateLiveKitRoom(
        url=staging_env.livekit_url,
        api_key=staging_env.livekit_api_key,
        api_secret=staging_env.livekit_api_secret,
        room_name=room_name,
    )
    service = make_service(
        room,
        avatar_ids=[staging_env.avatar_id],
        api_key=staging_env.api_key,
        api_url=staging_env.api_url,
        launch_timeout=LAUNCH_WAIT,
    )
    recorder = EventRecorder(service)

    try:
        await _drive_round_trip(service, recorder, expect_video=True)
        report("relayed join", recorder, service)
    finally:
        # The plugin deletes the room it owns on every path it controls; this proves it did, and
        # cleans up by exact name if a crash got in the way. A leaked room holds a session open.
        leaked = await room_exists(staging_env, room_name)
        if leaked:
            api = lk_api.LiveKitAPI(
                url=staging_env.livekit_url,
                api_key=staging_env.livekit_api_key,
                api_secret=staging_env.livekit_api_secret,
            )
            try:
                await api.room.delete_room(lk_api.DeleteRoomRequest(room=room_name))
            finally:
                await api.aclose()
        assert not leaked, f"the plugin left its private room {room_name} behind (now deleted)"


@pytest.mark.skipif(
    not os.environ.get("SYNTHESIA_STAGING_DIRECT_JOIN"),
    reason="the direct join costs a second billed session; set SYNTHESIA_STAGING_DIRECT_JOIN=1 to include it",
)
async def test_direct_join_round_trip(staging_env, room_name) -> None:
    """Direct join: the avatar joins the bot's own LiveKit room and publishes there.

    Behind its own flag because it is a second billed session and a second concurrent slot, and the
    relayed-join run already covers the API client, the token recipe and the worker protocol. What this
    adds is the borrowed transport room: the bot's transport owns it, and the plugin must not delete it
    (*borrowed-room rule*).
    """
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

    credentials = LiveKitCredentials(
        api_key=staging_env.livekit_api_key, api_secret=staging_env.livekit_api_secret
    )
    # A direct join's bot token as the docs tell developers to mint it: the agent kind and the agent
    # grant (*agent-kind requirement*). Not `mint_bot_token`, which mints the private room's standard bot.
    bot_token = (
        lk_api.AccessToken(credentials.api_key, credentials.api_secret)
        .with_identity("pcs-staging-bot")
        .with_kind("agent")
        .with_grants(lk_api.VideoGrants(room_join=True, room=room_name, agent=True))
        .to_jwt()
    )
    transport = LiveKitTransport(
        staging_env.livekit_url,
        bot_token,
        room_name,
        LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    room = LiveKitTransportRoom(transport, credentials=credentials)
    service = make_service(
        room,
        avatar_ids=[staging_env.avatar_id],
        api_key=staging_env.api_key,
        api_url=staging_env.api_url,
        launch_timeout=LAUNCH_WAIT,
    )
    recorder = EventRecorder(service)

    # `room.input_filter()` directly after `transport.input()` is mandatory in a direct join, and
    # a staging run is where its absence would show up as the bot transcribing the avatar.
    pipeline = Pipeline([transport.input(), room.input_filter(), service, transport.output()])

    try:
        # No video assertion: in a direct join the avatar publishes into the room, so nothing reaches
        # the pipeline.
        await _drive_round_trip(service, recorder, expect_video=False, processor=pipeline)
        report("direct join", recorder, service)
    finally:
        try:
            await transport.cleanup()
        finally:
            api = lk_api.LiveKitAPI(
                url=staging_env.livekit_url,
                api_key=staging_env.livekit_api_key,
                api_secret=staging_env.livekit_api_secret,
            )
            try:
                # The bot owns this room, not the plugin, so deleting it is our job here.
                await api.room.delete_room(lk_api.DeleteRoomRequest(room=room_name))
            finally:
                await api.aclose()
