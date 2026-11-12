"""Direct join: the avatar in the bot's own room, behind a real ``LiveKitTransport``.

The direct join cannot be tested anywhere else. Its three behaviours are all facts about things we do
not own: that ``lk.publish_on_behalf`` must equal the identity **Pipecat's** transport joined with, that
livekit-server reports the bot as an agent only if the token carried the top-level ``kind`` claim, and
that ``mute_participant`` + ``set_subscribed(False)`` really do stop the avatar's media reaching the bot.
Every one of them passes against a fake transport and fails in production if it is wrong, so these tests
build a **real** ``LiveKitTransport``, put the service in a real pipeline behind it, and let the whole
thing connect to the dev server.

The pipeline is nested inside ``tests/harness.py``'s recording source and sink, so ``h.down``/``h.up``
still see everything that left the pipeline in each direction, and the harness still owns the teardown
and the dangling-task check. A fresh transport per test: its client caches the room object and
reference-counts connect/disconnect.
"""

from __future__ import annotations

import re

import pytest
from pipecat.frames.frames import ErrorFrame, UserAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.runner.livekit import generate_token_with_agent

from pipecat_synthesia.livekit._transport_access import get_room
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY
from pipecat_synthesia.protocol import WORKER_AUDIO_TRACK_NAME, WORKER_VIDEO_TRACK_NAME
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_api import ScriptedResponse
from tests.harness import bidirectional_pipeline, make_service
from tests.integration._join_modes import (
    AGENT,
    AVATAR_IDS,
    DIRECT_JOIN_BOT,
    STANDARD,
    WAIT,
    FrameProbe,
    direct_join_bot_token,
    livekit_transport,
    speak,
    transport_room,
)
from tests.integration._launcher import SESSION_BODY, Launcher
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent
from tests.support.livekit_dev import LiveKitDev, connect_room, list_participants, mint_dev_token
from tests.support.waiting import until

HUMAN = "human-client"
#: A loud 440 Hz tone from the avatar, so "the bot heard nothing" is a fact about the filter and the
#: unsubscribe rather than a fact about the avatar having published digital silence.
TONE_HZ = 440.0


def _avatar_audio(frames) -> list[UserAudioRawFrame]:
    """Input audio frames from the avatar, which the bot must never transcribe (*self-hearing guard*)."""
    return [
        frame
        for frame in frames
        if isinstance(frame, UserAudioRawFrame) and frame.user_id == DEFAULT_AVATAR_IDENTITY
    ]


@pytest.mark.integration
async def test_avatar_joins_transport_room_and_bot_does_not_hear_it(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """The avatar joins the bot's own room, and the bot neither hears nor downloads it.

    ``lk.publish_on_behalf == transport.participant_id`` is the one assertion that can only be made
    here: the bot identity is Pipecat's, read out of a private attribute by ``_transport_access``, and a
    version of Pipecat that changed where it keeps it would make the worker silently drop every stream
    we sent, with no error anywhere. It is checked through the fake's own
    ``token_violations``/``publish_on_behalf`` recorders: the counterparty's word, not ours.

    The *self-hearing guard* has two mechanisms, and this test proves the **outcome** of both together:
    no avatar ``UserAudioRawFrame`` gets past ``room.input_filter()``, and both of the avatar's
    publications end up unsubscribed. The frame filter on its own is proven in the unit tier; what is only
    provable here is that ``mute_participant`` (``async def`` at both layers in Pipecat 1.10) and
    ``set_subscribed(False)`` actually reach the SFU, and that the video publication is dropped too (it is
    up to 4 Mbps the bot never renders).
    """
    dev = livekit_server
    api = await launcher.api(tone_hz=TONE_HZ)
    transport = livekit_transport(
        dev, room_name=room_name, token=direct_join_bot_token(dev, room_name=room_name)
    )
    room = transport_room(dev, transport)
    service = make_service(room, avatar_ids=AVATAR_IDS, api_url=api.base_url)
    probe = FrameProbe(name="after-input-filter")
    rec = EventRecorder(service)

    # The documented placement: the filter immediately after `transport.input()`.
    pipeline = Pipeline([transport.input(), room.input_filter(), probe, service, transport.output()])

    async with bidirectional_pipeline(pipeline) as h:
        # Registered before the avatar can join: purely diagnostic, but "was this track ever
        # subscribed?" is the first question a failure below raises, and it is unanswerable after
        # the fact from `publication.subscribed` alone.
        subscriptions: list[str] = []
        lk_room = get_room(transport)
        lk_room.on(
            "track_subscribed",
            lambda _track, publication, participant: subscriptions.append(
                f"{participant.identity}:{publication.name}"
            ),
        )

        (info,) = await rec.wait("on_avatar_connected", timeout=WAIT)
        fake = launcher.fakes[0]

        # --- the worker publishes on behalf of the identity Pipecat actually joined with ----------
        assert transport.participant_id == DIRECT_JOIN_BOT, launcher.debug_dump()
        assert fake.publish_on_behalf == transport.participant_id, launcher.debug_dump()
        assert fake.token_violations == [], launcher.debug_dump()
        assert fake.room_name == room_name, launcher.debug_dump()
        assert info.room_name == room_name, launcher.debug_dump()
        assert info.session_id == SESSION_BODY["id"], launcher.debug_dump()
        assert room.filter_installed is True, "the pipeline set the filter up, so no warning is due"

        # --- the server's own view: bot and avatar, both agents, in the bot's room -----------------
        kinds = {p.identity: p.kind for p in await list_participants(dev, room_name)}
        assert kinds == {DIRECT_JOIN_BOT: AGENT, DEFAULT_AVATAR_IDENTITY: AGENT}, launcher.debug_dump()

        # --- the positive control, and it is load-bearing -------------------------------------------
        # rtc's `RemoteTrackPublication.subscribed` starts False and only becomes True when the
        # `track_subscribed` event lands, so "both publications are unsubscribed" is ALSO true of a room
        # that has simply not auto-subscribed yet: asserted straight after the join, it passed 5 runs in
        # 7 with `_unsubscribe_from_avatar` stubbed out. One full utterance round trip (bot -> SFU ->
        # worker -> `lk.playback_finished` RPC -> bot) is strictly more network work than the
        # auto-subscribe that was already in flight, so past this point "unsubscribed" is an end state
        # the product produced, not a state it never left.
        await speak(h, context_id="c1")
        await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="the utterance at the worker")
        await until(
            lambda: service.metrics.playback_finished_count == 1,
            timeout=WAIT,
            what="the worker's playback report back at the bot (a full round trip through the SFU)",
        )

        # --- self-hearing guard: both publications unsubscribed ------------------------------------
        def unsubscribed() -> list[str]:
            avatar = lk_room.remote_participants.get(DEFAULT_AVATAR_IDENTITY)
            if avatar is None:
                return []
            return sorted(
                publication.name
                for publication in avatar.track_publications.values()
                if publication.subscribed is False
            )

        await until(
            lambda: unsubscribed() == [WORKER_AUDIO_TRACK_NAME, WORKER_VIDEO_TRACK_NAME],
            timeout=WAIT,
            what=f"the avatar's audio AND video publications to be unsubscribed "
            f"(subscribed so far: {subscriptions}; currently unsubscribed: {unsubscribed()})",
        )

        # --- self-hearing guard: nothing of the avatar's got past the filter ------------------------
        assert _avatar_audio(probe.seen) == [], (
            "the bot transcribed its own avatar: "
            f"{len(_avatar_audio(probe.seen))} frame(s) past room.input_filter(); "
            f"subscriptions seen: {subscriptions}\n" + launcher.debug_dump()
        )
        await h.finish()

    assert _avatar_audio(probe.seen) == [], "not during teardown either"
    assert _avatar_audio(h.down) == []


#: How long the fake API holds the 201 after the worker has joined and published (the real gap is ~0.5 s).
HOLD_201_SECS = 3.0


@pytest.mark.integration
async def test_audio_reaches_the_worker_before_the_201_when_this_launchs_avatar_is_up(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """*Early start*, through a real livekit-server: the minted token's ``synthesia.launch`` attribute is
    visible on the avatar participant, so the session goes active on its video, and pre-launch audio
    reaches the worker while the 201 is still held. ``on_avatar_connected`` then fires at the 201, with
    the session id.

    Only provable here: that livekit-server delivers the avatar's token attributes to the bot, even once
    the worker has written its own readiness attributes. The fake API holds the 201 ``HOLD_201_SECS``
    after the worker has joined and published (``ScriptedResponse.delay`` runs after the launch hook).
    """
    dev = livekit_server
    api = await launcher.api([ScriptedResponse(201, dict(SESSION_BODY), delay=HOLD_201_SECS)])
    transport = livekit_transport(
        dev, room_name=room_name, token=direct_join_bot_token(dev, room_name=room_name)
    )
    room = transport_room(dev, transport)
    service = make_service(room, avatar_ids=AVATAR_IDS, api_url=api.base_url)
    rec = EventRecorder(service)
    pipeline = Pipeline([transport.input(), room.input_filter(), service, transport.output()])

    async with bidirectional_pipeline(pipeline) as h:
        await speak(h, context_id="greeting")
        await until(
            lambda: bool(launcher.fakes and launcher.fakes[0].streams),
            timeout=WAIT,
            what="the greeting at the worker",
        )
        assert rec.count("on_avatar_connected") == 0, "the audio waited for the 201"
        assert service.state is ServiceState.SESSION_ACTIVE
        assert service.session_id is None

        (info,) = await rec.wait("on_avatar_connected", timeout=WAIT)
        assert info.session_id == SESSION_BODY["id"]
        assert service.session_id == SESSION_BODY["id"]
        await h.finish()

    assert len(api.requests) == 1


@pytest.mark.integration
async def test_bot_standard_kind_rejected(livekit_server: LiveKitDev, room_name: str) -> None:
    """The dev runner's bot token is refused when the direct join is constructed, naming the call to add.

    A STANDARD bot is the defect with the longest fuse in the whole plugin: everything works, and then
    the worker never ends the session when the real human leaves, because it counts STANDARD
    participants and finds the bot (*agent-kind requirement*). Pipecat's dev runner mints its `-t
    livekit` bot token with ``generate_token_with_agent()``, which sets only the ``video.agent`` grant.
    That grant does **not** change the kind, and ``ListParticipants`` here is the proof that the
    distinction is real.

    The token is Pipecat's own, and the transport is a real ``LiveKitTransport``: what this pins is that
    the constructor reads the token a real transport holds before anything connects. The refusal is a
    ``ValueError`` at construction, so no service, no pipeline and no billed session ever exist. The
    error message is one of the few asserted by text: it exists to name the exact call a developer must
    add, so the substring is the contract.
    """
    dev = livekit_server
    runner_token = generate_token_with_agent(room_name, DIRECT_JOIN_BOT, dev.api_key, dev.api_secret)
    transport = livekit_transport(dev, room_name=room_name, token=runner_token)

    with pytest.raises(ValueError, match=re.escape('with_kind("agent")')):
        transport_room(dev, transport)

    # The server's own view of that token: the bot really would have joined STANDARD.
    async with connect_room(dev, runner_token):
        kinds = {p.identity: p.kind for p in await list_participants(dev, room_name)}
    assert kinds == {DIRECT_JOIN_BOT: STANDARD}, "generate_token_with_agent() now sets the kind"


@pytest.mark.integration_extended
async def test_human_leaves_ends_session_expectedly(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """The normal end of a call: the human hangs up, the worker stops, and nothing is an error.

    This is the one test of ``humans_present=False`` against a real room, and it needs a second rtc
    client because the whole question is what livekit-server reports about the participants that are
    *left*. Get it wrong and every normal call end pushes an ``ErrorFrame`` at the developer, which is
    why an unexpected leave is told apart from a no-humans-left end by the room's remaining STANDARD
    participants rather than by the disconnect reason.

    The worker's no-humans-left end (it leaves about 5 s after the last human) is modelled by stopping
    the fake once the human's leave has been **observed** rather than after a sleep: what matters is the
    order, not the delay, because durations are inputs, never assertions.
    """
    dev = livekit_server
    api = await launcher.api()
    transport = livekit_transport(
        dev, room_name=room_name, token=direct_join_bot_token(dev, room_name=room_name)
    )
    room = transport_room(dev, transport)
    service = make_service(room, avatar_ids=AVATAR_IDS, api_url=api.base_url)
    rec = EventRecorder(service)

    pipeline = Pipeline([transport.input(), room.input_filter(), service, transport.output()])

    async with bidirectional_pipeline(pipeline) as h:
        await rec.wait("on_avatar_connected", timeout=WAIT)
        fake = launcher.fakes[0]
        lk_room = get_room(transport)

        human_token = mint_dev_token(dev, identity=HUMAN, room=room_name, kind=None)
        async with connect_room(dev, human_token):
            await until(lambda: HUMAN in lk_room.remote_participants, timeout=WAIT, what="the human to join")
            assert any(
                p.identity == HUMAN and p.kind == STANDARD for p in await list_participants(dev, room_name)
            ), "the human must be STANDARD, or this test proves nothing"
        await until(lambda: HUMAN not in lk_room.remote_participants, timeout=WAIT, what="the human to leave")

        # The worker notices the room has no humans left and ends its session.
        await fake.stop()

        (reason,) = await rec.wait("on_avatar_disconnected", timeout=WAIT)
        assert reason == "avatar_left_room", launcher.debug_dump()
        (ended,) = await rec.wait("on_session_ended", timeout=WAIT)
        assert ended == "no_humans_left", launcher.debug_dump()
        assert service.state is ServiceState.ENDED
        assert service.is_usable is True
        await h.finish()

    assert_absent(h.up, ErrorFrame, where="upstream of an expected session end")
    assert rec.count("on_error") == 0
    assert rec.count("on_avatar_disconnected") == 1
