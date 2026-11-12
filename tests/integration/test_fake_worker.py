"""Self-tests for ``FakeAvatarWorker`` and the ``Launcher``: one per detector, each driven by a deliberately
wrong sender.

``FakeAvatarWorker`` is the oracle for the worker protocol, and **an oracle that says yes to everything
makes the whole integration tier decorative**. Every rule the fake enforces therefore has a test here that
breaks that one rule with a raw-``rtc`` sender and asserts the fake noticed. The sabotage check is the
inverse: swap a ``BadSender`` call for the equivalent ``GoodSender`` one and the test must fail.

These are tests *of the fake*, kept in their own file rather than mixed into a product suite. The last
test covers the ``Launcher``: it hooks only successful responses, so a retried launch is one worker and
not one per HTTP attempt.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from livekit import rtc

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from pipecat_synthesia.protocol import (
    ATTR_AVATAR_READY,
    ATTR_AVATAR_STATUS,
    RPC_SWAP_AVATAR,
    SWAP_ERROR_CODES,
    SWAP_RESPONSE_STATUS_OK,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
)
from tests.fakes.fake_api import ScriptedResponse
from tests.integration._launcher import SESSION_BODY, Launcher
from tests.integration._senders import BadSender, GoodSender
from tests.integration.fake_worker import (
    DEFAULT_AVATAR_IDENTITY,
    SWAP_CODE_AVATAR_PREPARING,
    FakeAvatarWorker,
)
from tests.support.livekit_dev import LiveKitDev, connect_room, list_participants, mint_dev_token
from tests.support.secrets import API_KEY
from tests.support.waiting import until

pytestmark = pytest.mark.integration

BOT_IDENTITY = "bot"
INTRUDER_IDENTITY = "intruder"
#: 2 KiB of non-repeating bytes: enough that a truncated read is visible as a byte-equality failure, small
#: enough to cross the SFU in one chunk.
PAYLOAD = bytes(range(256)) * 8
#: Every network-observable wait in the integration tier. A backstop, not a budget.
WAIT = 10.0


@pytest.fixture
async def bot(livekit_server: LiveKitDev, room_name: str) -> AsyncIterator[rtc.Room]:
    """The bot's own connected ``rtc.Room``, with ``kind=agent`` (*agent-kind requirement*)."""
    token = mint_dev_token(livekit_server, identity=BOT_IDENTITY, room=room_name, kind="agent")
    async with connect_room(livekit_server, token) as room:
        yield room


@contextlib.asynccontextmanager
async def running_fake(dev: LiveKitDev, room_name: str, **options: Any) -> AsyncIterator[FakeAvatarWorker]:
    """A started ``FakeAvatarWorker`` bound to ``BOT_IDENTITY``, stopped however the test ends."""
    fake = FakeAvatarWorker.for_bot(dev, room_name=room_name, bot_identity=BOT_IDENTITY, **options)
    try:
        await fake.start()
        yield fake
    finally:
        await fake.stop()


async def test_good_sender_is_accepted(livekit_server: LiveKitDev, room_name: str, bot: rtc.Room) -> None:
    """A sender that obeys the protocol is recorded verbatim and flagged for nothing.

    The first thing to prove about a detector is that it does not fire on correct input: every other test
    in this file is meaningless if this one can pass while the fake rejects everything.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = GoodSender(bot, avatar_identity=fake.identity)
        sender.register_handlers()

        await sender.send(PAYLOAD)

        await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="one recorded stream")
        stream = fake.streams[0]
        assert stream["identity"] == BOT_IDENTITY, fake.debug_dump()
        # Byte-exact, not "contains": both must be decimal *strings* taken from the frame.
        assert stream["attributes"] == {"sample_rate": "24000", "num_channels": "1"}, fake.debug_dump()
        assert stream["bytes"] == PAYLOAD, fake.debug_dump()

        await until(lambda: len(sender.finished_calls) == 1, timeout=WAIT, what="playback_finished")
        assert sender.started_calls == [""], "lk.playback_started carries an empty payload"
        assert sender.finished_calls[0]["interrupted"] is False
        assert "playback_position" in sender.finished_calls[0]

        assert fake.overlapping_streams == 0, fake.debug_dump()
        assert fake.token_violations == [], fake.debug_dump()
        assert fake.attribute_violations == [], fake.debug_dump()
        assert fake.rejected_stream_identities == [], fake.debug_dump()
        assert fake.fatal_unsupported_playback_finished is False, fake.debug_dump()
        assert fake.connected is True, fake.debug_dump()


async def test_missing_playback_finished_is_fatal(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """No ``lk.playback_finished`` handler on the bot ends the worker's session.

    The real receiver retries three times and then *raises*, which ends the session. A fake that merely logged
    this would let the single most damaging regression in the plugin ship green.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = BadSender(bot, avatar_identity=fake.identity)

        await sender.send_without_playback_finished(PAYLOAD)

        await until(
            lambda: fake.fatal_unsupported_playback_finished,
            timeout=WAIT,
            what="the fake to give up on lk.playback_finished",
        )
        await until(lambda: fake.connected is False, timeout=WAIT, what="the fake to disconnect")
        assert sender.started_calls == [""], "lk.playback_started was registered and must have arrived"


async def test_raising_playback_started_handler_is_fatal(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """A registered ``lk.playback_started`` handler that *raises* also ends the session.

    The receiver tolerates only ``UNSUPPORTED_METHOD`` for this method. A handler
    that is present but broken, which rtc reports as ``APPLICATION_ERROR``, takes the same
    retry-three-times-then-raise path as ``lk.playback_finished``.

    ``fatal_unsupported_playback_finished`` deliberately stays ``False`` here: that recorder means "the bot
    never registered the handler". This bot did register one; the session dies anyway.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = BadSender(bot, avatar_identity=fake.identity)

        await sender.send_with_raising_playback_started(PAYLOAD)

        await until(lambda: fake.connected is False, timeout=WAIT, what="the fake to end its session")
        assert fake.fatal_unsupported_playback_finished is False, fake.debug_dump()


async def test_missing_attributes_kill_the_session(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """A stream without ``sample_rate`` is fatal, and its bytes are never read.

    The receiver's ``int(attrs["sample_rate"])`` raises ``ValueError``, which propagates to the worker and
    ends the session.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = BadSender(bot, avatar_identity=fake.identity)
        sender.register_handlers()

        # The fake closes the reader and leaves; the writer's own close can fail on the way out, exactly
        # as it would for a real bot whose worker just died. The assertions are on the fake.
        with contextlib.suppress(Exception):
            await sender.send_without_sample_rate(PAYLOAD)

        await until(lambda: len(fake.attribute_violations) == 1, timeout=WAIT, what="one attribute violation")
        await until(lambda: fake.connected is False, timeout=WAIT, what="the fake to disconnect")
        assert fake.streams == [], fake.debug_dump()
        assert sender.finished_calls == [], "a stream the receiver refused never reports playback"


async def test_foreign_identity_stream_is_ignored(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """A stream from anyone but ``lk.publish_on_behalf`` is dropped silently, and the identity recorded.

    This is the failure mode that makes a mis-minted token invisible: the worker does not complain, it
    simply never speaks. The fake records the identity instead of nothing.
    """
    intruder_token = mint_dev_token(livekit_server, identity=INTRUDER_IDENTITY, room=room_name, kind=None)
    async with running_fake(livekit_server, room_name) as fake:
        async with connect_room(livekit_server, intruder_token) as intruder_room:
            intruder = GoodSender(intruder_room, avatar_identity=fake.identity)
            intruder.register_handlers()

            await intruder.send(PAYLOAD)

            await until(
                lambda: fake.rejected_stream_identities == [INTRUDER_IDENTITY],
                timeout=WAIT,
                what="the intruder's identity to be recorded",
            )
            assert fake.streams == [], fake.debug_dump()
            assert intruder.finished_calls == [], "a dropped stream reports no playback"


async def test_overlapping_streams_are_counted(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """A second stream opened before the first closed is counted (*one-open-stream rule*).

    The avatar worker's receiver has no supersede and no idle timeout: readers are processed
    strictly one at a time, so the second stream queues behind a first that may never end.
    ``overlapping_streams`` is how that becomes an assertion instead of a hang.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = BadSender(bot, avatar_identity=fake.identity)
        sender.register_handlers()

        await sender.send_overlapping(PAYLOAD)

        await until(lambda: fake.overlapping_streams == 1, timeout=WAIT, what="one overlap")
        await until(lambda: len(fake.streams) == 2, timeout=WAIT, what="both streams recorded")


@pytest.mark.parametrize(
    ("kind", "attributes", "expected_in_violation"),
    [
        pytest.param(None, {TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY}, "kind", id="standard-kind"),
        pytest.param("agent", None, TOKEN_ATTR_PUBLISH_ON_BEHALF, id="no-publish-on-behalf"),
    ],
)
async def test_token_violations_recorded_not_fatal(
    livekit_server: LiveKitDev,
    room_name: str,
    bot: rtc.Room,
    kind: str | None,
    attributes: dict[str, str] | None,
    expected_in_violation: str,
) -> None:
    """A malformed avatar token is recorded, and the fake still joins.

    Both of these break a *real* worker: a standard-kind avatar confuses the worker's count of human
    participants, and a missing ``lk.publish_on_behalf`` rejects the launch. The fake joins anyway on
    purpose: a fake that refused to start would make the product's failure show up as a fixture error with
    no recorder to read.
    """
    token = mint_dev_token(
        livekit_server,
        identity=DEFAULT_AVATAR_IDENTITY,
        room=room_name,
        kind=kind,
        attributes=attributes,
    )
    fake = FakeAvatarWorker(livekit_server.url, token)
    try:
        await fake.start()

        assert len(fake.token_violations) == 1, fake.debug_dump()
        assert expected_in_violation in fake.token_violations[0]
        assert fake.connected is True, "a bad token is recorded, never fatal to the fake"
    finally:
        await fake.stop()


async def test_clear_buffer_from_foreign_identity_rejected(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """``lk.clear_buffer`` from anyone but the bound sender returns ``"reject"`` and does nothing.

    The receiver applies the same identity gate that guards the byte streams, and a fake that honoured any
    caller's barge-in would hide a bot that interrupted the wrong avatar.
    """
    intruder_token = mint_dev_token(livekit_server, identity=INTRUDER_IDENTITY, room=room_name, kind=None)
    async with running_fake(livekit_server, room_name) as fake:
        async with connect_room(livekit_server, intruder_token) as intruder_room:
            intruder = GoodSender(intruder_room, avatar_identity=fake.identity)

            assert await intruder.clear_buffer() == "reject"
            assert fake.clear_buffer_calls == 0, fake.debug_dump()


async def test_clear_buffer_mid_stream_sends_interrupted_finished(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """Barge-in mid-utterance reports ``interrupted: true`` once, and not again at EOF.

    The worker sends ``playback_finished(interrupted=True)`` from its clear-buffer path and resets its
    position, so the utterance's normal end-of-stream report is suppressed. A second,
    ``interrupted: false`` report after a barge-in is how a bot ends up believing the avatar is still
    speaking after it was cut off.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = GoodSender(bot, avatar_identity=fake.identity)
        sender.register_handlers()

        writer = await sender.open()
        await writer.write(PAYLOAD)
        # `lk.playback_started` is the public signal that the fake has bytes for this utterance; waiting
        # on it rather than sleeping is what keeps this test honest on a loaded runner.
        await until(lambda: len(sender.started_calls) == 1, timeout=WAIT, what="playback_started")

        assert await sender.clear_buffer() == "ok"

        await until(lambda: fake.clear_buffer_calls == 1, timeout=WAIT, what="the clear_buffer to land")
        await until(
            lambda: len(sender.finished_calls) == 1, timeout=WAIT, what="an interrupted playback_finished"
        )
        assert sender.finished_calls[0]["interrupted"] is True, fake.debug_dump()

        await writer.aclose()
        await until(lambda: len(fake.streams) == 1, timeout=WAIT, what="the cleared stream to be recorded")
        assert len(sender.finished_calls) == 1, "a cleared segment reports playback exactly once"


async def test_clear_buffer_between_utterances_reports_nothing(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """A barge-in with nothing playing is accepted, and produces no playback report.

    The worker sends no ``lk.playback_finished`` when nothing is playing, so a ``lk.clear_buffer`` that
    lands *between* utterances gets no report at all. A fake that answered it with an interrupted report
    for the utterance that just ended would hand the bot a second stop for an utterance it had already
    stopped, which the product reads as an extra ``BotStoppedSpeakingFrame`` and an extra interruption
    metric.

    The second utterance is the synchronisation point rather than a sleep: its own report cannot arrive
    before a spurious one, so asserting on the *sequence* of ``interrupted`` flags catches an extra report
    wherever it was emitted.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = GoodSender(bot, avatar_identity=fake.identity)
        sender.register_handlers()

        await sender.send(PAYLOAD)
        await until(lambda: len(sender.finished_calls) == 1, timeout=WAIT, what="the first utterance")

        assert await sender.clear_buffer() == "ok"
        await until(lambda: fake.clear_buffer_calls == 1, timeout=WAIT, what="the barge-in to land")

        await sender.send(PAYLOAD)
        await until(lambda: len(sender.finished_calls) == 2, timeout=WAIT, what="the second utterance")
        assert [call["interrupted"] for call in sender.finished_calls] == [False, False], fake.debug_dump()


async def test_a_stream_blocks_the_one_behind_it(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """Streams are read one at a time, so an unclosed stream stalls every stream behind it.

    This is the failure the *one-open-stream rule* is really about. The avatar worker's receiver
    has no idle timeout and no supersede, so a stream the bot forgets to close does not lose its own
    tail: it silently swallows every utterance after it, for the rest of the session. An oracle that read
    streams concurrently would report healthy playback for all of them, and the regression would only
    appear in production.

    ``len(sender.started_calls) == 1`` is the positive synchronisation point: the fake has begun an
    utterance and can begin no other, so the "nothing recorded yet" check below is a statement about a
    settled state rather than a race with one.
    """
    async with running_fake(livekit_server, room_name) as fake:
        sender = GoodSender(bot, avatar_identity=fake.identity)
        sender.register_handlers()

        first = await sender.open()
        second = await sender.open()
        await first.write(PAYLOAD)
        await second.write(PAYLOAD)
        # The second utterance is complete on the wire; the first is not.
        await second.aclose()

        await until(lambda: len(sender.started_calls) == 1, timeout=WAIT, what="the first segment to open")
        await until(lambda: fake.overlapping_streams == 1, timeout=WAIT, what="the overlap to be counted")
        assert fake.streams == [], "the second stream must wait behind the first"
        assert fake.open_streams == 2, fake.debug_dump()

        await first.aclose()

        await until(lambda: len(fake.streams) == 2, timeout=WAIT, what="both streams, once unblocked")
        assert [stream["bytes"] for stream in fake.streams] == [PAYLOAD, PAYLOAD], fake.debug_dump()
        await until(lambda: fake.open_streams == 0, timeout=WAIT, what="both streams closed")


async def test_swap_preparing_returns_code(livekit_server: LiveKitDev, room_name: str, bot: rtc.Room) -> None:
    """``swapAvatar`` to an avatar that is still preparing carries ``AVATAR_PREPARING``.

    The code is the whole point: a plain ``error`` means "that avatar does not exist here", while this one
    means "ask again later", and only the code distinguishes them. ``status_delay`` is far longer than the
    test so the second avatar cannot become ready underneath the assertion.
    """
    async with running_fake(livekit_server, room_name, avatar_ids=["a", "b"], status_delay=30.0) as fake:
        # `perform_rpc` resolves the destination against this room's own remote-participant map and
        # raises RECIPIENT_NOT_FOUND if the avatar is not in it yet. The bot connected first, so it
        # learns of the avatar from an event rather than from its join response — on a loaded runner
        # that event can still be in flight when `start()` returns.
        await until(
            lambda: fake.identity in bot.remote_participants, timeout=WAIT, what="the avatar to be visible"
        )
        response = await bot.local_participant.perform_rpc(
            destination_identity=fake.identity,
            method=RPC_SWAP_AVATAR,
            payload=json.dumps({"avatar_id": "b"}),
            response_timeout=5.0,
        )

        body = json.loads(response)
        assert body["code"] == SWAP_CODE_AVATAR_PREPARING, fake.debug_dump()
        # The fake emits the worker's code; `protocol.py` is our list of the codes we recognise. If these
        # ever disagree, the product is reading a code nobody sends.
        assert body["code"] in SWAP_ERROR_CODES
        assert fake.swap_calls == [{"avatar_id": "b"}], fake.debug_dump()


async def test_readiness_attributes_published(
    livekit_server: LiveKitDev, room_name: str, bot: rtc.Room
) -> None:
    """The fake publishes ``avatar_status`` and ``avatar_ready`` on itself, keyed by unprefixed avatar id.

    Observed from the *bot's* room, not from the fake's own state: the product reads these through
    ``participant_attributes_changed`` on a remote participant, so this also checks that livekit-server
    actually propagates them. The trailing ``swapAvatar`` proves the handler that returns
    ``AVATAR_PREPARING`` above returns success once the avatar is ready; without it, a fake that rejected
    every swap would pass that test too.
    """
    async with running_fake(livekit_server, room_name, avatar_ids=["av_a", "av_b"], status_delay=0.5) as fake:

        def ready() -> bool:
            participant = bot.remote_participants.get(fake.identity)
            return participant is not None and participant.attributes.get(ATTR_AVATAR_READY) == "true"

        await until(ready, timeout=WAIT, what="avatar_ready=true seen from the bot's room")

        attributes = bot.remote_participants[fake.identity].attributes
        # Keys are unprefixed avatar ids (*unprefixed ids on the wire*).
        assert json.loads(attributes[ATTR_AVATAR_STATUS]) == {
            "a": {"state": "ready"},
            "b": {"state": "ready"},
        }, fake.debug_dump()
        # The token attributes are still there after the readiness write. This locks the *server's*
        # merge semantics, not the fake's re-sending of its whole attribute map: on livekit-server 1.9.1,
        # a `set_attributes` call that omits a key leaves that key alone, so dropping the re-send would
        # not break this. The fake re-sends because the real worker does, not because it is load-bearing.
        assert attributes[TOKEN_ATTR_PUBLISH_ON_BEHALF] == BOT_IDENTITY

        response = await bot.local_participant.perform_rpc(
            destination_identity=fake.identity,
            method=RPC_SWAP_AVATAR,
            payload=json.dumps({"avatar_id": "b"}),
            response_timeout=5.0,
        )
        assert json.loads(response) == {"status": SWAP_RESPONSE_STATUS_OK, "avatar_id": "b"}


async def test_leave_after_disconnects(livekit_server: LiveKitDev, room_name: str, bot: rtc.Room) -> None:
    """``leave_after`` makes the avatar go away, visibly, from the bot's side.

    This is the knob behind every "the avatar left mid-call" integration test, so it needs to be the real
    thing: an actual ``participant_disconnected`` from livekit-server, not a flag on the fake.
    """
    departed: list[str] = []
    bot.on("participant_disconnected", lambda participant: departed.append(participant.identity))

    async with running_fake(livekit_server, room_name, leave_after=0.5) as fake:
        await until(
            lambda: fake.identity in departed,
            timeout=WAIT,
            what="participant_disconnected for the avatar",
        )
        assert fake.connected is False, fake.debug_dump()


async def test_launcher_hooks_only_successful_responses(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """A ``[503, 201]`` script launches exactly **one** fake worker.

    A retried launch is one launch. The API client resolves the avatar token once and re-sends it
    verbatim, so a fixture that launched a worker per HTTP *attempt* would put two workers on the same
    identity for reasons that have nothing to do with the product, and hide the same-identity
    replacement test behind a fixture artefact.

    Driven through a real ``SessionApiClient``, not by calling the hook: what is being tested is that
    the retry the client actually performs produces one worker.
    """
    dev = livekit_server
    api = await launcher.api([ScriptedResponse(503), ScriptedResponse(201, dict(SESSION_BODY))])
    token = mint_dev_token(
        dev,
        identity=DEFAULT_AVATAR_IDENTITY,
        room=room_name,
        kind="agent",
        attributes={TOKEN_ATTR_PUBLISH_ON_BEHALF: BOT_IDENTITY},
    )

    async with aiohttp.ClientSession() as http:
        client = SessionApiClient(
            api_key=API_KEY,
            api_url=api.base_url,
            client=build_client_info(),
            http_session=http,
            backoff_seconds=(0.01,),
        )
        created = await client.create_session(avatar_ids=["av_a"], livekit_url=dev.url, livekit_token=token)

    assert created.session_id == SESSION_BODY["id"]
    assert len(api.requests) == 2, launcher.debug_dump()
    assert len(launcher.fakes) == 1, launcher.debug_dump()
    fake = launcher.fakes[0]
    assert fake.connected is True, launcher.debug_dump()
    # The 201's hook awaited `start()`, so both tracks were published before the client saw the body,
    # as the real API promises.
    assert fake.publish_on_behalf == BOT_IDENTITY
    assert fake.token_violations == [], launcher.debug_dump()
    assert [p.identity for p in await list_participants(dev, room_name)] == [DEFAULT_AVATAR_IDENTITY]
