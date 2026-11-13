"""*Early start*: the launch returns when this launch's avatar publishes its video, not at the 201.

The Synthesia API's 201 typically arrives about half a second after the avatar worker has joined and
published its video, and the plugin used to start sending audio only at the 201. Now the session mints a
random `synthesia.launch` id into the avatar token, and the launch returns as soon as the avatar carrying
that id publishes its video; `wait_session_created()` takes the 201 later.

What they protect: the launch returns before the 201 only for this launch's avatar, never for a stale avatar
with the same identity or one joined with a caller-provided token (which carries no launch id); the session
id and `session_created` are stamped when the 201 lands; a 201 that fails after an early start is raised by
`wait_session_created()`; a close before the 201 cancels the request and leaves nothing waiting on it; a
retried request before this launch's video means no early start, and one after it holds sending until the
201 lands.

`FakeSessionApiClient(gate=...)` holds the 201 open. The launch id is pinned by replacing the session
module's `secrets`, so an avatar can carry it before the request goes out.
"""

from __future__ import annotations

import asyncio
from time import monotonic
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import session as session_module
from pipecat_synthesia.protocol import RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE, TOKEN_ATTR_LAUNCH
from tests.fakes.fake_rtc import FakeLocalParticipant
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._session_rig import BOT, ROOM, SESSION_ID, Rig, SessionRigRoom, make_avatar, settle

LAUNCH = "0123456789abcdef"
OTHER_LAUNCH = "fedcba9876543210"
#: How long "and the launch still hasn't returned" is watched for; an input, not an outcome.
NOTHING_HAPPENS = 0.2


@pytest.fixture
def launch_id(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(session_module, "secrets", SimpleNamespace(token_hex=lambda _nbytes: LAUNCH))
    return LAUNCH


def _held_rig(**kwargs) -> tuple[Rig, asyncio.Event]:
    gate = asyncio.Event()
    return Rig(api=FakeSessionApiClient(gate=gate), **kwargs), gate


@pytest.mark.unit
async def test_the_launch_returns_when_this_launchs_avatar_publishes_before_the_201(launch_id: str) -> None:
    rig, gate = _held_rig(avatar_present=False)
    task = rig.launch_task()
    try:
        await until(lambda: rig.api.calls, what="the session request")
        assert rig.tokens.calls[0]["launch_id"] == launch_id
        avatar = rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: launch_id}, video=False))
        rig.publish_video(avatar)
        session = await asyncio.wait_for(task, timeout=5)
    finally:
        await settle(task)

    try:
        assert rig.api.returned == 0, "the launch returned before the 201"
        assert session.timings.session_created is None
        with pytest.raises(RuntimeError):
            _ = session.session_id

        gate.set()
        await session.wait_session_created()
        assert session.session_id == SESSION_ID
        assert session.timings.session_created is not None
        assert session.timings.session_created >= session.timings.session_requested
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_the_avatar_join_is_timed_when_the_avatar_joins_not_when_its_video_starts_the_session(
    launch_id: str,
) -> None:
    """`timings.avatar_joined` is the moment the avatar participant joined the room, not the later moment
    the launch stopped waiting (its video, after an early start; the 201 otherwise). The avatar joins, its
    video comes a measurable time later, and the recorded join must be the earlier of the two."""
    rig, gate = _held_rig(avatar_present=False)
    task = rig.launch_task()
    try:
        await until(lambda: rig.api.calls, what="the session request")
        joined_from = monotonic()
        avatar = rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: launch_id}, video=False))
        joined_by = monotonic()
        await asyncio.sleep(NOTHING_HAPPENS)  # the worker prepares before it publishes
        published_from = monotonic()
        rig.publish_video(avatar)
        session = await asyncio.wait_for(task, timeout=5)
    finally:
        await settle(task)

    try:
        assert joined_from <= session.timings.avatar_joined <= joined_by, "the join was not timed at the join"
        assert session.timings.avatar_joined < published_from, "the join was timed at the video"
    finally:
        gate.set()
        await session.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("start", ["at-the-201", "early"])
async def test_the_join_is_this_avatars_when_it_replaced_a_stale_one(launch_id: str, start: str) -> None:
    """An earlier launch's avatar, with the same identity, is still in the room when the session attaches;
    this launch's avatar then replaces it. `timings.avatar_joined` is the join of the avatar the launch
    ended up with, not the moment the stale one was first seen."""
    rig, gate = _held_rig(attributes={TOKEN_ATTR_LAUNCH: OTHER_LAUNCH})  # the stale avatar, present
    task = rig.launch_task()
    try:
        await until(lambda: rig.api.calls, what="the session request")
        assert rig.avatar is not None
        rig.leave(rig.avatar)
        joined_from = monotonic()
        fresh = rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: launch_id}, video=False))
        joined_by = monotonic()
        if start == "early":
            rig.publish_video(fresh)
        else:
            gate.set()
        session = await asyncio.wait_for(task, timeout=5)
    finally:
        await settle(task)

    try:
        assert session.avatar_participant is fresh
        assert joined_from <= session.timings.avatar_joined <= joined_by, "timed at the stale avatar"
    finally:
        gate.set()
        await session.aclose()


@pytest.mark.unit
async def test_this_launchs_avatar_already_publishing_when_the_room_is_attached(launch_id: str) -> None:
    """The avatar joined and published while the bot was connecting: seen at attach, not by an event."""
    rig, gate = _held_rig(attributes={TOKEN_ATTR_LAUNCH: launch_id})
    session = await asyncio.wait_for(rig.launch(), timeout=5)
    try:
        assert rig.api.returned == 0
        gate.set()
        await session.wait_session_created()
        assert session.session_id == SESSION_ID
    finally:
        await session.aclose()


@pytest.mark.unit
@pytest.mark.parametrize(
    "attributes",
    [{}, {TOKEN_ATTR_LAUNCH: OTHER_LAUNCH}],
    ids=["no-launch-id(provided-token-or-stale)", "another-launchs-avatar"],
)
async def test_an_avatar_that_is_not_this_launchs_waits_for_the_201(
    launch_id: str, attributes: dict[str, str]
) -> None:
    """An avatar with the same identity but without this launch's id, already publishing: a stale avatar
    from an earlier session, or one joined with a caller-provided token. The launch waits for the 201."""
    rig, gate = _held_rig(avatar_present=False)
    task = rig.launch_task()
    try:
        await until(lambda: rig.api.calls, what="the session request")
        avatar = rig.join(make_avatar(attributes=attributes, video=False))
        rig.publish_video(avatar)
        await asyncio.sleep(NOTHING_HAPPENS)
        assert not task.done(), "the launch returned before the 201 for an avatar that is not this launch's"

        gate.set()
        session = await asyncio.wait_for(task, timeout=5)
    finally:
        await settle(task)
    try:
        assert session.session_id == SESSION_ID
        assert session.timings.session_created is not None
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_a_201_that_fails_after_an_early_start_is_raised_by_wait_session_created(
    launch_id: str,
) -> None:
    gate = asyncio.Event()
    failure = SynthesiaError("the API dropped the connection", type=ErrorType.CONNECTION)
    rig = Rig(api=FakeSessionApiClient(gate=gate, error=failure), attributes={TOKEN_ATTR_LAUNCH: launch_id})
    session = await asyncio.wait_for(rig.launch(), timeout=5)
    try:
        gate.set()
        with pytest.raises(SynthesiaError) as raised:
            await session.wait_session_created()
        assert raised.value is failure
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_closing_before_the_201_cancels_the_request(launch_id: str) -> None:
    rig, gate = _held_rig(attributes={TOKEN_ATTR_LAUNCH: launch_id})
    before = asyncio.all_tasks()
    session = await asyncio.wait_for(rig.launch(), timeout=5)
    await session.aclose()

    with pytest.raises(SynthesiaError) as raised:
        await session.wait_session_created()
    assert raised.value.type is ErrorType.CONNECTION
    assert rig.api.returned == 0
    leftover = {t for t in asyncio.all_tasks() - before if not t.done()}
    assert not leftover, f"still running after the close: {leftover}"
    gate.set()  # nothing is waiting on it any more
    rig.assert_torn_down()


# --- the session request is retried ----------------------------------------------------------------
#
# `SessionApiClient` retries with the same token, and a failed attempt may still have launched the worker
# now in the room: the retry's worker then joins as the same identity and replaces it (seen live: a 502
# after 3.9 s with the worker already publishing, then a second worker 2 s later). That avatar carries
# this launch's id but may not be the one that stays.

#: 20 ms of 24 kHz mono s16le.
PCM = b"\x01\x00" * 480


@pytest.mark.unit
async def test_a_retry_before_this_launchs_video_means_no_early_start(launch_id: str) -> None:
    """The request was retried before the avatar published: the launch waits for the 201, as without
    an early start."""
    gate = asyncio.Event()
    rig = Rig(api=FakeSessionApiClient(gate=gate, retries=1), avatar_present=False)
    task = rig.launch_task()
    try:
        await until(lambda: rig.api.retries_reported == 1, what="the retry reported")
        avatar = rig.join(make_avatar(attributes={TOKEN_ATTR_LAUNCH: launch_id}, video=False))
        rig.publish_video(avatar)
        await asyncio.sleep(NOTHING_HAPPENS)
        assert not task.done(), "an early start after the request was retried"

        gate.set()
        session = await asyncio.wait_for(task, timeout=5)
    finally:
        await settle(task)
    try:
        assert session.session_id == SESSION_ID
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_a_retry_after_an_early_start_holds_sending_until_the_201_lands(launch_id: str) -> None:
    """Audio flows after the early start; once the request is retried, `sending` is False until the 201,
    which releases it as it lands, whether or not anything waits in `wait_session_created()`."""
    gate, retry = asyncio.Event(), asyncio.Event()
    rig = Rig(
        api=FakeSessionApiClient(gate=gate, retries=1, retry_gate=retry),
        attributes={TOKEN_ATTR_LAUNCH: launch_id},
    )
    session = await asyncio.wait_for(rig.launch(), timeout=5)
    try:
        assert session.sending
        await session.send_audio(PCM, 24000, 1)
        assert len(rig.participant.streams) == 1, "audio flows after the early start"

        retry.set()
        await until(lambda: rig.api.retries_reported == 1, what="the retry reported")
        assert not session.sending
        waiting = asyncio.ensure_future(session.wait_sending())
        await asyncio.sleep(NOTHING_HAPPENS)
        assert not waiting.done(), "the hold ended before the 201"

        gate.set()  # the 201 lands; nothing is in wait_session_created()
        await asyncio.wait_for(waiting, timeout=5)
        assert session.sending
        assert session.timings.session_created is not None
    finally:
        await session.aclose()


@pytest.mark.unit
async def test_a_close_ends_a_hold(launch_id: str) -> None:
    gate, retry = asyncio.Event(), asyncio.Event()
    rig = Rig(
        api=FakeSessionApiClient(gate=gate, retries=1, retry_gate=retry),
        attributes={TOKEN_ATTR_LAUNCH: launch_id},
    )
    session = await asyncio.wait_for(rig.launch(), timeout=5)
    retry.set()
    await until(lambda: rig.api.retries_reported == 1, what="the retry reported")
    waiting = asyncio.ensure_future(session.wait_sending())
    await asyncio.sleep(0)
    await session.aclose()
    await asyncio.wait_for(waiting, timeout=5)  # returns: nothing hangs on a closed session


@pytest.mark.unit
async def test_log_lines_after_the_201_carry_the_session_id_after_an_early_start(launch_id: str) -> None:
    """Every logger the session hands out carries the session id once the 201 is in, including those
    built at the early start, before it, such as the audio stream's: a structured sink reads it there."""
    seen: list[Any] = []
    sink = logger.add(lambda message: seen.append(message.record), level="DEBUG")
    gate = asyncio.Event()
    rig = Rig(api=FakeSessionApiClient(gate=gate), attributes={TOKEN_ATTR_LAUNCH: launch_id})
    session = await asyncio.wait_for(rig.launch(), timeout=5)
    try:
        gate.set()
        await session.wait_session_created()
        await session.send_audio(PCM, 24000, 1)  # after the 201
        opened = next(r for r in seen if r["message"].startswith("opened audio stream"))
        assert opened["extra"].get("session_id") == SESSION_ID
    finally:
        logger.remove(sink)
        await session.aclose()


class _TheClearBufferLetsThe201Land(FakeLocalParticipant):
    """The bot's participant, whose clear buffer lets the held 201 land and returns once it has."""

    def __init__(self, gate: asyncio.Event, api: FakeSessionApiClient) -> None:
        super().__init__(BOT)
        self._gate, self._api = gate, api
        self.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)

    async def perform_rpc(self, **kwargs: Any) -> str:
        if kwargs["method"] == RPC_CLEAR_BUFFER:
            self._gate.set()
            await until(lambda: self._api.returned == 1, what="the 201")
        return await super().perform_rpc(**kwargs)


@pytest.mark.unit
async def test_a_201_that_lands_after_the_service_stopped_waiting_still_names_the_session(
    launch_id: str,
) -> None:
    """The launch timeout runs out with the 201 still out, and the 201 lands while the avatar is being
    stopped. The service has the 201's id from then; the session's own records carry it too."""
    gate = asyncio.Event()
    api = FakeSessionApiClient(gate=gate)
    rig = Rig(
        api=api,
        participant=_TheClearBufferLetsThe201Land(gate, api),
        attributes={TOKEN_ATTR_LAUNCH: launch_id},
    )
    service = make_service(SessionRigRoom(rig), required=False, launch_timeout=0.5, stop_drain_timeout=0.2)
    rec = EventRecorder(service)
    seen: list[Any] = []
    sink = logger.add(lambda message: seen.append(message.record), level="DEBUG")
    try:
        async with bidirectional_pipeline(service) as h:
            await rec.wait("on_session_ended")
            await until(
                lambda: any(r["message"].endswith("avatar session closed") for r in seen), what="the close"
            )
            await h.finish()
    finally:
        logger.remove(sink)

    assert service.metrics.session_id == SESSION_ID
    closed = next(r for r in seen if r["message"].endswith("avatar session closed"))
    assert closed["extra"].get("room") == ROOM, "a record from the session's own logger"
    assert closed["extra"].get("session_id") == SESSION_ID
