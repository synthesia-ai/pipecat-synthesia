"""`LiveKitAvatarSession` on `FakeRoom` and `FakeLocalParticipant`, through `_session_rig.py`.

What these protect: in a room the bot is already connected to, both playback handlers are registered
before `POST /sessions` (a direct join's *connect overlap* is `test_connect_overlap.py`'s); the join wait; an
interruption sends `clear_buffer` and then closes the stream; readiness comes from the avatar's attributes
only; a swap's wait, bounds and failure types; rollback, teardown and release are *single flight*,
bounded, never disconnect a borrowed room, and finish through any cancel (*cancel-proof cleanup*); leave
classification and the *replacement grace*; no cancellation is swallowed; and the callback rules: nothing
follows `on_avatar_left`, and no callback waits for another.

Every asynchronous outcome is awaited with `until()`. A duration is shrunk only where the test is about it
elapsing: `_types.REPLACEMENT_GRACE_SECS` in the grace tests, `launch_timeout` in the join-wait timeouts,
`_types.POST_CREATE_JOIN_WAIT_SECS` in the backstop test, `_types.CLOSE_HOOK_SECS` in the hook-timeout test
and a swap's `timeout` in the tests about it running out; where the point is that something did not wait,
the duration is raised far past `until()`'s 5 s instead (`LONG_TIMEOUT`, the 30 s join wait).
`ROOM_DISCONNECT_SECS` and `ROOM_DELETE_SECS` stay real because they bound real operations. Waiting 10x a
shrunk grace is how a test asserts "and nothing happened".
"""

from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic
from typing import Any, cast

import aiohttp
import pytest
from livekit import rtc as lk_rtc

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.api import SessionApiClient
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._types import LiveKitCredentials, MintedAvatarToken
from pipecat_synthesia.livekit.rpc import PlaybackRpcHandlers
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_NAME, decode_unsigned
from pipecat_synthesia.protocol import (
    ATTR_AVATAR_STATUS,
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_REJECT,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    RPC_SWAP_AVATAR,
    STREAM_ATTR_NUM_CHANNELS,
    STREAM_ATTR_SAMPLE_RATE,
    SWAP_DEFAULT,
    SWAP_ERROR_CODE_AVATAR_FAILED,
)
from tests.fakes.fake_api import RecordedRequest, ScriptedResponse
from tests.fakes.fake_rtc import (
    FakeLocalParticipant,
    FakeRemoteParticipant,
    FakeTrackPublication,
    never_returns,
)
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.support.callbacks import CallbackRecorder
from tests.support.secrets import API_KEY, LK_API_SECRET, LK_TOKEN_MARKER, assert_no_secrets
from tests.support.waiting import until
from tests.unit._session_rig import (
    A_READY_B_PREPARING,
    AGENT,
    ALL_READY,
    AUDIO_SID,
    AVATAR,
    AVATAR_IDS,
    BOT,
    GRACE,
    GRACE_MARGIN,
    KIND_VIDEO,
    PREPARING,
    ROOM,
    SESSION_ID,
    STANDARD,
    URL,
    VIDEO_SID,
    Rig,
    StubTokenSource,
    launched,
    make_avatar,
    settle,
    status_attr,
)

CHUNK = b"\x01\x02" * 480  # 20 ms of 24 kHz mono s16le
DUPLICATE_IDENTITY = lk_rtc.DisconnectReason.DUPLICATE_IDENTITY
CLIENT_INITIATED = lk_rtc.DisconnectReason.CLIENT_INITIATED
CODES = lk_rtc.RpcError.ErrorCode
SWAPPED_TO_B = json.dumps({"status": "ok", "avatar_id": "b"})
#: Far longer than `until()`'s 5 s: an operation that finishes inside `until()` did not wait it out.
LONG_TIMEOUT = 60.0


def warnings_with(caplog: pytest.LogCaptureFixture, keyword: str) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and keyword in record.getMessage()
    ]


# ----------------------------------------------------------------------------------------- launch


async def test_launch_order() -> None:
    """In a room the bot is already connected to, both playback RPC handlers are registered, and the avatar
    token resolved, before `POST /sessions`: `launch()`'s order, which the relayed join keeps."""
    snapshots: list[tuple[list[str], int]] = []
    holder: list[Rig] = []

    async def on_create(avatar_ids, livekit_url, livekit_token) -> None:
        rig = holder[0]
        tokens = cast(StubTokenSource, rig.tokens)
        snapshots.append((list(rig.participant.registration_order), len(tokens.calls)))

    rig = Rig(api=FakeSessionApiClient(on_create=on_create))
    holder.append(rig)
    session = await rig.launch()
    try:
        assert len(snapshots) == 1
        registered_at_post, tokens_resolved_at_post = snapshots[0]
        assert {RPC_PLAYBACK_STARTED, RPC_PLAYBACK_FINISHED} <= set(registered_at_post)
        assert tokens_resolved_at_post == 1
        # The request body is a wire contract: exactly the three keys, the resolved token, the URL as given.
        assert rig.api.calls == [
            {"avatar_ids": AVATAR_IDS, "livekit_url": URL, "livekit_token": "stub-avatar-token"}
        ]
        resolve = cast(StubTokenSource, rig.tokens).calls[0]
        assert resolve["bot_identity"] == BOT
        assert resolve["room_name"] == ROOM
        assert resolve["avatar_identity"] == AVATAR
        assert resolve["avatar_name"] == DEFAULT_AVATAR_NAME
    finally:
        await session.aclose()


async def test_avatar_already_present_before_201() -> None:
    """The avatar can be in the room before the 201, when a launch retried with the same token replaced a
    stalled worker. No join event comes, so none is waited for: with a 30 s join wait, returning inside
    `until()` means the wait was not waited out."""
    rig = Rig(launch_timeout=_types.POST_CREATE_JOIN_WAIT_SECS)
    task = rig.launch_task()
    try:
        await until(task.done, what="launch() to return without waiting out the join wait")
    finally:
        await settle(task)
    session = task.result()
    try:
        assert len(rig.api.calls) == 1
        assert session.session_id == SESSION_ID
        assert session.avatar_identity == AVATAR
        assert session.room_name == ROOM
        assert session.active_avatar_id == "av_a"
        assert session.avatar_participant is rig.avatar
        assert session.is_alive is True
        # timed when the session first saw it in the room, before the 201, not when launch() stopped waiting
        assert session.timings.avatar_joined <= session.timings.session_created
        assert session.timings.room_connected == rig.handle.connected_at
        assert session.avatar_status == {}
        assert session.all_ready is False
    finally:
        await session.aclose()


async def test_avatar_joins_after_201(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """The join arrives as an event after the 201; `track_published` reaches `on_track_published`."""
    rig = Rig(avatar_present=False)
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="the 201")
    avatar = rig.join()

    session = await task
    try:
        assert session.avatar_participant is avatar
        assert session.timings.session_created <= session.timings.avatar_joined
        assert any(
            record.levelno == logging.INFO and "joined" in record.getMessage()
            for record in caplog_loguru.records
        )
        for sid in (VIDEO_SID, AUDIO_SID):
            rig.room.emit("track_published", avatar.track_publications[sid], avatar)
        await until(lambda: sorted(rig.rec.tracks) == ["audio", "video"], what="both track kinds")
        # a foreign participant's publications are not the avatar's
        human = FakeRemoteParticipant("human", kind=STANDARD)
        rig.room.emit("track_published", FakeTrackPublication("TR_h", kind=KIND_VIDEO), human)
        await asyncio.sleep(0)
        assert len(rig.rec.tracks) == 2
    finally:
        await session.aclose()


async def test_avatar_that_left_again_is_refetched() -> None:
    """The object that resolved the future is never used; a participant that left again is waited for."""
    rig = Rig(avatar_present=False)
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="the 201")
    first = rig.join()
    rig.leave(first)  # gone again in the same tick, before launch() could look
    await asyncio.sleep(0)
    assert not task.done(), "launch() must wait for a participant that is actually in the room"

    second = rig.join()
    session = await task
    try:
        assert session.avatar_participant is second
        assert session.avatar_participant is not first
    finally:
        await session.aclose()


async def test_an_avatar_that_left_again_before_it_was_seen_is_not_a_join() -> None:
    """With nothing coming back, the join wait runs to its end (`launch_timeout`, shrunk: the test is
    about it running out). test_avatar_that_left_again_is_refetched checks only one loop turn of waiting
    and that a later join is used; this one that the gone participant never ends the wait."""
    rig = Rig(avatar_present=False, launch_timeout=0.3)
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="launch() to be waiting for the join")
    rig.leave(rig.join())  # gone again in the same tick, before launch() could look

    with pytest.raises(SynthesiaError) as excinfo:
        await task

    assert excinfo.value.type is ErrorType.TIMEOUT
    assert excinfo.value.session_id == SESSION_ID
    rig.assert_torn_down()


async def test_join_wait_times_out() -> None:
    """No avatar within min(POST_CREATE_JOIN_WAIT_SECS, launch_timeout): typed, stamped, cleaned up."""
    rig = Rig(avatar_present=False, launch_timeout=0.2)
    expected_wait = min(_types.POST_CREATE_JOIN_WAIT_SECS, 0.2)

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.launch()

    err = excinfo.value
    assert err.type is ErrorType.TIMEOUT
    assert str(err) == f"avatar did not join within {expected_wait}s"
    assert err.session_id == "ses_fake"
    rig.assert_torn_down()
    assert rig.room.disconnect_calls == 0, "a borrowed room is never disconnected"


async def test_join_wait_is_capped_by_the_post_create_backstop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wait is min(POST_CREATE_JOIN_WAIT_SECS, launch_timeout); here the backstop is the smaller.

    The one test that shrinks the backstop, because its outcome is that constant elapsing first.
    """
    monkeypatch.setattr(_types, "POST_CREATE_JOIN_WAIT_SECS", 0.2)
    rig = Rig(avatar_present=False, launch_timeout=30.0)

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.launch()

    assert excinfo.value.type is ErrorType.TIMEOUT
    assert str(excinfo.value) == "avatar did not join within 0.2s"
    rig.assert_torn_down()


async def test_join_wait_rechecks_the_room_before_giving_up() -> None:
    """An avatar that is in the room when the wait expires is used even if its join event was lost.

    `wait_for` cancels the join future on timeout, so an event landing in that same loop iteration can
    no longer resolve it; the room is the source of truth. Modelled by adding the participant without
    emitting anything.
    """
    rig = Rig(avatar_present=False, launch_timeout=0.2)
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="launch() to be waiting for the join")
    silent = rig.room.add_participant(make_avatar())  # present, but no participant_connected event

    session = await task
    try:
        assert session.avatar_participant is silent
        assert session.timings.avatar_joined >= session.timings.session_created
    finally:
        await session.aclose()


async def test_api_error_cleans_up_and_is_stamped() -> None:
    """A failing `create_session` re-raises the same error; the owned room is disconnected and deleted."""
    original = SynthesiaError("unauthenticated", type=ErrorType.AUTH, status=401)
    rig = Rig(owns_room=True, api=FakeSessionApiClient(error=original))

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.launch()

    assert excinfo.value is original
    assert excinfo.value.session_id is None, "no session was created, so nothing to stamp"
    rig.assert_torn_down()
    assert rig.room.disconnect_calls == 1
    assert rig.room.connected is False
    assert rig.delete_calls == 1


async def test_cancellation_cleans_up() -> None:
    """The service's launch timeout cancels us mid-join: cleanup runs, then CancelledError propagates."""
    rig = Rig(avatar_present=False)
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="launch() to be waiting for the join")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    rig.assert_torn_down()
    assert rig.room.disconnect_calls == 0


async def test_second_cancel_still_deletes_room() -> None:
    """A second cancel while the owned-room disconnect is in flight does not abort the cleanup
    (*cancel-proof cleanup*)."""
    rig = Rig(owns_room=True, avatar_present=False)
    rig.room.disconnect_event = asyncio.Event()  # holds `disconnect()` until the test releases it
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="launch() to be waiting for the join")

    task.cancel()
    await rig.room.disconnect_started.wait()  # the cleanup is inside room.disconnect() right now
    task.cancel()  # cancel_task's wait_for timeout, from the inside
    assert rig.delete_calls == 0, "the delete must not have run while the disconnect is in flight"
    rig.room.disconnect_event.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert rig.room.disconnect_calls == 1
    assert rig.delete_calls == 1
    rig.assert_torn_down()


async def test_cancellation_while_the_request_is_in_flight_cancels_it() -> None:
    """The session request runs in a task of its own, so a launch cancelled while the request is in flight
    must not leave it running: it is cancelled and waited for, and the owned room is still released."""
    before = {task for task in asyncio.all_tasks() if not task.done()}
    gate = asyncio.Event()
    rig = Rig(owns_room=True, avatar_present=False, api=FakeSessionApiClient(gate=gate))
    task = rig.launch_task()
    try:
        await until(lambda: len(rig.api.calls) == 1, what="the session request")
        task.cancel()
        await until(task.done, what="the rollback to finish")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.wait({task})

    assert task.cancelled(), "the cancellation must propagate once the rollback has finished"
    assert {t for t in asyncio.all_tasks() if not t.done()} - before == set(), "a task outlived the launch"
    gate.set()
    await asyncio.sleep(0)
    assert rig.api.returned == 0, "the session request was left to finish"
    assert rig.room.disconnect_calls == 1
    assert rig.delete_calls == 1
    rig.assert_torn_down()


class UnregisterRecordingParticipant(FakeLocalParticipant):
    """Records the task each playback handler is unregistered from: the one running the close."""

    def __init__(self, identity: str) -> None:
        super().__init__(identity)
        self.unregistering_tasks: list[asyncio.Task[object] | None] = []

    def unregister_rpc_method(self, method: str, /) -> None:
        self.unregistering_tasks.append(asyncio.current_task())
        super().unregister_rpc_method(method)


async def test_a_cancel_during_a_failed_launch_rollback_still_releases_the_owned_room() -> None:
    """The caller, cancelled while the rollback disconnects the room, sees its own cancellation only after
    the disconnect and the delete, and not the launch error it no longer waits for."""
    failure = SynthesiaError("unauthenticated", type=ErrorType.AUTH, status=401)
    rig = Rig(owns_room=True, api=FakeSessionApiClient(error=failure))
    rig.room.disconnect_event = asyncio.Event()  # holds the disconnect until released below
    task = rig.launch_task()
    await until(rig.room.disconnect_started.is_set, what="the rollback to be inside room.disconnect()")

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "the rollback absorbs the cancel"
    assert rig.delete_calls == 0, "the delete must not run while the disconnect is in flight"
    rig.room.disconnect_event.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert rig.room.disconnect_calls == 1
    assert rig.delete_calls == 1
    rig.assert_torn_down()


async def test_session_requested_is_the_first_attempts_time(fake_api_factory) -> None:
    """`session_requested` is stamped once, just before the first attempt. The API client retries inside
    `create_session`, so a retry never moves it. The 503 is retried after a tiny backoff (an input)."""
    attempts: list[float] = []

    async def stamp(_request: RecordedRequest) -> None:
        attempts.append(monotonic())

    script = [
        ScriptedResponse(503, {"code": "service_unavailable", "detail": "busy"}, before=stamp),
        ScriptedResponse(201, {"id": "ses_retried"}, before=stamp),
    ]
    async with fake_api_factory(script) as server, aiohttp.ClientSession() as http_session:
        client = SessionApiClient(
            api_key=API_KEY,
            api_url=server.base_url,
            client=build_client_info(),
            http_session=http_session,
            backoff_seconds=(0.01,),
        )
        async with launched(api=cast(Any, client)) as (_rig, session):
            timings = session.timings
            assert len(attempts) == 2, attempts
            assert timings.session_requested <= attempts[0] < attempts[1] <= timings.session_created


async def test_a_cancelled_caller_of_aclose_still_releases_the_owned_room() -> None:
    rig = Rig(owns_room=True)
    session = await rig.launch()
    rig.room.disconnect_event = asyncio.Event()
    closing = asyncio.ensure_future(session.aclose())
    await until(rig.room.disconnect_started.is_set, what="the release to be inside room.disconnect()")

    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done(), "the caller waits for the release to finish"
    assert rig.delete_calls == 0
    rig.room.disconnect_event.set()

    with pytest.raises(asyncio.CancelledError):
        await closing

    assert rig.room.disconnect_calls == 1
    assert rig.delete_calls == 1
    assert session.is_alive is False
    rig.assert_torn_down()


async def test_cancelling_the_close_task_itself_still_finishes_the_release() -> None:
    """`aclose()` runs the close in a task of its own; cancelling that task, not the caller, still lets the
    release finish before the cancel is honoured. The product gives no handle on the close task, so it is
    found through the fake: it is the task `unregister_rpc_method` is called from
    (`UnregisterRecordingParticipant`)."""
    participant = UnregisterRecordingParticipant(BOT)
    rig = Rig(owns_room=True, participant=participant)
    session = await rig.launch()
    rig.room.disconnect_event = asyncio.Event()
    closing = asyncio.ensure_future(session.aclose())
    await until(rig.room.disconnect_started.is_set, what="the release to be inside room.disconnect()")
    (releasing,) = set(participant.unregistering_tasks)
    assert releasing is not None

    releasing.cancel()
    await asyncio.sleep(0)
    assert not releasing.done(), "the release absorbs the cancel"
    assert rig.delete_calls == 0
    rig.room.disconnect_event.set()

    with pytest.raises(asyncio.CancelledError):
        await closing

    assert releasing.cancelled(), "the cancel is honoured once the release has finished"
    assert rig.room.disconnect_calls == 1
    assert rig.delete_calls == 1
    rig.assert_torn_down()


# --------------------------------------------------------------------------------------- audio path


class OrderingParticipant(FakeLocalParticipant):
    """Records, at each `perform_rpc`, whether the latest stream was still open: an interruption must
    clear the buffer before it closes the stream."""

    def __init__(self, identity: str) -> None:
        super().__init__(identity)
        self.rpc_events: list[tuple[str, bool | None]] = []

    async def perform_rpc(self, *, method: str, **kwargs: Any) -> str:
        latest_open = (not self.streams[-1].closed) if self.streams else None
        self.rpc_events.append((method, latest_open))
        return await super().perform_rpc(method=method, **kwargs)


async def test_send_end_interrupt_sequence() -> None:
    """One byte stream per utterance, with exact attributes; an interruption sends clear_buffer, then
    closes the stream."""
    participant = OrderingParticipant(BOT)
    participant.script_rpc(RPC_CLEAR_BUFFER, RPC_HANDLER_RESPONSE)
    async with launched(participant=participant) as (rig, session):
        await session.send_audio(CHUNK, 24000, 1)
        await session.send_audio(CHUNK, 24000, 1)
        assert len(participant.streams) == 1
        first = participant.streams[0]
        assert first.attributes == {STREAM_ATTR_SAMPLE_RATE: "24000", STREAM_ATTR_NUM_CHANNELS: "1"}
        assert first.destination_identities == [AVATAR]
        assert first.data == CHUNK * 2

        await session.end_utterance()
        assert first.closed is True

        await session.send_audio(CHUNK, 24000, 1)
        assert len(participant.streams) == 2
        second = participant.streams[1]
        assert second.closed is False

        await session.interrupt()
        assert participant.rpc_events == [(RPC_CLEAR_BUFFER, True)], "clear_buffer before the close"
        assert participant.perform_rpc_calls[-1]["destination_identity"] == AVATAR
        assert second.closed is True
        assert participant.max_concurrent_open_streams == 1
        assert participant.unclosed_streams() == []
        _ = rig


async def test_playback_rpcs_reach_the_callbacks_and_reject_foreign_callers() -> None:
    """The worker's playback RPCs are forwarded; a foreign identity is rejected and not forwarded."""
    async with launched() as (rig, session):
        participant = rig.participant
        assert await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR) == RPC_HANDLER_RESPONSE
        payload = json.dumps({"playback_position": 1.5, "interrupted": True})
        assert (
            await participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=payload)
            == RPC_HANDLER_RESPONSE
        )
        await until(lambda: rig.rec.started == 1 and rig.rec.finished == [(1.5, True)], what="both callbacks")

        assert (
            await participant.invoke(RPC_PLAYBACK_STARTED, caller_identity="intruder") == RPC_HANDLER_REJECT
        )
        await asyncio.sleep(0)
        assert rig.rec.started == 1
        assert session.is_alive


# ----------------------------------------------------------------------------------------- readiness


async def test_readiness_flow(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """Attributes at join -> `on_avatar_status`; a later change -> status + `all_ready`."""
    async with launched(attributes=PREPARING) as (rig, session):
        assert rig.avatar is not None
        await until(lambda: len(rig.rec.status) == 1, what="the status seen at join")
        assert rig.rec.status[0] == ({"av_a": "preparing", "av_b": "preparing"}, False)
        assert session.avatar_status == {"av_a": "preparing", "av_b": "preparing"}
        assert session.all_ready is False

        # another participant's attributes are not the avatar's, even with the same keys; the session
        # folds attribute changes in synchronously, so the line after the event sees the result
        human = rig.room.add_participant(FakeRemoteParticipant("human", kind=STANDARD))
        rig.change_attributes(human, ALL_READY)
        assert session.all_ready is False
        assert session.avatar_status == {"av_a": "preparing", "av_b": "preparing"}

        rig.change_attributes(rig.avatar, ALL_READY)
        await until(lambda: len(rig.rec.status) == 2, what="the readiness update")
        assert rig.rec.status[1] == ({"av_a": "ready", "av_b": "ready"}, True)
        assert session.all_ready is True
        assert any(
            record.levelno == logging.INFO and "ready" in record.getMessage()
            for record in caplog_loguru.records
        )


# ---------------------------------------------------------------------------------------------- swap


async def test_swap() -> None:
    """default -> first id, unprefixed on the wire; the worker's error body and an RpcError are typed and
    stamped. Waiting, timeouts and the other failures have tests of their own below."""
    async with launched(attributes=A_READY_B_PREPARING) as (rig, session):
        assert rig.avatar is not None
        participant = rig.participant
        # `script_rpc` is a queue (the last entry repeats): every swap response, in consumption order.
        participant.script_rpc(
            RPC_SWAP_AVATAR,
            json.dumps({"status": "ok", "avatar_id": "a"}),
            json.dumps({"error": "still preparing", "code": "AVATAR_PREPARING"}),
            lk_rtc.RpcError(CODES.RECIPIENT_NOT_FOUND, "gone"),
        )

        assert await session.swap_avatar(SWAP_DEFAULT, timeout=5) == "av_a"
        assert json.loads(participant.perform_rpc_calls[-1]["payload"]) == {"avatar_id": "a"}
        assert session.active_avatar_id == "av_a"

        # the worker's error body: typed AVATAR_SWAP_FAILED with its code, stamped
        with pytest.raises(SynthesiaError) as excinfo:
            await session.swap_avatar("av_a", timeout=5)
        assert excinfo.value.type is ErrorType.AVATAR_SWAP_FAILED
        assert excinfo.value.code == "AVATAR_PREPARING"
        assert excinfo.value.session_id == "ses_fake"

        # an RpcError: mapped by error_from_rpc, stamped
        with pytest.raises(SynthesiaError) as excinfo:
            await session.swap_avatar("av_a", timeout=5)
        assert excinfo.value.type is ErrorType.CONNECTION
        assert excinfo.value.session_id == "ses_fake"
        assert session.active_avatar_id == "av_a", "a failed swap leaves the active id alone"


async def test_swap_to_a_failed_avatar_fails_without_an_rpc() -> None:
    """A swap to an avatar in the `failed` state fails locally with the worker's own code, stamped, and
    sends no RPC."""
    failed = {ATTR_AVATAR_STATUS: status_attr(a="ready", b="failed")}
    async with launched(attributes=failed) as (rig, session):
        with pytest.raises(SynthesiaError) as excinfo:
            await session.swap_avatar("av_b", timeout=5)
        assert excinfo.value.type is ErrorType.AVATAR_SWAP_FAILED
        assert excinfo.value.code == SWAP_ERROR_CODE_AVATAR_FAILED
        assert excinfo.value.session_id == "ses_fake"
        assert rig.participant.perform_rpc_calls == []


async def test_a_swap_to_a_preparing_avatar_waits_then_swaps_once_it_is_ready() -> None:
    async with launched(attributes=A_READY_B_PREPARING) as (rig, session):
        assert rig.avatar is not None
        rig.participant.script_rpc(RPC_SWAP_AVATAR, SWAPPED_TO_B)
        swap = asyncio.ensure_future(session.swap_avatar("av_b", timeout=LONG_TIMEOUT))
        try:
            await asyncio.sleep(0)  # one loop turn: the swap reaches its readiness wait
            assert rig.participant.perform_rpc_calls == [], "no RPC while b is preparing"

            rig.change_attributes(rig.avatar, ALL_READY)
            await until(swap.done, what="the swap to go ahead once b is ready")
        finally:
            await settle(swap)

        assert swap.result() == "av_b"
        assert json.loads(rig.participant.perform_rpc_calls[-1]["payload"]) == {"avatar_id": "b"}
        assert session.active_avatar_id == "av_b"


async def test_a_swap_whose_avatar_fails_while_it_waits_is_avatar_swap_failed() -> None:
    async with launched(attributes=A_READY_B_PREPARING) as (rig, session):
        assert rig.avatar is not None
        swap = asyncio.ensure_future(session.swap_avatar("av_b", timeout=LONG_TIMEOUT))
        try:
            await asyncio.sleep(0)  # one loop turn: the swap reaches its readiness wait
            rig.change_attributes(rig.avatar, {ATTR_AVATAR_STATUS: status_attr(a="ready", b="failed")})
            await until(swap.done, what="the swap to fail once b has failed")
        finally:
            await settle(swap)

        error = swap.exception()
        assert isinstance(error, SynthesiaError)
        assert error.type is ErrorType.AVATAR_SWAP_FAILED
        assert error.code == SWAP_ERROR_CODE_AVATAR_FAILED
        assert error.session_id == SESSION_ID
        assert rig.participant.perform_rpc_calls == []
        assert session.active_avatar_id == "av_a"


async def test_a_swap_to_an_avatar_still_preparing_when_time_runs_out_is_timeout() -> None:
    async with launched(attributes=A_READY_B_PREPARING) as (rig, session):
        swap = asyncio.ensure_future(session.swap_avatar("av_b", timeout=0.2))
        try:
            await until(swap.done, what="the swap's timeout to run out")
        finally:
            await settle(swap)

        error = swap.exception()
        assert isinstance(error, SynthesiaError)
        assert error.type is ErrorType.TIMEOUT
        assert error.session_id == SESSION_ID
        assert rig.participant.perform_rpc_calls == []


async def test_a_short_swap_timeout_still_waits_for_readiness_that_arrives_in_time() -> None:
    async with launched(attributes=A_READY_B_PREPARING) as (rig, session):
        assert rig.avatar is not None
        rig.participant.script_rpc(RPC_SWAP_AVATAR, SWAPPED_TO_B)
        swap = asyncio.ensure_future(session.swap_avatar("av_b", timeout=1.0))
        try:
            await asyncio.sleep(0)  # one loop turn: the swap reaches its readiness wait
            assert not swap.done(), "the swap is waiting for b, not giving up on it"
            rig.change_attributes(rig.avatar, ALL_READY)
            assert await swap == "av_b"
        finally:
            await settle(swap)


async def test_a_swap_to_an_avatar_with_no_reported_readiness_goes_straight_to_the_rpc() -> None:
    async with launched() as (rig, session):
        assert session.avatar_status == {}
        rig.participant.script_rpc(RPC_SWAP_AVATAR, SWAPPED_TO_B)

        assert await session.swap_avatar("av_b", timeout=1.0) == "av_b"
        assert len(rig.participant.perform_rpc_calls) == 1


async def test_the_swap_rpc_is_bounded_by_the_swap_timeout() -> None:
    async with launched(attributes=ALL_READY) as (rig, session):
        rig.participant.script_rpc(RPC_SWAP_AVATAR, SWAPPED_TO_B)

        await session.swap_avatar("av_b", timeout=5.0)

        call = rig.participant.perform_rpc_calls[-1]
        assert 0 < call["response_timeout"] <= 5.0
        assert 0 < call["max_round_trip_latency"] <= 5.0


@pytest.mark.parametrize(
    ("response", "expected", "timeout"),
    [
        pytest.param(never_returns, ErrorType.TIMEOUT, 0.1, id="never-answers"),
        pytest.param(
            lk_rtc.RpcError(CODES.RESPONSE_TIMEOUT, "timeout"), ErrorType.TIMEOUT, 5.0, id="rpc-timeout"
        ),
        pytest.param(
            lk_rtc.RpcError(CODES.RECIPIENT_DISCONNECTED, "gone"), ErrorType.CONNECTION, 5.0, id="worker-gone"
        ),
        pytest.param(
            RuntimeError("the room is closed"), ErrorType.CONNECTION, 5.0, id="rpc-failed-otherwise"
        ),
        pytest.param("not json", ErrorType.AVATAR_SWAP_FAILED, 5.0, id="unparseable-answer"),
        pytest.param(
            json.dumps({"error": "avatar b not prepared"}), ErrorType.AVATAR_SWAP_FAILED, 5.0, id="error"
        ),
        pytest.param(
            json.dumps({"status": "ok", "avatar_id": ""}),
            ErrorType.AVATAR_SWAP_FAILED,
            5.0,
            id="empty-avatar-id",
        ),
    ],
)
async def test_every_swap_rpc_failure_is_typed_and_carries_the_session_id(
    response: object, expected: ErrorType, timeout: float
) -> None:
    """Only the never-answers case is about a timeout running out, so only it gets a short one."""
    async with launched(attributes=ALL_READY) as (rig, session):
        rig.participant.script_rpc(RPC_SWAP_AVATAR, response)

        with pytest.raises(SynthesiaError) as excinfo:
            await session.swap_avatar("av_b", timeout=timeout)

        assert excinfo.value.type is expected
        assert excinfo.value.session_id == SESSION_ID
        assert session.active_avatar_id == "av_a", "a failed swap leaves the active avatar alone"


async def test_a_confirmed_avatar_id_outside_the_pattern_is_still_a_success() -> None:
    """The worker has already swapped when it answers (*tolerant reader*)."""
    async with launched(attributes=ALL_READY) as (rig, session):
        rig.participant.script_rpc(
            RPC_SWAP_AVATAR, json.dumps({"status": "ok", "avatar_id": "not a valid id!"})
        )

        assert await session.swap_avatar("av_b", timeout=5.0) == "av_not a valid id!"
        assert session.active_avatar_id == "av_not a valid id!"


@pytest.mark.parametrize("avatar_id", ["av_zzz", "not a valid id!", ""])
async def test_a_swap_to_an_id_outside_the_session_avatars_is_unknown_avatar(avatar_id: str) -> None:
    async with launched(attributes=ALL_READY) as (rig, session):
        with pytest.raises(SynthesiaError) as excinfo:
            await session.swap_avatar(avatar_id, timeout=5.0)

        assert excinfo.value.type is ErrorType.UNKNOWN_AVATAR
        assert excinfo.value.session_id == SESSION_ID
        assert rig.participant.perform_rpc_calls == []


# -------------------------------------------------------------------------------------- avatar left


async def test_participant_disconnected_reports_left_once() -> None:
    """A plain disconnect is an immediate leave, reported once even when more leave events follow."""
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        assert session.is_alive is False, "the flag flips synchronously, before any await"
        assert session.avatar_participant is None

        await until(lambda: len(rig.rec.left) == 1, what="on_avatar_left")
        assert rig.rec.left == [("avatar_left_room", False)]

        # the same leave also produces track_unpublished and disconnected; still exactly once
        rig.room.emit("disconnected", CLIENT_INITIATED)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert rig.rec.left == [("avatar_left_room", False)]


async def test_room_disconnected_reports_left() -> None:
    """The room going away is a leave with reason `room_connection_lost`."""
    async with launched() as (rig, session):
        rig.room.emit("disconnected", CLIENT_INITIATED)
        await until(lambda: len(rig.rec.left) == 1, what="on_avatar_left")
        assert rig.rec.left == [("room_connection_lost", False)]
        assert session.is_alive is False


async def test_video_unpublished_reports_left_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """`track_unpublished(video)` arms the *replacement grace*; the leave is reported once it elapses."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.unpublish(rig.avatar, VIDEO_SID)
        await asyncio.sleep(0)
        assert rig.rec.left == [], "the grace exists: no leave on the next yield"
        assert session.is_alive is True

        await until(lambda: len(rig.rec.left) == 1, timeout=2, what="the grace to elapse")
        assert rig.rec.left == [("avatar_video_stopped", False)]
        assert session.is_alive is False


async def test_video_that_returns_within_the_grace_leaves_later_losses_reportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grace re-checks the room and finds the video back; a later loss starts a fresh grace."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        avatar = rig.avatar
        assert avatar is not None

        rig.unpublish(avatar, VIDEO_SID)
        rig.publish_video(avatar)  # the same participant, never gone: no re-join to cancel the grace
        await asyncio.sleep(GRACE_MARGIN)
        assert rig.rec.left == []
        assert session.is_alive is True

        rig.unpublish(avatar, VIDEO_SID)
        await asyncio.sleep(0)
        assert rig.rec.left == [], "the second loss starts a fresh grace: no leave on the next yield"
        assert session.is_alive is True
        await until(lambda: len(rig.rec.left) == 1, what="the second video loss to be reported")
        assert rig.rec.left == [("avatar_video_stopped", False)]
        assert session.is_alive is False


async def test_audio_unpublished_alone_is_not_a_leave(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the video publication going away arms the grace, as in the LiveKit plugin."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.unpublish(rig.avatar, AUDIO_SID)
        await asyncio.sleep(GRACE_MARGIN)
        assert rig.rec.left == []
        assert session.is_alive is True


async def test_same_identity_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """DUPLICATE_IDENTITY then a same-tick re-join is a *same-identity replacement*, not a leave."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched(attributes=PREPARING) as (rig, session):
        old = rig.avatar
        assert old is not None
        await until(lambda: len(rig.rec.status) == 1, what="the status seen at join")

        # the new worker's join kicks the old participant; its video goes first, then it disconnects,
        # then the replacement joins, all before the loop turns (the ordering margin is unbounded)
        rig.unpublish(old, VIDEO_SID)
        rig.leave(old, reason=DUPLICATE_IDENTITY)
        new = rig.join(make_avatar(attributes=ALL_READY))

        await asyncio.sleep(GRACE_MARGIN)
        assert rig.rec.left == []
        assert session.is_alive is True
        assert session.avatar_participant is new
        # readiness was re-read from the replacement's attributes
        await until(lambda: any(all_ready for _, all_ready in rig.rec.status), what="readiness re-read")
        assert session.all_ready is True


async def test_unstated_disconnect_reason_is_treated_as_a_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset disconnect reason arms the grace too: it is what real rtc reports for a replacement.

    `DUPLICATE_IDENTITY` is what the kicked client sees about itself; livekit-server tells the other
    participants nothing about the same replacement (`UNKNOWN_REASON`, surfaced as `None`; measured
    against livekit-server 1.9.1). A grace armed only by `DUPLICATE_IDENTITY` would never arm for us,
    and every retried launch would be reported as an unexpected leave.

    rtc announces a participant before its tracks, so the replacement joins with nothing published: a
    re-join that only shortened the wait, rather than cancelling the grace, would find no video and
    report a leave. The replacement's tracks and readiness are the session's afterwards.
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched(attributes=PREPARING) as (rig, session):
        old = rig.avatar
        assert old is not None
        await until(lambda: len(rig.rec.status) == 1, what="the readiness seen at the join")

        rig.depart(old, reason=None)
        await asyncio.sleep(0)
        assert rig.rec.left == [], "an unstated reason must not be an immediate leave"
        new = rig.join(make_avatar(video=False, audio=False))

        await asyncio.sleep(GRACE_MARGIN)
        assert rig.rec.left == [], "the re-join must cancel the grace"
        assert session.is_alive is True
        assert session.avatar_participant is new

        rig.publish_video(new)
        await until(lambda: "video" in rig.rec.tracks, what="the replacement's video")
        rig.change_attributes(new, ALL_READY)
        await until(
            lambda: any(all_ready for _, all_ready in rig.rec.status), what="the replacement's readiness"
        )
        assert session.all_ready is True


async def test_unstated_disconnect_reason_without_rejoin_is_a_leave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grace resolves the ambiguity: nothing came back, so it was a leave after all.

    This is what bounds the cost of treating an unstated reason as ambiguous: a real but unexplained
    leave is still reported, `REPLACEMENT_GRACE_SECS` later.

    Driven through `depart()`, so the `track_unpublished` that real rtc always sends first arms the
    timer before the disconnect reaches it. The reported reason is still `avatar_left_room`:
    the disconnect sharpens the pending timer's reason without extending its deadline
    (`_start_grace`). Report "avatar_video_stopped" here and the developer is told the avatar stopped
    publishing video when it has in fact left the room.
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.depart(rig.avatar, reason=None)

        await until(lambda: len(rig.rec.left) == 1, timeout=2, what="the grace to elapse")
        assert rig.rec.left == [("avatar_left_room", False)]
        assert session.is_alive is False


async def test_video_unpublished_alone_keeps_its_own_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video unpublish with no disconnect behind it still reports `avatar_video_stopped`.

    The other side of `_start_grace`'s ranking: it sharpens a reason when a more specific signal
    arrives, and invents nothing when none does.
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.unpublish(rig.avatar, VIDEO_SID)

        await until(lambda: len(rig.rec.left) == 1, timeout=2, what="the grace to elapse")
        assert rig.rec.left == [("avatar_video_stopped", False)]
        assert session.is_alive is False


async def test_left_reason_is_forgotten_when_a_replacement_cancels_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a replacement cycle, a later video unpublish reports its own reason, not a stale one.

    The `participant_disconnected` that sharpened the pending timer belongs to the leave that armed
    it; once a re-join cancels that timer nothing of it may survive into the next grace, or the
    replacement worker dropping only its video would be reported as "the avatar left the room".
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        old = rig.avatar
        assert old is not None
        rig.depart(old, reason=None)
        new = rig.join(make_avatar())
        await asyncio.sleep(GRACE_MARGIN)
        assert rig.rec.left == []

        rig.unpublish(new, VIDEO_SID)  # the new worker drops its video but stays in the room
        await until(lambda: len(rig.rec.left) == 1, timeout=2, what="the grace to elapse")
        assert rig.rec.left == [("avatar_video_stopped", False)]
        assert session.is_alive is False


async def test_duplicate_identity_without_rejoin_is_a_leave(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a re-join inside the grace, DUPLICATE_IDENTITY is a leave after the grace."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.leave(rig.avatar, reason=DUPLICATE_IDENTITY)
        await asyncio.sleep(0)
        assert rig.rec.left == [], "DUPLICATE_IDENTITY must not be an immediate leave"
        assert session.is_alive is True

        await until(lambda: len(rig.rec.left) == 1, timeout=2, what="the grace to elapse")
        assert rig.rec.left == [("avatar_left_room", False)]
        assert session.is_alive is False


@pytest.mark.parametrize(
    ("owns_room", "human_present", "expected"),
    [
        pytest.param(False, True, True, id="borrowed-room-human-present"),
        pytest.param(False, False, False, id="borrowed-room-nobody-left"),
        pytest.param(True, False, None, id="owned-room-is-none"),
    ],
)
async def test_leave_reports_humans_present(
    owns_room: bool, human_present: bool, expected: bool | None
) -> None:
    """`humans_present`: whether any standard-kind participant remains in a borrowed room; None for an
    owned room."""
    async with launched(owns_room=owns_room) as (rig, session):
        assert rig.avatar is not None
        if human_present:
            rig.room.add_participant(FakeRemoteParticipant("human", kind=STANDARD))
        # another agent (a second bot, an egress) is not a human
        rig.room.add_participant(FakeRemoteParticipant("other-agent", kind=AGENT))

        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        await until(lambda: len(rig.rec.left) == 1, what="on_avatar_left")
        assert rig.rec.left == [("avatar_left_room", expected)]
        _ = session


async def test_closing_with_a_grace_pending_reports_no_leave_afterwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    rig = Rig()
    session = await rig.launch()
    assert rig.avatar is not None
    rig.unpublish(rig.avatar, VIDEO_SID)  # starts the grace

    await session.aclose()
    await asyncio.sleep(GRACE_MARGIN)

    assert rig.rec.left == [], "nothing is delivered after aclose()"
    rig.assert_torn_down()


async def test_join_events_are_handled_without_an_error_in_the_room_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first join (during the join wait) and a re-join. rtc's emitter logs and swallows a listener's
    exception (the fake records it in `callback_errors`), so a half-run listener goes unnoticed elsewhere."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    rig = Rig(avatar_present=False)
    task = rig.launch_task()
    await until(lambda: rig.api.returned == 1, what="launch() to be waiting for the join")
    first = rig.join()
    session = await task
    try:
        rig.depart(first, reason=None)
        rig.join(make_avatar())
        await asyncio.sleep(GRACE_MARGIN)

        assert rig.room.callback_errors == []
        assert rig.rec.left == []
        assert rig.room.remote_participants[AVATAR] is session.avatar_participant
    finally:
        await session.aclose()


async def test_nothing_is_delivered_after_left() -> None:
    """`on_avatar_left` is the last callback; later status/playback events are dropped."""
    async with launched() as (rig, session):
        assert rig.avatar is not None
        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        await until(lambda: len(rig.rec.left) == 1, what="on_avatar_left")

        rig.change_attributes(rig.avatar, ALL_READY)
        assert (
            await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR) == RPC_HANDLER_RESPONSE
        )
        rig.room.emit("track_published", FakeTrackPublication("TR_late", kind=KIND_VIDEO), rig.avatar)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert rig.rec.status == []
        assert rig.rec.started == 0
        assert rig.rec.tracks == []
        assert session.is_alive is False


# -------------------------------------------------------------------------------------------- aclose


async def test_aclose_idempotent_and_borrowed_room_untouched() -> None:
    """A borrowed room is never disconnected (*borrowed-room rule*); a second or concurrent aclose is a
    no-op (*single flight*)."""
    rig = Rig()
    session = await rig.launch()
    await session.send_audio(CHUNK, 24000, 1)
    hook_runs: list[int] = []

    async def hook() -> None:
        hook_runs.append(1)

    session.add_close_hook(hook)

    await asyncio.gather(session.aclose(), session.aclose())
    await session.aclose()

    assert rig.room.disconnect_calls == 0
    assert rig.room.connected is True
    assert rig.delete_calls == 0
    assert session.is_alive is False
    assert rig.participant.unclosed_streams() == [], "the open utterance is closed on aclose"
    assert rig.participant.streams[0].close_count == 1, "single flight: the stream is closed once"
    assert hook_runs == [1], "single flight: the hooks ran once for three aclose() calls"
    rig.assert_torn_down()


async def test_aclose_owned_room_disconnects_and_deletes_once() -> None:
    """An owned room is disconnected then deleted exactly once, listeners off first."""
    rig = Rig(owns_room=True)
    session = await rig.launch()
    hook_runs: list[int] = []

    async def hook() -> None:
        hook_runs.append(1)

    session.add_close_hook(hook)

    await asyncio.gather(session.aclose(), session.aclose())

    assert rig.room.disconnect_calls == 1
    assert rig.room.connected is False
    assert rig.delete_calls == 1
    assert hook_runs == [1], "single flight: the hook ran once for two concurrent aclose() calls"
    # `FakeRoom.disconnect()` emits `disconnected` as rtc does; our listeners were already off, so no
    # spurious leave was reported
    assert rig.rec.left == []
    assert session.is_alive is False
    rig.assert_torn_down()


async def test_aclose_cancelled_caller_does_not_abort_or_finish_the_close() -> None:
    """Cancelling the task that called aclose() neither aborts the close nor marks it done.

    A later caller joins the still-running close; the owned room is disconnected and deleted exactly
    once; the cancelled caller sees its CancelledError only once the close has finished.
    """
    rig = Rig(owns_room=True)
    session = await rig.launch()
    entered = asyncio.Event()
    release = asyncio.Event()
    hook_runs: list[int] = []

    async def slow_hook() -> None:
        entered.set()
        await release.wait()
        hook_runs.append(1)

    session.add_close_hook(slow_hook)

    first = asyncio.ensure_future(session.aclose())
    await entered.wait()  # the close is inside the hook right now
    first.cancel()
    second = asyncio.ensure_future(session.aclose())
    await asyncio.sleep(0)
    assert not first.done() and not second.done(), "both callers wait for the one close to finish"
    assert rig.room.disconnect_calls == 0, "the close was neither aborted nor skipped ahead"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    assert hook_runs == [1]
    assert rig.room.disconnect_calls == 1
    assert rig.delete_calls == 1
    assert session.is_alive is False
    rig.assert_torn_down()


async def test_send_after_aclose_is_noop(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """After aclose(), send_audio drops silently with one WARNING and opens nothing."""
    rig = Rig()
    session = await rig.launch()
    await session.send_audio(CHUNK, 24000, 1)
    await session.aclose()
    streams_before = len(rig.participant.streams)

    await session.send_audio(CHUNK, 24000, 1)
    await session.send_audio(CHUNK, 24000, 1)

    assert len(rig.participant.streams) == streams_before, "no stream after aclose"
    assert rig.participant.unclosed_streams() == []
    assert len(warnings_with(caplog_loguru, "dropping audio")) == 1


async def test_aclose_hooks_bounded(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """Close hooks run inside aclose, each bounded by CLOSE_HOOK_SECS (shrunk: the test is about it)."""
    monkeypatch.setattr(_types, "CLOSE_HOOK_SECS", 0.1)
    ran: list[str] = []

    async def quick() -> None:
        ran.append("quick")

    async def never() -> None:
        await asyncio.Event().wait()

    async def boom() -> None:
        raise RuntimeError("hook blew up")

    rig = Rig()
    session = await rig.launch()
    session.add_close_hook(never)
    session.add_close_hook(boom)
    session.add_close_hook(quick)

    await session.aclose()

    assert ran == ["quick"], "a hung hook and a failing hook do not stop the next one"
    assert len(warnings_with(caplog_loguru, "close hook")) == 2
    assert session.is_alive is False
    rig.assert_torn_down()


async def test_send_after_left_is_noop(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """After the avatar left, `send_audio` drops silently with one WARNING; teardown calls are safe."""
    async with launched() as (rig, session):
        assert rig.avatar is not None
        await session.send_audio(CHUNK, 24000, 1)
        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        await until(lambda: len(rig.rec.left) == 1, what="on_avatar_left")
        assert rig.participant.unclosed_streams() == [], "the open utterance is aborted on leave"

        await session.send_audio(CHUNK, 24000, 1)
        await session.send_audio(CHUNK, 24000, 1)
        assert len(rig.participant.streams) == 1, "no new stream after the avatar left"
        assert len(warnings_with(caplog_loguru, "dropping audio")) == 1

        await session.end_utterance()
        await session.interrupt()
        assert rig.participant.perform_rpc_calls == [], "no clear_buffer to a participant that is gone"


# ------------------------------------------------------------------------------------ cancellation


class ParkingRecorder(CallbackRecorder):
    """Holds `on_avatar_status` and `on_playback_started` open, recording the task each runs in."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.delivered_in: dict[str, asyncio.Task[object] | None] = {}

    async def on_avatar_status(self, status: dict[str, str], all_ready: bool) -> None:
        self.delivered_in["status"] = asyncio.current_task()
        await self.release.wait()
        await super().on_avatar_status(status, all_ready)

    async def on_playback_started(self) -> None:
        self.delivered_in["playback_started"] = asyncio.current_task()
        await self.release.wait()
        await super().on_playback_started()


async def test_cancelling_an_interruption_during_the_clear_buffer_rpc_propagates() -> None:
    """Pipecat cancels the service's tasks at interruption and teardown; none may swallow the cancel."""
    async with launched() as (rig, session):
        rig.participant.script_rpc(RPC_CLEAR_BUFFER, never_returns)
        interrupting = asyncio.ensure_future(session.interrupt())
        try:
            await until(lambda: len(rig.participant.perform_rpc_calls) == 1, what="the clear buffer RPC")
            interrupting.cancel()
            await asyncio.wait({interrupting})
        finally:
            await settle(interrupting)

        assert interrupting.cancelled(), "the interruption ends with its own cancellation"


async def test_cancelling_a_swap_during_its_rpc_propagates() -> None:
    async with launched(attributes=ALL_READY) as (rig, session):
        rig.participant.script_rpc(RPC_SWAP_AVATAR, never_returns)
        swapping = asyncio.ensure_future(session.swap_avatar("av_b", timeout=5.0))
        try:
            await until(lambda: len(rig.participant.perform_rpc_calls) == 1, what="the swap RPC")
            swapping.cancel()
            await asyncio.wait({swapping})
        finally:
            await settle(swapping)

        assert swapping.cancelled(), "the swap ends with its own cancellation"
        assert session.active_avatar_id == "av_a"


async def test_a_cancelled_status_delivery_ends_cancelled() -> None:
    """Each room event is delivered in a task of its own, here the readiness seen at the join."""
    recorder = ParkingRecorder()
    async with launched(attributes=PREPARING, recorder=recorder):
        try:
            await until(lambda: "status" in recorder.delivered_in, what="the status delivery")
            delivery = recorder.delivered_in["status"]
            assert delivery is not None
            delivery.cancel()
            await asyncio.wait({delivery})

            assert delivery.cancelled()
        finally:
            recorder.release.set()


async def test_a_cancelled_playback_event_delivery_ends_cancelled() -> None:
    recorder = ParkingRecorder()
    async with launched(recorder=recorder) as (rig, _session):
        try:
            await rig.participant.invoke(RPC_PLAYBACK_STARTED, caller_identity=AVATAR)
            await until(lambda: "playback_started" in recorder.delivered_in, what="the playback delivery")
            delivery = recorder.delivered_in["playback_started"]
            assert delivery is not None
            delivery.cancel()
            await asyncio.wait({delivery})

            assert delivery.cancelled()
        finally:
            recorder.release.set()


# ------------------------------------------------------------------------------------------- secrets


async def test_no_secrets_logged(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """A real minted token flows through launch and close and never reaches a log line (*redaction rule*)."""
    token_source = MintedAvatarToken(LiveKitCredentials(api_key="lk-key", api_secret=LK_API_SECRET))
    rig = Rig(token_source=token_source, avatar_name=LK_TOKEN_MARKER, attributes=PREPARING)
    session = await rig.launch()
    try:
        token = rig.api.calls[0]["livekit_token"]
        assert decode_unsigned(token).name == LK_TOKEN_MARKER, "the marker must be in the token payload"
        await session.send_audio(CHUNK, 24000, 1)
        await session.end_utterance()
    finally:
        await session.aclose()

    # the failure paths log too
    failing = Rig(
        token_source=token_source,
        avatar_name=LK_TOKEN_MARKER,
        api=FakeSessionApiClient(error=SynthesiaError("nope", type=ErrorType.AUTH)),
    )
    with pytest.raises(SynthesiaError) as excinfo:
        await failing.launch()

    assert_no_secrets(caplog_loguru.text, str(excinfo.value), repr(excinfo.value))


# --------------------------------------------------------------------------------- cleanup robustness


async def test_roll_back_launch_never_raises(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """Every cleanup step failing still re-raises the original error, stamped, with a WARNING per step."""
    rig = Rig(owns_room=True, avatar_present=False, launch_timeout=0.2)

    def bad_unregister(self: PlaybackRpcHandlers) -> None:
        raise RuntimeError("unregister exploded")

    def bad_off(event: Any, callback: Any) -> None:
        raise RuntimeError("off exploded")

    async def bad_disconnect() -> None:
        raise RuntimeError("disconnect exploded")

    async def bad_delete() -> None:
        raise RuntimeError("delete exploded")

    monkeypatch.setattr(PlaybackRpcHandlers, "unregister", bad_unregister)
    monkeypatch.setattr(rig.room, "off", bad_off)
    monkeypatch.setattr(rig.room, "disconnect", bad_disconnect)
    rig.handle.release_room = bad_delete

    with pytest.raises(SynthesiaError) as excinfo:
        await rig.launch()

    assert excinfo.value.type is ErrorType.TIMEOUT, "the original error, not a cleanup failure"
    assert excinfo.value.session_id == "ses_fake"
    for keyword in ("unregister exploded", "off exploded", "disconnect exploded", "delete exploded"):
        assert warnings_with(caplog_loguru, keyword), f"no WARNING for the failed step: {keyword}"


async def test_callbacks_are_not_serialised() -> None:
    """Callbacks are not serialised: a blocked `on_playback_finished` does not delay a later
    `on_avatar_status`."""
    async with launched(attributes=PREPARING) as (rig, session):
        assert rig.avatar is not None
        await until(lambda: len(rig.rec.status) == 1, what="the status seen at join")
        rig.rec.hold_finished = asyncio.Event()

        payload = json.dumps({"playback_position": 2.0, "interrupted": False})
        await rig.participant.invoke(RPC_PLAYBACK_FINISHED, caller_identity=AVATAR, payload=payload)
        rig.change_attributes(rig.avatar, ALL_READY)

        await until(lambda: len(rig.rec.status) == 2, what="the status update, despite the blocked callback")
        assert rig.rec.finished == [], "the finished callback is still blocked"

        rig.rec.hold_finished.set()
        await until(lambda: rig.rec.finished == [(2.0, False)], what="the released callback")
        assert session.all_ready is True
