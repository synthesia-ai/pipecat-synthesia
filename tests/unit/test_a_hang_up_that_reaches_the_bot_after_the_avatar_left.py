"""A direct join where the user hangs up: the session ends as `no_humans_left`, even when the avatar's leave
reaches the bot before the user's own.

When the last human hangs up, the avatar worker leaves at once, and its leave can reach the bot about a
second before the bot's own `participant_disconnected` for that human (measured in a live call,
2026-11-10 UTC). Read at that instant, the room still holds the human, so the leave
looked unexpected: an error pushed, the voice-only handover started in an emptying room, and the end reason
`avatar_left`. So when the avatar leaves while humans still appear present, the service holds the error and
the end reason for the *no-humans-left window*, and reports `no_humans_left` with no error if the last human
leaves inside it. The state and the hold on later TTS audio change at once, before any await
(*state-before-await*).

The first tests run the real `LiveKitAvatarSession` behind the service, on the session rig (`_session_rig.py`,
`SessionRigRoom`), because the window watches its room. `GatedOpenParticipant` holds the first chunk's byte
stream open, so the utterance is still queued when the avatar leaves, and a voice-only handover would show as
TTS audio at the user transport. The window's length is an input (`_types.NO_HUMANS_LEFT_WINDOW_SECS`): `LONG`
where the human's leave must end it, so that waiting it out shows as a timeout, and `SHORT` where it must run
out. The last tests drive the service alone with `FakeAvatarSession(humans_window=)`, a window the test ends.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from livekit import rtc as lk_rtc
from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom, FakeAvatarSession
from tests.fakes.fake_rtc import FakeRemoteParticipant
from tests.harness import Harness, bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, count, frames_of
from tests.support.waiting import until
from tests.unit._service_helpers import AUDIO_1, AUDIO_2, AUDIO_3, processed, records_at, tts_audio
from tests.unit._session_rig import AGENT, STANDARD, Rig, SessionRigRoom, launched, settle
from tests.unit._slow_byte_streams import GatedOpenParticipant

CLIENT_INITIATED = lk_rtc.DisconnectReason.CLIENT_INITIATED
_NAMES = {AUDIO_1: "AUDIO_1", AUDIO_2: "AUDIO_2", AUDIO_3: "AUDIO_3"}
#: Far longer than `until()`'s 5 s: a window the human's leave must end, or the test times out.
LONG = 60.0
#: A window that runs out with the human still there.
SHORT = 0.05
#: How long a window that must stay open is watched for closing by mistake.
STAYS_OPEN = 0.1


def _audio_downstream(h: Harness) -> list[str]:
    """The TTS audio at the user transport by chunk name, so a failure shows the order, not 2 KB of bytes."""
    return [
        _NAMES.get(f.audio, f"<{len(f.audio)} other bytes>")
        for f in frames_of(h.down, TTSAudioRawFrame)
        if isinstance(f, TTSAudioRawFrame)
    ]


async def _queue_an_utterance_held_at_the_avatar(h: Harness, participant: GatedOpenParticipant) -> None:
    """The first chunk's byte stream is opening, held by the gate; the other two wait in the send queue."""
    await h.queue(TTSStartedFrame(context_id="c1"))
    for chunk in (AUDIO_1, AUDIO_2, AUDIO_3):
        await h.queue(tts_audio("c1", chunk))
    await processed(h, "the utterance queued")
    await until(lambda: len(participant.streams) == 1, what="the first chunk's byte stream opening")


# --- the real session behind the service ------------------------------------------------------------------


@pytest.mark.unit
async def test_a_human_leaving_after_the_avatar_ends_the_session_as_no_humans_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order seen live: the avatar leaves, then the human. Nothing is reported as an error, the queued
    audio is not handed over to an empty room, and the end waits for the human's leave, not the window."""
    monkeypatch.setattr(_types, "NO_HUMANS_LEFT_WINDOW_SECS", LONG)
    participant = GatedOpenParticipant()
    rig = Rig(participant=participant)
    human = rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))
    service = make_service(SessionRigRoom(rig))
    rec = EventRecorder(service)
    gate = participant.open_gate = asyncio.Event()

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_an_utterance_held_at_the_avatar(h, participant)

            assert rig.avatar is not None
            rig.leave(rig.avatar, reason=CLIENT_INITIATED)  # the avatar worker's leave arrives first
            gate.set()
            await rec.wait("on_avatar_disconnected")
            await processed(h, "the leave handled as far as it goes before the human leaves")
            assert rec.count("on_session_ended") == 0, "the leave was classified before the human's own leave"
            assert_absent(h.up, ErrorFrame, where="while the human still appears present")

            rig.leave(human, reason=CLIENT_INITIATED)  # the bot hears of the hang-up after the avatar's leave
            await rec.wait("on_session_ended")
        finally:
            gate.set()
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "no_humans_left"
    assert_absent(h.up, ErrorFrame, where="after a hang-up")
    assert _audio_downstream(h) == [], (
        "the queued audio was handed over voice-only to a room with no one in it"
    )


@pytest.mark.unit
async def test_a_human_leaving_before_the_avatar_ends_the_session_as_no_humans_left_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other order: the human's leave arrives first, so the avatar's leave finds no human, and the end
    does not wait for any window."""
    monkeypatch.setattr(_types, "NO_HUMANS_LEFT_WINDOW_SECS", LONG)
    participant = GatedOpenParticipant()
    rig = Rig(participant=participant)
    human = rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))
    service = make_service(SessionRigRoom(rig))
    rec = EventRecorder(service)
    gate = participant.open_gate = asyncio.Event()

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_an_utterance_held_at_the_avatar(h, participant)

            rig.leave(human, reason=CLIENT_INITIATED)
            assert rig.avatar is not None
            rig.leave(rig.avatar, reason=CLIENT_INITIATED)
            gate.set()
            await rec.wait("on_session_ended")
        finally:
            gate.set()
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "no_humans_left"
    assert_absent(h.up, ErrorFrame, where="after a hang-up")
    assert _audio_downstream(h) == []


@pytest.mark.unit
async def test_a_window_that_runs_out_with_a_human_present_is_an_unexpected_leave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The avatar leaves and the human stays: when the window runs out, the end is `avatar_left` with one
    error, and the queued audio reaches the user voice-only, in order, as before the window existed."""
    monkeypatch.setattr(_types, "NO_HUMANS_LEFT_WINDOW_SECS", SHORT)
    participant = GatedOpenParticipant()
    rig = Rig(participant=participant)
    rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))
    service = make_service(SessionRigRoom(rig))
    rec = EventRecorder(service)
    gate = participant.open_gate = asyncio.Event()

    async with bidirectional_pipeline(service) as h:
        try:
            await rec.wait("on_avatar_connected")
            await _queue_an_utterance_held_at_the_avatar(h, participant)

            assert rig.avatar is not None
            rig.leave(rig.avatar, reason=CLIENT_INITIATED)
            gate.set()
            await rec.wait("on_session_ended")
        finally:
            gate.set()
        await h.finish()

    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert count(h.up, ErrorFrame) == 1
    assert _audio_downstream(h) == ["AUDIO_1", "AUDIO_2", "AUDIO_3"], (
        "the audio in flight or queued at the leave did not reach the user, in order"
    )


@pytest.mark.unit
async def test_a_lost_room_connection_with_a_human_listed_is_an_unexpected_leave_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot's own connection ends, and rtc keeps listing the human it can no longer hear leave: the
    window is skipped, and the end is `avatar_left` with its error at once, not after the window."""
    monkeypatch.setattr(_types, "NO_HUMANS_LEFT_WINDOW_SECS", LONG)
    rig = Rig()
    rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))
    service = make_service(SessionRigRoom(rig))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        rig.room.connected = False
        rig.room.emit("disconnected", lk_rtc.DisconnectReason.SIGNAL_CLOSE)
        await rec.wait("on_session_ended")
        await h.finish()

    assert rec.payload("on_avatar_disconnected")[0] == "room_connection_lost"
    assert rec.payload("on_session_ended")[0] == "avatar_left"
    assert count(h.up, ErrorFrame) == 1


@pytest.mark.unit
async def test_only_a_human_leaving_ends_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another agent in the room (a second bot, an egress) leaving inside the window is not the hang-up;
    the last standard-kind participant leaving is."""
    monkeypatch.setattr(_types, "NO_HUMANS_LEFT_WINDOW_SECS", LONG)
    async with launched() as (rig, session):
        human = rig.room.add_participant(FakeRemoteParticipant("user", kind=STANDARD))
        other_agent = rig.room.add_participant(FakeRemoteParticipant("other-agent", kind=AGENT))
        assert rig.avatar is not None
        rig.leave(rig.avatar, reason=CLIENT_INITIATED)
        await until(lambda: len(rig.rec.left) == 1, what="on_avatar_left")
        assert rig.rec.left == [("avatar_left_room", True)]

        listening = len(rig.room.listeners.get("participant_disconnected", ()))
        window = asyncio.ensure_future(session.wait_no_humans_left())
        try:
            await until(
                lambda: len(rig.room.listeners.get("participant_disconnected", ())) > listening,
                what="the window listening for leaves",
            )
            rig.leave(other_agent, reason=CLIENT_INITIATED)
            # the timeout is the assertion: the window must still be open well after the agent's leave
            done, _ = await asyncio.wait({window}, timeout=STAYS_OPEN)
            assert not done, "an agent leaving ended the window"
            rig.leave(human, reason=CLIENT_INITIATED)
            await until(window.done, what="the window to end on the human's leave")
            assert window.result() is True
        finally:
            await settle(window)
    rig.assert_torn_down()  # nothing left listening on the room, the window included


# --- the service alone, with a window the test ends ---------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("end", "reason"),
    [(EndFrame, "pipeline_ended"), (CancelFrame, "pipeline_cancelled")],
    ids=["EndFrame", "CancelFrame"],
)
async def test_the_pipeline_ending_during_the_window_ends_the_session_with_its_own_reason(
    end: type[Frame], reason: str, caplog_loguru: pytest.LogCaptureFixture
) -> None:
    """The window never ends by itself here. The pipeline's end cuts it short and takes the shutdown path
    it takes today: the pipeline's reason, no error, no handover, and the leave's one INFO line."""
    window: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    session = FakeAvatarSession(humans_window=window)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        session.emit_avatar_left("avatar_left_room", humans_present=True)
        await until(lambda: ("wait_no_humans_left",) in session.calls, what="the window to start")
        await h.queue(TTSStartedFrame(context_id="c1"))
        await h.queue(tts_audio("c1", AUDIO_1))
        await processed(h, "audio during the window")

        await h.queue(end())
        await rec.wait("on_session_ended")
        await until(lambda: all(task.done() for task in session.emitted), what="the leave to finish")
        await h.finish(send_end_frame=False)

    assert not window.done(), "the test's window was ended; the pipeline's end should have cut it short"
    assert rec.payload("on_session_ended")[0] == reason
    assert rec.count("on_session_ended") == 1
    assert_absent(h.up, ErrorFrame, where="when the pipeline ends during the window")
    assert _audio_downstream(h) == [], "audio held during the window was handed over at a shutdown"
    assert len(records_at(caplog_loguru, logging.INFO, "avatar left during shutdown")) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("last_human_left", "end_reason", "handed_over"),
    [
        pytest.param(False, "avatar_left", ["AUDIO_1", "AUDIO_2"], id="window-runs-out"),
        pytest.param(True, "no_humans_left", [], id="the-last-human-leaves"),
    ],
)
async def test_audio_during_the_window_is_held(
    last_human_left: bool, end_reason: str, handed_over: list[str]
) -> None:
    """TTS audio that arrives during the window goes neither to the avatar that left nor past the hold.
    If the window runs out, it is handed over to the user, in order; if the last human leaves, there is no
    one to hear it, and it is dropped as at any no-humans-left end."""
    window: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    session = FakeAvatarSession(humans_window=window)
    service = make_service(FakeAvatarRoom(JoinMode.DIRECT, session=session))
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        session.emit_avatar_left("avatar_left_room", humans_present=True)
        await until(lambda: ("wait_no_humans_left",) in session.calls, what="the window to start")
        await h.queue(TTSStartedFrame(context_id="c1"))
        for chunk in (AUDIO_1, AUDIO_2):
            await h.queue(tts_audio("c1", chunk))
        await processed(h, "audio during the window")

        assert session.send_audio_calls == [], "audio went to the avatar that left"
        assert _audio_downstream(h) == [], "audio passed the hold during the window"
        assert rec.count("on_session_ended") == 0
        assert service.state is ServiceState.ENDED, "the state waited for the window"

        window.set_result(last_human_left)
        await rec.wait("on_session_ended")
        await h.finish()

    assert rec.payload("on_session_ended")[0] == end_reason
    assert count(h.up, ErrorFrame) == (0 if last_human_left else 1)
    assert _audio_downstream(h) == handed_over
    assert session.send_audio_calls == []
