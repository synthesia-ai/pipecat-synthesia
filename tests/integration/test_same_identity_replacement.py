"""Same-identity replacement and a stated departure, on a borrowed room of the real livekit-server.

The replacement grace rests on how livekit-server tells the other participants about each: a replacement
is a disconnect with no reason and then the new join, a real departure states its reason. The unit fakes
only model that order. The retried launch reuses the session's own request, so it carries the very token
the session minted. `REPLACEMENT_GRACE_SECS` is raised for the replacement, where it bounds the new
worker's real connect and publish, and put far past the wait backstop for the stated departure, so a
report inside the backstop came from the immediate path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from time import monotonic
from typing import Any

import pytest
from livekit import rtc

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit._types import LiveKitCredentials, MintedAvatarToken, RoomHandle
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY, DEFAULT_AVATAR_NAME
from pipecat_synthesia.protocol import SESSIONS_PATH
from tests.fakes.fake_api import RecordedRequest
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.integration._launcher import Launcher
from tests.support.callbacks import CallbackRecorder, make_launch_context
from tests.support.livekit_dev import LiveKitDev, connect_room, mint_dev_token
from tests.support.waiting import until

AVATAR = DEFAULT_AVATAR_IDENTITY
BOT = "bot"
AVATAR_IDS = ("av_a", "av_b")
CHUNK = b"\x01\x02" * 960  # 40 ms of 24 kHz mono s16le
#: Every network-observable wait. A backstop, not a budget.
WAIT = 10.0
#: The replacement grace, raised: see the module docstring.
REPLACEMENT_GRACE = 3.0
#: The "and nothing happened" wait after the replacement: a multiple of the grace (see its use).
NOTHING_HAPPENED_WAIT = 2 * REPLACEMENT_GRACE
#: Far above WAIT, so a leave reported within WAIT did not wait for the grace.
UNREACHABLE_GRACE = 60.0
#: The replacement worker publishes its readiness this long after it has joined, so its "preparing"
#: reaches the session as an attribute change of the new participant rather than with the join.
REPLACEMENT_STATUS_DELAY = 0.5


def _has_video(participant: rtc.RemoteParticipant | None) -> bool:
    return participant is not None and any(
        publication.kind == rtc.TrackKind.KIND_VIDEO
        for publication in participant.track_publications.values()
    )


class _Launches:
    """`create_session` that launches a fake worker through the launcher and keeps its request.

    The request is what a retried launch sends again: `launcher.hook()` builds each fake worker from
    the request's own `livekitUrl`, `livekitToken` and `avatarIds`.
    """

    def __init__(self, launcher: Launcher, **fake_worker_options: Any) -> None:
        self.launcher = launcher
        self.options = fake_worker_options
        self.requests: list[RecordedRequest] = []

    async def __call__(self, avatar_ids: Sequence[str], livekit_url: str, livekit_token: str) -> None:
        request = RecordedRequest(
            method="POST",
            path=SESSIONS_PATH,
            headers={},
            json={"avatarIds": list(avatar_ids), "livekitUrl": livekit_url, "livekitToken": livekit_token},
        )
        self.requests.append(request)
        # Returns once the worker's tracks are published: "the 201 means the tracks are published".
        await self.launcher.hook(**self.options)(request)


async def _connect(
    dev: LiveKitDev, room: rtc.Room, room_name: str, launches: _Launches, rec: CallbackRecorder
) -> LiveKitAvatarSession:
    handle = RoomHandle(
        room=room,
        bot_identity=BOT,
        livekit_url=dev.url,
        room_name=room_name,
        owns_room=False,
        connected_at=monotonic(),
    )
    ctx = make_launch_context(
        FakeSessionApiClient(on_create=launches), avatar_ids=AVATAR_IDS, launch_timeout=30, recorder=rec
    )
    return await LiveKitAvatarSession.launch(
        handle,
        token_source=MintedAvatarToken(LiveKitCredentials(dev.api_key, dev.api_secret)),
        avatar_identity=AVATAR,
        avatar_name=DEFAULT_AVATAR_NAME,
        ctx=ctx,
    )


@pytest.mark.integration_extended
async def test_a_same_identity_replacement_is_not_a_leave(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch retried with the same token replaces the serving worker: the session carries on.

    The old avatar participant disconnects with no reason, the new one joins, and the session neither
    reports the avatar as left nor stops: it reads the new participant's readiness, swaps on it, and
    sends it the next utterance.
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", REPLACEMENT_GRACE)
    dev = livekit_server
    rec = CallbackRecorder()
    launches = _Launches(launcher)

    async with connect_room(dev, mint_dev_token(dev, identity=BOT, room=room_name)) as room:
        disconnect_reasons: list[Any] = []

        def on_disconnected(participant: rtc.RemoteParticipant) -> None:
            if participant.identity == AVATAR:
                disconnect_reasons.append(participant.disconnect_reason)

        room.on("participant_disconnected", on_disconnected)
        session: LiveKitAvatarSession | None = None
        try:
            session = await _connect(dev, room, room_name, launches, rec)
            (first,) = launcher.fakes
            await until(
                lambda: any(all_ready for _, all_ready in rec.status),
                timeout=WAIT,
                what="the first worker's readiness",
            )
            before = len(rec.status)
            replaced = room.remote_participants.get(AVATAR)

            # The retried launch: the same request, so the same token and the same avatar identity.
            await launcher.hook(status_delay=REPLACEMENT_STATUS_DELAY)(launches.requests[0])
            second = launcher.fakes[1]
            await until(
                lambda: first.connected is False, timeout=WAIT, what="the server to drop the old worker"
            )
            await until(
                lambda: (
                    (current := room.remote_participants.get(AVATAR)) is not replaced and _has_video(current)
                ),
                timeout=WAIT,
                what="the new avatar participant and its video in the bot's room",
            )
            # "And nothing happened": the grace's deadline is fixed by its first signal, the old worker's
            # disconnect, which came before this point, so one grace would do by construction; twice that
            # is the margin for a loaded runner's late timers. The grace itself cannot shrink here: it
            # bounds the new worker's real connect and publish.
            await asyncio.sleep(NOTHING_HAPPENED_WAIT)

            assert rec.left == [], launcher.debug_dump()
            assert session.is_alive is True, launcher.debug_dump()
            assert disconnect_reasons and disconnect_reasons[0] in (
                None,
                rtc.DisconnectReason.UNKNOWN_REASON,
            ), f"a replacement should reach other participants with no reason, got {disconnect_reasons}"

            # The new participant's readiness reaches the session: its "preparing", then its "ready".
            await until(
                lambda: any(not all_ready for _, all_ready in rec.status[before:]),
                timeout=WAIT,
                what="the new worker's preparing status",
            )
            await until(lambda: rec.status[-1][1] is True, timeout=WAIT, what="the new worker's ready status")
            assert await session.swap_avatar("av_b", timeout=WAIT) == "av_b"
            assert second.swap_calls[-1]["avatar_id"] == "b", launcher.debug_dump()

            await session.send_audio(CHUNK, 24000, 1)
            await session.end_utterance()
            await until(
                lambda: len(second.streams) == 1, timeout=WAIT, what="the utterance at the new worker"
            )
            assert second.streams[0]["bytes"] == CHUNK, launcher.debug_dump()
            assert rec.left == [], launcher.debug_dump()
        finally:
            if session is not None:
                await session.aclose()


@pytest.mark.integration
async def test_a_departure_with_a_stated_reason_is_reported_without_waiting_for_the_grace(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker leaving on its own states `CLIENT_INITIATED`, and is reported once, at once.

    The worker's video unpublish reaches the session first and starts a grace of its own; the stated
    disconnect behind it must end that grace with one report, not wait for it.
    """
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", UNREACHABLE_GRACE)
    dev = livekit_server
    rec = CallbackRecorder()
    launches = _Launches(launcher, leave_after=1.0)

    async with connect_room(dev, mint_dev_token(dev, identity=BOT, room=room_name)) as room:
        disconnect_reasons: list[Any] = []

        def on_disconnected(participant: rtc.RemoteParticipant) -> None:
            if participant.identity == AVATAR:
                disconnect_reasons.append(participant.disconnect_reason)

        room.on("participant_disconnected", on_disconnected)
        session: LiveKitAvatarSession | None = None
        try:
            session = await _connect(dev, room, room_name, launches, rec)

            await until(lambda: len(rec.left) >= 1, timeout=WAIT, what="the leave to be reported")
            assert rec.left == [("avatar_left_room", False)], launcher.debug_dump()
            assert session.is_alive is False
            assert disconnect_reasons == [rtc.DisconnectReason.CLIENT_INITIATED]

            await session.aclose()
            assert rec.left == [("avatar_left_room", False)], "reported exactly once"
        finally:
            if session is not None:
                await session.aclose()
