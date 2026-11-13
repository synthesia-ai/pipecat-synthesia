"""A failure from the LiveKit API client never reaches a message or a log line through its `repr`.

What this pins (*redaction rule*): the LiveKit API client runs on aiohttp, and the `repr` of aiohttp's
`ClientResponseError` echoes the request headers, Authorization included. So when creating, joining or
deleting a private room fails with one, the `SynthesiaError` the relayed join raises and the WARNING it
logs describe the failure by type and message only; and so do the session's own close warnings, for a
close hook or an owned-room release that fails the same way.

`PrivateLiveKitRoom.launch()` is driven through its `api_factory` and `room_factory` seams. The session's
close is driven on `LiveKitAvatarSession` directly, through the unit rig, because no public route can make
a close hook or a room release raise: the adapters supply both, and neither raises on its own.
"""

from __future__ import annotations

import dataclasses
import logging

import aiohttp
import pytest
from livekit import rtc as lk_rtc
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from pipecat_synthesia import ErrorType, SynthesiaError
from pipecat_synthesia.livekit import PrivateLiveKitRoom
from pipecat_synthesia.room import LaunchContext
from tests.fakes.fake_lk_api import LiveKitApiFactory, RoomFactory
from tests.fakes.fake_session_api import FakeSessionApiClient
from tests.support.callbacks import make_launch_context
from tests.support.secrets import LK_API_SECRET, LK_TOKEN_MARKER, assert_no_secrets
from tests.unit._service_helpers import records_at
from tests.unit._session_rig import Rig as SessionRig

pytestmark = pytest.mark.unit

ROOM = "private-room"
#: The LiveKit API key id the private room is built with; not a secret, unlike `LK_API_SECRET`.
LIVEKIT_KEY_ID = "APIlivekit"


def aiohttp_error_echoing(secret: str) -> aiohttp.ClientResponseError:
    """A real aiohttp error carrying `secret` in its request's Authorization header, as the LiveKit API
    client's requests do. Its `repr` shows the header; its `str` does not."""
    url = URL("https://livekit.example/twirp/livekit.RoomService/CreateRoom")
    headers = CIMultiDictProxy(CIMultiDict({"Authorization": f"Bearer {secret}"}))
    info = aiohttp.RequestInfo(url, "POST", headers, url)
    error = aiohttp.ClientResponseError(info, (), status=502, message="Bad Gateway")
    assert secret in repr(error) and secret not in str(error), "the premise of every test here"
    return error


def private_room(
    *, rooms: RoomFactory | None = None, apis: LiveKitApiFactory | None = None
) -> PrivateLiveKitRoom:
    return PrivateLiveKitRoom(
        url="wss://livekit.example",
        api_key=LIVEKIT_KEY_ID,
        api_secret=LK_API_SECRET,
        room_name=ROOM,
        room_factory=rooms or RoomFactory(),
        api_factory=apis or LiveKitApiFactory(),
    )


async def _drop_media(_data: bytes, _a: int, _b: int) -> None:
    return None


def relayed_context() -> LaunchContext:
    """A launch context with the two media callbacks a relayed join requires; nothing here reaches them."""
    ctx = make_launch_context(FakeSessionApiClient())
    callbacks = dataclasses.replace(ctx.callbacks, on_video_frame=_drop_media, on_audio_frame=_drop_media)
    return dataclasses.replace(ctx, callbacks=callbacks)


async def test_a_create_failure_names_the_error_without_its_request_headers(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    error = aiohttp_error_echoing(LK_TOKEN_MARKER)
    room = private_room(apis=LiveKitApiFactory(create_error=error))

    with pytest.raises(SynthesiaError) as caught:
        await room.launch(relayed_context())

    assert caught.value.type is ErrorType.CONNECTION
    assert "ClientResponseError" in str(caught.value), "the error still says what failed"
    assert_no_secrets(str(caught.value), repr(caught.value), caplog_loguru.text)


async def test_a_join_failure_names_the_error_without_its_request_headers(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    room = private_room(rooms=RoomFactory(connect_error=aiohttp_error_echoing(LK_TOKEN_MARKER)))

    with pytest.raises(SynthesiaError) as caught:
        await room.launch(relayed_context())

    assert caught.value.type is ErrorType.CONNECTION
    assert "ClientResponseError" in str(caught.value)
    assert_no_secrets(str(caught.value), repr(caught.value), caplog_loguru.text)


async def test_a_delete_failure_is_logged_without_its_request_headers(
    caplog_loguru: pytest.LogCaptureFixture,
) -> None:
    """The join fails, so the room is deleted on the way out, and the delete fails too."""
    room = private_room(
        rooms=RoomFactory(connect_error=lk_rtc.ConnectError("no route")),
        apis=LiveKitApiFactory(delete_error=aiohttp_error_echoing(LK_TOKEN_MARKER)),
    )

    with pytest.raises(SynthesiaError):
        await room.launch(relayed_context())

    assert records_at(caplog_loguru, logging.WARNING, "delete"), "the failed delete was logged"
    assert_no_secrets(caplog_loguru.text)


@pytest.mark.parametrize("failing", ["close hook", "room release"])
async def test_a_session_close_failure_is_logged_without_its_request_headers(
    caplog_loguru: pytest.LogCaptureFixture, failing: str
) -> None:
    error = aiohttp_error_echoing(LK_TOKEN_MARKER)

    async def fail() -> None:
        raise error

    rig = SessionRig(owns_room=True)
    if failing == "room release":
        rig.handle = dataclasses.replace(rig.handle, release_room=fail)
    session = await rig.launch()
    if failing == "close hook":
        session.add_close_hook(fail)

    await session.aclose()

    assert records_at(caplog_loguru, logging.WARNING, "ClientResponseError"), "the failure was logged"
    assert_no_secrets(caplog_loguru.text)
