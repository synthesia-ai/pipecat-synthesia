"""A real `LiveKitTransport` subclass for the direct-join tests.

Nothing here is monkeypatched: `FakeLiveKitTransport` **is** a `LiveKitTransport` and its `_client`
**is** a `LiveKitTransportClient`, so `_transport_access._check()` passes for the same reasons it will
pass in production, not because a test relaxed it. Only three things are swapped for recorders:

* `mute_participant()` records the identity **inside the coroutine body**, so a caller that forgot the
  `await` records nothing and the test that asserts on `mute_calls` fails (the project-wide
  `error:coroutine .* was never awaited` filter also turns the forgotten await into a hard error);
* `_client._room` is assigned directly instead of by `setup()`: a `FakeRoom`, or `None` to model the
  pre-setup transport whose `room` property raises a plain `Exception`;
* `_client._participant_id` is assigned by `set_connected()`, which is what `connect()` does.

`super().__init__(url, token, "fake-room", LiveKitParams())` is called for real: at Pipecat 1.10.0 the
constructor only assigns attributes and registers event names (no `rtc.Room()`, no loop, no I/O), which
is why this module can be used from a sync fixture.

`_client` is a `CountingRoomClient`, a genuine `LiveKitTransportClient` subclass whose `room` property
counts its reads and otherwise delegates to `super()`. That counter is what lets
`test_check_passes_before_setup_and_never_evaluates_room` prove that the guard inspects the *class* with
`inspect.getattr_static` and never evaluates the property. A test cannot express that with the base
client, and patching Pipecat to add the counter would defeat the point of a real subclass.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from livekit import api as lk_api
from livekit import rtc as lk_rtc
from pipecat.transports.livekit.transport import (
    LiveKitParams,
    LiveKitTransport,
    LiveKitTransportClient,
)

from tests.fakes.fake_rtc import FakeRoom
from tests.support.secrets import LK_API_SECRET, LK_TOKEN_MARKER

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "FAKE_AGENT_TOKEN",
    "FAKE_STANDARD_TOKEN",
    "FAKE_URL",
    "CountingRoomClient",
    "FakeLiveKitTransport",
    "fake_bot_token",
]

#: The transport's URL: `https://` on purpose, so a test can see `launch()` normalise it to `wss://`.
FAKE_URL = "https://fake.livekit"
FAKE_ROOM_NAME = "fake-room"


def fake_bot_token(
    *, identity: str = "pipecat-bot", room: str = FAKE_ROOM_NAME, kind: str | None = "agent"
) -> str:
    """A signed bot token, with or without the top-level `kind` claim (*agent-kind requirement*).

    Real `livekit.api.AccessToken`, never a hand-rolled JWT: the claim layout of a STANDARD bot token
    is the thing under test, so it has to be produced the way a developer would produce it. The name
    claim carries `LK_TOKEN_MARKER` so any line that echoes the token fails `assert_no_secrets`.
    """
    token = (
        lk_api.AccessToken("fake-lk-key", LK_API_SECRET)
        .with_identity(identity)
        .with_name(f"{identity} {LK_TOKEN_MARKER}")
        .with_grants(lk_api.VideoGrants(room_join=True, room=room, agent=True))
        .with_ttl(timedelta(hours=1))
    )
    if kind is not None:
        token = token.with_kind(kind)  # type: ignore[arg-type]  # a plain str, so a test can pass a bogus kind
    return token.to_jwt()


#: What Pipecat's developers *should* mint: `kind="agent"` (*agent-kind requirement*).
FAKE_AGENT_TOKEN = fake_bot_token()
#: What `pipecat.runner.livekit.generate_token_with_agent()` actually mints: `video.agent`, no kind.
FAKE_STANDARD_TOKEN = fake_bot_token(kind=None)


class CountingRoomClient(LiveKitTransportClient):
    """A real transport client whose `room` property counts its reads (see the module docstring)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.room_reads = 0
        super().__init__(*args, **kwargs)

    @property
    def room(self) -> lk_rtc.Room:
        self.room_reads += 1
        # Pipecat's own property, unchanged: raises `Exception("... missing room object ...")` while
        # `_room is None`.
        return super().room


class FakeLiveKitTransport(LiveKitTransport):
    """A `LiveKitTransport` that records `mute_participant` and can fire its events on demand."""

    def __init__(
        self,
        *,
        url: str = FAKE_URL,
        token: str | None = FAKE_AGENT_TOKEN,
        room: FakeRoom | None = None,
        participant_id: str = "",
        room_raises_before_setup: bool = False,
    ) -> None:
        """Build the transport in whatever state the test needs.

        Args:
            url: The URL the transport "connected" with; read through `_transport_access.get_url`.
            token: The bot's token, as the developer passed it to `LiveKitTransport(...)`. `None`
                or `""` models a developer who passed no usable token, which is the only way the
                kind check can find nothing to read (a release that *removed* `_token` is caught
                by the version guard in `_check` instead).
            room: The room `_client.room` returns. Defaults to a fresh `FakeRoom`.
            participant_id: The bot identity. `""` (the default) is the pre-connect state.
            room_raises_before_setup: Leave `_client._room` as `None`, so the real `room` property
                raises — the pre-`setup()` transport our launch task races.
        """
        super().__init__(url, token or "", FAKE_ROOM_NAME, LiveKitParams())
        #: Every identity passed to `mute_participant`, recorded inside the coroutine body.
        self.mute_calls: list[str] = []
        #: Same, for `unmute_participant` (nothing in the plugin calls it; a test asserts that).
        self.unmute_calls: list[str] = []
        self.fake_room = None if room_raises_before_setup else (room or FakeRoom(FAKE_ROOM_NAME))

        # Re-create the client as the counting subclass with the arguments the transport just used, so
        # every `_transport_access` guard still sees a genuine `LiveKitTransportClient`.
        base = self._client
        self._client = CountingRoomClient(
            url, token or "", FAKE_ROOM_NAME, base._params, base._callbacks, self.name
        )
        self._client._room = self.fake_room  # type: ignore[assignment]  # a FakeRoom stands in for rtc.Room
        self._client._participant_id = participant_id
        if token is None:
            # A transport whose token we cannot read at all: `get_token` must return falsy, not "".
            self._client._token = None  # type: ignore[assignment]

    @property
    def room_reads(self) -> int:
        """How many times anything evaluated `client.room`; the transport access guard never should."""
        return self._client.room_reads if isinstance(self._client, CountingRoomClient) else -1

    async def mute_participant(self, participant_id: str) -> None:
        """Record the call, *inside* the coroutine, so a caller that forgets the `await` records nothing."""
        self.mute_calls.append(participant_id)

    async def unmute_participant(self, participant_id: str) -> None:
        self.unmute_calls.append(participant_id)

    def set_connected(self, identity: str) -> None:
        """What `connect()` does to the client; fire `on_connected` separately with `fire()`."""
        self._client._participant_id = identity

    async def fire(self, event: str, *args: Any) -> None:
        """Fire a transport event and await the tasks it spawned.

        Pipecat runs non-sync handlers as detached tasks, so a test that only called
        `_call_event_handler` would assert before the handler had run. Draining them makes the event
        synchronous from the test's point of view, and leaves nothing running at the end of the test.

        It does **not** surface a handler's exception: Pipecat catches `Exception` inside
        `_run_handler` and logs it at ERROR, so the task completes successfully either way, exactly as
        in production. A test that cares
        whether a handler blew up must assert on the recorded effect, or look for that ERROR record
        in `caplog_loguru`.
        """
        await self._call_event_handler(event, *args)
        tasks = [task for _name, task in list(self._event_tasks)]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def handlers_for(self, event: str) -> list[Callable[..., Any]]:
        """The handlers currently registered for `event` — how a test sees a close hook take effect."""
        handler = self._event_handlers.get(event)
        return list(handler.handlers) if handler is not None else []
