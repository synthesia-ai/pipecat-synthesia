"""`_transport_access`: the checked reads of `LiveKitTransport` privates (*Pipecat-internals rule*).

`check_transport` runs once, when the direct join is constructed; the accessors trust it. `hasattr(client,
"room")` is not a safe check: the property raises a plain `Exception` before the transport's `setup()` has
run, and `hasattr` swallows only `AttributeError`. `FakeLiveKitTransport` is a real `LiveKitTransport`
subclass with a real `LiveKitTransportClient`, so the check passes or fails for production reasons; its
`room_reads` counter turns "never evaluates the property" into an assertion.

`wait_connected`'s timeout is shrunk only where a correct implementation never reaches it, or where the
timeout elapsing is the behaviour under test.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from pipecat_synthesia._compat import PIPECAT_VERSION, TESTED_PIPECAT_VERSIONS
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._transport_access import (
    bot_identity,
    check_transport,
    get_room,
    get_token,
    get_url,
    wait_connected,
)
from pipecat_synthesia.version import __version__
from tests.fakes.fake_transport import (
    FAKE_AGENT_TOKEN,
    FAKE_URL,
    FakeLiveKitTransport,
)
from tests.support.secrets import assert_no_secrets
from tests.support.waiting import until

BOT = "pipecat-bot"


def _without_client(transport: FakeLiveKitTransport) -> None:
    transport._client = object()  # type: ignore[assignment]


def _without_token(transport: FakeLiveKitTransport) -> None:
    del transport._client._token


def _without_url(transport: FakeLiveKitTransport) -> None:
    del transport._client._url


#: The ways a Pipecat release could move what we read, each on an otherwise genuine `LiveKitTransport`.
MOVED_INTERNALS: list[tuple[str, Callable[[FakeLiveKitTransport], None]]] = [
    ("client-replaced", _without_client),
    ("token-moved", _without_token),
    ("url-moved", _without_url),
]


# ------------------------------------------------------------------------------------- the check


@pytest.mark.unit
@pytest.mark.parametrize("subject", [object, MagicMock], ids=["plain-object", "magicmock"])
def test_check_rejects_a_non_livekit_transport(subject: Callable[[], Any]) -> None:
    """Anything that is not a `LiveKitTransport` is a `TypeError` naming the adapter to use instead,
    never the version message: a Daily transport is a wrong choice of join mode, not a Pipecat upgrade."""
    with pytest.raises(TypeError) as excinfo:
        check_transport(subject())

    text = str(excinfo.value)
    assert "PrivateLiveKitRoom" in text, text
    assert PIPECAT_VERSION not in text, text


@pytest.mark.unit
@pytest.mark.parametrize("move", [m for _, m in MOVED_INTERNALS], ids=[n for n, _ in MOVED_INTERNALS])
def test_check_rejects_moved_internals(move: Callable[[FakeLiveKitTransport], None]) -> None:
    """A `LiveKitTransport` whose privates are not where this version reads them is a `TypeError` naming
    both versions and the tested set, never an `AttributeError` from deep inside an accessor."""
    transport = FakeLiveKitTransport(participant_id=BOT)
    move(transport)

    with pytest.raises(TypeError) as excinfo:
        check_transport(transport)

    text = str(excinfo.value)
    assert __version__ in text, text
    assert PIPECAT_VERSION in text, text
    assert TESTED_PIPECAT_VERSIONS[0] in text, text


@pytest.mark.unit
def test_check_passes_before_setup_and_never_evaluates_room() -> None:
    """On a transport whose `setup()` has not run (`_room is None`), the guard passes and nothing reads
    the `room` property, which would raise a plain `Exception`, not `AttributeError`."""
    transport = FakeLiveKitTransport(room_raises_before_setup=True, participant_id=BOT)

    check_transport(transport)
    assert get_url(transport) == FAKE_URL
    assert get_token(transport) == FAKE_AGENT_TOKEN
    assert bot_identity(transport) == BOT
    assert transport.room_reads == 0, "the guard evaluated client.room"


@pytest.mark.unit
def test_get_room_before_setup_raises_connection_error() -> None:
    """The one accessor that must touch the property reports Pipecat's own diagnosis as CONNECTION."""
    transport = FakeLiveKitTransport(room_raises_before_setup=True, participant_id=BOT)

    with pytest.raises(SynthesiaError) as excinfo:
        get_room(transport)

    assert "missing room object" in str(excinfo.value)
    assert excinfo.value.type is ErrorType.CONNECTION
    assert transport.room_reads == 1


@pytest.mark.unit
def test_get_room_returns_the_transports_own_room() -> None:
    """The borrowed room is the transport's own, not a copy."""
    transport = FakeLiveKitTransport(participant_id=BOT)

    assert get_room(transport) is transport.fake_room


@pytest.mark.unit
def test_get_token_is_falsy_when_the_transport_has_no_usable_token() -> None:
    """`_token` is whatever the developer handed `LiveKitTransport(...)`, so it can be `None` or `""`.
    That must read as "no token" rather than as a token we failed to parse, so the constructor can say
    which it is.

    A Pipecat release that removes `_token` deliberately does not take this path: `check_transport`
    requires the attribute, so that case raises the version message, the louder and more accurate
    answer."""
    transport = FakeLiveKitTransport(token=None, participant_id=BOT)

    assert get_token(transport) is None


# ----------------------------------------------------------------------------------- wait_connected


@pytest.mark.unit
async def test_wait_connected_immediate() -> None:
    """Already connected (`participant_id` set): returns without registering anything."""
    transport = FakeLiveKitTransport(participant_id=BOT)

    await wait_connected(transport, timeout=30.0)

    assert transport.handlers_for("on_connected") == []


@pytest.mark.unit
async def test_wait_connected_after_event() -> None:
    """Not connected yet, the normal case because Pipecat connects in its own concurrent `setup()`: the
    wait ends on `on_connected`, and the handler is removed afterwards."""
    transport = FakeLiveKitTransport(participant_id="")
    waiter = asyncio.ensure_future(wait_connected(transport, timeout=30.0))
    await until(lambda: transport.handlers_for("on_connected") != [], what="the on_connected handler")
    assert not waiter.done(), "returned before the transport connected"

    transport.set_connected(BOT)
    await transport.fire("on_connected")

    await waiter
    assert transport.handlers_for("on_connected") == [], "handler left installed"


@pytest.mark.unit
async def test_wait_connected_when_connect_lands_while_registering() -> None:
    """A connect that lands while the handler is being installed is not missed. `on_connected` fires
    exactly once, so the handler goes on first and `participant_id` is read again after it.

    This transport connects during registration and never fires the event, which is exactly that
    window. The timeout is shrunk because a correct implementation never reaches it; the shrink only
    turns a regression from a 30 s hang into an immediate failure."""

    class ConnectsWhileRegistering(FakeLiveKitTransport):
        def add_event_handler(self, event_name: str, handler: Any) -> None:
            super().add_event_handler(event_name, handler)
            if event_name == "on_connected":
                self.set_connected(BOT)

    transport = ConnectsWhileRegistering(participant_id="")

    await wait_connected(transport, timeout=0.05)

    assert transport.handlers_for("on_connected") == [], "handler left installed"


@pytest.mark.unit
async def test_wait_connected_before_setup_does_not_touch_room() -> None:
    """The wait path never evaluates `client.room`, so it is safe to run while Pipecat's own `setup()`
    is still building the room."""
    transport = FakeLiveKitTransport(room_raises_before_setup=True, participant_id="")
    waiter = asyncio.ensure_future(wait_connected(transport, timeout=30.0))
    await until(lambda: transport.handlers_for("on_connected") != [], what="the on_connected handler")

    transport.set_connected(BOT)
    await transport.fire("on_connected")
    await waiter

    assert transport.room_reads == 0


@pytest.mark.unit
async def test_wait_connected_timeout_message(caplog_loguru: pytest.LogCaptureFixture) -> None:
    """On timeout: CONNECTION, naming what to check and Pipecat's own connect retries, and no token in
    the message. Here the timeout elapsing is the behaviour under test."""
    transport = FakeLiveKitTransport(participant_id="")

    with pytest.raises(SynthesiaError) as excinfo:
        await wait_connected(transport, timeout=0.05)

    message = str(excinfo.value)
    assert "did not connect within 0.05s" in message
    assert "check url/token" in message
    assert "retries connect 3 times" in message
    assert excinfo.value.type is ErrorType.CONNECTION
    assert transport.handlers_for("on_connected") == []
    assert_no_secrets(message, repr(excinfo.value), caplog_loguru.text)
