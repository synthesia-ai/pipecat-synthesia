"""The reads of `LiveKitTransport`'s private attributes, all in one module (*Pipecat-internals rule*).

Pipecat (1.14 to 1.16) gives `LiveKitTransport` no public accessor for its `rtc.Room`, server URL or
token, and a direct join needs all three. Keeping the reads here means one module to re-check on each
Pipecat release, and one error message when a release moves them.

`check_transport()` runs once, from `LiveKitTransportRoom.__init__`, so a wrong transport or a Pipecat that
moved these attributes fails at startup (*misuse vs session errors*). The accessors below trust that check
and do not repeat it. Two traps:

* `LiveKitTransportClient.room` is a property that raises a plain `Exception`, not `AttributeError`, until
  the transport's own `setup()` has built the room (Pipecat 1.14 `transports/livekit/transport.py:284-295`).
  So `hasattr(client, "room")` raises on a healthy transport that has not been set up yet, and the
  constructor always runs before that `setup()`. `check_transport()` inspects the class with
  `inspect.getattr_static` instead, which never runs the property; only `get_room()` evaluates it, at
  launch.
* The launch starts while Pipecat is still setting up the pipeline, so "not connected yet" is normal.
  `wait_connected()` waits on the transport's public `on_connected` event and `participant_id`, and
  touches neither the client nor the room.

Pipecat's LiveKit transport module is imported inside `check_transport()`, so a relayed join never imports
it.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from pipecat_synthesia._compat import PIPECAT_VERSION, TESTED_PIPECAT_VERSIONS
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.version import __version__

if TYPE_CHECKING:
    from livekit import rtc as lk_rtc
    from pipecat.transports.livekit.transport import LiveKitTransport

__all__ = ["bot_identity", "check_transport", "get_room", "get_token", "get_url", "wait_connected"]


def check_transport(transport: Any) -> None:
    """Verify this is a `LiveKitTransport` whose privates are where Pipecat 1.14.0 put them.

    Class-level checks only: nothing here evaluates `client.room` (see the module docstring), so the
    check passes on a transport whose `setup()` has not run yet.

    Raises:
        TypeError: Not a `LiveKitTransport` at all, naming `PrivateLiveKitRoom` as the fix; or a
            `LiveKitTransport` whose private attributes are not where this version expects them, naming
            both versions.
    """
    # Imported here, so importing `pipecat_synthesia.livekit` for a relayed join never imports Pipecat's
    # LiveKit transport.
    from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitTransportClient

    if not isinstance(transport, LiveKitTransport):
        raise TypeError(
            f"LiveKitTransportRoom needs Pipecat's LiveKitTransport, not {type(transport).__name__}; "
            "for Daily or SmallWebRTC, use PrivateLiveKitRoom"
        )
    client = getattr(transport, "_client", None)
    if (
        isinstance(client, LiveKitTransportClient)
        and isinstance(inspect.getattr_static(type(client), "room", None), property)
        and hasattr(client, "_url")
        and hasattr(client, "_token")
    ):
        return
    raise TypeError(
        f"pipecat-synthesia {__version__} does not know how to access this LiveKitTransport "
        f"(pipecat-ai {PIPECAT_VERSION}); tested with {TESTED_PIPECAT_VERSIONS}"
    )


def get_room(transport: LiveKitTransport) -> lk_rtc.Room:
    """The `rtc.Room` the transport is connected to.

    Raises:
        SynthesiaError: `CONNECTION` if the transport has not built its room yet (its own message,
            verbatim).
    """
    try:
        return transport._client.room
    except Exception as exc:
        # Pipecat's own wording names the cause ("missing room object (pipeline not started?)");
        # keeping it is more useful than anything we could write here.
        raise SynthesiaError(str(exc), type=ErrorType.CONNECTION) from exc


def get_url(transport: LiveKitTransport) -> str:
    """The LiveKit server URL the transport connects with (`transport.py:177`; no public accessor)."""
    return transport._client._url


def get_token(transport: LiveKitTransport) -> str | None:
    """The bot's own access token (`transport.py:178`), set by `LiveKitTransport.__init__`. Never logged."""
    return transport._client._token


def bot_identity(transport: LiveKitTransport) -> str:
    """The bot's LiveKit identity — public, and `""` until `connect()` completed (`transport.py:275`)."""
    return transport.participant_id


async def wait_connected(transport: LiveKitTransport, *, timeout: float) -> None:  # noqa: ASYNC109
    """Return once the transport has joined its room; never touches `client.room`.

    Pipecat connects inside `input()/output().setup()`, concurrently with ours, and populates
    `participant_id` at the end of it. The handler is registered *before* the
    second `participant_id` check so a connect that completes between the two is not missed.

    Args:
        transport: The bot's transport.
        timeout: Seconds to wait (the adapter's `connect_timeout`; 30 s by default).

    Raises:
        SynthesiaError: `CONNECTION` on timeout, naming Pipecat's own retry budget.
    """
    if transport.participant_id:
        return

    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def _on_connected(_transport: Any) -> None:
        if not fut.done():
            fut.set_result(None)

    transport.add_event_handler("on_connected", _on_connected)
    try:
        # Re-checked under the handler: `connect()` may have finished while we were registering, and
        # `on_connected` fires exactly once (`transport.py:1271`).
        if transport.participant_id:
            return
        await asyncio.wait_for(fut, timeout)
    except TimeoutError:
        raise SynthesiaError(
            f"LiveKitTransport did not connect within {timeout}s (check url/token; Pipecat retries "
            "connect 3 times with 4 s waits)",
            type=ErrorType.CONNECTION,
        ) from None
    finally:
        transport.remove_event_handler("on_connected", _on_connected)
