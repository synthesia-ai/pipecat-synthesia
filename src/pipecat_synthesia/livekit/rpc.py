"""The playback event handlers, and the RPCs the bot calls on the avatar worker (clear buffer, swap)."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from pipecat_synthesia._async import CallbackTasks
from pipecat_synthesia.config import unprefixed_avatar_id
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.livekit._rtc import lk_rtc
from pipecat_synthesia.livekit._types import CLEAR_BUFFER_SECS, RPC_ACK_SECS
from pipecat_synthesia.protocol import (
    AVATAR_ID_PREFIX,
    RPC_CLEAR_BUFFER,
    RPC_HANDLER_REJECT,
    RPC_HANDLER_RESPONSE,
    RPC_PLAYBACK_FINISHED,
    RPC_PLAYBACK_STARTED,
    RPC_SWAP_AVATAR,
    SWAP_DEFAULT,
    TOKEN_ATTR_PUBLISH_ON_BEHALF,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pipecat_synthesia.livekit._rtc_protocols import LocalParticipantLike


class PlaybackRpcHandlers:
    """Registers the handlers for the avatar worker's playback events on the bot's local participant.

    Both handlers are synchronous and return at once, running the callback in a callback task: the worker
    retries `lk.playback_finished` three times and then ends its session, so a handler must never block
    or raise.
    """

    def __init__(
        self,
        participant: LocalParticipantLike,
        *,
        expected_caller: str,
        on_started: Callable[[], Awaitable[None]],
        on_finished: Callable[[float, bool], Awaitable[None]],
        log,
    ) -> None:
        self._participant = participant
        self._expected_caller = expected_caller
        self._on_started_cb = on_started
        self._on_finished_cb = on_finished
        self._log = log
        self._registered = False
        self._callbacks = CallbackTasks("playback callback")

    def register(self) -> None:
        """Register both playback methods, started first. Idempotent.

        Both must exist before the first audio goes out: without `lk.playback_finished`, the worker ends
        its session after the first utterance.
        """
        if self._registered:
            return

        def _on_started(data: lk_rtc.RpcInvocationData) -> str:
            try:
                if data.caller_identity != self._expected_caller:
                    self._log.warning(
                        f"{RPC_PLAYBACK_STARTED} from unexpected identity {data.caller_identity!r}"
                    )
                    return RPC_HANDLER_REJECT
                self._callbacks.schedule(self._on_started_cb(), log=self._log)
            except Exception:
                # A raise here reaches the worker as APPLICATION_ERROR, which for `lk.playback_finished`
                # ends its session after three retries. Never propagate.
                self._log.exception("playback_started handler failed")
            return RPC_HANDLER_RESPONSE

        def _on_finished(data: lk_rtc.RpcInvocationData) -> str:
            try:
                if data.caller_identity != self._expected_caller:
                    self._log.warning(
                        f"{RPC_PLAYBACK_FINISHED} from unexpected identity {data.caller_identity!r}"
                    )
                    return RPC_HANDLER_REJECT
                payload = _parse_json_object(data.payload)
                pos = payload.get("playback_position", 0.0)
                # *tolerant reader*: the avatar worker's receiver has sent different keys across its
                # versions, and the payload gains keys over time (`synchronized_transcript`).
                if isinstance(pos, bool) or not isinstance(pos, int | float):
                    pos = 0.0
                self._callbacks.schedule(
                    self._on_finished_cb(float(pos), bool(payload.get("interrupted", False))), log=self._log
                )
            except Exception:
                self._log.exception("playback_finished handler failed")
            return RPC_HANDLER_RESPONSE

        # Positionally: rtc names the parameter `method_name`.
        self._participant.register_rpc_method(RPC_PLAYBACK_STARTED, _on_started)
        self._participant.register_rpc_method(RPC_PLAYBACK_FINISHED, _on_finished)
        self._registered = True
        self._log.debug(f"registered {RPC_PLAYBACK_STARTED} and {RPC_PLAYBACK_FINISHED}")

    def unregister(self) -> None:
        """Best-effort unregister of both methods. Idempotent."""
        if not self._registered:
            return
        self._registered = False
        for method in (RPC_PLAYBACK_STARTED, RPC_PLAYBACK_FINISHED):
            try:
                self._participant.unregister_rpc_method(method)
            except Exception as exc:
                self._log.debug(f"could not unregister {method}: {exc!r}")


def _parse_json_object(payload: str | None) -> dict[str, Any]:
    """Parse an RPC payload into a dict; anything else (blank, `null`, a list, junk) becomes `{}`."""
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def perform_clear_buffer(
    participant: LocalParticipantLike,
    *,
    avatar_identity: str,
    timeout: float = CLEAR_BUFFER_SECS,  # noqa: ASYNC109
    log,
) -> bool:
    """Tell the worker to drop the audio it has buffered. Returns whether it accepted; never raises.

    rtc waits up to `max_round_trip_latency` for the worker's acknowledgement, then up to `response_timeout`
    for its answer, so neither bounds the whole call, and it is wrapped in `wait_for`. rtc sends the worker's
    handler a deadline of `response_timeout - max_round_trip_latency`, at least 1 s: at the default timeout,
    the 1 s floor (see `CLEAR_BUFFER_SECS` and `RPC_ACK_SECS`).
    """
    try:
        resp = await asyncio.wait_for(
            participant.perform_rpc(
                destination_identity=avatar_identity,
                method=RPC_CLEAR_BUFFER,
                payload="",
                response_timeout=timeout,
                max_round_trip_latency=min(RPC_ACK_SECS, timeout / 2),
            ),
            timeout + 0.5,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # RpcError, asyncio.TimeoutError, anything
        log.warning(f"clear_buffer failed: {exc!r}")
        return False
    if resp == RPC_HANDLER_REJECT:
        log.warning(
            "clear_buffer rejected: the worker is bound to a different bot identity "
            f"({TOKEN_ATTR_PUBLISH_ON_BEHALF})"
        )
        return False
    return resp == RPC_HANDLER_RESPONSE


async def perform_swap_avatar(
    participant: LocalParticipantLike,
    *,
    avatar_identity: str,
    avatar_id: str,
    timeout: float,  # noqa: ASYNC109
) -> str:
    """Swap the rendered avatar, and return the avatar id the worker confirmed.

    Sends the unprefixed id, because the worker keys avatars that way and does not strip the prefix itself
    (*unprefixed ids on the wire*). `"default"` goes through as it is; the worker resolves it to the
    first of `avatarIds`. Raises `SynthesiaError`; the session stamps the session id on it.
    """
    raw = SWAP_DEFAULT if avatar_id == SWAP_DEFAULT else unprefixed_avatar_id(avatar_id)
    try:
        resp = await asyncio.wait_for(
            participant.perform_rpc(
                destination_identity=avatar_identity,
                method=RPC_SWAP_AVATAR,
                payload=json.dumps({"avatar_id": raw}),
                response_timeout=timeout,
                max_round_trip_latency=min(RPC_ACK_SECS, timeout / 2),
            ),
            # As in `perform_clear_buffer`, neither rtc timeout bounds the whole call, and the worker's
            # handler can block for up to 180 s while the target avatar prepares. rtc sends the handler a
            # deadline of `timeout` minus the acknowledgement allowance: `timeout - 2 s`, at least half of
            # `timeout` and at least 1 s (see `RPC_ACK_SECS`). rtc itself still waits the full `timeout`.
            timeout + 0.5,
        )
    except lk_rtc.RpcError as exc:
        raise error_from_rpc(exc, method=RPC_SWAP_AVATAR) from exc
    except TimeoutError as exc:
        raise SynthesiaError(f"{RPC_SWAP_AVATAR} timed out after {timeout}s", type=ErrorType.TIMEOUT) from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # `swap_avatar` is called by the developer, so it may raise only SynthesiaError
        # (*permanent-error rule*); anything else rtc raises is wrapped here.
        raise SynthesiaError(f"{RPC_SWAP_AVATAR} failed: {exc!r}", type=ErrorType.CONNECTION) from exc

    try:
        parsed = json.loads(resp)
    except ValueError as exc:
        raise _malformed_swap_response() from exc
    if not isinstance(parsed, dict):
        raise _malformed_swap_response()
    error = parsed.get("error")
    if error:
        raise SynthesiaError(
            f"avatar swap failed: {error}",
            type=ErrorType.AVATAR_SWAP_FAILED,
            code=parsed.get("code"),
        )
    confirmed = parsed.get("avatar_id")
    if not isinstance(confirmed, str) or not confirmed.strip():
        raise _malformed_swap_response()
    # The worker has swapped by now, so an id the avatar-id pattern would reject is still a success
    # (*tolerant reader*): it is only given the prefix every id inside the package carries.
    confirmed = confirmed.strip()
    return confirmed if confirmed.startswith(AVATAR_ID_PREFIX) else AVATAR_ID_PREFIX + confirmed


def _malformed_swap_response() -> SynthesiaError:
    return SynthesiaError("avatar swap returned a malformed response", type=ErrorType.AVATAR_SWAP_FAILED)


def error_from_rpc(exc: lk_rtc.RpcError, *, method: str) -> SynthesiaError:
    """Map an `rtc.RpcError` to a `SynthesiaError`, by its LiveKit RPC error code."""
    codes = lk_rtc.RpcError.ErrorCode
    code = str(exc.code)
    if exc.code == codes.UNSUPPORTED_METHOD:
        return SynthesiaError(f"worker does not support {method}", type=ErrorType.CONNECTION, code=code)
    if exc.code in (codes.RECIPIENT_NOT_FOUND, codes.RECIPIENT_DISCONNECTED):
        return SynthesiaError(f"avatar is not in the room ({method})", type=ErrorType.CONNECTION, code=code)
    if exc.code in (codes.RESPONSE_TIMEOUT, codes.CONNECTION_TIMEOUT):
        return SynthesiaError(f"{method} timed out", type=ErrorType.TIMEOUT, code=code)
    if exc.code == codes.APPLICATION_ERROR:
        return SynthesiaError(exc.message, type=ErrorType.AVATAR_SWAP_FAILED, code=code)
    return SynthesiaError(
        f"{method} failed with LiveKit RPC code {exc.code}: {exc.message}",
        type=ErrorType.CONNECTION,
        code=code,
    )
