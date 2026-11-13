"""Per-avatar readiness, read from the avatar worker's participant attributes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence

from pipecat_synthesia.config import normalize_avatar_id
from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.protocol import (
    ATTR_AVATAR_READY,
    ATTR_AVATAR_STATUS,
    AVATAR_STATE_FAILED,
    AVATAR_STATE_READY,
    SWAP_ERROR_CODE_AVATAR_FAILED,
)


class AvatarReadiness:
    """Readiness of the session avatars, fed by the avatar worker's attribute updates (*tolerant reader*).

    The worker keys its attributes by unprefixed id, so keys are normalised on the way in (*unprefixed
    ids on the wire*). `avatar_ready` is only ever `"true"` or absent, never `"false"`. An `avatar_status`
    state other than the three known ones is kept as it is and read as "not ready yet", so a state a newer
    worker adds is never mistaken for a failure.
    """

    def __init__(self, avatar_ids: Sequence[str], log) -> None:
        self._log = log
        self._known = set(avatar_ids)  # the session avatars, already normalised
        self._status: dict[str, str] = {}
        self._event = asyncio.Event()
        self._warned_malformed = False

    def update(self, attributes: Mapping[str, str]) -> bool:
        """Fold one attribute map into the state; returns whether anything changed."""
        changed = False
        raw_status = attributes.get(ATTR_AVATAR_STATUS)
        if raw_status:
            parsed: object
            try:
                parsed = json.loads(raw_status)
            except ValueError:
                parsed = None
            if not isinstance(parsed, dict):
                if not self._warned_malformed:
                    self._warned_malformed = True
                    self._log.warning(
                        f"ignoring a malformed {ATTR_AVATAR_STATUS} attribute from the avatar worker"
                    )
            else:
                for key, value in parsed.items():
                    try:
                        prefixed = normalize_avatar_id(key)
                    except ValueError:
                        continue
                    # A bare value where the status object should be is kept as text: it is not one of
                    # the known states, so it reads as "not ready yet" and stays visible for diagnostics.
                    state = value.get("state") if isinstance(value, dict) else str(value)
                    if not isinstance(state, str) or not state:
                        continue
                    if self._status.get(prefixed) != state:
                        self._status[prefixed] = state
                        changed = True

        if attributes.get(ATTR_AVATAR_READY) == "true":
            for prefixed in self._known:
                if self._status.get(prefixed) not in (AVATAR_STATE_FAILED, AVATAR_STATE_READY):
                    self._status[prefixed] = AVATAR_STATE_READY
                    changed = True

        if changed:
            self._event.set()
        return changed

    @property
    def status(self) -> dict[str, str]:
        return dict(self._status)

    @property
    def all_ready(self) -> bool:
        return bool(self._known) and all(
            self._status.get(prefixed) == AVATAR_STATE_READY for prefixed in self._known
        )

    def state_of(self, avatar_id: str) -> str | None:
        return self._status.get(avatar_id)

    async def wait_ready(self, avatar_id: str, *, timeout: float) -> None:  # noqa: ASYNC109
        """Wait until `avatar_id` is `ready`; raise if it failed, is unknown, or the wait ran out.

        `"default"` is resolved by the caller (`LiveKitAvatarSession` maps it to the first of `avatar_ids`),
        so this only ever sees an avatar id.
        """
        if avatar_id not in self._known:
            raise SynthesiaError(
                f"avatar {avatar_id} is not part of this session", type=ErrorType.UNKNOWN_AVATAR
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            state = self._status.get(avatar_id)
            if state == AVATAR_STATE_READY:
                return
            if state == AVATAR_STATE_FAILED:
                raise SynthesiaError(
                    f"avatar {avatar_id} failed to prepare",
                    type=ErrorType.AVATAR_SWAP_FAILED,
                    code=SWAP_ERROR_CODE_AVATAR_FAILED,
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise SynthesiaError(
                    f"avatar {avatar_id} is still preparing after {timeout}s", type=ErrorType.TIMEOUT
                )
            # The event is the "some update landed" signal, not "this avatar is ready": clear it and
            # re-read the state, so a change to another avatar simply costs one extra loop.
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), remaining)
            except TimeoutError:
                continue
