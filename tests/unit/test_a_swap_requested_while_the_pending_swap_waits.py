"""A swap requested while the launch's pending swap is still waiting for its avatar to be ready.

An `avatar_id` change received during the launch, or a constructor `settings.avatar_id` other than the
first avatar id, is a pending swap, applied once the session is active. A swap to an avatar that is still
preparing waits for it to be ready before the avatar worker is asked. So right after the session becomes
active, the pending swap can still be waiting when the developer asks for another swap. This file pins
the outcome: the avatar asked for last is the active avatar at the end, at the avatar worker and in
`active_avatar_id`, and the two agree.

Here the pending swap is to `av_b`, still preparing. The later request is `swap_avatar("av_c")`, as from
an `on_avatar_connected` handler, to an avatar that is already ready; a settings change on the pipeline
takes the same path. Then `av_b` becomes ready.

It needs the real `LiveKitAvatarSession`, whose readiness wait holds the pending swap, on the session rig
(`_session_rig.py`) behind a direct-join `AvatarRoom`, as in `test_service_with_a_leaving_livekit_session.py`.
`_SwappingWorker` answers `swapAvatar` for the avatar worker: it swaps to the avatar it is asked for, at
once, in the order the requests arrive, and confirms it. 0.2 s is the "and nothing more happened" wait
after `av_b` becomes ready: the fake answers without waiting, so a swap still to come has long gone out.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from pipecat_synthesia.frames import SynthesiaVideoSettings
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.protocol import ATTR_AVATAR_READY, ATTR_AVATAR_STATUS, RPC_SWAP_AVATAR
from pipecat_synthesia.room import AvatarRoom, AvatarSession, JoinMode, LaunchContext
from tests.fakes.fake_rtc import FakeLocalParticipant
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.waiting import until
from tests.unit._session_rig import AVATAR, BOT, Rig, settle, status_attr

SETTLE = 0.2
B_PREPARING = {ATTR_AVATAR_STATUS: status_attr(a="ready", b="preparing", c="ready")}
ALL_READY = {ATTR_AVATAR_STATUS: status_attr(a="ready", b="ready", c="ready"), ATTR_AVATAR_READY: "true"}


class _SwappingWorker(FakeLocalParticipant):
    """The bot's participant, with the avatar worker's side of `swapAvatar`: swap at once and confirm."""

    def __init__(self) -> None:
        super().__init__(BOT)
        #: The unprefixed avatar ids the avatar worker swapped to, in the order it was asked.
        self.swapped_to: list[str] = []

    async def perform_rpc(
        self,
        *,
        destination_identity: str,
        method: str,
        payload: str,
        response_timeout: float | None = None,
        max_round_trip_latency: float | None = None,
    ) -> str:
        if method != RPC_SWAP_AVATAR:
            return await super().perform_rpc(
                destination_identity=destination_identity,
                method=method,
                payload=payload,
                response_timeout=response_timeout,
                max_round_trip_latency=max_round_trip_latency,
            )
        requested = json.loads(payload)["avatar_id"]
        self.swapped_to.append(requested)
        return json.dumps({"avatar_id": requested})


class _SessionRigRoom(AvatarRoom):
    """A direct join whose `launch()` is a real `LiveKitAvatarSession` on the session rig."""

    join_mode = JoinMode.DIRECT

    def __init__(self, rig: Rig) -> None:
        self._rig = rig

    @property
    def avatar_identity(self) -> str:
        return AVATAR

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        rig = self._rig
        return await LiveKitAvatarSession.launch(
            rig.handle,
            token_source=rig.tokens,
            avatar_identity=AVATAR,
            avatar_name=rig.avatar_name,
            # the service's callbacks and avatar ids, the rig's fake Synthesia API
            ctx=dataclasses.replace(ctx, api=rig.api),
        )


@pytest.mark.unit
async def test_the_avatar_asked_for_last_is_active_after_a_pending_swap_that_waited() -> None:
    worker = _SwappingWorker()
    rig = Rig(participant=worker, attributes=B_PREPARING)
    service = make_service(
        _SessionRigRoom(rig),
        avatar_ids=("av_a", "av_b", "av_c"),
        settings=SynthesiaVideoSettings(avatar_id="av_b"),  # a pending swap from the start
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:
        await rec.wait("on_avatar_connected")
        assert worker.swapped_to == [], "the pending swap to av_b went ahead while av_b was preparing"

        swap_to_c = asyncio.ensure_future(service.swap_avatar("av_c"))
        try:
            await asyncio.sleep(0)  # one loop turn: the swap to av_c is asked for
            assert rig.avatar is not None
            rig.change_attributes(rig.avatar, ALL_READY)
            await until(swap_to_c.done, what="the swap to av_c")
            await rec.wait("on_avatar_ready")
            await h.sleep(SETTLE)
        finally:
            await settle(swap_to_c)
        active = service.active_avatar_id
        await h.finish()

    assert swap_to_c.result() == "av_c"
    assert worker.swapped_to, "no swap reached the avatar worker"
    assert active == "av_" + worker.swapped_to[-1], "the service and the avatar worker disagree"
    assert active == "av_c", (
        f"the avatar asked for last was undone: the avatar worker swapped to {worker.swapped_to}"
    )
