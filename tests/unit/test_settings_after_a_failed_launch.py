"""A settings change after a failed `required=True` launch does not revive the service.

Pipecat's own settings handling marks a service usable again on any change, because new settings may be
the fix. After a failed launch with `required=True` they are not: the avatar never renders, the error stays
permanent and the service ends the pipeline (*permanent-error rule*). What this file pins: whatever the
change carries (a model, with or without an `avatar_id`, as a delta or as the deprecated settings dict),
the service stays unusable, `on_usable_changed` fires only for the failure, no swap is attempted, nothing
more is reported, and the frame is consumed. `test_service_events.py` covers a change carrying only an
`avatar_id`.

The settings frame is queued from `on_error`, which Pipecat awaits before it pushes the `ErrorFrame`, so
it enters the pipeline ahead of the `EndFrame` the failure leads to; `hold_launch` keeps the launch in
progress until that handler is registered.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import ErrorFrame, TextFrame

from pipecat_synthesia.errors import ErrorType, SynthesiaError
from pipecat_synthesia.frames import SynthesiaVideoSettings, SynthesiaVideoUpdateSettingsFrame
from pipecat_synthesia.room import JoinMode
from pipecat_synthesia.service import ServiceState
from tests.fakes.fake_room import FakeAvatarRoom
from tests.harness import bidirectional_pipeline, make_service
from tests.support.events import EventRecorder
from tests.support.frames import assert_absent, count


@pytest.mark.unit
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda: SynthesiaVideoUpdateSettingsFrame(delta=SynthesiaVideoSettings(model="model-2")),
            id="model-only",
        ),
        pytest.param(
            lambda: SynthesiaVideoUpdateSettingsFrame(
                delta=SynthesiaVideoSettings(model="model-2", avatar_id="av_b")
            ),
            id="model-and-avatar-id",
        ),
        pytest.param(
            lambda: SynthesiaVideoUpdateSettingsFrame(settings={"model": "model-2"}),
            id="deprecated-dict",
        ),
    ],
)
async def test_a_settings_change_after_a_required_launch_failure_leaves_the_service_unusable(change) -> None:
    gate = asyncio.Event()
    room = FakeAvatarRoom(
        JoinMode.DIRECT,
        fail_with=SynthesiaError("bad key", type=ErrorType.AUTH, status=401),
        hold_launch=gate,
    )
    service = make_service(room, avatar_ids=("av_a", "av_b"), required=True)
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:

        @service.event_handler("on_error")
        async def _settings_after_failure(_service, _error) -> None:
            await h.queue(change())
            await h.queue(TextFrame("marker"))

        gate.set()
        await h.frame(TextFrame)  # the settings frame ahead of it has been processed
        assert service.state is ServiceState.FAILED
        assert service.is_usable is False
        assert service.active_avatar_id == "av_a"
        await h.finish()

    assert service.is_usable is False
    assert rec.count("on_usable_changed") == 1
    assert rec.payload("on_usable_changed")[0] is False
    assert room.session.swap_calls == []
    assert count(h.up, ErrorFrame) == 1, "the settings change in FAILED must report nothing more"
    assert_absent(h.down, SynthesiaVideoUpdateSettingsFrame)
