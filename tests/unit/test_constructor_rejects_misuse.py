"""The constructor refuses an argument it would otherwise drop, and a `room` that is not a room.

What this pins: a keyword that neither `SynthesiaVideoService` nor Pipecat's constructor chain declares is a
`TypeError` naming it, never its value, with a pointer to the right argument for the two most likely
mistakes (`join_timeout` from the LiveKit plugin, `avatar_id` for `avatar_ids`); every keyword Pipecat's
chain declares at 1.14 to 1.16 still reaches Pipecat; and `room=None`, or anything that is not an
`AvatarRoom`, is a `TypeError` naming both room classes. All of them are misuse errors, raised at
construction rather than as a `SynthesiaError` minutes into a call (*misuse vs session errors*).
"""

from __future__ import annotations

from typing import Any

import pytest
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics

from pipecat_synthesia import JoinMode, SynthesiaError, SynthesiaVideoService
from tests.fakes.fake_room import FakeAvatarRoom
from tests.support.secrets import API_KEY, LK_API_SECRET, assert_no_secrets

pytestmark = pytest.mark.unit


def _construct(**kwargs: Any) -> SynthesiaVideoService:
    return SynthesiaVideoService(
        avatar_ids=["av_a"], room=FakeAvatarRoom(JoinMode.DIRECT), api_key=API_KEY, **kwargs
    )


@pytest.mark.parametrize(
    ("keyword", "hint"),
    [
        ("join_timeout", "use launch_timeout, which bounds the whole launch"),
        ("avatar_id", "use avatar_ids"),
    ],
)
def test_a_likely_mistaken_keyword_names_the_argument_to_use(keyword: str, hint: str) -> None:
    with pytest.raises(TypeError) as caught:
        _construct(**{keyword: 5})

    message = str(caught.value)
    assert repr(keyword) in message
    assert hint in message


def test_an_unknown_keyword_is_named_but_its_value_never_echoed() -> None:
    """A secret passed under the wrong name stays out of the message (*redaction rule*)."""
    with pytest.raises(TypeError) as caught:
        _construct(api_secret=LK_API_SECRET)

    assert "'api_secret'" in str(caught.value)
    assert_no_secrets(str(caught.value), repr(caught.value))


def test_every_unknown_keyword_is_named_at_once() -> None:
    with pytest.raises(TypeError) as caught:
        _construct(join_timeout=5, avatar_id="av_b", colour="blue")

    message = str(caught.value)
    for keyword in ("join_timeout", "avatar_id", "colour"):
        assert repr(keyword) in message
    assert "launch_timeout" in message and "avatar_ids" in message


def test_the_keywords_pipecat_declares_still_reach_pipecat() -> None:
    """`name`, `enable_direct_mode`, `metrics` and `task_manager` are declared by Pipecat's
    `FrameProcessor` and `BaseObject` at 1.14 to 1.16; none of them is refused."""
    metrics = FrameProcessorMetrics()
    service = _construct(name="avatar", enable_direct_mode=False, metrics=metrics, task_manager=None)

    assert service.name == "avatar"


@pytest.mark.parametrize("room", [None, object(), "room"], ids=["none", "object", "string"])
def test_a_room_that_is_not_an_avatar_room_names_both_room_classes(room: object) -> None:
    with pytest.raises(TypeError) as caught:
        SynthesiaVideoService(avatar_ids=["av_a"], room=room, api_key=API_KEY)  # type: ignore[arg-type]

    message = str(caught.value)
    assert not isinstance(caught.value, SynthesiaError)
    assert "LiveKitTransportRoom" in message and "direct join" in message
    assert "PrivateLiveKitRoom" in message and "relayed join" in message
