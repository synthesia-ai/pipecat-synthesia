"""`AvatarAudioFilter` and `filter_installed`: the frame half of the *self-hearing guard*.

The processor runs in a real `Pipeline` under a real `PipelineWorker` (`tests/harness.py`), so
"forwarded unchanged in its direction" is asserted where Pipecat would actually drop it. Frame lists are
filtered rather than compared exactly, so a new frame type does not break these tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    UserAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia.livekit._types import LiveKitCredentials
from pipecat_synthesia.livekit.audio_filter import AvatarAudioFilter
from pipecat_synthesia.livekit.token import DEFAULT_AVATAR_IDENTITY
from pipecat_synthesia.livekit.transport_room import LiveKitTransportRoom
from tests.fakes.fake_transport import FakeLiveKitTransport
from tests.harness import bidirectional_pipeline, run_bidirectional
from tests.support.frames import assert_absent, count, frames_of
from tests.support.secrets import LK_API_SECRET

AVATAR = DEFAULT_AVATAR_IDENTITY
HUMAN = "human-1"
CREDENTIALS = LiveKitCredentials(api_key="fake-lk-key", api_secret=LK_API_SECRET)
DOWN = FrameDirection.DOWNSTREAM
UP = FrameDirection.UPSTREAM
CHUNK = b"\x01\x02" * 160  # 20 ms of 16 kHz mono s16le


def audio_from(user_id: str) -> UserAudioRawFrame:
    """What the LiveKit input transport pushes for a subscribed participant."""
    return UserAudioRawFrame(user_id=user_id, audio=CHUNK, sample_rate=16000, num_channels=1)


def avatar_audio(frames: Sequence[Frame]) -> list[UserAudioRawFrame]:
    """Every `UserAudioRawFrame` in `frames` that the avatar produced."""
    audio = [cast(UserAudioRawFrame, frame) for frame in frames_of(frames, UserAudioRawFrame)]
    return [frame for frame in audio if frame.user_id == AVATAR]


def audio_users(frames: Sequence[Frame]) -> list[str]:
    """The `user_id` of every `UserAudioRawFrame` in `frames`, in arrival order."""
    return [cast(UserAudioRawFrame, frame).user_id for frame in frames_of(frames, UserAudioRawFrame)]


@pytest.mark.unit
async def test_filter_drops_only_avatar_audio() -> None:
    """The avatar's own audio never reaches STT; the human's audio, text and interruptions do."""
    processor = AvatarAudioFilter(avatar_identity=AVATAR)

    down, _up = await run_bidirectional(
        processor,
        [
            (DOWN, audio_from(AVATAR)),
            (DOWN, audio_from(HUMAN)),
            (DOWN, audio_from(AVATAR)),
            (DOWN, TextFrame("hello")),
            (DOWN, InterruptionFrame()),
        ],
    )

    assert audio_users(down) == [HUMAN], "the avatar's audio reached the bot's input path"
    assert count(down, UserAudioRawFrame) == 1, "both avatar frames must be gone, the human's kept"


@pytest.mark.unit
async def test_filter_passes_other_users_and_all_non_audio_frames() -> None:
    """Everything that is not the avatar's audio passes through untouched, including the frame types a
    pipeline cannot work without."""
    processor = AvatarAudioFilter(avatar_identity=AVATAR)
    human = audio_from(HUMAN)
    text = TextFrame("hello")

    down, _up = await run_bidirectional(
        processor,
        [(DOWN, human), (DOWN, text), (DOWN, InterruptionFrame()), (DOWN, audio_from(AVATAR))],
    )

    assert human in down, "a human's audio frame was not forwarded"
    assert text in down, "a TextFrame was not forwarded"
    assert count(down, InterruptionFrame) == 1
    assert_absent(avatar_audio(down), UserAudioRawFrame, where="downstream of the filter")


@pytest.mark.unit
async def test_filter_forwards_upstream_frames_upstream() -> None:
    """Upstream frames are forwarded upstream: the half a downstream-only test cannot see.

    The filter sits right after `transport.input()`, on the return path of the whole pipeline.
    `push_error` sends `ErrorFrame` upstream, which is how the `PipelineWorker`'s
    `processor_unusable_policy` hears that a downstream service failed. A filter that dropped the
    `direction` argument would reflect those frames back *down* the pipeline, and every downstream-only
    test in this file would still pass."""
    processor = AvatarAudioFilter(avatar_identity=AVATAR)
    upstream_text = TextFrame("from downstream")
    upstream_error = ErrorFrame("stt exploded")
    human = audio_from(HUMAN)

    down, up = await run_bidirectional(
        processor,
        [(UP, upstream_text), (UP, upstream_error), (DOWN, human), (DOWN, audio_from(AVATAR))],
    )

    assert upstream_text in up, "an upstream TextFrame did not reach the head of the pipeline"
    assert upstream_error in up, "an upstream ErrorFrame did not reach the head of the pipeline"
    assert upstream_text not in down, "an upstream frame was reflected downstream"
    assert upstream_error not in down, "an upstream frame was reflected downstream"
    # The downstream contract still holds in the same run.
    assert audio_users(down) == [HUMAN]


@pytest.mark.unit
async def test_filter_only_drops_its_own_avatar_identity() -> None:
    """Two avatars in one room: a second plugin's identity is not this filter's business."""
    processor = AvatarAudioFilter(avatar_identity="avatar-a")

    down, _up = await run_bidirectional(
        processor, [(DOWN, audio_from("avatar-a")), (DOWN, audio_from("avatar-b"))]
    )

    assert audio_users(down) == ["avatar-b"]


@pytest.mark.unit
async def test_filter_installed_flag() -> None:
    """The developer places the filter, so the room reports whether it ever reached a pipeline: that flag
    is what lets `service.start()` warn about a missing filter (*self-hearing guard*)."""
    transport = FakeLiveKitTransport(participant_id="pipecat-bot")
    room = LiveKitTransportRoom(cast(Any, transport), credentials=CREDENTIALS)

    assert room.filter_installed is False, "nothing has been set up yet"
    processor = room.input_filter()
    assert room.input_filter() is processor, "input_filter() must return the same instance"
    assert room.filter_installed is False, "constructing the filter is not installing it"

    async with bidirectional_pipeline(processor) as harness:
        assert room.filter_installed is True, "setup() did not report back to the room"
        await harness.queue(audio_from(AVATAR))

    assert_absent(harness.down, UserAudioRawFrame, where="downstream of an installed filter")
