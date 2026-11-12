"""Relayed join: the avatar's video and audio keep reaching the pipeline across a same-identity replacement.

A launch retried with the same avatar token starts a second avatar worker on the same identity. When it
joins, livekit-server removes the worker that was serving the session, and the session carries on: the
*replacement grace* treats the two as one avatar (`test_relayed_join.py` pins that half, with no leave
reported and the next utterance at the surviving worker). This pins the half the user sees: track capture
moves to the new avatar participant's tracks, so its video and audio still reach the user transport.

The retried launch is the session's own request sent again through `launcher.hook()`, as in
`test_same_identity_replacement.py`, so the second worker joins at a moment the test chooses: once media is
flowing from the first. The hook returns once the second worker's tracks are published.

Frames counted after the server has dropped the first worker can still include a few of its own, already
buffered in the bot: at most `VIDEO_STREAM_CAPACITY` video and `AUDIO_STREAM_CAPACITY` audio frames, plus
the one each consumer is delivering. So the test waits for more than twice that, which only the
replacement's tracks can supply. `REPLACEMENT_GRACE_SECS` is raised, as in the other replacement tests,
because in this tier it bounds the new worker's real connect and publish.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import OutputAudioRawFrame, OutputImageRawFrame, OutputTransportReadyFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat_synthesia.livekit import _types
from pipecat_synthesia.livekit.track_capture import AUDIO_STREAM_CAPACITY, VIDEO_STREAM_CAPACITY
from pipecat_synthesia.service import ServiceState
from tests.harness import bidirectional_pipeline, make_service
from tests.integration._join_modes import AVATAR_IDS, WAIT, private_room
from tests.integration._launcher import Launcher
from tests.support.events import EventRecorder
from tests.support.frames import frames_of
from tests.support.livekit_dev import LiveKitDev
from tests.support.waiting import until

UP = FrameDirection.UPSTREAM
#: The *replacement grace*, raised from the real 1 s; see the module docstring.
GRACE = 3.0
#: More frames than the first worker can still have buffered in the bot; see the module docstring.
FRESH_VIDEO = 2 * VIDEO_STREAM_CAPACITY + 1
FRESH_AUDIO = 2 * AUDIO_STREAM_CAPACITY + 1


@pytest.mark.integration_extended
async def test_avatar_media_keeps_flowing_after_a_same_identity_replacement(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the replacement, the new worker's video and audio reach the pipeline; no leave is reported."""
    monkeypatch.setattr(_types, "REPLACEMENT_GRACE_SECS", GRACE)
    dev = livekit_server
    api = await launcher.api()
    service = make_service(
        private_room(dev, room_name=room_name), avatar_ids=AVATAR_IDS, api_url=api.base_url
    )
    rec = EventRecorder(service)

    async with bidirectional_pipeline(service) as h:

        def video() -> int:
            return len(frames_of(h.down, OutputImageRawFrame))

        def audio() -> int:
            # Exactly `OutputAudioRawFrame`: the relayed avatar audio, not TTS audio passing through.
            return sum(1 for frame in h.down if type(frame) is OutputAudioRawFrame)

        await rec.wait("on_avatar_connected", timeout=WAIT)
        (first,) = launcher.fakes
        await h.queue(OutputTransportReadyFrame(), UP)
        await h.frame(OutputTransportReadyFrame, direction=UP, timeout=WAIT)
        await until(
            lambda: video() > 0 and audio() > 0, timeout=WAIT, what="the first worker's video and audio"
        )

        # The retried launch: the same request, so the same token and the same avatar identity.
        await launcher.hook()(api.requests[0])
        await until(
            lambda: first.connected is False, timeout=WAIT, what="the server to drop the first worker"
        )
        video_before, audio_before = video(), audio()

        try:
            await until(
                lambda: video() >= video_before + FRESH_VIDEO and audio() >= audio_before + FRESH_AUDIO,
                timeout=WAIT,
                what="the replacement's video and audio in the pipeline",
            )
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}: video frames {video_before} -> {video()}, audio frames {audio_before} -> {audio()}; "
                f"on_avatar_disconnected={rec.count('on_avatar_disconnected')}, state={service.state.name}\n"
                + launcher.debug_dump()
            ) from None

        assert rec.count("on_avatar_disconnected") == 0, launcher.debug_dump()
        assert service.state is ServiceState.SESSION_ACTIVE, launcher.debug_dump()
        await h.finish()
