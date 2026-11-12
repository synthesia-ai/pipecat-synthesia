"""The avatar audio filter: in a direct join, drops the avatar's audio on the bot's input path.

It is the half of the *self-hearing guard* that the developer places, right after `transport.input()`,
because the service sits after TTS and never sees input frames. Pipecat's LiveKit transport subscribes
to every participant and filters none, so the avatar's audio reaches STT. The automatic half, the avatar
unsubscribe, cannot act until the avatar's audio track exists; this filter catches whatever arrives
before then. `setup()` reports back through `on_setup`, which is how the service warns when the filter
was never placed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.frames.frames import UserAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor

from pipecat_synthesia._log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipecat.frames.frames import Frame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup


class AvatarAudioFilter(FrameProcessor):
    """Input-side filter that keeps the bot from hearing the avatar it is driving."""

    def __init__(self, *, avatar_identity: str, on_setup: Callable[[], None] | None = None, **kwargs) -> None:
        """Initialise the filter.

        Args:
            avatar_identity: The LiveKit identity whose `UserAudioRawFrame`s are dropped. Pipecat's
                `user_id` is the participant **identity**, not the sid.
            on_setup: Called from `setup()`; how `LiveKitTransportRoom.filter_installed` learns that
                the filter reached a pipeline.
            **kwargs: Forwarded to `FrameProcessor` (e.g. `name`).
        """
        super().__init__(**kwargs)
        self._avatar_identity = avatar_identity
        self._on_setup = on_setup
        self._dropped = 0
        self._log = get_logger(processor=self.name, avatar=avatar_identity)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        if self._on_setup is not None:
            self._on_setup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserAudioRawFrame) and frame.user_id == self._avatar_identity:
            self._dropped += 1
            if self._dropped == 1:
                # Per-instance, not `_once`: "the filter is working" is worth one line per pipeline,
                # and the rest are per-frame, hence TRACE.
                self._log.debug("dropping the avatar's own audio on the bot's input path")
            else:
                self._log.trace("dropped avatar audio frame {n}", n=self._dropped)
            return
        await self.push_frame(frame, direction)
