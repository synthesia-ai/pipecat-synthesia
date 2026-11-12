"""Recorder for `AvatarCallbacks` and a `LaunchContext` builder, for tests below the service.

`tests/support/events.py` records the *service's* events; this records what a room or session delivers to
the service through `AvatarCallbacks`. Every callback appends and returns. The session runs each in its own
task (*state-before-await*), so "the callback fired" is asserted with `until()` on the recorded lists,
never on the line after the event that caused it.

One deliberate knob: `hold_finished`. When set, `on_playback_finished` blocks on it, which is how a test
proves that one slow callback does not delay another (there is no lock and no queue between callbacks).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from typing import Any, cast

from loguru import logger

from pipecat_synthesia._client_info import build_client_info
from pipecat_synthesia.room import AvatarCallbacks, LaunchContext

__all__ = ["CallbackRecorder", "make_launch_context", "make_relayed_launch_context"]


class CallbackRecorder:
    """Records every `AvatarCallbacks` delivery, in arrival order.

    Attributes:
        started: Number of `on_playback_started` deliveries.
        finished: `(playback_position, interrupted)` per `on_playback_finished`.
        lost: Number of `on_playback_lost` deliveries.
        status: `(status, all_ready)` per `on_avatar_status` (the dict is copied).
        left: `(reason, humans_present)` per `on_avatar_left`.
        tracks: The `kind` of every `on_track_published`.
        order: Every delivery's name in arrival order, for cross-callback ordering assertions.
        hold_finished: When set, `on_playback_finished` waits on it before recording.
    """

    def __init__(self) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []
        self.lost = 0
        self.status: list[tuple[dict[str, str], bool]] = []
        self.left: list[tuple[str, bool | None]] = []
        self.tracks: list[str] = []
        self.order: list[str] = []
        self.hold_finished: asyncio.Event | None = None

    async def on_playback_started(self) -> None:
        self.started += 1
        self.order.append("playback_started")

    async def on_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        if self.hold_finished is not None:
            await self.hold_finished.wait()
        self.finished.append((playback_position, interrupted))
        self.order.append("playback_finished")

    async def on_playback_lost(self) -> None:
        self.lost += 1
        self.order.append("playback_lost")

    async def on_avatar_status(self, status: dict[str, str], all_ready: bool) -> None:
        self.status.append((dict(status), all_ready))
        self.order.append("status")

    async def on_avatar_left(self, reason: str, humans_present: bool | None) -> None:
        self.left.append((reason, humans_present))
        self.order.append("left")

    async def on_track_published(self, kind: str) -> None:
        self.tracks.append(kind)
        self.order.append(f"track:{kind}")

    def as_callbacks(self) -> AvatarCallbacks:
        """The `AvatarCallbacks` a `LaunchContext` carries; media callbacks are `None` (direct join)."""
        return AvatarCallbacks(
            on_playback_started=self.on_playback_started,
            on_playback_finished=self.on_playback_finished,
            on_playback_lost=self.on_playback_lost,
            on_avatar_status=self.on_avatar_status,
            on_avatar_left=self.on_avatar_left,
            on_track_published=self.on_track_published,
            on_video_frame=None,
            on_audio_frame=None,
        )


def make_launch_context(
    api: Any,
    *,
    avatar_ids: Sequence[str] = ("av_a", "av_b"),
    launch_timeout: float = 5.0,
    audio_out_sample_rate: int = 24000,
    recorder: CallbackRecorder | None = None,
    log: Any = None,
) -> LaunchContext:
    """A `LaunchContext` for a test that drives a room/session directly (no service in the loop).

    `api` is anything with `create_session` (normally a `FakeSessionApiClient`); it is not type-checked
    because the context's field is the concrete `SessionApiClient` and tests deliberately pass a fake.
    """
    return LaunchContext(
        api=cast(Any, api),
        avatar_ids=tuple(avatar_ids),
        launch_timeout=launch_timeout,
        audio_out_sample_rate=audio_out_sample_rate,
        audio_out_channels=1,
        client=build_client_info(),
        callbacks=(recorder or CallbackRecorder()).as_callbacks(),
        log=log if log is not None else logger.bind(component="test"),
    )


async def _drop_media(data: bytes, first: int, second: int) -> None:
    """A relayed join's media callback that drops what it is given."""


def make_relayed_launch_context(api: Any, **kwargs: Any) -> LaunchContext:
    """`make_launch_context` plus the two media callbacks a relayed join requires, which drop the media."""
    ctx = make_launch_context(api, **kwargs)
    callbacks = dataclasses.replace(ctx.callbacks, on_video_frame=_drop_media, on_audio_frame=_drop_media)
    return dataclasses.replace(ctx, callbacks=callbacks)
