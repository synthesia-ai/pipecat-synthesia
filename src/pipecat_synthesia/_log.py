"""The package logger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import loguru


def get_logger(**fields) -> loguru.Logger:
    """Return the package logger bound with ``component`` plus the given fields.

    Per-frame logs use ``.trace()`` so the default INFO/DEBUG sinks stay readable.
    """
    return logger.bind(component="pipecat_synthesia", **fields)
