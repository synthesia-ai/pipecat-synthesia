"""Pipecat version facts, and one of the package's two Pipecat private reads (*Pipecat-internals rule*)."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from pipecat_synthesia.protocol import TRANSPORT_OTHER, TRANSPORT_PACKAGES, TRANSPORT_UNKNOWN

if TYPE_CHECKING:
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams

TESTED_PIPECAT_VERSIONS = ("1.14.0", "1.15.0", "1.16.0")
PIPECAT_VERSION: str = importlib.metadata.version("pipecat-ai")


def _minor(version: str) -> tuple[int, int] | None:
    """``(major, minor)`` of ``version`` (``"1.17.0.dev3"`` is ``(1, 17)``); None if it does not parse."""
    try:
        release = Version(version).release
    except InvalidVersion:
        return None
    return (release[0], release[1] if len(release) > 1 else 0)


def untested_newer_pipecat() -> str | None:
    """The installed ``pipecat-ai`` version if it is a newer minor than every tested one, else None.

    A patch of a tested minor (``1.16.3``) and an older minor than the newest tested one (``1.15.0``) are
    not newer; a pre-release or dev build of a newer minor (``1.17.0rc1``, ``1.17.0.dev3``) is. Reads
    ``PIPECAT_VERSION`` at call time, so a test can set it.
    """
    installed = _minor(PIPECAT_VERSION)
    newest = max(m for v in TESTED_PIPECAT_VERSIONS if (m := _minor(v)) is not None)
    return PIPECAT_VERSION if installed is not None and installed > newest else None


_MAX_HOPS = 64


def _output_transport(processor) -> BaseOutputTransport | None:
    """The first output transport downstream of ``processor``, or None; never raises.

    Follows the public ``next`` links, which a pipeline sets before any processor's ``setup()``.
    ``BaseOutputTransport`` is imported lazily, so importing this package never imports a transport module.
    """
    try:
        from pipecat.transports.base_output import BaseOutputTransport
    except Exception:
        return None
    try:
        current = processor
        for _ in range(_MAX_HOPS):
            # `next` is a plain property on FrameProcessor at pipecat 1.14 to 1.16, not a method.
            current = getattr(current, "next", None)
            if current is None:
                return None
            if isinstance(current, BaseOutputTransport):
                return current
    except Exception:
        return None
    return None


def output_transport_params(processor) -> TransportParams | None:
    """The ``TransportParams`` of the first output transport downstream of ``processor``, or None.

    Reads the transport's private ``_params``. Used only for the one-time warning about the transport's
    ``video_out_*`` settings.
    """
    return getattr(_output_transport(processor), "_params", None)


def transport_name(processor) -> str:
    """The user transport downstream of ``processor``, as client identification reports it.

    The first class in the output transport's MRO whose module is in a package under ``pipecat.transports``
    names it, so a developer's subclass of a Pipecat transport reads as that transport's package. A package
    outside ``TRANSPORT_PACKAGES``, or a transport from no Pipecat package, is ``"other"``; no output
    transport downstream is ``"unknown"``.
    """
    transport = _output_transport(processor)
    if transport is None:
        return TRANSPORT_UNKNOWN
    prefix = "pipecat.transports."
    for cls in type(transport).__mro__:
        module = getattr(cls, "__module__", None) or ""
        package, dot, _ = module.removeprefix(prefix).partition(".")
        if module.startswith(prefix) and dot:  # not a base module such as pipecat.transports.base_output
            return package if package in TRANSPORT_PACKAGES else TRANSPORT_OTHER
    return TRANSPORT_OTHER
