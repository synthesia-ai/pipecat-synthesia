"""Fixtures for the integration suite: a real ``livekit-server``, a room that cleans itself up, and a
``Launcher`` that turns a scripted Synthesia 201 into a fake worker joining that room.

The livekit-server the integration suite talks to is started with ``DOCKER_RUN`` below: by this
fixture, by hand with the justfile's `livekit-up`, and by `.github/actions/livekit-dev/action.yml` in CI.
The three MUST stay byte-identical (`tests/unit/test_structure.py` checks).

Port 7880 carries only signalling. Media goes over 7881/tcp and 7882/udp (the `--dev` defaults), so both
are published too, and `--node-ip 127.0.0.1` makes the server offer the host's loopback address in its ICE
candidates, where those published ports are, instead of the container's own address, which the host
cannot reach. With only 7880 published, joining a room works but media can stall, and integration tests
fail at random.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator

import pytest

from pipecat_synthesia.livekit import token as livekit_token
from tests.integration._launcher import Launcher
from tests.support.livekit_dev import LiveKitDev, delete_room, unique_room_name

#: Byte-identical to `.github/actions/livekit-dev/action.yml`, which owns it. If you change one, change
#: the other in the same commit: a fixture that starts a different server from the one CI starts turns
#: "green locally, red in CI" into a protocol mystery.
# keep in sync with .github/actions/livekit-dev/action.yml and the justfile's `livekit-up`
DOCKER_RUN = (
    "docker run -d --rm --name livekit -p 7880:7880 -p 7881:7881/tcp -p 7882:7882/udp "
    "livekit/livekit-server:v1.9.1 --dev --bind 0.0.0.0 --node-ip 127.0.0.1"
)
CONTAINER_NAME = "livekit"

DEFAULT_HTTP_URL = "http://127.0.0.1:7880"
DEFAULT_WS_URL = "ws://127.0.0.1:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "secret"
#: The media port ``DOCKER_RUN`` publishes over TCP. Only checked on a server this fixture reuses, since
#: one started before the media ports were published still answers on 7880. Its UDP sibling, 7882, cannot
#: be checked without a server that replies, so TCP stands in for both.
MEDIA_TCP_ADDRESS = ("127.0.0.1", 7881)

READY_TIMEOUT_SECS = 20.0
READY_POLL_SECS = 0.5
#: `docker run` also *pulls* the image when it is not cached, which on a first-run laptop is minutes of
#: silence. Generous rather than tight, because a timeout here aborts a legitimate first pull; what it buys
#: is that a wedged daemon fails with a message instead of hanging until pytest's own `--timeout` kills the
#: run with no explanation.
DOCKER_RUN_TIMEOUT_SECS = 300.0


def _answers(http_url: str, *, timeout: float = 1.0) -> bool:
    """True if something is serving HTTP at ``http_url``.

    livekit-server answers ``GET /`` with ``200 OK`` (body ``OK``); the check is deliberately "anything
    answered", including a 4xx, because the point is "the port is a live HTTP server", not "this exact
    route exists" — a future server version that moves the health route must not make every integration
    test skip.
    """
    try:
        with urllib.request.urlopen(http_url, timeout=timeout):  # noqa: S310 - fixed localhost URL
            return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def _wait_ready(http_url: str) -> bool:
    """Poll ``http_url`` until it answers or ``READY_TIMEOUT_SECS`` elapses."""
    deadline = time.monotonic() + READY_TIMEOUT_SECS
    while time.monotonic() < deadline:
        if _answers(http_url):
            return True
        time.sleep(READY_POLL_SECS)
    return False


def _accepts_tcp(address: tuple[str, int], *, timeout: float = 1.0) -> bool:
    """True if something accepts a TCP connection at ``address``."""
    try:
        with socket.create_connection(address, timeout=timeout):
            return True
    except OSError:
        return False


def _docker_usable() -> bool:
    """True if a docker daemon is reachable — not merely if the client binary is installed."""
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture(scope="session")
def livekit_server() -> Iterator[LiveKitDev]:
    """A running ``livekit-server`` in ``--dev`` mode, however this machine can provide one.

    **Synchronous on purpose.** pytest-asyncio 1.x gives every test a fresh function-scoped loop; a
    session-scoped *async* fixture would either need ``loop_scope="session"`` and then share one loop
    with every test in the session, or quietly keep a closed loop's objects alive. Nothing here needs
    asyncio: docker is a subprocess and the readiness check is ``urllib``.

    Four paths, in order:

    1. ``LIVEKIT_TEST_URL`` set — use it, and **never skip**. CI sets it precisely so that a server that
       failed to start fails the job loudly instead of silently skipping the whole tier.
    2. A server already answering on 7880 — reuse it. This is the developer who left the container
       running, and starting a second one would fail on the port anyway. If its TCP media port is
       closed, fail rather than reuse it: a container started with the older 7880-only command answers
       on 7880, and tests against it fail at random, far from the cause.
    3. Docker available — run the container, poll, and stop it on the way out. If a container of that
       name is already running, wait for it to answer, then check its media port as in 2.
    4. Neither — skip with a message that says what to install.
    """
    env_url = os.environ.get("LIVEKIT_TEST_URL")
    if env_url:
        dev = LiveKitDev(
            url=env_url,
            api_key=os.environ.get("LIVEKIT_TEST_API_KEY", DEFAULT_API_KEY),
            api_secret=os.environ.get("LIVEKIT_TEST_API_SECRET", DEFAULT_API_SECRET),
            http_url=_http_url_for(env_url),
        )
        yield dev
        return

    dev = LiveKitDev(
        url=DEFAULT_WS_URL,
        api_key=DEFAULT_API_KEY,
        api_secret=DEFAULT_API_SECRET,
        http_url=DEFAULT_HTTP_URL,
    )

    if _answers(dev.http_url):
        _fail_without_media_port(dev)
        yield dev
        return

    if not _docker_usable():
        pytest.skip("integration tests need docker or LIVEKIT_TEST_URL")

    try:
        started = subprocess.run(
            DOCKER_RUN.split(), capture_output=True, text=True, timeout=DOCKER_RUN_TIMEOUT_SECS
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"`{DOCKER_RUN}` did not return within {DOCKER_RUN_TIMEOUT_SECS}s. If the image is not "
            "cached this includes the pull; run the command by hand once to see its progress."
        )
    # A container of this name is already running but was not answering a moment ago: it may still be
    # booting. Poll for it rather than racing to start a second one, check its media port as on the
    # reuse path above, and do not stop what we did not start.
    owned = started.returncode == 0
    if not owned and "already in use" not in (started.stderr or ""):
        pytest.fail(f"`{DOCKER_RUN}` failed:\n{started.stderr.strip()}")

    try:
        if not _wait_ready(dev.http_url):
            logs = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True).stdout
            pytest.fail(f"livekit-server did not answer {dev.http_url} within {READY_TIMEOUT_SECS}s.\n{logs}")
        if not owned:
            _fail_without_media_port(dev)
        yield dev
    finally:
        if owned:
            subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, timeout=60)


def _fail_without_media_port(dev: LiveKitDev) -> None:
    """Fail if the reused server at ``dev`` does not accept connections on its TCP media port."""
    if not _accepts_tcp(MEDIA_TCP_ADDRESS):
        host, port = MEDIA_TCP_ADDRESS
        pytest.fail(
            f"a livekit-server answers {dev.http_url} but nothing accepts connections on its media "
            f"port {host}:{port}, so tests would stall at random. It was probably started with an "
            f"older command that published only 7880. Stop it and start it again with:\n{DOCKER_RUN}"
        )


def _http_url_for(ws_url: str) -> str:
    """``ws://host:port`` -> ``http://host:port`` (and ``wss`` -> ``https``), for ``LiveKitAPI``."""
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://") :]
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://") :]
    return ws_url


@pytest.fixture
def room_name(request: pytest.FixtureRequest, livekit_server: LiveKitDev) -> Iterator[str]:
    """A room name unique to this test, deleted server-side when the test ends.

    Synchronous, and the deletion runs in a throwaway ``asyncio.run`` loop on purpose: by the time a
    function-scoped fixture finalises, the test's own loop may already be closed, and cleanup that
    depends on the loop it is cleaning up after is cleanup that does not happen on the failure paths
    that need it most, where a room left behind can collide with a later test's.
    """
    name = unique_room_name(request)
    try:
        yield name
    finally:
        asyncio.run(delete_room(livekit_server, name))


@pytest.fixture(autouse=True)
def _accept_ws_livekit_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dev server speaks ``ws://``, which the room adapters accept only through their test seam."""
    monkeypatch.setattr(livekit_token, "_ALLOW_INSECURE_WS", True)


# ======================================================================= the launcher (class in _launcher.py)


@pytest.fixture
async def launcher(livekit_server: LiveKitDev) -> AsyncIterator[Launcher]:
    """A :class:`Launcher` bound to this session's dev server, torn down however the test ends.

    Function-scoped and async: only the server fixture is session-scoped, and it is synchronous. Every
    fake worker, fake API server and room a test's launches created is released here, including on the
    failure paths: a room left behind is the classic way an integration suite starts failing tests that
    have nothing wrong with them.
    """
    launcher = Launcher(livekit_server)
    try:
        yield launcher
    finally:
        await launcher.aclose()
