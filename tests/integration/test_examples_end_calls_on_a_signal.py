"""Ctrl-C or SIGTERM mid-call ends the relayed-join example's session in order, then the process exits.

Marker: ``integration_extended``, with the other example tests. It runs ``examples/bot_relayed_join.py``
unmodified, in its own process, under Pipecat's development runner (``-t webrtc``, the web server), with
the scripted brain, the fake Synthesia API and ``FakeAvatarWorker``. A WebRTC client in this process
makes the call. Mid-call, the test signals the process, then requires:

- the example's own log line for the session ending as a cancel (``pipeline_cancelled``), which is the
  plugin tearing the session down in order rather than leaving it billing until the worker notices;
- the process exiting as that signal (a shell shows 130 for SIGINT, 143 for SIGTERM), within a bound
  that only backstops the example's own wait (``_shutdown.SHUTDOWN_SECS``).

The process is a wrapper around ``runpy``, for two reasons. The dev server speaks ``ws://``, which the
room adapters accept only through the test seam this suite's conftest sets, and a subprocess needs it
set in its own interpreter. And ``PYTHON_DOTENV_DISABLED`` keeps the development runner's
``load_dotenv(override=True)`` from loading a credentials file found above the virtualenv, which would
send this test's calls to the real services.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription

from tests.integration._join_modes import AVATAR_IDS, WAIT
from tests.integration._launcher import Launcher
from tests.support.livekit_dev import LiveKitDev
from tests.support.secrets import API_KEY
from tests.support.waiting import until

pytestmark = pytest.mark.integration_extended

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RELAYED = EXAMPLES / "bot_relayed_join.py"

#: A backstop for the call to be up, not a budget: the server starts, the call connects, the fake
#: worker joins the private room.
CALL_WAIT = 3 * WAIT
#: A backstop for the process to exit, not a budget: longer than the example's own bounded wait for
#: its calls to end (10 s).
EXIT_WAIT = 30.0

SESSION_ENDED_IN_ORDER = "avatar session ended: pipeline_cancelled"

#: Run the example as a script would run, with the seam a ``ws://`` dev server needs.
WRAPPER = """
import runpy, sys
import pipecat_synthesia.livekit.token as token
token._ALLOW_INSECURE_WS = True
sys.path.insert(0, {examples!r})
sys.argv = [{script!r}, *sys.argv[1:]]
runpy.run_path({script!r}, run_name="__main__")
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _environment(dev: LiveKitDev, api_url: str) -> dict[str, str]:
    """This process's environment without any Synthesia or LiveKit settings, plus the local ones."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SYNTHESIA_", "LIVEKIT_"))}
    env.update(
        PYTHON_DOTENV_DISABLED="1",
        PYTHONUNBUFFERED="1",
        LIVEKIT_URL=dev.url,
        LIVEKIT_API_KEY=dev.api_key,
        LIVEKIT_API_SECRET=dev.api_secret,
        SYNTHESIA_API_KEY=API_KEY,
        SYNTHESIA_API_URL=api_url,
        SYNTHESIA_AVATAR_ID=AVATAR_IDS[0],
    )
    return env


class Example:
    """The example's process and everything it has printed so far."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.output = ""

    def printed(self, text: str) -> bool:
        return text in self.output

    def tail(self, count: int = 60) -> str:
        return "\n".join(self.output.splitlines()[-count:])

    async def read(self) -> None:
        """Drain the pipe in chunks, so a long log line can never stop the reader and block the example."""
        assert self.process.stdout is not None
        while chunk := await self.process.stdout.read(65536):
            self.output += chunk.decode(errors="replace")


@asynccontextmanager
async def _running_example(env: dict[str, str], port: int, cwd: Path) -> AsyncIterator[Example]:
    """The relayed-join example under the dev runner's web server; killed on the way out if still running."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        WRAPPER.format(examples=str(EXAMPLES), script=str(RELAYED)),
        "-t",
        "webrtc",
        "--fake-brain",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    example = Example(process)
    reader = asyncio.create_task(example.read(), name="example-output")
    try:
        yield example
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await asyncio.wait([reader], timeout=WAIT)


@asynccontextmanager
async def _call(port: int) -> AsyncIterator[RTCPeerConnection]:
    """A WebRTC client calling the example, as the dev runner's browser page would."""
    pc = RTCPeerConnection()
    pc.addTransceiver("audio", direction="sendrecv")
    pc.addTransceiver("video", direction="sendrecv")
    try:
        await pc.setLocalDescription(await pc.createOffer())
        offer = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        async with aiohttp.ClientSession() as http:
            async with http.post(f"http://127.0.0.1:{port}/api/offer", json=offer) as response:
                response.raise_for_status()
                answer = await response.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        yield pc
    finally:
        await pc.close()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
async def test_a_signal_mid_call_ends_the_session_in_order_then_exits(
    sig: signal.Signals, livekit_server: LiveKitDev, launcher: Launcher, tmp_path: Path
) -> None:
    api = await launcher.api()
    port = _free_port()

    async with _running_example(_environment(livekit_server, api.base_url), port, tmp_path) as example:
        await until(
            lambda: example.printed("Uvicorn running on") or example.process.returncode is not None,
            timeout=CALL_WAIT,
            what="the example's web server to start",
        )
        assert example.process.returncode is None, example.tail()

        async with _call(port):
            await until(
                lambda: example.printed("avatar connected: session="),
                timeout=CALL_WAIT,
                what=f"the call's avatar to connect\n{example.tail()}\n{launcher.debug_dump()}",
            )

            example.process.send_signal(sig)
            await until(
                lambda: example.process.returncode is not None,
                timeout=EXIT_WAIT,
                what=f"the example to exit after {sig.name}\n{example.tail()}",
            )

    assert example.printed(SESSION_ENDED_IN_ORDER), example.tail()
    # asyncio reports a process ended by a signal as minus the signal number; a shell shows 128 + it.
    assert example.process.returncode == -sig, example.tail()
