"""The integration suite's `livekit_server` fixture, when it finds a livekit-server it did not start.

It reuses that server only if the server's TCP media port accepts connections too. A container started
with the older command that published only 7880 answers on 7880, and tests against it stall at random,
so the fixture fails with the command to restart it instead. That holds whether the server answers at
once or, as a container still booting does, only after `docker run` has found it already running.

Each test runs the real fixture in an inner pytest session (pytester), against a local HTTP server
standing in for livekit-server's 7880 and a TCP port that is open or closed. Docker is either reported
missing, so a fixture that stopped reusing the server would skip, or replaced by a fake that says a
container named `livekit` is already running; no test starts a real container.
"""

from __future__ import annotations

import http.server
import socket
import subprocess
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from tests.integration import conftest as livekit_fixture

pytest_plugins = ["pytester"]

INNER_TEST = """
from tests.integration.conftest import livekit_server  # noqa: F401


def test_gets_a_server(livekit_server):
    print("SERVER", livekit_server.http_url)
"""


class _Ok(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def signalling_url() -> Iterator[str]:
    """A local HTTP server answering ``GET /`` like livekit-server's 7880 does."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Ok)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def open_port() -> Iterator[int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener.getsockname()[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def closed_port() -> int:
    return _free_port()


def _run_fixture(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    http_url: str,
    media_port: int,
    *,
    docker_usable: bool = False,
) -> pytest.RunResult:
    monkeypatch.delenv("LIVEKIT_TEST_URL", raising=False)
    monkeypatch.setattr(livekit_fixture, "DEFAULT_HTTP_URL", http_url)
    monkeypatch.setattr(livekit_fixture, "DEFAULT_WS_URL", http_url.replace("http://", "ws://"))
    monkeypatch.setattr(livekit_fixture, "MEDIA_TCP_ADDRESS", ("127.0.0.1", media_port))
    monkeypatch.setattr(livekit_fixture, "_docker_usable", lambda: docker_usable)
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    pytester.makepyfile(test_inner=INNER_TEST)
    return pytester.runpytest_inprocess("-s", "-p", "no:cacheprovider")


@pytest.mark.unit
def test_a_running_server_with_its_media_port_open_is_reused(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, signalling_url: str, open_port: int
) -> None:
    result = _run_fixture(pytester, monkeypatch, signalling_url, open_port)

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines([f"*SERVER {signalling_url}*"])


@pytest.mark.unit
def test_a_running_server_without_its_media_port_fails_with_the_command_to_restart_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, signalling_url: str, closed_port: int
) -> None:
    result = _run_fixture(pytester, monkeypatch, signalling_url, closed_port)

    result.assert_outcomes(errors=1)
    output = result.stdout.str()
    assert f"127.0.0.1:{closed_port}" in output, output
    assert "published only 7880" in output, output
    assert livekit_fixture.DOCKER_RUN in output, output


@pytest.mark.unit
def test_a_container_already_running_that_answers_late_without_its_media_port_fails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, closed_port: int
) -> None:
    """Nothing answers at first; `docker run` reports a container named `livekit` already running, which
    then starts answering. The fixture did not start it, so it checks the media port, and fails."""
    signalling_port = _free_port()
    late_servers: list[http.server.HTTPServer] = []
    docker_calls: list[list[str]] = []

    def fake_docker(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        docker_calls.append(list(args))
        if args[:2] == ["docker", "run"]:
            server = http.server.HTTPServer(("127.0.0.1", signalling_port), _Ok)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            late_servers.append(server)
            conflict = 'Conflict. The container name "/livekit" is already in use by container "0123abcd".'
            return subprocess.CompletedProcess(
                args, 125, "", f"docker: Error response from daemon: {conflict}"
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(
        livekit_fixture,
        "subprocess",
        SimpleNamespace(
            run=fake_docker,
            TimeoutExpired=subprocess.TimeoutExpired,
            SubprocessError=subprocess.SubprocessError,
        ),
    )
    try:
        result = _run_fixture(
            pytester, monkeypatch, f"http://127.0.0.1:{signalling_port}", closed_port, docker_usable=True
        )
    finally:
        for server in late_servers:
            server.shutdown()
            server.server_close()

    assert [call[:2] for call in docker_calls] == [["docker", "run"]], docker_calls
    result.assert_outcomes(errors=1)
    output = result.stdout.str()
    assert f"127.0.0.1:{closed_port}" in output, output
    assert "published only 7880" in output, output
    assert livekit_fixture.DOCKER_RUN in output, output
