"""An in-process, scripted HTTP server standing in for the Synthesia API."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from tests.support.waiting import until

__all__ = ["FakeSynthesiaApi", "RecordedRequest", "ScriptedResponse"]


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    json: Any


@dataclass
class ScriptedResponse:
    status: int
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    before: Callable[[RecordedRequest], Awaitable[None]] | None = None


class FakeSynthesiaApi:
    """A tiny scripted `aiohttp.web` server standing in for the Synthesia public API.

    Every request is recorded, then answered by the next scripted response (the last entry repeats
    once the script is exhausted), following the sequence: record -> pick scripted entry -> await
    its `before` hook (if any) -> sleep `delay` -> respond.
    """

    def __init__(self, script: list[ScriptedResponse]) -> None:
        if not script:
            raise ValueError("FakeSynthesiaApi needs at least one ScriptedResponse")
        self._script = list(script)
        self.requests: list[RecordedRequest] = []
        self.hook_errors: list[BaseException] = []
        self._app = web.Application()
        # Every method, so that a request a redirect turned into a GET is recorded too.
        self._app.router.add_route("*", "/{tail:.*}", self._handle)
        self._server = TestServer(self._app)

    async def _handle(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except ValueError:
            payload = None
        recorded = RecordedRequest(
            method=request.method,
            path=request.path,
            headers=dict(request.headers),
            json=payload,
        )
        self.requests.append(recorded)
        index = min(len(self.requests) - 1, len(self._script) - 1)
        scripted = self._script[index]

        if scripted.before is not None:
            try:
                await scripted.before(recorded)
            except Exception as exc:  # noqa: BLE001 -- deliberately loud: a broken test fixture must fail
                self.hook_errors.append(exc)
                return web.json_response({"code": "internal", "detail": repr(exc)}, status=500)

        if scripted.delay:
            await asyncio.sleep(scripted.delay)

        return self._respond(scripted)

    @staticmethod
    def _respond(scripted: ScriptedResponse) -> web.Response:
        if scripted.body is None:
            return web.Response(status=scripted.status, headers=scripted.headers)
        if isinstance(scripted.body, str):
            return web.Response(
                status=scripted.status,
                text=scripted.body,
                headers=scripted.headers,
                content_type="text/plain",
            )
        return web.json_response(scripted.body, status=scripted.status, headers=scripted.headers)

    async def __aenter__(self) -> FakeSynthesiaApi:
        await self._server.start_server()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._server.close()

    @property
    def base_url(self) -> str:
        return str(self._server.make_url("")).rstrip("/")

    async def wait_for_requests(self, n: int, *, timeout: float = 5.0) -> None:
        """Poll until at least `n` requests have been recorded."""
        await until(lambda: len(self.requests) >= n, timeout=timeout, what=f"{n} requests recorded")
