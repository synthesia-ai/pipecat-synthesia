# Deployment

Deploying a Pipecat bot that renders a Synthesia avatar is deploying an ordinary Pipecat bot, plus
outbound access to two services and a handful of environment variables. This page covers the parts
that are specific to the avatar.

## Container

This is **your bot's** Dockerfile, not this repository's. `pipecat-synthesia` is an ordinary
dependency of it.

```dockerfile
FROM python:3.11-slim

# uv for a fast, reproducible install from the lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root. The bot needs no privileged access; nothing here writes outside /app.
RUN useradd --create-home --uid 10001 bot
WORKDIR /app

# Dependency layer first, so a code change does not re-resolve the environment.
# --no-install-project is what makes that split work: without it `uv sync` also builds and
# installs your own package, which fails here because your source (and anything pyproject
# references, such as a readme) has not been copied yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=bot:bot . .
# Now the project itself, against the layer above.
RUN uv sync --frozen --no-dev \
    && chown -R bot:bot /app
USER bot

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
CMD ["python", "bot.py"]
```

Your `pyproject.toml` needs the plugin with its extra:

```toml
dependencies = ["pipecat-synthesia[livekit]>=1.0,<2", "pipecat-ai[runner,webrtc]>=1.14,<2"]
```

Any Pipecat 1.x from 1.14 is allowed, but only 1.14.0, 1.15.0 and 1.16.0 are tested on every
change, and a newer minor logs one warning at startup: for a container you rebuild unattended, pin
`pipecat-ai` to a tested minor (for example `pipecat-ai[runner,webrtc]==1.16.*`) so a new Pipecat
release reaches production only when you choose.

`uv add "pipecat-synthesia[livekit]>=1.0,<2"` adds it from PyPI. uv records the requirement in
`pyproject.toml` and `uv.lock`, so the `uv sync --frozen` above installs it from there.

**Handle SIGTERM, so that `docker stop` ends the session in order.** `WorkerRunner` installs a SIGINT
handler by default but **not** a SIGTERM one, and SIGTERM is what every container runtime sends to
stop a task. Without it the process dies before the avatar session is torn down, and the session
keeps running — and billing — until the avatar worker ends it, a few seconds after LiveKit notices
that the bot's participant has left: the transport room in a direct join, the private room in a
relayed join.

```python
runner = WorkerRunner(handle_sigterm=True)
await runner.add_workers(worker)
await runner.run()
```

Two calls rather than `run(worker)`: passing the worker straight to `run()` is deprecated
since Pipecat 1.3 and warns.

<!-- Verified in a live run: with handle_sigterm=True, SIGTERM ends the session as
     pipeline_cancelled, logs a full metrics line, and exits cleanly. -->

With it set, a stop produces an orderly `on_session_ended` (`pipeline_cancelled`) and `on_metrics`
before exit. Both this and the few seconds of billing without it are checked for a standalone bot
that runs its own `WorkerRunner`, as above. Under Pipecat's development runner, the examples' ordered
end is checked as described below.

Whatever stops the bot, end the pipeline and await it before the process exits. Exiting while a
relayed join is still connecting the bot to its private room cancels that connect, and the LiveKit
SDK can then end the process with SIGTERM.

Under Pipecat's development runner's web server (`-t webrtc`), each call is a task inside the server,
and that server owns the signals. The relayed-join example and the quickstart therefore run their calls
through `_shutdown.py`, which on SIGINT or SIGTERM ends every call in order and then exits. A bot run
that way without it dies on `docker stop`, and its session bills until the avatar worker ends it, a few
seconds after the bot's participant leaves. As PID 1 the process exits itself after the ordered end, so
the container needs no init.

It waits at most `SHUTDOWN_SECS` (10 s) for the calls to end: Docker's default stop grace, after which
it would be SIGKILLed anyway. An ordered end normally takes one to three seconds; the plugin's 20 s and
30 s bounds (`SESSION_ACLOSE_BACKSTOP_SECS`, `LAUNCH_CANCEL_SECS`) are backstops for a misbehaving SDK or
worker. The slow path is a signal while a relayed join is connecting the bot to its private room:
unwinding that can take the `connect_timeout` plus about 6 s, which outlasts the 10 s bound. If the bound
cuts a teardown short, the platform finishes it: the avatar worker ends the session about 5 s after the
bot's participant is gone (or at its 30 s join watchdog if the bot never arrived), and a private room
closes at its 20 s departure timeout (or its 120 s empty timeout if nobody joined). Under Kubernetes's
default 30 s grace, raise it to about 25 s. Going past about 30 s is not advised: beyond the plugin's 30 s
launch-cancel bound, waiting longer buys nothing, and the bot only looks hung.

The `[livekit]` extra is required for **both** join modes, not just the direct join:
`pipecat.transports.livekit.transport` imports `tenacity` at module import time and `tenacity` is
not in Pipecat core, so the package installed without the extra cannot import the LiveKit adapters.

Silero VAD downloads no weights at runtime — the ONNX model ships inside `pipecat-ai[silero]`, so
the image needs no model-fetch step and the container works with no egress to a model host.

## Environment

| Variable | Direct join: `LiveKitTransportRoom` | Relayed join: `PrivateLiveKitRoom` |
| --- | --- | --- |
| `SYNTHESIA_API_KEY` | required | required |
| `SYNTHESIA_API_URL` | optional | optional (non-production only) |
| `SYNTHESIA_ALLOW_ANY_API_HOST` | optional: `1` lets `SYNTHESIA_API_URL` be a host outside `synthesia.io` | the same |
| `LIVEKIT_URL` | the project your bot's transport is in | **required** — the project the private room is created in |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | required unless you pass `avatar_token` yourself | **required** — creates the room and mints both tokens |

If you pass `avatar_token=` to `LiveKitTransportRoom` yourself, mint it with what the plugin's own avatar
token carries:

- the grants `roomJoin` for the bot's room, `canPublish`, `canSubscribe`, `canPublishData` and
  **`canUpdateOwnMetadata`**;
- the attribute `lk.publish_on_behalf` set to the bot's identity;
- `kind="agent"` (recommended).

`canUpdateOwnMetadata` is the one that goes missing: livekit-api's `VideoGrants` leaves it out unless you
set `can_update_own_metadata=True`. Without it the avatar cannot report readiness, because
livekit-server silently drops the attributes it reports it with, so `on_avatar_ready` would never fire.
`LiveKitTransportRoom` raises `ValueError` at startup for a token without it, as it does for one without
`roomJoin` or the publish-on-behalf attribute, or for another room.

In a direct join the LiveKit project is the one your users are already in. In a relayed join it is
an implementation detail of the avatar: your users never connect to it, and it can be a different
project from anything else you run.

In a relayed join the plugin creates one private room per session and deletes it on teardown, on
every failure path it owns. Give it a project where creating and deleting short-lived rooms is
unremarkable.

## LiveKit Cloud

**URL scheme is normalised for you.** `https://` becomes `wss://`, so pasting the URL straight out
of the LiveKit Cloud dashboard works. A `ws://` or `http://` URL is rejected, because Synthesia's API
accepts only `wss://`. The room adapter checks the URL when you construct it, so a URL that is not
`wss://` after normalisation, or has no host, raises `ValueError` at startup rather than failing the
launch.

**The bot's token needs `kind="agent"` — not just the agent grant.** This is the single most
common direct-join misconfiguration:

```python
from livekit.api import AccessToken, VideoGrants

token = (
    AccessToken(api_key, api_secret)
    .with_identity("pipecat-bot")
    .with_kind("agent")  # <-- the part people miss
    .with_grants(VideoGrants(room_join=True, room=room_name, agent=True))
    .to_jwt()
)
```

`kind` is a top-level claim; `VideoGrants(agent=True)` is a grant. They are different things, and
the avatar worker reads the former. Pipecat's own `generate_token_with_agent()`
(`pipecat.runner.livekit`) sets only the grant, so a token from it is **not** sufficient, and that
includes the bot token Pipecat's dev runner mints for `-t livekit`. LiveKit Cloud's agent-dispatch
tokens do carry `kind="agent"`; hand-minted ones usually do not.

Get this wrong and the avatar worker classifies your bot as a human, so when the real human
leaves, the room still looks occupied and **the session never ends and keeps billing**. Nothing
about that fails loudly, so the plugin checks the token itself, when you construct
`LiveKitTransportRoom(transport)`: a token without the claim raises `ValueError` with a message
naming the fix, whatever `required` says, before any session is requested. So does a transport with
no token, or one that is not a LiveKit access token.

## Pipecat Cloud

Nothing avatar-specific is required. Two defaults to know about:

- **The 300-second idle timeout is Pipecat's, not ours.** `PipelineWorker(idle_timeout_secs=...)`
  defaults to 300 and cancels the pipeline when no `BotSpeakingFrame`, transcription or
  user-speaking frame has passed for that long. A demo where someone stares silently at the avatar
  for five minutes will be cancelled; that is Pipecat working as designed. Raise it or pass `None`
  to disable. <!-- verified against pipecat-ai 1.10.0: PipelineWorker.idle_timeout_secs default 300,
  idle_timeout_frames = (BotSpeakingFrame, InterimTranscriptionFrame, TranscriptionFrame,
  UserSpeakingFrame, UserStartedSpeakingFrame) -->
- **No `processor_unusable_policy` is needed for the avatar.** With `required=True` (the default),
  the service ends the pipeline itself when the avatar cannot start, under Pipecat's default
  `CONTINUE` policy too, so a bot never keeps running with a dead avatar.

Scale by session: one avatar session per pipeline, one pipeline per worker.

## Network egress and IP allowlisting

Outbound only; the bot accepts no inbound connections for the avatar.

| Destination | Protocol | Why |
| --- | --- | --- |
| `developers.synthesia.io` (or your `SYNTHESIA_API_URL`) | HTTPS 443 | Creating the session. A redirect is not followed. |
| Your LiveKit project host | WSS 443 | Signalling. |
| LiveKit media | UDP (WebRTC), TCP/TLS 443 fallback | The avatar's audio and video. |

If egress is restricted, allow UDP to your LiveKit region. Forcing everything over TCP/443 works but
adds latency to a path where latency is the product. LiveKit publishes per-region IP ranges; ask
Synthesia support for the current Interactive Avatar API ranges rather than resolving the hostname
once and pinning what you get.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| **The call ends about 4 s in, before the avatar says anything** | **An avatar id your workspace cannot use.** The Synthesia API refuses the session, and with `required=True` the service ends the call and plays none of the greeting. | Look for the `UNKNOWN_AVATAR` error, whose message lists the avatar ids sent, and the WARNING session-create line with `unknown_reference`. Note a **Studio** avatar id is not an **Interactive** avatar id; the two look alike and are not interchangeable. See "Avatar ids" in the [README](../README.md#avatar-ids). |
| **Avatar speaks once, then goes silent but the call continues** | Should be impossible. | A plugin bug — please [report it](../SUPPORT.md) with `session_id` and the `metrics.summary()` line. |
| **Bot interrupts itself; it "hears" the avatar** | A direct join without `room.input_filter()`. The avatar publishes audio into the same room, your STT transcribes it, and the bot replies to itself. | Put `room.input_filter()` immediately after `transport.input()`. |
| **`LiveKitTransportRoom(...)` raises `ValueError` with a `kind="agent"` message at startup** | The bot's token lacks `kind="agent"`, so the avatar worker would count it as a human and the session would never end after the human leaves (and keep billing). Pipecat's dev runner (`-t livekit`) mints exactly such a token. | `AccessToken(...).with_kind("agent")` — see LiveKit Cloud above. Don't use the dev runner for a direct join; `examples/mint_tokens.py` and `examples/bot_direct_join.py` show the alternative. |
| **No avatar video in the browser with `-t webrtc`** | `video_out_enabled` not set on the transport params. | Set all five `video_out_*` values: `video_out_enabled=True`, `video_out_is_live=True`, 512×512, 25 fps. |
| **Avatar stretched or soft; one core pinned; latency creeps up over a call** | Transport geometry disagrees with the avatar's 512×512 at 25 fps, so Pipecat PIL-resizes every frame. | Same fix. The plugin warns once at startup on a mismatch. |
| **`400 validation_error`** | A LiveKit URL or a token attribute the Synthesia API rejected. (A URL that is not `wss://` after normalisation is refused earlier, when the room adapter is constructed.) The body is `extra="forbid"`. | Check `LIVEKIT_URL`, and that you have not added fields to the session body. |
| **`429 concurrency_limit`** | More simultaneous sessions than your plan allows; see the [limits in Synthesia's developer docs](https://docs.synthesia.io/reference/ia-operational-trust). | Wait and retry; the error carries `retry_after` when the Synthesia API supplies it. Check for sessions leaked by a bot that did not tear down. |
| **Direct join fails with `503 service_unavailable`, "The session could not be started"** | Most likely the avatar worker could not join your LiveKit room, which the Synthesia API reports only as this generic 503: LiveKit credentials from a different project than the room's, or a URL that resolves to a private address. The error's message says so. | Take the LiveKit credentials, or the provided avatar token, from the project your bot's transport connects to, and give the avatar worker a publicly resolvable `wss://` URL. If neither applies, it may be a real outage: retry later. |
| **Direct join warns at startup that "the avatar token will be signed with a different LiveKit API key than the bot's token"** | The `credentials` (or `LIVEKIT_API_KEY`) that `LiveKitTransportRoom` mints the avatar token with are not the key that signed the bot's token. One LiveKit project can have several keys, so this is only a warning, but keys from different projects make the launch fail with the `503` above. | Take the credentials from the project your bot's transport connects to. If both keys belong to that project, ignore the warning. |
| **`ValueError` at startup: the Synthesia API URL "must be an https:// URL to synthesia.io or a host under it"** | `api_url` or `SYNTHESIA_API_URL` names a host outside `synthesia.io`, uses `http://` to a host that is not localhost, or carries a user name or password. The service would send your API key and a LiveKit token to it. | Leave both unset for production. For a proxy of your own, use `https://` and pass `allow_any_api_host=True` or set `SYNTHESIA_ALLOW_ANY_API_HOST=1`; the service logs a WARNING at startup saying so. |
| **`UNKNOWN` error "with no Synthesia error code"** | Something other than the Synthesia API answered the session request, such as a wrong host or a proxy or CDN page. The Synthesia API always sends an error code. The message gives the HTTP status and the start of the response body. A 3xx is a redirect, which the service does not follow. | Check `api_url` or `SYNTHESIA_API_URL` (leave both unset for production), and any proxy between the bot process and the Synthesia API. |
| **Disconnects at almost exactly 5 minutes** | Pipecat's 300 s idle timeout, not a network fault. | `PipelineWorker(idle_timeout_secs=...)`. |
| **`AssertionError` spam from `FfiHandle.__del__` on shutdown** (9–11 lines, "Exception ignored in …") | Cosmetic, and **not** this plugin. LiveKit's Python SDK disposes FFI handles via GC rather than explicitly ([python-sdks #524](https://github.com/livekit/python-sdks/issues/524), [agents #1449](https://github.com/livekit/agents/issues/1449)). Reproduces with Pipecat's LiveKit transport alone, with no avatar involved. | Ignore it. `Exception ignored in __del__` means Python swallowed it; the session has already closed cleanly, and on macOS the exit code is unaffected (on Linux, see the next row). The count scales with the number of media objects, so a run with an avatar shows more than a bare transport. |
| **Exit code 134 at shutdown on Linux**, after `panic in a function that cannot unwind` from a `tokio-rt-worker` thread | The same garbage-collected disposal as the row above. On Linux the SDK's Rust runtime can panic during interpreter shutdown and abort the process after all the work has finished. Seen once in this project's CI, under pytest on native x86_64 Ubuntu. Not reproduced for a bot process in 392 Linux container runs (303 native arm64, 89 emulated amd64; Python 3.11 and 3.12; `livekit` 1.1.18–1.1.20; `pipecat-ai` 1.10.0–1.12.0), all of which exited 0. Bot processes on native x86_64 are untested. | If your platform treats a non-zero exit as a crash (restarts, alerts), check your bot's exit code in its container. This project's test suite works around it by exiting with pytest's result before that shutdown (`tests/conftest.py`). |
| **Pipeline ended although `required=False`** | Should be impossible. With `required=False` the `ErrorFrame` carries a non-permanent category precisely so the pipeline survives. | A plugin bug — please [report it](../SUPPORT.md). |

Two of these are marked *should be impossible*. That is deliberate: they are invariants the test
suite enforces, so seeing one in the wild is information worth having, not something to work around.
