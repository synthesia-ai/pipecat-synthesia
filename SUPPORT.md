# Support

## Where to report

GitHub issues are turned off for this repository.

- **A security issue:** see [SECURITY.md](SECURITY.md) to report a vulnerability.
- **Anything else** (a bug, a question, help with an integration): email Synthesia Support at
  support@synthesia.io, say that it is about `pipecat-synthesia`, and include the details listed below.
  Synthesia's [developer docs](https://docs.synthesia.io/docs/need-help) list the other ways to get help.

## What to include

A report with these eight things can usually be diagnosed without a back-and-forth. A report
without the first two usually cannot be diagnosed at all.

1. **`session_id`** — once the Synthesia API has created the session (its 201 response): on every
   `SynthesiaError`, on the `on_avatar_connected` payload, in `metrics.summary()`, and in square
   brackets at the start of every line that the service and its session print. An error before the
   201, such as the launch timing out before the API answered or a network failure reaching it, has
   none.
2. **`request_id`** — `SynthesiaError.request_id`. For an error from the Synthesia API it is the
   failing request's id, prefixed in square brackets to the error message when the API's response
   carried one. For any other error after the session was created (the avatar left, a swap failed,
   the launch timed out waiting for the avatar) it is the request id of the call that created the
   session, and the message has no bracket prefix. An error before the 201 that the API didn't
   answer has none. Where it is set, it is what makes Synthesia's logs searchable.
3. **Plugin version** — `python -c "import pipecat_synthesia; print(pipecat_synthesia.__version__)"`
4. **Pipecat version** — `python -c "import pipecat; print(pipecat.__version__)"`
5. **Which join mode** — a relayed join (`PrivateLiveKitRoom`, the avatar in a private room) or a
   direct join (`LiveKitTransportRoom`, the avatar in your bot's room). They fail in different places.
6. **Which transport** — SmallWebRTC, Daily or LiveKit.
7. **The `metrics.summary()` line** — see below.
8. **Logs** — see the redaction note below.

## Getting `metrics.summary()`

The plugin emits it on `on_metrics`, once per session, right after `on_session_ended`:

```python
@avatar.event_handler("on_metrics")
async def _on_metrics(service, metrics):
    logger.info("avatar metrics: {}", metrics.summary())
```

It is a single dict of seconds-since-launch, and it localises a problem faster than a log dump —
it says whether time went into creating the session, the avatar joining, or the first frame
arriving.

A real one, from a two-minute direct-join call with three utterances on a pre-release version of
the plugin (which timed `avatar_joined` at the 201; it is now the avatar's join itself, before
`avatar_video_published`):

```
{'room_connected': 1.606, 'session_created': 4.462, 'avatar_joined': 4.462,
 'avatar_ready': 4.463, 'avatar_video_published': 3.983, 'transport_ready': 1.618,
 'ended': 140.273, 'end_reason': 'pipeline_cancelled',
 'playback_finished_count': 3, 'prelaunch_dropped_bytes': 0}
```

Every key is a delta in seconds from `launch_started`, except `end_reason`,
`playback_finished_count`, `prelaunch_dropped_bytes`, and `session_id` and `request_id`, which appear
once the Synthesia API has created
the session, even if the launch failed after that. The plugin also logs the summary itself, at INFO,
as `synthesia session summary {...}`.

For how long the avatar took to appear, read **`avatar_video_published`** and **`avatar_ready`**.
`session_created` is stamped when the API responds, which is often after the avatar's video is
already flowing: above, `avatar_video_published` (3.983) is *earlier* than `session_created` (4.462).

## Logs, and what is safe to paste

**The plugin never logs a credential.** API keys and LiveKit secrets are replaced with a constant
`<redacted>` placeholder that reveals not even their length, and LiveKit tokens are summarised as
`<jwt sub=... room=... kind=... exp=...>` rather than printed. Error bodies are redacted before
they are stored on the exception, so `repr(err)` is safe too.

That covers this plugin's own output. It does **not** cover your application's logs: if your code
logs a token it minted, or a traceback prints a config object, that is yours to check. Scan a log
for `Bearer `, `eyJ` (a JWT) and your key prefixes before pasting it anywhere.

Turn up detail with loguru:

```python
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

## Before you report

Two failure modes in the troubleshooting table of [docs/deployment.md](docs/deployment.md) are
documented as **"should be impossible"**: the avatar speaking once and then going silent, and the
pipeline ending even though `required=False`. If you hit either, that is a plugin bug and worth a
report on its own — please do not work around it silently.
