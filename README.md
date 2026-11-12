# pipecat-synthesia

A [Synthesia Interactive Avatar](https://docs.synthesia.io/reference/interactive-avatars) for
[Pipecat](https://github.com/pipecat-ai/pipecat) pipelines: a photorealistic, lip-synced face for
a voice bot you already have. You bring the brain — your STT, LLM and TTS, or a speech-to-speech
model — and this renders it as a person.

Drop `SynthesiaVideoService` in after your TTS. It consumes the speech your pipeline already
produces; everything upstream stays exactly as it was.

## Choosing a join mode

The avatar reaches your users in one of two ways, chosen by the `room` you pass. Start with the
direct join:

- **Direct join (`LiveKitTransportRoom`).** The avatar joins the LiveKit room your bot's
  `LiveKitTransport` is already in, and publishes its video and audio there itself. The avatar's
  video never passes through your pipeline, and there is no second room to run. If your bot runs
  Pipecat over LiveKit, this is the join mode to use. If you know you want Pipecat but have not yet
  chosen between LiveKit and Daily, choose LiveKit and a direct join.
- **Relayed join (`PrivateLiveKitRoom`).** For users who connect over a transport other than
  LiveKit. The avatar joins a private LiveKit room that the plugin creates and owns, and the plugin
  relays the avatar's video and audio into your pipeline as frames. Your transport delivers them.
  It is tested with SmallWebRTC (Pipecat's default dev transport, `python bot.py`) and Daily; other
  Pipecat transports are untested. What it costs you: a LiveKit project of your own for the private
  room (your users never connect to it), and an extra hop, because the avatar's video and audio
  travel through your bot process on their way to the user.

| | Direct join: `LiveKitTransportRoom` | Relayed join: `PrivateLiveKitRoom` |
| --- | --- | --- |
| Where the avatar runs | The room your bot is already in | A private LiveKit room the plugin creates and owns |
| How video reaches the user | Published directly into the room | Relayed through your pipeline as Pipecat frames |
| Works with | LiveKit only, with a bot token you mint (not the dev runner's) | SmallWebRTC and Daily, as tested — **not** `-t livekit` |
| You need | The room you already have | A LiveKit project of your own |
| Extra pipeline wiring | `room.input_filter()` | Five `video_out_*` transport settings |

A relayed join over Pipecat's LiveKit transport is untested (its video output exists since Pipecat
1.11). If your users are in a LiveKit room, use a direct join.

Use a direct join unless your users connect over Daily, SmallWebRTC or another transport that is
not LiveKit; for those, use a relayed join.

## Installation

Install it from PyPI with its `[livekit]` extra:

```bash
uv add "pipecat-synthesia[livekit]"

# or
pip install "pipecat-synthesia[livekit]"
```

It installs `pipecat-ai` 1.14 or later, below 2.0, if you don't already have it. Python 3.11 or later
is required. Check the install:

```bash
python -c "import pipecat_synthesia; print(pipecat_synthesia.__version__)"
```

Each release on PyPI carries PyPI attestations, which record that it was published by this
project's release workflow.

Tested with Pipecat 1.14, 1.15 and 1.16. Any later 1.x is allowed too, and logs one warning at
startup if it is a newer minor than these: see [Compatibility](#compatibility).

The `[livekit]` extra is required for **both** join modes: the avatar is reached through LiveKit
in either case.

Developing on this repo instead? Use `just sync`, not a bare `uv sync`: a plain sync installs no
optional extras and will uninstall `tenacity` and `livekit`, leaving a checkout where
`pipecat_synthesia.livekit` cannot import.

## Quick start

### Direct join (`LiveKitTransportRoom`)

`stt`, `llm`, `tts` and the context aggregators in `pair` are your pipeline's own, with VAD on the
user aggregator (see [Voice activity](#voice-activity)); the plugin adds `room` and `avatar`.

```python
from pipecat_synthesia import SynthesiaVideoService
from pipecat_synthesia.livekit import LiveKitTransportRoom

transport = LiveKitTransport(
    url, bot_token, room_name, LiveKitParams(audio_in_enabled=True, audio_out_enabled=True)
)
room = LiveKitTransportRoom(transport)
avatar = SynthesiaVideoService(avatar_ids=[avatar_id], room=room)

pipeline = Pipeline(
    [
        transport.input(),
        room.input_filter(),  # <- REQUIRED: drops the avatar's own audio off the bot's input
        stt,
        pair.user(),
        llm,
        tts,
        avatar,
        transport.output(),
        pair.assistant(),
    ]
)
```

No `video_out_*` here: the avatar publishes video into the room itself, so nothing passes through
your pipeline. A direct join needs three things a relayed join does not:

- **`room.input_filter()` directly after `transport.input()`.** The avatar's audio arrives on the
  bot's own input; without the filter the bot transcribes the avatar and replies to itself.
- **A bot token carrying `kind="agent"`.** Not the `agent` grant — the top-level claim. **Pipecat's
  dev runner (`-t livekit`) mints its bot token with
  `pipecat.runner.livekit.generate_token_with_agent()`, which sets only the grant, so a direct join
  refuses it at startup:** `LiveKitTransportRoom(transport)` raises `ValueError`, whatever `required`
  says. Mint the bot token yourself instead: `examples/mint_tokens.py` mints a correct pair, and
  `examples/bot_direct_join.py` runs a direct join without the dev runner. The check exists because
  without the claim the avatar worker counts your bot as a human, so the session never ends when the
  real human leaves and **keeps billing**.
- **LiveKit credentials from the bot's project.** When the plugin mints the avatar token from
  `credentials` (or `LIVEKIT_API_KEY`), `LiveKitTransportRoom` warns at startup if that key is not the
  one that signed the bot's token. Keys from two LiveKit projects mean the avatar worker cannot join
  the bot's room, which the Synthesia API reports only as a generic `503 service_unavailable`; two keys
  of the same project are fine. If you pass your own `avatar_token=` instead, it needs the grants
  [`docs/deployment.md`](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/docs/deployment.md#environment) lists, **`canUpdateOwnMetadata`** included:
  without it the avatar can never report that it is ready.

### Relayed join (`PrivateLiveKitRoom`)

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat_synthesia import SynthesiaVideoService
from pipecat_synthesia.livekit import PrivateLiveKitRoom

# VAD lives on the user aggregator, never on the transport params. See "Voice activity" below.
pair = LLMContextAggregatorPair(
    LLMContext([]),
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
avatar = SynthesiaVideoService(
    avatar_ids=[os.environ["SYNTHESIA_AVATAR_ID"]],  # SYNTHESIA_API_KEY from the environment
    room=PrivateLiveKitRoom(),  # LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
)
# `avatar` goes straight after the TTS; everything else is a stock Pipecat pipeline.
stages = [transport.input(), stt, pair.user(), llm, tts, avatar, transport.output(), pair.assistant()]
pipeline = Pipeline(stages)
```

The transport needs to be told the avatar's video geometry — see
[Video geometry](#video-geometry):

```python
transport_params = TransportParams(
    audio_in_enabled=True,
    audio_out_enabled=True,
    video_out_enabled=True,
    video_out_is_live=True,
    video_out_width=512,
    video_out_height=512,
    video_out_framerate=25,
)
```

Runnable versions of both are in [`examples/`](https://github.com/synthesia-ai/pipecat-synthesia/tree/main/examples): `bot_direct_join.py` and
`bot_relayed_join.py`. Their `--fake-brain` flag replaces the STT, LLM and TTS with a scripted
stand-in, so you need only the Synthesia and LiveKit credentials. Their Ctrl-C and SIGTERM handling
is unverified on Windows.

### Voice activity

VAD belongs on the **user aggregator**, not on the transport:

```python
LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
```

`TransportParams`, `LiveKitParams` and `DailyParams` have no `vad_analyzer` field, and as pydantic
models they **accept the keyword and silently discard it**. The result is a bot that never detects
speech and can never be interrupted, with nothing in any log explaining why. This repo fails its
build if the keyword appears anywhere else (`tests/unit/test_examples_static.py`).

## Pipeline position and frame contract

Place the service **immediately after your TTS**, before `transport.output()`.

| | |
| --- | --- |
| **Consumes** | `TTSAudioRawFrame` — the avatar speaks it. **Consumed, not forwarded, in both join modes**, so it never reaches `transport.output()` and the user never hears the raw TTS alongside the avatar. The one exception is voice-only, when the avatar cannot start with `required=False` or leaves unexpectedly: then the bot's speech goes to the transport instead (see `required` below). |
| **Emits (both join modes)** | `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame`, from the avatar's **real playout**, not from when audio was sent |
| **Emits (relayed join only)** | `OutputImageRawFrame` (the avatar's video) and plain audio frames |
| **Forwards** | Everything else, unchanged |

The `Bot*Speaking` frames coming from real playout rather than from send time is the point of the
whole design: interruption timing, transcripts and turn-taking all key off when the avatar's mouth
actually moved.

## Configuration

```python
SynthesiaVideoService(*, avatar_ids, room, api_key=None, api_url=None, allow_any_api_host=None,
                      launch_timeout=90.0, required=True, session=None, settings=None)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `avatar_ids` | *required* | 1–5 ids; the first is active. The `av_` prefix is added for you. |
| `room` | *required* | `LiveKitTransportRoom(transport)` or `PrivateLiveKitRoom()`. |
| `api_key` | `$SYNTHESIA_API_KEY` | |
| `api_url` | `$SYNTHESIA_API_URL`, else `https://developers.synthesia.io` | Override only for non-production. Must be `https://` to `synthesia.io` or a host under it (`http://` only for localhost); any other host is refused unless `allow_any_api_host` is set. |
| `allow_any_api_host` | `$SYNTHESIA_ALLOW_ANY_API_HOST`, else `False` | Lets `api_url` be an `https://` host outside `synthesia.io`, such as a proxy of your own. The service sends your API key and a LiveKit token to that host, and logs a WARNING at construction saying the URL points outside synthesia.io. The variable takes `1`, `true`, `yes` or `on` for `True`, and `0`, `false`, `no` or `off` (or unset) for `False`; any other value raises `ValueError` at construction. |
| `launch_timeout` | `90.0` | Bounds **the whole launch**, not just the wait for the avatar: the room connect, `POST /sessions` with its retries, and the avatar joining, under one timer. |
| `required` | `True` | Whether the avatar is required to start. `True` ends the call if it cannot; `False` continues voice-only. See below. |
| `session` | `None` | Bring your own `aiohttp.ClientSession`. |
| `settings` | `None` | Initial `SynthesiaVideoSettings`. |

Invalid constructor arguments (avatar ids, API key, `api_url`, `launch_timeout`, LiveKit credentials,
URLs and tokens) raise `ValueError` or `TypeError` immediately; failures once running raise or push
`SynthesiaError`. A `room` that is not one of the two room adapters, and a keyword argument the
service does not take, such as the LiveKit plugin's `join_timeout`, raise `TypeError` the same way.

**`required` means required to start.** It decides what happens when the avatar cannot start, and
nothing else.

- **`required=True` (the default) ends the call.** The service reports the error, marks itself
  unusable, and ends the pipeline itself. None of the bot's speech plays from then on, neither what was
  queued for the avatar nor what the TTS is still producing, so a bot that is about to hang up doesn't
  go on talking. (If the session request fails after an early start, the avatar may already have
  started speaking: see Events.) You don't need `processor_unusable_policy=ProcessorUnusablePolicy.END` for
  this. Guaranteed whatever the policy or the Pipecat version: the pipeline ends, the service is
  unusable, and `on_session_ended` fires once with `launch_failed`; rely on that handler. Under
  `ProcessorUnusablePolicy.CANCEL`, Pipecat cancels the pipeline instead of ending it; whether
  downstream processors see a `CancelFrame` is up to Pipecat: a launch that fails while the pipeline is
  still starting is torn down without one. On that path `on_pipeline_started` may or may
  not fire and `on_pipeline_finished` does not, but `runner.run()` still returns, so do final cleanup
  after it returns or in `on_session_ended`.
- **`required=False` means voice-only, not silent failure.** The error is still reported, but the
  service stays usable and the bot keeps talking without a face: speech still queued for the avatar is
  handed to the transport as ordinary `TTSAudioRawFrame`s, and later speech passes straight through.

**An unexpected leave continues voice-only, whatever `required` says.** When the avatar leaves
mid-call (the avatar worker crashed, the session reached its length cap, or the credits ran out), the
user is already talking to the bot, so the bot finishes the call without a face: queued speech is
handed to the transport as above, and you get `on_avatar_disconnected`. To end the call instead:

```python
from pipecat.frames.frames import EndWorkerFrame
from pipecat.processors.frame_processor import FrameDirection


@avatar.event_handler("on_avatar_disconnected")
async def on_avatar_disconnected(service, reason):
    await service.push_frame(EndWorkerFrame(reason=reason), FrameDirection.UPSTREAM)
```

After a direct join's leave with humans still present (any reason but `room_connection_lost`), this
normally ends the session as `pipeline_ended`, with no error pushed, rather than `avatar_left`: the
plugin waits up to 2 s to see whether the last human leaves (see Limits and lifetimes), and the
pipeline's own end cuts that wait short, provided the `EndFrame` reaches this service within the 2 s.
Pipecat queues that `EndFrame` at the start of the pipeline, behind any frames the processors before
this service are still working through; if it arrives after the 2 s, the session ends as
`avatar_left`, with its error, and the pipeline's end follows. `on_avatar_disconnected` fires once
either way.

In a direct join, the user's app sees the avatar leave as its video track disappearing from the
room: the plugin cannot replace it with anything, so your app has to cope with a participant whose
video goes away mid-call.

**Speech produced before the session is active is buffered, not lost.** Starting the service does
not block your pipeline: if your bot begins talking while the avatar is still launching, that audio
is held and sent once the session is active, which can be up to about half a second before
`on_avatar_connected` (see Events). The buffer is bounded, so a very long head start is truncated
rather than growing without limit — when that happens it is reported as `prelaunch_dropped_bytes` in
the metrics rather than passing silently.

The user speaking first does not lose it either. Pipecat interrupts the bot at every user turn start,
even when the bot is not speaking. Before the session is active nothing has played, so the first time
the user speaks while audio is buffered, the plugin keeps that audio. With a TTS service that sets
context ids, it drops the rest of an utterance the TTS was still producing; one that sets none, such
as OpenAI Realtime, may still send that rest, and whatever of it arrives after the user's latest turn
plays, as it would once the avatar is live. A greeting
therefore plays once the avatar is live, and the answer to the user's question follows it. If the user
speaks again before then, whatever was buffered after the kept audio is dropped, as an interruption
would drop it later. The plugin cannot tell a greeting from a reply: if the user first speaks before
anything is buffered, the reply to that turn can be kept along with the greeting, or in place of one,
and play before the current answer. That is usually one reply at most, because Pipecat cancels a
reply still being produced when the user speaks again. Three consequences:
- The answer waits behind the whole greeting, so a long greeting delays it by its own length.
- If the user is still talking when the session becomes active, the greeting plays over them. Pipecat
  interrupts only when a user turn starts, so nothing stops it until their next one.
- Once the session is active, an interruption clears audio the avatar has not begun playing, as any
  interruption of unplayed audio does in Pipecat. The session usually becomes active as the avatar's
  video appears, so a user who starts speaking in the first second or so after seeing the avatar, before
  its first words play, still cuts off a greeting they have not heard.

Environment variables are listed in [`.env.example`](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/.env.example).

## Events

```python
@avatar.event_handler("on_avatar_connected")
async def on_avatar_connected(service, info):
    logger.info(f"avatar is live: session {info.session_id}")


@avatar.event_handler("on_session_ended")
async def on_session_ended(service, reason):
    logger.info(f"avatar session ended: {reason}")
```

| Event | Fires | Handler arguments |
| --- | --- | --- |
| `on_avatar_connected` | Once, when the session is active and the Synthesia API has returned it; audio can start up to about half a second earlier (below) | `service, info`: an `AvatarConnected` with `session_id`, `avatar_identity`, `active_avatar_id`, `room_name` |
| `on_avatar_ready` | Once, when every avatar in `avatar_ids` is ready; never before `on_avatar_connected` | `service, info`: an `AvatarReady` with `session_id`, `status` |
| `on_avatar_disconnected` | At most once, when the avatar leaves an active session | `service, reason`: a leave reason (below) |
| `on_session_ended` | Exactly once per instance | `service, reason`: an end reason (below) |
| `on_metrics` | Exactly once, right after `on_session_ended` | `service, metrics`: a `SessionMetrics` |
| `on_error` | Per error | `service, frame`: an `ErrorFrame`. Pipecat's own event, not registered by us |

**What your app sees during a launch, in order.**
1. The session becomes active when this launch's avatar starts publishing its video, or when the
   Synthesia API returns the session (its 201), whichever comes first. With an avatar token you
   provide, it is always the 201. From then on the bot's audio goes to the avatar, and
   `service.state` is `SESSION_ACTIVE`.
2. `on_avatar_connected` fires once the 201 is in, usually about half a second after the session
   became active. Until then `service.session_id` is `None`, and `swap_avatar()` raises.
3. `on_avatar_ready` fires after `on_avatar_connected`, once every avatar is ready.

If the session request fails after the session became active, or is still out when `launch_timeout`
runs out, `on_avatar_connected` never fires: the avatar is stopped, and the service ends as a failed
launch, as `required` decides (`launch_failed`, `launch_timeout` or `degraded`). By then the avatar may
have started speaking. If the avatar leaves after the session became active, even before the 201, that
is an unexpected leave (`avatar_left`): you get `on_avatar_disconnected`, with no `on_avatar_connected`
before it if the 201 hadn't arrived, and the call continues voice-only, even with `required=True`, which
requires the avatar to reach its video, not the 201. If the request is retried after the session became
active, the retry can start a second avatar worker that replaces the first, so the bot's audio waits until
the request settles (its 201 or its failure), at most until `launch_timeout` runs out. During a graceful
stop the hold waits for the request to settle, bounded by the drain timeout; a request still out at that
bound loses the goodbye. Whatever the first worker had been sent before the retry, often the greeting,
is cut off if it is replaced. Audio waiting this way counts against the same cap as audio buffered
during the launch.

**Leave reasons**, passed to `on_avatar_disconnected`:

| Reason | Meaning |
| --- | --- |
| `avatar_left_room` | The avatar participant left the room. |
| `avatar_video_stopped` | The avatar stopped publishing its video. |
| `room_connection_lost` | The bot's connection to the room was lost (in a direct join, the `LiveKitTransport`'s connection). |

An avatar worker that disconnects and is immediately replaced under the same identity has not
left: the plugin waits one second for the replacement before it reports anything.

**End reasons**, passed to `on_session_ended` and recorded as `metrics.end_reason`:

| Reason | Meaning | Error pushed |
| --- | --- | --- |
| `pipeline_ended` | The pipeline ended normally (`EndFrame`). | none |
| `pipeline_cancelled` | The pipeline was cancelled (`CancelFrame`). A stop by signal through `WorkerRunner` ends this way. | none |
| `pipeline_cleanup` | Pipecat cleaned up the service without ending or cancelling it first. | none |
| `no_humans_left` | Direct join only: the avatar worker ended the session because no human participants were left. A normal end. | none |
| `avatar_left` | The avatar left unexpectedly: in a direct join, while humans were still there 2 s later (at once if the bot lost its own connection, `room_connection_lost`); in a relayed join, any leave. The bot continues voice-only. | yes, unless the pipeline's own ending cut that end short (see `on_error` below); the service stays usable |
| `launch_failed` | The launch failed, with `required=True`. The service ends the pipeline. | yes; marks the service unusable |
| `launch_timeout` | The launch timed out, with `required=True`: the whole launch exceeded `launch_timeout`, or a step inside it timed out (a Synthesia API request, or the avatar not joining within 30 s of the session being created). The service ends the pipeline. | yes; marks the service unusable |
| `degraded` | The launch failed or timed out, with `required=False`. The bot continues voice-only. | yes, unless the pipeline's own ending cut that end short (see `on_error` below); the service stays usable |

`on_error` hands you an `ErrorFrame` whose `.exception` is a `SynthesiaError`. **Read the category
off the exception, not off the frame.** Only a failed launch with `required=True` is pushed as
permanent, which marks the service unusable. Every other error is pushed with a non-permanent
category, so that Pipecat does not mark the processor unusable and tear the pipeline down; the true
classification stays on `exception.category`.

When an error ends the session, `on_error` comes before `on_session_ended` and `on_metrics`. Pipecat
runs `on_error` handlers one after another, inside the code that reports the error, so a handler that
awaits something holds back the handlers after it, and the end events too. When the pipeline ends, the
teardown gives an end already in progress up to `ON_ERROR_WAIT_SECS` (one second) to push its error,
whether it is still handing audio over or already inside the app's handlers; past that the error is
dropped at INFO if it was not yet pushed, and `on_session_ended`'s reason stands for it. That can happen
after an unexpected leave, or a failed launch with `required=False`, which hand queued speech over
voice-only before they push their error. An error already inside your handlers is not dropped: the end
events fire anyway, and handlers still to run may run after them, or not at all.

## Errors

One exception type, `SynthesiaError`, with an `ErrorType` enum. Category is derived from the HTTP
status when there is one; the table below is the fallback used when there is not.

| `ErrorType` | Category | Retryable |
| --- | --- | --- |
| `AUTH` | `AUTHENTICATION` | no |
| `FEATURE_NOT_IN_PLAN` | `AUTHORIZATION` | no |
| `INVALID_ROOM_TOKEN` | `INVALID_REQUEST` | no |
| `LIVEKIT_CREDENTIALS_REJECTED` | `AUTHENTICATION` | no |
| `INVALID_SESSION_REQUEST` | `INVALID_REQUEST` | no |
| `UNKNOWN_AVATAR` | `INVALID_REQUEST` | no |
| `QUOTA_EXCEEDED` | `QUOTA` | no |
| `RATE_LIMITED` | `RATE_LIMIT` | **yes** |
| `CONCURRENCY_LIMIT` | `RATE_LIMIT` | **yes** |
| `TIMEOUT` | `CONNECTIVITY` | **yes** |
| `CONNECTION` | `CONNECTIVITY` | **yes** |
| `CLIENT_UPGRADE_REQUIRED` | `INVALID_REQUEST` | no |
| `AVATAR_SWAP_FAILED` | `APPLICATION` | no |
| `UNKNOWN` | `UNKNOWN` | no |

An error code from the Synthesia API that this version has never seen becomes `UNKNOWN`, with the API's `detail`
preserved verbatim — never reclassified into a guess.

So does an error response with no code at all. The Synthesia API always sends one, so something else
answered: a wrong `api_url`, or a proxy or CDN in between. The message gives the HTTP status and the start
of the response body. A redirect is one of these too: the service does not follow it, because it could take
your LiveKit token to another host. Two statuses mean the same whoever sends them, so they keep
retryable types: a 429 without a code is `RATE_LIMITED`, with its `Retry-After`, and a 5xx is `CONNECTION`
(`TIMEOUT` for a 504).

Advisories from the Synthesia API, such as notice that something this plugin version relies on is
being retired, appear once each as WARNING log lines beginning `Synthesia API advisory` or
`Synthesia API warning`, and are worth acting on.

## Swapping the avatar

```python
new_id = await avatar.swap_avatar("av_other", timeout=10.0)
```

The id must be one of the `avatar_ids` passed to the constructor (with or without `av_`), or
`"default"` for the first of them. `timeout` defaults to 15 s and must be positive (a `ValueError`
otherwise); a swap can take up to about a second longer than it. To answer, the avatar worker gets what is
left of `timeout` after any readiness wait, minus two seconds, at least half of that and at least one second.
A swap it answers later than that fails with `TIMEOUT`, and an acknowledgement slower than two seconds also
fails the swap with `TIMEOUT` (LiveKit's connection-timeout code 1501 rather than 1502), which the app may
retry.

`swap_avatar()` needs an active session: called before `on_avatar_connected`, it raises
`SynthesiaError` (`CONNECTION`) and reports nothing through `on_error`. To ask for a swap before
then, change the `avatar_id` setting instead, with `settings=` in the constructor or a
`SynthesiaVideoUpdateSettingsFrame` (both importable from `pipecat_synthesia`); the swap is made
once `on_avatar_connected` has fired. The session always starts on the first of `avatar_ids`, so list the
avatar you want to start with first.

A failed swap raises, reports the error once through `on_error` with a non-permanent category, and
leaves the pipeline running with the previous avatar. The type says why: `UNKNOWN_AVATAR` (not one
of `avatar_ids`), `AVATAR_SWAP_FAILED` (the avatar worker refused), `TIMEOUT` or `CONNECTION`.

## Metrics

`on_metrics` delivers a `SessionMetrics`; `summary()` gives seconds-since-launch:

```
{'room_connected': 1.606, 'session_created': 4.462, 'avatar_joined': 4.462,
 'avatar_ready': 4.463, 'avatar_video_published': 3.983, 'transport_ready': 1.618,
 'ended': 140.273, 'end_reason': 'pipeline_cancelled',
 'playback_finished_count': 3, 'prelaunch_dropped_bytes': 0}
```

This one is from a pre-release version of the plugin, which timed `avatar_joined` when the launch
stopped waiting (here the 201); it is now the avatar's join itself, which comes before
`avatar_video_published`.

Time to first byte (`utterance_ttfb_seconds`) and interruption latency (`interruption_latency_seconds`)
are not recorded, because the avatar worker's playback events don't say which audio stream they belong
to. The fields stay on `SessionMetrics`, always empty, and `summary()` leaves them out.

For how long the avatar took to appear, read `avatar_video_published` and `avatar_ready`.
`session_created` is stamped when the API responds, which is often after the avatar's video is
already flowing: note `avatar_video_published` above is *earlier* than `session_created`.
`session_requested` is when the session request went out. In a direct join it goes out while the bot
is still connecting, so it normally comes before `room_connected`.

## Limits and lifetimes

- **Concurrent sessions:** your Synthesia plan sets how many sessions can run at once; see the
  [limits in Synthesia's developer docs](https://docs.synthesia.io/reference/ia-operational-trust). Going over it is a `429` with code
  `concurrency_limit`.
- **Ending a session:** there is no end-session API. The session ends, and billing stops, about 5 s
  after LiveKit reports the bot gone from the room or, in a direct join, after it reports the last human
  gone, or at once if that human's client said it was leaving (LiveKit's disconnect reason
  `CLIENT_INITIATED`, as when your app calls `disconnect()`). In the second case, if your
  pipeline is still running when the avatar leaves, the plugin reports `on_avatar_disconnected`,
  then `on_session_ended` with `no_humans_left`, and pushes no error. The avatar's leave can reach the
  bot before the human's own, so when humans still appear present the plugin waits up to 2 s for them
  to leave before it reports an unexpected leave (`avatar_left`); the bot's speech is held, not sent,
  meanwhile. It does not wait when the bot itself lost the room (`room_connection_lost`), since it can no
  longer hear anyone leave. So an avatar that crashes and a last human who hangs up within those 2 s
  end as `no_humans_left`, with no error: the crash is not reported. And after a hang-up, a human who
  rejoins before the plugin has recorded the end (within about 2 s of the avatar leaving) gets the
  voice-only bot, but one who rejoins after the plugin has recorded the end, as on a page reload whose
  leave is client-initiated (not checked), finds the session already ended as `no_humans_left`; the
  direct-join example then ends the call, although that human is back. (The direct-join example ends its
  pipeline 6 s after the last human leaves, a second after the avatar worker's default 5 s, so it
  normally ends with `no_humans_left`. That assumes the avatar has left within that second, which
  has not been measured; if it has not, the example's own end comes first and the session ends with
  `pipeline_ended`. The relayed-join example ends with `pipeline_ended`.)
  This is why a direct join's bot token must carry `kind="agent"`. In a direct join the bot gets the
  5 s however it leaves, even when your app calls `disconnect()`: the worker skips it only for a
  human. (In a relayed join,
  the plugin deletes the private room when the session closes, by default, which ends the session at
  once. A relayed join's bot that drops without the room being deleted is treated like a human: at
  once if its client said it was leaving, otherwise after 5 s.)
- **A human who leaves during the launch is not noticed.** In a direct join, the avatar worker notices
  the last human leaving only once it has joined. If the last human leaves during the launch, before
  the worker joins, the session runs, and bills, until the bot leaves or the session length cap. End
  the call yourself when nobody is left, as the examples' presence code (`examples/_presence.py`) does.
- **Session length cap:** plan-dependent and enforced by Synthesia.
- **Pipecat's idle timeout is 300 s**, and it is not ours: `PipelineWorker` cancels the pipeline
  after that long with no speech or transcription frames. Change it with
  `PipelineWorker(idle_timeout_secs=...)`, or `None` to disable.

## Video geometry

Relayed join only. The avatar arrives as **512×512 at 25 fps**, so tell the transport:

```python
video_out_enabled=True, video_out_is_live=True,
video_out_width=512, video_out_height=512, video_out_framerate=25
```

Leave Pipecat's defaults in place and every frame goes through a PIL resize: a distorted aspect
ratio (512×512 is square), one core pinned, and latency that grows over a call.
`video_out_is_live=True` marks the stream as continuous rather than a fixed-length clip. The
service warns once at startup if the transport disagrees.

## Avatar ids

`avatar_ids` takes **Interactive Avatar** ids. An avatar id from Synthesia Studio is not one: the two
look alike, and the Synthesia API refuses a Studio id with `404 unknown_reference`, which the plugin
reports as `UNKNOWN_AVATAR`, naming the ids it sent. There is **no endpoint that lists** the avatars
available to your workspace. Interactive Avatar ids come from:

- **the stock avatars** in Synthesia's Interactive Avatar quickstarts. The
  [quickstart](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/quickstarts/pipecat/README.md) here starts with one.
- **custom avatars**, which you create with a `POST` to `/api/interactive-avatars/avatars` on the
  Synthesia API. This plugin deliberately does not wrap that call — it is a one-off setup step, not
  part of a live session.

## Known limitations

- **No relayed join over a LiveKit transport.** `PrivateLiveKitRoom` over `-t livekit` is untested
  (Pipecat's LiveKit transport gained video output in 1.11, after this plugin's design). Use
  `LiveKitTransportRoom` instead.
- **No rejoin.** If the avatar leaves unexpectedly, the session is over; the plugin reports it
  rather than reconnecting. An avatar worker that restarts more than a second after it left (cause not
  established; rare) is ignored: nothing more is sent to it. In a direct join it joins your room
  again: your users see the avatar back, silent, and it bills until the bot or the last human leaves.
  Until a follow-up release, the bot also downloads its video, a few Mbps, for the rest of the call.
- **One avatar per room.**
- **A late avatar after a direct join's launch times out with `required=False`.** The bot carries on
  voice-only, but the avatar worker can still join the room afterwards and stay there, idle, until
  the session ends, when the bot or the last human leaves. The session bills while it runs.
- **`swap_avatar` is untested against the live API** at the time of writing.
- **The avatar worker replaced mid-call (rare).** A worker can be replaced by another with the same
  identity: its host can restart it, or a retried session request (after a 500, 502, 503 or 504, a timeout,
  a connection error, or a 2xx whose body was lost) can start a second one. The bot is reported stopped
  0.35 s later, but the reply in flight is lost and the user may have to repeat; after a retried request the
  plugin holds the bot's audio until the request settles (see Events). A graceful stop close to a
  replacement can end before the bot's last words play, or wait up to 5 s for audio no worker will play.
- **The avatar leaving during a graceful stop.** The rest of the bot's last words is not heard: what the
  avatar had been sent leaves with it, and the part not yet sent is dropped rather than played voice-only.
  The pipeline ends as `pipeline_ended`, with no error; `on_avatar_disconnected` still fires.
- **A graceful stop during the launch.** Speech queued for the avatar before the session is active
  is dropped: the stop abandons the launch, and that speech plays neither through the avatar nor
  voice-only. The pipeline ends as `pipeline_ended`, with no error.
- **A 201 that arrives during a graceful stop.** After the session became active on the avatar's
  video, before the Synthesia API returned it, a stop that starts before the 201 arrives means
  `on_avatar_connected` and `on_avatar_ready` never fire, even if the 201 lands while the bot's last
  words are still playing. `on_session_ended` and `on_metrics` fire as usual.
- **A hang-up just after the avatar joins can end the launch with an error (to be fixed).** In a
  direct join, if the last human hangs up after the avatar worker has joined but before the session
  is active (a short window, not measured), the worker leaves at once, but the plugin does not yet
  treat that as a leave. It waits up to 30 s after the Synthesia API returns the session (or until
  `launch_timeout`, if it never does) for the avatar to join again, then ends the launch as
  `launch_timeout`, or as `launch_failed` if the leave lands as the launch finishes (`degraded` for
  either with `required=False`), with an error, rather than as `no_humans_left`. The worker has
  already left, so billing has stopped. An app that ends the call once nobody is left, as the
  direct-join example does, usually ends first, as `pipeline_ended`. In a relayed join, an avatar
  worker that leaves after joining and before its video gets the same 30 s wait.

## Compatibility

Pipecat support has three tiers:

- **Allowed:** any Pipecat 1.x from 1.14 (`pipecat-ai>=1.14,<2`).
- **Tested on every change:** **Pipecat 1.14.0, 1.15.0 and 1.16.0**.
- **Tested nightly:** Pipecat's newest release and its main branch.

On a newer Pipecat minor than the tested ones, the plugin logs one WARNING at startup, when the first
`SynthesiaVideoService` is constructed, naming the installed version and the tested ones. A new Pipecat
minor can break the plugin before it has been tested, so if you need stability, pin `pipecat-ai` in
your app (for example `pipecat-ai==1.16.*`).

The plugin follows SemVer: a breaking change to its public API comes only in a new major version,
so pin it with `pipecat-synthesia[livekit]>=1.0,<2`.

Synthesia has not yet committed to a compatibility window, that is, how old a plugin version the
avatar worker keeps accepting.

The wire protocol is additive-only and the rules are written down in
[docs/protocol-compatibility.md](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/docs/protocol-compatibility.md), so a plugin and an avatar worker
from different releases are expected to interoperate.

## What the plugin sends

Every request the plugin makes to the Synthesia API identifies it, in its `User-Agent` and
`X-Synthesia-Client` headers: the plugin's, Pipecat's and Python's versions, the operating system and
processor architecture (such as `Linux; x86_64`), the join mode, and the kind of user transport
(`livekit`, `daily`, `smallwebrtc` and so on, never a class name). Synthesia uses them to see which plugin
versions are in use before changing anything those versions depend on. An avatar token the plugin mints
carries the plugin's name and version, its protocol level, the join mode and a random id for each launch,
as participant attributes that everyone in the room can see; an avatar token you provide carries only
what you put in it. The full list is in
[docs/protocol-compatibility.md](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/docs/protocol-compatibility.md),
sections 4 and 6.

## Deployment

See [docs/deployment.md](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/docs/deployment.md) — container, environment, LiveKit Cloud and Pipecat
Cloud notes, egress rules, and a troubleshooting table.

## Support

See [SUPPORT.md](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/SUPPORT.md) for where to report and what to include.

## Attribution

Maintained by Synthesia.

## License

MIT; see [LICENSE](https://github.com/synthesia-ai/pipecat-synthesia/blob/main/LICENSE).
