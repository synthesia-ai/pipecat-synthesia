# Changelog

All notable changes to `pipecat-synthesia` are recorded here. The project follows SemVer: a breaking change comes
only in a new major version. Breaking changes are marked **BREAKING** (see `RELEASING.md`).

## Unreleased

## 1.0.0 - 2026-11-12

First public release, and the first on PyPI. It follows a private beta whose releases, 0.1.0b1 to 0.1.0b4, were
not published on PyPI. The first list says what the package does; the second lists what changed since 0.1.0b1,
for anyone moving from a beta.

### What the package does

- `SynthesiaVideoService`: a Pipecat service that renders a Synthesia Interactive Avatar speaking your bot's
  TTS output. It goes right after the TTS and needs `pipecat-ai` 1.14 or later.
- Two join modes:
  - **Direct join** (`LiveKitTransportRoom`), for Pipecat over LiveKit and the one to start with: the avatar
    joins the LiveKit room your bot's `LiveKitTransport` is in. `room.input_filter()` keeps the bot from
    hearing the avatar. The constructor checks the transport, and refuses a bot token without the
    `kind="agent"` claim that lets the session end when the last human leaves, whatever `required` says;
    Pipecat's dev runner (`-t livekit`) mints such a token, so a direct join needs a bot token you mint
    yourself. A provided avatar token is checked against the bot and its room at construction too, and the
    constructor warns if the avatar token will be signed with a different LiveKit API key than the bot's.
  - **Relayed join** (`PrivateLiveKitRoom`), for a user transport other than LiveKit: the avatar runs in a
    private LiveKit room the plugin creates and deletes, and its video and audio are relayed through the
    pipeline, so the user transport delivers them. Tested with SmallWebRTC and Daily.
  - Both room adapters check the LiveKit URL at construction: `https://` becomes `wss://`, and `ws://` is
    rejected, because the Synthesia API accepts only `wss://`.
- `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` come from the avatar's real playout, and an
  interruption clears the avatar's buffered speech.
- Speech produced before the session is active is buffered and sent once it is. The buffer is bounded; any
  truncation is reported as `prelaunch_dropped_bytes`.
- `launch_timeout` bounds the whole launch. `required` means required to start: with `required=True`, a failed
  launch marks the service unusable and the service ends the pipeline itself, playing none of the bot's
  speech, with no `processor_unusable_policy` needed; with `required=False`, it leaves the service usable and
  the bot talking voice-only. After an unexpected leave, the bot continues voice-only whatever `required`
  says.
- Up to five avatars per session, switched with `swap_avatar()` or an `avatar_id` settings update.
- Mistakes in the calling code fail at once: a keyword argument the constructor does not take (such as the
  LiveKit plugin's `join_timeout`) or a `room` that is not a room adapter raises `TypeError`, and a
  `swap_avatar()` timeout that is not positive raises `ValueError`.
- Everything a pipeline uses is importable from `pipecat_synthesia`: `SynthesiaVideoService`, its
  `SynthesiaVideoSettings` and `SynthesiaVideoUpdateSettingsFrame`, `ServiceState`, the event payloads
  `AvatarConnected` and `AvatarReady`, `SessionMetrics`, `SynthesiaError` and `ErrorType`.
- Events: `on_avatar_connected`, `on_avatar_ready`, `on_avatar_disconnected` (with a leave reason),
  `on_session_ended` (with an end reason) and `on_metrics` (`SessionMetrics`).
- One exception type, `SynthesiaError`, with an `ErrorType`, a category and a retryable flag. Only a failed
  launch with `required=True` marks the service unusable. The likely setup mistakes get a message that names
  the cause: `UNKNOWN_AVATAR` lists the avatar ids sent and warns that a Studio avatar id is not an Interactive
  Avatar id, and a direct join's `503 service_unavailable` points at the LiveKit credentials and URL. An error
  response without a Synthesia error code, from a wrong `api_url` or a proxy, is `UNKNOWN` (a 429 stays
  `RATE_LIMITED` and a 5xx a retryable `CONNECTION`), and its message gives the HTTP status and the start of the body. Failed
  session-create attempts are logged at WARNING.
- The package ships `py.typed`, so type checkers read its annotations.
- Credentials never reach the logs: API keys and LiveKit secrets are redacted, and LiveKit tokens are
  summarised rather than printed. An error response that echoes the request's secrets has them scrubbed
  wherever they appear in its body before the body is stored on the error.

### Changes since 0.1.0b1

- The plugin now needs `aiohttp` 3.14.5 or later and, with the `[livekit]` extra, `pyjwt` 2.15.1 or later: the
  first releases that fix the security advisories known on 2026-11-12. aiohttp carries the API key and the
  LiveKit token, and pyjwt reads LiveKit tokens.
- In a direct join, a user hanging up now ends the session as `no_humans_left`, with no error. Since 0.1.0b1,
  the avatar worker's leave could reach the bot about a second before the bot heard that the user had left, so
  the plugin took a normal hang-up for an unexpected leave: it pushed "Synthesia avatar left the room; continuing
  voice-only", handed the queued audio over to an empty room, and ended with `avatar_left`, where the README
  promises `no_humans_left`; the direct-join example then stayed in the empty room until its own presence grace
  ran out. When the avatar leaves while humans still appear present, the plugin now waits up to 2 s for them to
  leave before it calls the leave unexpected. A real unexpected leave, with someone still in the room, reports
  `avatar_left` and its error 2 s later than before; the bot's speech in those 2 s is held and then played
  voice-only, as before. A leave because the bot lost its own connection to the room (`room_connection_lost`)
  does not wait. Three consequences: an app that ends the call from `on_avatar_disconnected` after an
  unexpected leave, as the README suggests, normally ends as `pipeline_ended` with no error rather than
  `avatar_left`, provided the `EndFrame` reaches the service within the 2 s (otherwise it still ends as
  `avatar_left`, with its error); an avatar crash followed within 2 s by the last human hanging up ends as
  `no_humans_left`, without the crash's error; and in the direct-join example, a rejoin after the plugin has
  recorded the end, such as a page reload, now ends the call instead of getting the voice-only bot.
- **BREAKING** The Synthesia API URL (`api_url` or `SYNTHESIA_API_URL`) must now be `https://` to `synthesia.io`
  or a host under it, or go to a loopback host; any other host raises `ValueError` at construction. The
  service sends the API key and a LiveKit token (valid for 6 hours when the plugin mints it) to that URL, so a
  wrong or malicious value would hand both to another host. To point the plugin at a host of your own, such
  as a proxy, pass `allow_any_api_host=True` or set `SYNTHESIA_ALLOW_ANY_API_HOST=1`: the URL must still be
  `https://`, and the service logs one WARNING at construction saying the Synthesia API URL points outside
  synthesia.io. The variable takes `1`, `true`, `yes` or `on`, or `0`, `false`, `no` or `off`; any other
  value raises `ValueError` at construction. A URL with a user name or password in it is refused too: aiohttp
  could not send it with the API key, and failed only once the launch had started. The default URL and
  loopback URLs need no change.
- **BREAKING** The session request no longer follows a redirect: a 3xx answer is an `UNKNOWN` error with the
  HTTP status in its message, and is not retried. Following it would resend the request to whatever host the
  redirect names, and for a 307 or 308 its body, LiveKit token included. Point `api_url` at the address that
  answers directly.
- Releases to PyPI carry PyPI attestations (PEP 740), which record the repository and workflow that
  published each file. The release workflow checks that the tagged commit is on `main` before running any of its
  code, and the job that runs the build and tests can no longer publish; a separate job, approved by a person,
  publishes what it built. The person approving it, and a ruleset on `v*` tags, are what keep releases to commits on
  `main`: the check runs inside the workflow, which the tagged commit supplies.
- A greeting queued while the avatar launches is no longer lost when the user starts speaking first: a user
  turn before the session is active keeps the first audio held when the user spoke, which plays once the
  session is active, ahead of the answer. With a TTS service that sets context ids, the rest of an utterance it
  was still producing is dropped. A later turn before the session is active drops what was queued after the
  kept audio. The plugin cannot tell a greeting from a reply, so when the user's first turn finds nothing
  queued, the reply to it can play before the current one, usually one reply at most.
  Measured live: the greeting was heard 0 of 30 times on 0.1.0b1 when the user spoke first, and 14 of 14 with
  the queued audio kept.
- A live `swap_avatar()` no longer fails with `TIMEOUT` after about a second regardless of its `timeout`.
  Since 0.1.0b1 the avatar worker had one second to answer a swap, whatever `timeout` was: the deadline
  LiveKit sends to the worker's handler is `max(response_timeout - max_round_trip_latency, 1 s)`, and the
  plugin passed the timeout as both values. To answer, the avatar worker now gets what is left of `timeout`
  after any readiness wait, minus two seconds, at least half of that and at least one second: 13 s at the
  default of 15 s with no wait. A swap it answers later than that fails with `TIMEOUT`, and an acknowledgement
  slower than two seconds also fails the swap with `TIMEOUT` (LiveKit's connection-timeout code 1501 rather
  than 1502), which the app may retry. The clear buffer sent on an interruption is changed the same
  way; at its 2 s timeout the worker still has 1 s to answer it, as before, which is enough.
- **BREAKING** `interruption_latency_seconds` is no longer recorded, so `summary()` leaves it out: the avatar
  worker's playback events don't say which interruption they answer, so 0.1.0b1's values could be wrong. An
  interruption just after the session started, before the worker had bound the bot, recorded the wait for
  the bind; and two interruptions close together, with the first interrupted reply's late
  `playback_started` arriving between them, recorded a too-short sample. Code or dashboards that read the
  summary key should treat it as absent; the field stays on `SessionMetrics`, always empty.
- **BREAKING** `pipecat-ai` 1.14 or later is required, and any 1.x from there is allowed (`>=1.14,<2`). 0.1.0b1
  allowed 1.10 to 1.12: Pipecat 1.13 and older require Pillow<13, which would keep a Pillow with open security
  advisories in the lock this package is tested with. A newer Pipecat than the tested ones no longer stops the
  plugin installing, and the plugin no longer stops you upgrading Pipecat. On a newer Pipecat minor than any
  tested, constructing the first `SynthesiaVideoService` logs one WARNING that names the installed version and the
  tested ones. A new Pipecat minor can break the plugin before it is tested: pin `pipecat-ai` in your app if you
  need stability. CI tests Pipecat's newest release every night, as well as its main branch. The plugin now
  depends on `packaging`, which it uses to read the installed Pipecat version.
- Tested on every change with `pipecat-ai` 1.14.0, 1.15.0 and 1.16.0. Nothing in the plugin changed for them. The
  startup order Pipecat introduced in 1.13 was already handled, and is now pinned: a cancel before the worker's
  start task has returned (even after `on_pipeline_started` fired) is resolved by cleanup alone, with no
  `CancelFrame` and no `on_pipeline_finished`.
- A relayed join cancelled while its bot connects no longer kills the bot process: `Room.connect()` is never
  cancelled, so such a cancel can take up to `connect_timeout` plus about six seconds.
- A relayed join's session no longer ends about two minutes in: the bot now joins its private room as a
  standard participant rather than an agent, so LiveKit keeps the room open while the bot is in it.
- A pipeline no longer finishes while the service is still closing a session from the launch, one whose avatar
  left as the launch finished or one that launched as the pipeline ended: the end now waits for that close, as
  long as it would wait for any session close (20 s at most). In a relayed join that close deletes the private
  room, which a bot process that exits when its pipeline ends used to cut short.
- `on_session_ended` and `on_metrics` now fire, once each and after `on_error`, when the pipeline ends while an
  `on_error` handler is still awaiting something for the error that ended the session. After a failed launch
  they were lost, with the session summary log line; after an unexpected leave they came only once the handler
  returned, after the pipeline had finished. The end now waits up to about a second for the `on_error`
  handlers; past that, the end events fire anyway, and handlers still to run may run after them or not at all.
- An error whose end the pipeline's own ending cuts short is no longer reported after `on_session_ended`: an
  unexpected leave, or a failed launch with `required=False`, that is still handing queued speech over
  voice-only when the pipeline ends, and still is a second later, does not push its error;
  `on_session_ended`'s reason stands for it, and one INFO line records it.
- An unexpected leave whose handling was still under way when the pipeline started ending (an app's
  `on_before_push_frame` handler awaiting on the leave's `BotStoppedSpeakingFrame`, say) no longer ends the
  session as `avatar_left` with an error: the end is the pipeline's own, with its reason and no error, one INFO
  line records the leave, and speech the leave had queued for voice-only is dropped.
- The package is installed from PyPI: `pip install "pipecat-synthesia[livekit]"`. The README, the deployment
  guide and the quickstart say so, and the ImportError for a missing `[livekit]` extra gives that command.
- Docs and warnings corrected: a bot killed without a teardown bills until the avatar worker ends the session, a
  few seconds after the bot leaves the room.
- Audio starts about half a second sooner (*early start*): the session goes active when this launch's avatar
  publishes its video, instead of when the Synthesia API's 201 arrives, which trails it. A minted avatar token
  carries a random `synthesia.launch` attribute, so a stale avatar with the same identity is never taken for this
  launch's; with an avatar token you provide, the session still goes active at the 201. Not a breaking change:
  every event keeps its payload and its place in the order, except that an avatar leaving before the 201 now gives
  `on_avatar_disconnected` with no `on_avatar_connected` before it. `on_avatar_connected` still fires at the 201,
  so the bot's audio can now start up to about half a second before it; until then `session_id` is `None` and
  `swap_avatar()` raises. A session request that fails after the session went active, or is still out when
  `launch_timeout` runs out, ends the service as a failed launch (`required` decides) with no
  `on_avatar_connected`; the avatar, which may have started speaking, is stopped. With `required=True`, "required"
  now means required to reach this launch's video: an avatar that leaves after that, even before the 201, is an
  unexpected leave (`avatar_left`) and the call continues voice-only, where 0.1.0b1 ended the pipeline as a failed
  launch; a session request that fails after that still ends it. A request retried after the session went active
  can start a second avatar worker that replaces the first, so the bot's audio is held from the retry until the
  request settles (its 201 or its failure), at most until `launch_timeout` runs out. During a graceful stop the
  hold waits for the request to settle, bounded by the drain timeout; a request still out at that bound loses the
  goodbye. What the first worker was sent before the retry, often the greeting, is still cut off if it is
  replaced, and the held audio shares the pre-launch buffer's cap. Measured live: the avatar spoke 0.47 s after
  its video appeared instead of 0.97 s, sooner in 20 of 20 paired sessions.
- `avatar_joined` in the metrics is now the moment the avatar joined the room (or, if it was there before the
  bot, the moment the bot first saw it). It recorded the moment the launch stopped waiting, the 201 (or, after
  an early start, the avatar's video), up to about a second later.
- The bot is no longer left speaking when the avatar worker is replaced mid-utterance by one with the same
  identity: the replaced worker never reports the utterance finished, so the bot now stops speaking after the
  stop hysteresis, unless the new worker starts playing first.
- The bot is no longer left speaking when the interrupted utterance's `playback_started` arrives late, after
  the interruption: with no audio sent since, its interrupted `playback_finished` stops it like any other.
- Examples: `--fake-brain` gives one transcript per utterance, not one per audio chunk.
- Examples: a direct-join or Daily call ends once nobody has been in it for 6 s, the example's own wait, so a
  rejoin keeps it. In a direct join the avatar worker usually ends the session first: 5 s after the last human
  leaves, or at once if their client said it was leaving.
- Examples: under Pipecat's development runner, Ctrl-C or SIGTERM ends every call in order, then exits.
- The two warnings logged when the bot fails to connect as the avatar token expects now say the avatar worker
  ends the session within about 30 s. 0.1.0b1 said about two minutes.
- **BREAKING** `utterance_ttfb_seconds` is no longer recorded, so `summary()` leaves it out, and the service
  pushes no Pipecat TTFB `MetricsFrame`: the avatar worker's playback events don't say which audio stream they
  belong to, so its values could be wrong when speech overlapped. Code or dashboards that read the
  summary key or the service's TTFB metric should treat it as absent; the field stays on `SessionMetrics`,
  always empty.
- With Pipecat's metrics enabled, the service no longer gets Pipecat's initial empty time-to-first-byte and
  processing metrics (always 0.0) at pipeline start; 0.1.0b1 emitted them.
- The `X-Synthesia-Client` header always carries eight keys, adding `join_mode` (`direct` or `relayed`) and
  `transport` (the user transport, from a closed set), and a minted avatar token carries `synthesia.join_mode`.
- A launch that fails, or that the pipeline ends, before the Synthesia API's 201 no longer reports
  `avatar_video_published`: the avatar in the room until then may be a stale one with the same identity.
- Every line the service and its session print after the 201 starts with `[<session_id>]`, with loguru's
  default format too, and the `synthesia session summary` line prints its values; `session_id` and the
  summary's fields stay bound for a structured sink, and the service's own lines have `session_id` bound from
  the 201 on, not only once the session is active.
- The session and request ids from the Synthesia API's 201 are in `metrics.summary()` and on every error
  the service pushes, even when the launch fails after the 201.
- When the avatar leaves mid-call and the bot goes on voice-only, the chunk of speech being sent at that moment
  (20 ms to about a second, depending on the TTS service) is no longer lost: it now starts the voice-only
  audio. Audio the avatar had received but not yet played is still lost.
