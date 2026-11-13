# Architecture

How `pipecat-synthesia` is put together, the terms its code and docs use, and the design rules the
code relies on. Code comments cite these rules by name (for example "see *replacement grace*"), so
each rule's reasoning is written here once rather than at every place that depends on it.

## What the package does

`SynthesiaVideoService` is a Pipecat processor that sits right after your TTS service. It consumes the
bot's speech (`TTSAudioRawFrame`s) and sends it to a Synthesia **avatar worker**, which renders an
avatar speaking it. The avatar worker always joins a LiveKit room. What differs is *which* room, and
how the avatar's video and audio reach the user:

- **Direct join** (`LiveKitTransportRoom`), for Pipecat over LiveKit. The avatar joins the LiveKit room
  your bot's transport is already connected to, and publishes straight to the user. The service emits
  no video or audio frames of its own.
- **Relayed join** (`PrivateLiveKitRoom`), for Pipecat over Daily or SmallWebRTC. The plugin creates a
  private LiveKit room for the bot and the avatar worker, which the user never joins. It captures the
  avatar's video and audio there and relays them through your pipeline as frames, so your transport
  delivers them.

Today the user transport decides the join mode. A relayed join over `LiveKitTransport` is untested:
its video output exists since Pipecat 1.11, after this design, and nothing here has been run against it.

## Who runs what

The package runs only in the **bot process**, on the developer's own infrastructure. Synthesia runs the
Synthesia API and the avatar worker. The rooms live in the developer's **LiveKit project**, and its API
secret never leaves the bot process: the plugin mints the avatar token there, and only that signed token
reaches Synthesia.

```
 user's app ──► bot process (developer's) ──► Synthesia API ──► avatar worker (Synthesia's)
                  └── LiveKit project (developer's): the room(s) the bot and avatar worker join
```

## The pieces

```
 your pipeline:   … ─► TTS ─► SynthesiaVideoService ─► … ─► user transport
                                  │  frame contract, service states, send queue,
                                  │  speaking state, events, metrics
                                  ▼
                   AvatarRoom.launch() → AvatarSession                        room.py
               ┌──────────────────┴───────────────────┐
     LiveKitTransportRoom                     PrivateLiveKitRoom
     direct join: borrows the transport       relayed join: creates and owns a
     room, unsubscribes the bot from the      private room; AvatarTrackCapture
     avatar; AvatarAudioFilter                turns the avatar's tracks into frames
               └──────────────────┬───────────────────┘
                                  ▼
                        LiveKitAvatarSession                          livekit/session.py
                        launch, leave detection, close
                          ├─ UtteranceStreamer    TTS audio → worker, one byte stream per utterance
                          ├─ PlaybackRpcHandlers  playback events ← worker
                          ├─ AvatarReadiness      avatar_status / avatar_ready ← worker
                          └─ token.py             the avatar token
                                  │
                                  ▼
                        SessionApiClient ─► the Synthesia API (create the session)    api.py
```

| Module | What it holds |
|---|---|
| `service.py` | `SynthesiaVideoService`: the frame contract, service states, speaking state, events, teardown |
| `room.py` | `AvatarRoom` / `AvatarSession`, the interface between the service and a room adapter; `JoinMode`, `AvatarCallbacks`, `LaunchContext` |
| `_send_queue.py` | the send queue between `process_frame` and the send loop |
| `api.py`, `errors.py` | the Synthesia API client; `SynthesiaError` and `ErrorType` |
| `protocol.py` | every wire name, in one place |
| `config.py`, `events.py`, `metrics.py`, `frames.py` | constructor config and avatar-id helpers; event names and payloads; `SessionMetrics`; the settings frame |
| `_async.py`, `_once.py`, `_redact.py`, `_client_info.py`, `_compat.py`, `_log.py` | shared helpers |
| `livekit/transport_room.py`, `audio_filter.py`, `_transport_access.py` | direct join |
| `livekit/private_room.py`, `track_capture.py` | relayed join |
| `livekit/session.py` | `LiveKitAvatarSession`, shared by both join modes |
| `livekit/utterance_streamer.py`, `rpc.py`, `readiness.py`, `token.py` | the LiveKit side of the worker protocol |
| `livekit/_types.py` | the time bounds, `RoomHandle`, the avatar token sources |
| `livekit/_rtc.py`, `_rtc_protocols.py` | the LiveKit SDK import guard; the narrow SDK surface the test fakes implement |

## Lifecycle

The service states (`ServiceState`) are `INIT → LAUNCHING → SESSION_ACTIVE → ENDED`, or
`DEGRADED` / `FAILED` when the launch fails: before the session is active, or, after an *early start*,
when the session request fails once the session is already active.

1. **Launch.** `setup()` starts the launch in a task and returns at once, because Pipecat gives all
   processors' setup a shared 20 s. The whole launch is bounded by `launch_timeout` (default 90 s):
   - A direct join creates the session while the bot is still connecting (*connect overlap*).
     `LiveKitAvatarSession.launch_while_connecting()` resolves the avatar token for the bot identity and
     room the bot token names, and sends the session request to the Synthesia API. Then it waits for the
     transport to connect and borrows its room, checks that the bot connected as that identity to that
     room, registers the playback RPC handlers, and waits for the 201 and the avatar participant.
   - A relayed join creates a private room and connects the bot to it first. `LiveKitAvatarSession.launch()`
     then registers the playback RPC handlers, resolves the avatar token, creates the session, and waits
     for the avatar participant.
   - Either way the handlers exist before the session is active: the worker ends its session if
     `lk.playback_finished` is missing when an utterance finishes, and no utterance is sent before then.
   - The adapter finishes: a direct join unsubscribes the bot from the avatar; a relayed join starts
     track capture.

   Speech that arrives during the launch is **pre-launch audio**. It waits in the send queue. The first
   interruption that finds audio queued keeps it, and later ones drop what came after it (*pre-launch
   interruption*).
   - *Early start*: if this launch's avatar publishes its video before the 201, the launch finishes
     without waiting for the 201, which typically trails the video by about half a second.
2. **Session active.** The send loop starts writing the send queue to the worker, one byte stream per
   utterance. The service fires `on_avatar_connected` once the 201 is in, which after an early start is
   just after; `on_avatar_ready` fires after it, once every session avatar is prepared.
3. **End.** The pipeline stops or is cancelled, or the avatar leaves.
   - A graceful stop first lets the send queue **drain** and the last utterance finish playing, for up
     to 5 s.
   - A cancel interrupts the avatar at once, and the teardown lets that interruption reach the avatar
     worker, for up to 3 s, before it closes the session. The `CancelFrame` itself is not held.
   - `on_session_ended(reason)` fires exactly once, followed by `on_metrics`.

If the launch fails, `required` decides what happens:
- With `required=True` (the default), the service pushes a permanent error, so Pipecat marks it
  unusable (FAILED). It drops the send queue and then ends the pipeline itself by pushing an
  `EndWorkerFrame` upstream, whatever the pipeline's `processor_unusable_policy`. Under `CANCEL`,
  Pipecat cancels the pipeline instead, and it still ends only once.
- With `required=False`, the bot continues **voice-only** (DEGRADED): TTS audio, including any
  pre-launch audio already queued, passes through to the user transport.

`required` governs the launch only: after an unexpected leave, the bot continues voice-only whatever
`required` says.

## Glossary

A term may be one word or several. Terms marked *(rule)* are stated in full under
[Design rules](#design-rules).

### People and processes

- **developer**: the person or company that builds and runs the bot, with this package installed in
  it; Synthesia's customer. Public docs address them as "you". Writing about the code, such as PR
  descriptions, says "the developer", never "you", which there would read as Synthesia. "Customer"
  is only for commercial topics, such as billing and support.
- **bot**: the Pipecat pipeline, and its LiveKit participant.
- **bot process**: the developer's process that runs the bot, and this package with it.
- **user**: the person the bot talks to. Their software, which connects through the user transport,
  is the **user's app** (Pipecat calls it the client, as in `on_client_connected`).
- **human participant**: any participant whose LiveKit kind is standard rather than agent. That is
  the user, or anyone else in the room. The worker counts these (see *no-humans-left rule*). One such
  participant is not actually a human: the bot in a private room is a standard participant on purpose,
  so the worker counts it as human, as intended (see *agent-kind requirement*).
- **avatar worker**: Synthesia's hosted process. It joins the room, receives the bot's audio, renders
  the avatar, and publishes its video and audio.
  - User docs always say "avatar worker", because Pipecat has its own `PipelineWorker`.
  - Plain "worker" is fine inside `livekit/`.
- **avatar participant**, **avatar identity**: the avatar worker's presence in a room, and its LiveKit
  identity (`avatar_identity`, default `synthesia-avatar-agent`). "The avatar joined" and "the avatar
  left" refer to the avatar participant.
- **avatar**: the character the user sees.
- **the Synthesia API**: the public HTTP API that creates sessions.
- **the LiveKit plugin**: `livekit-agents[synthesia]`, which this package keeps parity with. Don't call
  it "upstream": in Pipecat, upstream is a frame direction. It is not the archived
  `livekit-plugins-synthesia` repository, its prototype from before it moved into livekit-agents, whose
  API differs.

### Avatars

- **avatar id**: an id with the `av_` prefix. This is the only form used inside the package and in its
  API.
- **unprefixed avatar id**: an avatar id without `av_`, used only on the wire *(rule: unprefixed ids on
  the wire)*.
- **session avatars**: the 1–5 avatars in `avatar_ids`. The worker prepares all of them at launch, and
  one is active at a time.
- **active avatar**: the avatar being rendered now (`active_avatar_id`). At launch, it is the first in
  `avatar_ids`.
- **readiness**, **avatar ready**: the worker's preparation state for each avatar (`preparing`,
  `ready` or `failed`). `on_avatar_ready` fires when every session avatar is ready.
- **swap**: make another session avatar the active one (`swap_avatar()`).
- **pending swap**: an `avatar_id` settings change received during the launch. It is applied once the
  session is active.

### Rooms and transports

- **LiveKit project**: the developer's LiveKit Cloud project or self-hosted LiveKit server. Both kinds
  of room live in it, and its API key and secret stay in the bot process.
- **user transport**: the Pipecat transport the user connects through (Daily, SmallWebRTC or
  LiveKit). "Pipecat over LiveKit" and "Pipecat over Daily" name it.
- **join mode**: direct join or relayed join (`JoinMode.DIRECT` / `JoinMode.RELAYED`), as described
  above.
- **transport room**: the LiveKit room a `LiveKitTransport` is connected to. The user is in it. It is
  **borrowed**.
- **private room**: the LiveKit room a relayed join creates for the bot and the avatar worker. It is
  **owned**.
- **borrowed**, **owned**: whether the plugin controls a room's lifetime (`RoomHandle.owns_room`)
  *(rules: borrowed-room rule, room ownership handoff)*.

### Session lifecycle

- **session**: one Synthesia interactive-avatar session (`session_id`, `ses_…`). `AvatarSession` is
  the package's handle on one.
- **HTTP session**: the `aiohttp` session. The public argument is named `session=`, following Pipecat's
  convention, but internally it is `http_session`.
- **launch**: everything from `setup()` until the avatar participant joins *(rule: launch timeout)*.
- **connect overlap**: in a direct join, sending the session request while the bot is still connecting
  (`LiveKitAvatarSession.launch_while_connecting()`) *(rule)*.
- **early start**: finishing the launch when this launch's avatar publishes its video, before the 201.
  The minted avatar token carries a random `synthesia.launch` id, and only the avatar carrying it counts:
  never a stale avatar with the same identity, nor one joined with a caller-provided token (no id), which
  wait for the 201. `AvatarSession.wait_session_created()` then waits for the 201, bounded by what is left of
  the *launch timeout*; what needs the session id (`on_avatar_connected`, the metrics' ids, the log lines'
  prefix) waits for it. A request that fails then, or runs out of time, ends the service as a failed launch,
  run as an unexpected leave is: terminal state first, the avatar stopped with a clear buffer, then the
  close. A leave after an early start is an unexpected leave, even before the 201 and with
  `required=True`. A retried request (`on_retry`) ends the trust: a failed attempt may have launched the
  avatar in the room, and the retry's worker, with the same token, can replace it. So a retry before the
  video means no early start, and one after it holds sending (`AvatarSession.sending`) until the request
  settles (its 201 or its failure), bounded by the launch timeout, and during a graceful stop by the drain
  timeout (a request still out at that bound loses the goodbye): the send loop leaves held audio queued, for
  the voice-only handover or an interruption to find, and at most one chunk, whose write was already under
  way, reaches the avatar after the retry. Held audio shares the pre-launch cap. What the avatar was sent
  before the retry, often the greeting, is cut off if it is replaced (accepted, as rare). When the API
  deduplicates retries the hold only wastes one round trip, and should go.
- **join**, **join wait**: the avatar participant arriving in a room, and the wait for it after the
  session is created. The bot *connects* to a room; the avatar *joins* it.
- **session active** (`ServiceState.SESSION_ACTIVE`): the avatar participant has joined and audio can
  be sent to it. It does not mean that video is flowing yet, or that every avatar is ready, or, after an
  *early start*, that the 201 has arrived.
- **ready**: only ever *avatar ready* (see Avatars) or Pipecat's *transport ready*
  (`OutputTransportReadyFrame`).
- **leave**, **leave reason**: the avatar participant left, or it stopped publishing video, or the bot
  lost the room. The reasons passed to `on_avatar_disconnected` are `avatar_left_room`,
  `avatar_video_stopped` and `room_connection_lost`.
- **unexpected leave**: a leave while human participants remain, once the *no-humans-left window* has
  run out. An error is pushed, and the bot continues voice-only.
- **no-humans-left end**: the worker ends the session when the last human participant leaves: at once
  when that participant's client says it is leaving (a hang-up), otherwise about 5 s after. This is a
  normal end, not an error.
- **no-humans-left window**: in a direct join, when the avatar leaves while humans still appear
  present, the wait of up to 2 s for the last human's own leave before the leave is called unexpected
  *(rule)*. Skipped when the leave reason is `room_connection_lost`. For a genuine unexpected leave,
  `SessionMetrics.ended` is stamped at the end of the window, up to two seconds after the avatar left.
- **same-identity replacement**: a new avatar participant joins with the same identity, and livekit-server
  removes the old one. This happens when a launch retried with the same token replaces a stalled worker
  *(rule: replacement grace)*.
- **end reason**: the value passed to `on_session_ended`. One of `pipeline_ended`,
  `pipeline_cancelled`, `pipeline_cleanup`, `no_humans_left`, `avatar_left`, `launch_failed`,
  `launch_timeout`, or `degraded`.
- **voice-only**: the avatar is unavailable, and TTS audio passes through to the user transport, so the
  bot keeps talking.
- **voice-only handover**: when that happens and the pipeline continues, the audio still in the send
  queue is pushed to the user transport too, ahead of any later TTS audio, so none of it is lost or
  overtaken. A failed launch with `required=True` ends the pipeline instead, so its queued audio is
  dropped.
- **teardown**: the service releasing everything, once.
- **close**: one object releasing its own resources.
- **rollback**: undoing a launch that failed part-way.
- **release**: disconnecting from and deleting a private room, and closing the LiveKit API client.
  Deleting can be turned off with `delete_room_on_close=False`.

### Audio to the avatar

- **utterance**: the audio of one TTS response (`TTSStartedFrame` … `TTSStoppedFrame`), sent as one
  byte stream. livekit-agents and the worker call this a *segment*.
- **chunk**: the audio of one `TTSAudioRawFrame`.
- **byte stream**: the LiveKit byte stream (topic `lk.audio_stream`) that carries one utterance
  *(rule: one-open-stream rule)*.
- **send queue**, **send loop**: the FIFO of chunks and utterance ends between `process_frame` and the
  worker, and the task that writes it to the session.
- **pre-launch audio**: speech that arrives before the session is active. It waits in the send queue,
  up to a cap, dropping the oldest audio first, and the amount dropped is counted in
  `prelaunch_dropped_bytes`.
- **idle close**: ending an utterance after 1 s with no new chunk, for TTS services that never send
  `TTSStoppedFrame`.
- **orphaned open**: a byte-stream open whose caller was cancelled mid-flight. It must still be closed.
- **detached close**: a close that keeps running after its caller stops waiting for it.
- **interruption**: Pipecat's `InterruptionFrame`, and the service's response to it:
  - drop the queued audio, once the session is active (before that, see *pre-launch interruption*);
  - mark the TTS context stale;
  - end the speaking state at once, with no stop hysteresis;
  - send **clear buffer** (`lk.clear_buffer`), so the worker drops the audio it holds;
  - end the utterance;
  - restart the send loop.
- **pre-launch interruption**: an interruption before the session is active. Pipecat sends one at every
  user turn start, even when the bot is not speaking, and nothing has played yet:
  - The first that finds pre-launch audio queued keeps it, and it plays once the session is active: the first
    audio held when the user spoke, normally the greeting. One that finds no audio queued (nothing, or only
    utterance ends) keeps nothing and sets no boundary. The plugin cannot tell a greeting from a reply: when
    the user's first turn finds no audio, the reply to it can be kept along with the greeting, or in place of
    one, and play before the current one. That is usually one reply at most, because Pipecat cancels a reply
    still being produced when the user speaks again; the plugin itself does not enforce it. The rest of the
    interrupted utterance is a stale context, dropped as it arrives, so the service queues an utterance end
    after the kept part, unless the last queued item is already one, so the next utterance does not join its
    byte stream. A TTS service without context ids (OpenAI Realtime) has nothing stale. The rest of an
    interrupted utterance can still arrive and be queued, after the first turn or a later one, and it then
    plays, as it would once the session is active. Its own end may follow the added one; an end with no byte
    stream open closes nothing.
  - A later one drops what was queued since the kept audio (`SendQueue.truncate_after`): it answered a turn
    this one supersedes, as an interruption would drop it once the session is active.
  - There is no clear buffer, as there is no session yet. What is dropped is not counted in
    `prelaunch_dropped_bytes`, which counts only what the cap drops; one INFO line names the bytes kept.
- **stale context**: an interrupted TTS `context_id`. Audio that arrives for it late is dropped. Its late
  `TTSStartedFrame` and `TTSStoppedFrame` are forwarded but otherwise ignored: they neither change the
  current context nor end the utterance being sent.
- **generation**: a counter that lets async work started before an interruption recognise that it is
  stale.

### What the worker sends back

- **playback events**: the worker's `lk.playback_started` and `lk.playback_finished` RPCs.
- **speaking state**: `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame` and `BotSpeakingFrame`,
  derived from the playback events *(rule: speaking-state ownership)*.
- **stop hysteresis**: the 0.35 s wait after `playback_finished` before `BotStoppedSpeakingFrame`, so
  that back-to-back utterances do not flap.
- **interruption latency**: the time from an interruption to the worker's
  `playback_finished(interrupted=True)` (`interruption_latency_seconds`). Not recorded in this release
  (docs/protocol-compatibility.md, section 10): a finish doesn't say which clear buffer it answers, and a
  chunk written before an interruption may be read by the worker after its clear buffer, so the plugin
  can't tell which interruption a finish measures. The worker answers a clear buffer only once it has read
  the first frame of an utterance's audio.
- **discarded early audio**: the worker also sends `playback_finished(interrupted=True)` at position 0.0
  for audio it held before binding the bot and then threw away unplayed, and the plugin's speaking state
  does not change. With no interruption pending the finish only arms the stop timer, which the next
  `playback_started` cancels; with one pending (a clear buffer sent before the bind, which the worker
  answers with no finish of its own) it is that interruption's answer. In a log,
  such a finish with no `playback_started` before it means audio sent before the bind never played, for one
  of four reasons: it was older than 20 s by default, a clear buffer before the bind cancelled it (the user
  interrupted the greeting), the session ended before the worker bound the bot, or the bot is not the
  participant the worker binds to. The first two arrive at the bind, the third when the worker closes, and
  the last at the bind of whoever the worker binds, or at the close if nobody.
  If no audio ever plays and the session ends about 30 s in, check that the avatar token's
  `lk.publish_on_behalf` equals the bot's identity.
- **tolerant reader**: see the rule of the same name.

### Relayed join and direct join

- **track capture**: in a relayed join, turning the avatar's video and audio tracks in the private room
  into Pipecat frames (`AvatarTrackCapture`).
- **media gate**: avatar frames are pushed only once the session is active and the transport is ready.
- **self-hearing**: in a direct join, the bot transcribing its own avatar, because the avatar is an
  ordinary participant in the transport room *(rule: self-hearing guard)*.
- **avatar unsubscribe**: the automatic half of the guard. The bot unsubscribes from the avatar's audio
  and video.
- **avatar audio filter**: the half the developer places. `AvatarAudioFilter`, returned by
  `room.input_filter()`, goes right after `transport.input()`.

### Tokens and identification

- **avatar token**, **bot token**: the LiveKit access tokens the avatar worker and the bot join with.
- **minted**, **provided** avatar token: minted from LiveKit API credentials (`MintedAvatarToken`), or
  supplied by the caller and validated (`ProvidedAvatarToken`).
- **publish-on-behalf**: the `lk.publish_on_behalf` token attribute, which binds the avatar to the bot.
- **client identification**: the `User-Agent` and `X-Synthesia-Client` headers, and the
  `synthesia.client` / `synthesia.protocol` / `synthesia.join_mode` token attributes (`ClientInfo`).
- **advisories**: the Synthesia API's `Deprecation`, `Sunset`, `Warning` and deprecation `Link`
  response headers, and a successful response's `warnings[]`. Each is logged once, at WARNING.
- **installed-version channels**: client identification, advisories, `client_upgrade_required`,
  `Retry-After` and the `API_VERSION_HEADER` slot: the ways Synthesia can reach a version of the
  package that is already installed *(rule: installed-version channels)*.

### Errors

- **error type**, **category**: our `ErrorType` (its values match the LiveKit plugin's) and Pipecat's
  `ErrorCategory`. Say "type" for ours and "category" for Pipecat's.
- **permanent**, **unusable**: Pipecat marks a processor unusable when it pushes a permanent error:
  one with a permanent category, or pushed with `force_treat_as_permanent` *(rule: permanent-error
  rule)*.
- **misuse error**, **session error**: a mistake in the calling code, such as a bad constructor argument
  (`ValueError` / `TypeError`), versus any failure once running (`SynthesiaError`) *(rule: misuse vs
  session errors)*.
- **problem details**: the RFC 9457 error body the Synthesia API returns.

## Design rules

### Concurrency and teardown

**state-before-await.** Room callbacks run concurrently, each on its own callback task. There is no
lock and no queue between them. So every callback makes all its state changes before its first
`await`, and re-checks state after every `await`. (`AvatarCallbacks`, `CallbackTasks`)

**cancel-proof cleanup.** Rollback, release and close run in a task of their own and are awaited in a
loop that absorbs a second cancel. The caller's cancellation is re-raised only after the cleanup has
finished. This is needed because Pipecat's `cancel_task` can cancel twice, and cancelling
`room.disconnect()` wedges the LiveKit room: every later call on it blocks for its full timeout.
Nor is a private room's in-flight `Room.connect()` ever cancelled, since that can make the SDK send the
bot process SIGTERM. A cancel, or the adapter's own budget, deletes the room, waits (bounded) for the
connect to end, and disconnects the room if it connected. (`_async.await_cancel_proof`)

**bounded waits.** Every wait during launch, rollback, release and teardown has a time bound. The
LiveKit-side bounds live in one table (`livekit/_types.py`), and the service's own live at the top of
`service.py`, which cannot import `livekit/` (see *LiveKit-free core*). A bound around cancel-proof
cleanup is written with `_async.wait_bounded`, never `asyncio.wait_for`: the cleanup absorbs the cancel
that `wait_for` sends when its timeout expires, so the wait would last as long as the cleanup. Past the
bound, the cleanup carries on as a detached close, held by a strong reference until it finishes.

Below Python 3.12, Pipecat replaces `asyncio.wait_for` with `wait_for2.wait_for` for the whole process.
There, a cancel that races the waited-for coroutine's return raises a `CancelledError` that carries the
result instead of returning it. So a `wait_for` around a coroutine that returns a resource must not drop
it on a cancel: the service holds the launch's session from inside the coroutine it waits for, and closes
it if a cancel arrives instead (`_launch_and_hold`).

**single flight.** Teardown, session close and private-room release each run once. A later caller
joins the run already in progress.

**launch timeout.** `launch_timeout` bounds the entire launch, enforced by one `wait_for` around
`AvatarRoom.launch()`. Adapters cap their own waits by what is left of it, so that a failure inside an
adapter produces its own error rather than a bare timeout.

### Frames and speaking state

**frame contract.**
- The service consumes `TTSAudioRawFrame` and forwards every other frame unchanged.
- In `DEGRADED` and `ENDED`, TTS audio passes through (voice-only).
- In `FAILED`, TTS audio is dropped. The service is ending the pipeline, and the `EndFrame` enters at
  the pipeline source, behind whatever the TTS service is still producing, so passing that audio
  through would play the greeting of a bot that is about to hang up.
- `process_frame` never awaits the network: audio goes into the send queue, and the send loop does the
  I/O.

**speaking-state ownership.** In both join modes, the service is the only emitter of the bot speaking
frames, and it derives them from playback events. In a relayed join, the avatar's audio is pushed as
plain `OutputAudioRawFrame` rather than a TTS or speech audio frame, because the output transport emits
bot speaking frames for those and would emit a second set.

### Errors and logging

**permanent-error rule.**
- Only a failed launch with `required=True` is pushed as permanent (`force_treat_as_permanent`,
  keeping its own category), which marks the service unusable. The service then ends the pipeline
  itself; the error stays permanent anyway, so that `is_usable` is truthful for `on_error` handlers
  and a `CANCEL` policy is honoured.
- Every other error is pushed with a non-permanent category (a permanent one is downgraded to
  `APPLICATION`), so the pipeline keeps running. The true type and category stay on
  `ErrorFrame.exception`.
- `push_error` is called only from `_push_synthesia_error`.
- Only coroutines the developer calls raise, and a failure in one is always a `SynthesiaError`. A bad
  argument to one is a misuse error instead (`swap_avatar(timeout=0)` is a `ValueError`).

**misuse vs session errors.** Constructor mistakes raise `ValueError` or `TypeError` immediately,
so they surface at startup rather than minutes into a call. That includes a keyword the constructor does
not take: Pipecat's `BaseObject` would accept and drop it, so the service checks its keywords against
the ones Pipecat's constructor chain declares, read from the signatures. A failure once running is a
`SynthesiaError`. So a direct join checks at construction everything it can know before the call:
the transport's type, the Pipecat internals it reads, the bot token's kind, the LiveKit URL the
avatar worker is told, and a provided avatar token against the bot identity and room the bot token
names. Only the tokens' expiry waits for the launch, a provided avatar token's and the bot token's,
because a token can expire in between. A relayed join checks its LiveKit URL and credentials at
construction too. In both join modes the service checks the Synthesia API URL at construction: it must
be `https://` to `synthesia.io` or a host under it, or go to a loopback host, unless the developer sets
`allow_any_api_host`, because the service sends the API key and a LiveKit token to it.

**redaction rule.** No secret appears in a log line, in `str()` or `repr()` of an error, or in
`SynthesiaError.body`. Never `repr()` an aiohttp error: its repr echoes the request headers.
(`_redact.py`)

### Rooms

**borrowed-room rule.** The plugin never connects, disconnects or deletes a borrowed room. Pipecat's
transport owns its connection, and closing it would break the pipeline.

**room ownership handoff.** A private room belongs to `PrivateLiveKitRoom` until it is handed to
`LiveKitAvatarSession.launch()`. From then on, only the session releases it, so the room is never
deleted twice and never leaked.

**self-hearing guard.** In a direct join the bot must never transcribe its avatar.
- The avatar unsubscribe happens automatically.
- The avatar audio filter must be placed by the developer, because the service sits after TTS and
  never sees input frames. The service warns once if the filter is missing.

**agent-kind requirement** and **no-humans-left rule.** The worker ends the session when the last human
participant leaves: at once if that participant's client said it was leaving, otherwise about 5 s after.
So in a direct join, the bot's token must carry the top-level
`kind="agent"` claim.
Without it, the worker counts the bot as a human, the session never ends, and billing never stops.
Pipecat's token helpers set only the `video.agent` grant, which does not change the kind, and its dev
runner's `-t livekit` bot token comes from one of them. `LiveKitTransportRoom`'s constructor reads the
transport's token and refuses one without the claim, whatever `required` says: `required` decides
what a failed launch does, not how strictly the configuration is checked.

In a relayed join the bot is a standard participant in its private room, on purpose: neither the agent
kind nor the `video.agent` grant. livekit-server counts a participant with either as *dependent*, and
keeps a room open only while a participant that is not dependent is in it. The avatar worker is an
agent, so with an agent bot the private room would close `empty_timeout` (120 s) after its creation,
however busy it was. With a standard bot, the room stays open while the bot is connected, and closes
`departure_timeout` (20 s) after the bot leaves or dies. The worker binds the bot by identity,
through `lk.publish_on_behalf`, whatever its kind, and ends the session when it leaves.
(`token.mint_bot_token`)

### The worker protocol

**one-open-stream rule.** At most one byte stream is open to the worker at a time, and every stream
that was opened is closed, including an orphaned open. An unclosed stream wedges the avatar worker's
receiver, and a second concurrent stream drops the tail of the first.
(`UtteranceStreamer`)

One exception is accepted: an orphaned open slower than `STREAM_CLOSE_SECS`. Its close cannot be
requested until the open returns a writer, and the streamer stops waiting at the budget, so the next
utterance's stream can open while the orphaned one is still open. The orphaned stream carries no audio,
so the overlap costs a delay or an empty tail, depending on the receiver, and it happens only when
the data channel is already stalled. The orphaned stream is still closed, as a detached close.

**connect overlap.** In a direct join, the session request goes out while Pipecat's transport is still
connecting, so a launch costs about the longer of the two rather than their sum. Nothing in the request
needs the connection: the avatar token takes the bot identity and room from the bot token's own claims,
and the avatar worker publishes and acknowledges without waiting for the bot. Two constraints keep it
safe:
- *The playback RPC handlers are registered before the session is active*, not before the request. The
  worker calls `lk.playback_finished` only once an utterance finishes, and the send loop sends nothing
  until the session is active.
- *After the connect, the bot's identity and room are checked* against the bot token's claims. The worker
  binds to the identity in the avatar token, so a bot that connected as anyone else, or to another room,
  would lose the session silently, about 30 s later, when the worker's agent-join watchdog runs out.

An expired bot token fails the launch before the request, unless the bot has already connected with it
(LiveKit checks a token only at join): it could not connect, while the avatar worker's join would still
succeed. A bot token that names no room falls back to sending the request after the
connect. If the connect fails after the request went out, the request is abandoned with one warning: the
avatar worker may already have joined, and it ends that session itself when the bot never arrives. A
relayed join does not overlap; it registers the handlers, then sends the request.
(`LiveKitAvatarSession.launch_while_connecting`, `LiveKitTransportRoom.launch`)

**unprefixed ids on the wire.** Avatar ids carry the `av_` prefix everywhere in the package and its
API. Only `swapAvatar` payloads and `avatar_status` keys use the unprefixed id, because the worker
keys avatars that way and does not strip the prefix itself. (`config.py`)

**no-humans-left window.** In a direct join, when the avatar leaves while human participants still
appear present, the service waits up to 2 s (`livekit/_types.NO_HUMANS_LEFT_WINDOW_SECS`) for the last
of them to leave before it calls the leave unexpected. If none remains inside the window, the end is
`no_humans_left` with no error; otherwise, when it runs out, the leave is unexpected, as before.
- *Why:* on a hang-up the worker leaves at once (*no-humans-left rule*), and its leave can reach the bot
  about a second before the bot's own `participant_disconnected` for that human, which LiveKit sends
  separately (measured in a live call, 2026-11-10 UTC). Read at the avatar's leave, the
  room still holds the human.
- *What waits:* only the error and the end reason. The terminal state, the retired send loop and the
  hold on later TTS audio start at the leave (*state-before-await*), so nothing reaches the avatar that
  left, and audio that arrives during the window is handed over in order if the window runs out.
- *Ending early:* a pipeline that stops or is cancelled during the window cuts it short, and the end is
  the pipeline's, as for any leave during shutdown. A leave that finds no human present does not wait.
- *Not after a lost connection:* a leave whose reason is `room_connection_lost` does not wait. The bot's
  own connection is gone, so its view of the room means nothing, and no `participant_disconnected` can
  arrive (rtc does not clear `remote_participants` on `disconnected`); waiting would only delay the error.
- *Where:* the service waits, because the state and the classification are its own; the session answers
  the room question (`AvatarSession.wait_no_humans_left()`), listening for itself, since the close that
  starts at the leave removes the session's own listeners.

**replacement grace.** A video unpublish, or a disconnect with no stated reason, starts a 1 s grace
before the avatar is declared left. A re-join inside the grace cancels it.
- *Why:* a same-identity replacement reaches other participants as a disconnect with no reason, and a
  real leave always states one. The grace resolves the ambiguity either way.
- *Deadline and reason:* the first signal fixes the deadline, so a chatty departure cannot postpone
  the report. The most specific signal fixes the leave reason.
- *Playback:* the worker that disconnected stopped whatever it was playing, and no `playback_finished`
  will come for it, so the disconnect also arms the stop timer, as a finish does (`on_playback_lost`). A
  replacement that starts playing within the stop hysteresis keeps the bot speaking.

**tolerant reader.** Worker payloads are read leniently:
- unknown keys are ignored;
- unknown avatar states are kept and treated as "not ready";
- malformed values are logged once and ignored;
- a success is never turned into a failure because of its format: a swap's confirmed avatar id is
  accepted if it is any non-empty string, and only given the `av_` prefix, because the worker has
  already swapped by the time it answers.

The worker adds fields over time, and livekit-agents versions differ in what they send.
(`readiness.py`, `rpc.py`)

### Packaging

**Pipecat-internals rule.** The package reads Pipecat private attributes in exactly two places:
`livekit/_transport_access.py` (the `LiveKitTransport`'s room, URL and token) and
`_compat.output_transport_params()`. Both are re-checked on every Pipecat release. The transport's
are checked once, when `LiveKitTransportRoom` is constructed; a release that moved them is a
`TypeError` naming both versions. The tests reach further, into private state of `LiveKitTransport`,
its `LiveKitTransportClient` and Pipecat's event-handler machinery, in `tests/fakes/fake_transport.py`,
`tests/unit/test_transport_access.py` and `tests/unit/test_transport_room.py`; those are re-checked with
the same releases.

**Pipecat range.** Any `pipecat-ai` 1.x from 1.14 is allowed, because a cap below the next minor would
refuse the plugin to a user already on a newer Pipecat. `_compat.TESTED_PIPECAT_VERSIONS` lists the
versions the PR CI matrix runs; the nightly `compat-main` job adds Pipecat's newest release and its main
branch. On a newer minor than every tested one (a pre-release of one included), constructing a
`SynthesiaVideoService` logs one WARNING per process; tested minors and their patches log nothing. The
floor is 1.14 because Pipecat 1.13 and older require Pillow<13. Git builds of Pipecat are not reliably
detected: a git install of Pipecat's main branch on 2026-11-09 reported `1.15.1.dev627`, a build of the
next patch rather than the next minor, so the warning reads it as 1.15 whatever unreleased work main
holds.

**Startup order at Pipecat 1.13+.** The worker pushes the `StartFrame` while processors are still being
set up; each processor holds it until its own `setup()` returns, so `setup()` still precedes `start()`
and `next` is linked before both. A cancel before the worker's start task has returned (the `StartFrame`
may already have reached the sink and `on_pipeline_started` fired), as when a `required=True` launch
fails inside `setup()`'s launch task under `ProcessorUnusablePolicy.CANCEL`, is resolved by cleanup
alone: no `CancelFrame` is pushed, `on_pipeline_finished` does not fire, and the service sees
`cleanup()` without `cancel()`, which the single-flight teardown already handles.
`tests/unit/test_service_lifecycle.py` pins both paths.

**LiveKit-free core.** The core package imports without the `[livekit]` extra. Only
`pipecat_synthesia.livekit` imports `livekit` or `jwt`.

**installed-version channels.** From its first release, the package carries every channel Synthesia
may later need to reach a version that is already installed:
- client identification: the `User-Agent` and `X-Synthesia-Client` headers;
- the `synthesia.*` token attributes, where an absent `synthesia.protocol` means `"1"`;
- the advisory response headers (`Deprecation`, `Sunset`, `Warning`, and a `Link` with
  `rel="deprecation"`) and a successful response's `warnings[]`, each logged once at WARNING;
- `client_upgrade_required`, mapped to a permanent `CLIENT_UPGRADE_REQUIRED` error whose message
  carries `min_supported_version` and `upgrade_url`;
- honouring `Retry-After`: waited for before a 5xx is retried, and kept on the error as
  `retry_after`;
- the single `API_VERSION_HEADER` slot.

*Why:* an installed version can never gain the ability to read a new header or error code. These
channels are how Synthesia tells a stale install what is changing, or turns it away with a clear
message, short of breaking it, so they have to ship before anything uses them. Several are unused
today; that is not a reason to remove them. (`api.py`, `errors.py`, `protocol.py`,
`_client_info.py`, `livekit/token.py`; docs/protocol-compatibility.md)

## Naming conventions

- **One term per concept, one concept per term.** A term may be one word ("utterance") or several
  ("replacement grace"). Add new terms to the glossary above.
- **Prose uses the full term**, introduced with a plain description the first time it appears in each
  document, PR or page.
- **Identifiers use the shortest name that is unambiguous** in the package's vocabulary, and let the
  enclosing enum, class or module supply the context.
  - Add a qualifier only when the bare word collides with another term. `SESSION_ACTIVE` is qualified
    because "active avatar" is also a term.
  - Never go below the term's head word: `JoinMode.RELAYED`, `REPLACEMENT_GRACE_SECS`.
- **Test every name with three questions:**
  1. Does it have exactly one reading?
  2. Does it promise no more than actually holds?
  3. Does it fit its siblings?
- **Wire names are verbatim** and live only in `protocol.py`.

Terms to avoid, because they are ambiguous:

| Avoid | Use |
|---|---|
| "ready" for the live state | session active |
| upstream (for the LiveKit plugin) | the LiveKit plugin |
| worker, alone, in user docs | avatar worker |
| agent (for the bot) | bot, except in `kind="agent"` |
| raw id | unprefixed avatar id |
| segment | utterance |
| client, server, client-side, server-side (each is relative: the bot process is a server to the user's app and a client of the Synthesia API) | the party by name: the user's app, the bot process, the Synthesia API, livekit-server. Identifiers such as `SessionApiClient` keep their names. |
| "you" / "your" for the developer, in writing about the code | the developer |
| internal Synthesia service names, in code and public docs | the Synthesia API |
