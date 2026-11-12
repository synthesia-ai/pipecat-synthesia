# Protocol compatibility

<!-- Wire constants: src/pipecat_synthesia/protocol.py is the single source of truth -- every name
     below is a constant there, not a string literal repeated. -->

This plugin and the Synthesia Interactive Avatar worker talk over a wire protocol that ships on two
independent release trains: the plugin is a PyPI package a customer pins, and the worker is a
service Synthesia deploys. A customer can therefore be running a months-old plugin against a worker
deployed this morning, or the reverse. This document states the rules that keep those combinations
working, so that neither side has to coordinate a release with the other.

The rules are constraints on **us**, not documentation of what happens to be true today.

Some of what follows exists before anything uses it: client identification, the `synthesia.*` token
attributes, the advisories, `client_upgrade_required`, honouring `Retry-After`, and the
`API_VERSION_HEADER` slot. An installed plugin can never gain the ability to read a new header or
error code, so these channels to already-installed versions have to ship in the first release, and
must not be removed for being unused (*installed-version channels* in
[architecture.md](architecture.md)).

## 1. Additive-only

**Never rename a wire name, never reuse one, never change what an existing value means.** Every
constant in `protocol.py` is permanent once released. Adding is free; changing is not.

This applies to RPC method names, data-stream topics and attributes, participant attributes, token
attributes, JSON field names in RPC payloads, and request/response body fields.

When something must change shape, add the new name alongside the old, teach both sides to accept
either, and retire the old one only after the compatibility window (see "API versioning
header") has passed.

## 2. Two namespaces, two owners

| Names | Owner | Rule |
| --- | --- | --- |
| `lk.*` — `lk.audio_stream`, `lk.clear_buffer`, `lk.playback_started`, `lk.playback_finished`, `lk.publish_on_behalf` | **LiveKit.** Part of the `livekit-agents` avatar data-stream protocol. | Use as specified; never extend. A name we invent under `lk.*` will collide with a future LiveKit release. |
| `swapAvatar`, `avatar_ready`, `avatar_status`, `avatar_audio`, `avatar_video` | **Synthesia.** Pre-existing worker protocol, inherited from `livekit-plugins-synthesia`. | Additive-only. These are already deployed and cannot be renamed. |
| `synthesia.*` — `synthesia.protocol`, `synthesia.client`, `synthesia.join_mode`, `synthesia.launch` | **Synthesia, new.** Anything this plugin introduces. | All new names go here. The namespace exists so that anything we add is unmistakably ours. |

Because the middle row was inherited rather than designed, it does not follow the `synthesia.*`
convention. That is a fact to work around, not a precedent to extend: **new names go in
`synthesia.*`.**

## 3. Tolerate what the other side does not understand

The plugin must keep working against a worker that predates any feature it knows about.

**Unknown RPC method.** A worker that has never heard of a method answers with LiveKit's
`UNSUPPORTED_METHOD`. That is a normal answer from an older worker, not a failure of the call, and
it must never take down a session. `error_from_rpc` maps it to `ErrorType.CONNECTION` with the
LiveKit code preserved on `.code`, so a caller can tell "this worker is too old" from "this worker
is broken".

The full mapping (`src/pipecat_synthesia/livekit/rpc.py`):

| `RpcError.ErrorCode` | `ErrorType` | Reading |
| --- | --- | --- |
| `UNSUPPORTED_METHOD` | `CONNECTION` | Worker predates the method. |
| `RECIPIENT_NOT_FOUND`, `RECIPIENT_DISCONNECTED` | `CONNECTION` | Avatar is not in the room. |
| `RESPONSE_TIMEOUT`, `CONNECTION_TIMEOUT` | `TIMEOUT` | Retryable. |
| `APPLICATION_ERROR` | `AVATAR_SWAP_FAILED` | The worker ran the method and it failed. |
| anything else | `CONNECTION` | Code and message preserved verbatim in the message. |

**Unknown error code.** A `code` the plugin has never seen maps to `ErrorType.UNKNOWN` with the
Synthesia API's `detail` preserved verbatim, never to a status-derived guess. A new code from the API
therefore reaches the developer as text they can act on, instead of being silently reclassified as
something it is not.

**Unknown participant-attribute values.** An `avatar_status` state the plugin does not recognise is
treated as "not ready yet" and waiting continues. Only `failed` is terminal.

**Absent means the old default.** `avatar_ready` is set to `"true"` once every avatar is ready and
is otherwise **absent** — never `"false"`. Readers must treat missing as not-ready.

## 4. Token attributes, and "absent = 1"

These attributes let a future worker tell which plugins it is serving, and branch on their protocol
level, without any change to the plugins already installed. Every avatar token this plugin mints carries:

| Attribute | Value | Purpose |
| --- | --- | --- |
| `lk.publish_on_behalf` | the bot's identity | LiveKit-owned; tells the room the avatar publishes for the bot. |
| `synthesia.protocol` | `PROTOCOL_VERSION`, currently `"1"` | Lets the worker branch on plugin protocol level. |
| `synthesia.client` | `"pipecat-synthesia/<version>"` | Which SDK and version minted this token. |
| `synthesia.join_mode` | `"direct"` or `"relayed"` | How the avatar joins. Not the user transport: participant attributes are visible to everyone in the room. |
| `synthesia.launch` | a random 16-hex-character id, new for every launch | Lets the plugin recognise this launch's avatar when it publishes its video, so audio can start before the 201 (*early start*). The worker doesn't read it; it must keep it when it writes its own attributes. |

**A worker that sees no `synthesia.protocol` attribute must read it as `"1"`.** Tokens minted by
`livekit-plugins-synthesia`, and by any plugin build older than this attribute, carry nothing; the
absent case is therefore the original protocol, not an error. Bump `PROTOCOL_VERSION` only for a
change a worker genuinely has to branch on.

## 5. Request body: send nothing the Synthesia API has not accepted

The session-creation body is **exactly** three keys:

```json
{"avatarIds": ["av_..."], "livekitUrl": "wss://...", "livekitToken": "..."}
```

The session endpoint validates with `extra="forbid"`: an unrecognised key is a `400
validation_error`, not a field the API ignores. So a field must not be added to the body until
the Synthesia API has shipped support for it — **API first, then plugin.** This is the one place where
"additive is free" does not hold, and it is why protocol negotiation lives in token attributes
(see "Token attributes") rather than in the request body.

## 6. Identification headers

Every request carries these, so Synthesia can attribute traffic to a plugin version
without guessing, and can see which old versions are still calling before retiring anything they
depend on:

| Header | Value |
| --- | --- |
| `Authorization` | `Bearer <api key>` |
| `User-Agent` | `pipecat-synthesia/<ver> pipecat-ai/<ver> python/<ver> (<platform>)` |
| `X-Synthesia-Client` | compact sorted JSON of eight fields: the same six, plus `join_mode` and `transport` |
| `Accept` | `application/json, application/problem+json` |
| `Content-Type` | `application/json` |

Every key of `X-Synthesia-Client` is always present and set: the service fills in both of its own fields
before its first request, so `null` is never sent. `join_mode` is `"direct"` or `"relayed"`. `transport`
is the user transport, from a closed set read from the Pipecat package of the first output transport
downstream of the service: `livekit`, `daily`, `smallwebrtc`, `websocket`, `local`, `other` for any
other transport, or `unknown` when there is none. It is never a class name. A service inside a nested
`Pipeline` or a `ParallelPipeline` reads `unknown`: the walk ends at the inner pipeline's sink.

`Request-Id` on the **response** is captured onto every `SynthesiaError` built from a Synthesia API
response and prefixed to its message, so a bug report carries the one identifier that makes
Synthesia's logs searchable. Other errors raised after the session was created carry the create
call's request id instead, without the prefix.

## 7. API versioning header

The slot exists now so that adding the header later is one edit in one place. Not yet defined. When the Synthesia API specifies one, it goes in the single existing slot and
nowhere else:

```python
# src/pipecat_synthesia/api.py
API_VERSION_HEADER: tuple[str, str] | None = None
```

Set it to `("<Header-Name>", "<value>")` and `SessionApiClient.headers()` includes it on every
request. One slot, one edit, no second mechanism — that is the whole point of reserving it now.

The **compatibility window** — how old a plugin the worker will keep accepting — is still
open and will be stated in the README's Compatibility section once Synthesia commits to it.

## 8. Advisories

Before Synthesia breaks anything, it needs a way to tell the plugins already installed. The Synthesia
API can say so on any response, and the plugin logs what it says:

| Channel | Logged |
| --- | --- |
| `Deprecation`, `Sunset` and `Warning` response headers | `Synthesia API advisory <header>: <value>`, once per process for each header and value |
| a `Link` response header with `rel="deprecation"` | `Synthesia API advisory Link: <value>`, once per process for each value |
| `warnings[]` in a successful response body, each `{"code", "message", "sunset"?, "url"?}` | `Synthesia API warning <code>: <message> (sunset <sunset>, <url>)`, once per process for each code |

Each is a WARNING on the package's logger, and nothing else: an advisory never fails a request or
changes what the plugin does. The advisories reach the developer only as log lines, so the README
tells them to act on these. (`_log_advisories` and `_log_body_warnings` in
`src/pipecat_synthesia/api.py`)

`Retry-After` is the same kind of channel, for load. Before retrying a 5xx the plugin waits as long
as it says, capped at 10 s, even though the Synthesia API sends it only on a 429 today; a 429 is
never retried, and its value is kept on the error as `retry_after`.

## 9. What a breaking change looks like

This is the last resort, for when an advisory is not enough; the code is reserved now because a
plugin already installed could not learn it later. If the worker ever must reject an old plugin, it returns `client_upgrade_required`
(`ERROR_CODE_CLIENT_UPGRADE_REQUIRED`, reserved now, unused today). The plugin maps it to
`ErrorType.CLIENT_UPGRADE_REQUIRED` and — unusually — forces the category to `INVALID_REQUEST` so
that it is **permanent**, rather than letting Pipecat's status-code classifier map 426/410 to
`UNKNOWN` and make a version gate look retryable.

<!-- ErrorCategory and classify_http_status_code come from pipecat.utils.errors (Pipecat 1.10);
     is_permanent is what decides whether Pipecat retries. -->

The error message includes `min_supported_version` and `upgrade_url` from the response body when
the Synthesia API supplies them, so the developer is told what to upgrade to rather than just that they
must.

## 10. What we would ask for: playback events that name their byte stream

Not a dependency of any release. `lk.playback_started` and `lk.playback_finished` don't say which byte
stream they are for, so the plugin can't always match a `playback_started` to the utterance it starts, or a
`playback_finished(interrupted=True)` to the interruption it answers, and records neither time to first
byte nor interruption latency. The ask: one additive field, under a `synthesia.` key (section 2), in both
events' payloads, echoing the name of the byte stream the event is for (`AUDIO_` + hex, as the plugin opens
it). Installed plugins read these payloads leniently (*tolerant reader*) and would ignore it.
