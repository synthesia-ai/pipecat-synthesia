# Pipecat + Synthesia Interactive Avatar

Give a [Pipecat](https://github.com/pipecat-ai/pipecat) voice bot a photorealistic, lip-synced
face. You bring the brain — STT, an LLM and TTS — and Synthesia renders it as a person you can
interrupt mid-sentence.

Two files: [`bot.py`](bot.py), the bot, and [`_shutdown.py`](_shutdown.py), which ends calls in order on
Ctrl-C or SIGTERM. Roughly a dozen of `bot.py`'s lines are avatar-specific; the rest is a stock Pipecat
pipeline.

## Run it

```bash
cp .env.example .env    # then fill it in
uv run python bot.py -t webrtc
```

`uv run` installs what `pyproject.toml` lists first, `pipecat-synthesia[livekit]` from PyPI included.

Open <http://localhost:7860> and allow microphone access.

`pyproject.toml` allows any Pipecat 1.x from 1.14; the plugin is tested with 1.14, 1.15 and
1.16, logs one warning at startup on a newer minor, and a new minor can break it before it
is tested, so pin `pipecat-ai` if you need stability.

## What you need

| Variable | Where from |
| --- | --- |
| `SYNTHESIA_API_KEY` | [Synthesia developer portal](https://app.synthesia.io/#/developers) |
| `SYNTHESIA_AVATAR_ID` | An **Interactive** avatar id (see the note below) |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | A [LiveKit Cloud](https://cloud.livekit.io) project — free tier is fine |
| `OPENAI_API_KEY` | The LLM |
| `CARTESIA_API_KEY` | Speech to text *and* text to speech |

The LiveKit project hosts the avatar's **private room**. Your users never connect to it: the
avatar's video and audio are relayed through your Pipecat pipeline and delivered over your own
transport (SmallWebRTC here). That is a **relayed join**, the route for users on a transport other
than LiveKit. If your bot runs Pipecat over LiveKit, or you have not yet chosen between LiveKit and
Daily, the plugin's main route is a **direct join** instead: see "Going further".

> **A Studio avatar id is not an Interactive avatar id.** They look alike and are not
> interchangeable. The wrong one fails about four seconds in with `404 unknown_reference`, which
> presents as "the call ends before the avatar says anything" rather than as a bad id. A
> known-good stock avatar to start with is Jenny: `8788bef1-8020-46e0-a8f4-510ea9989b25`.

## How it fits together

```
transport.input() → STT → user aggregator → LLM → TTS → [ avatar ] → transport.output()
```

The avatar goes **immediately after the TTS**. It consumes the speech your pipeline already
produces and emits the avatar's video and audio; everything upstream is unchanged.

`BotStartedSpeakingFrame` and `BotStoppedSpeakingFrame` come from the avatar's *real playout*, not
from when audio was sent — which is what makes interruption feel right.

## Two things people get wrong

**VAD goes on the user aggregator, not the transport.**

```python
LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
```

`TransportParams` has no `vad_analyzer` field and silently discards one passed to it. The result
is a bot that never detects speech and can never be interrupted, with nothing in any log to say
why.

**Tell the transport the avatar's geometry.** It renders 512×512 at 25 fps. Leave Pipecat's
defaults and every frame goes through a resize: wrong aspect ratio (512×512 is square), a pinned
CPU core, and latency that grows over a call.

## Going further

- **Pipecat over LiveKit?** Use `LiveKitTransportRoom` (a direct join) instead: the avatar joins
  your users' room, publishing its own video track, with no private room and no relay. It needs `room.input_filter()` in
  the pipeline and a bot token carrying `kind="agent"`. Pipecat's dev runner (`-t livekit`) mints a
  bot token without that claim, so a direct join refuses it at startup: mint the bot token yourself,
  as the plugin's `examples/mint_tokens.py` does.
- **Speech-to-speech?** Swap STT + LLM + TTS for a realtime model; it emits the same TTS frames
  the avatar consumes, so it drops into the same slot.

Both are covered in the plugin's README.

## Docs

- [Synthesia Interactive Avatars](https://docs.synthesia.io/reference/interactive-avatars)
- [Pipecat](https://docs.pipecat.ai)
