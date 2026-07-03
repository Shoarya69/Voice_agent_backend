# Exotel Voicebot WebSocket Server

A production-ready **FastAPI WebSocket server** that acts as a real-time AI voice
agent for the **Exotel Voicebot Applet**. It receives live 8kHz mu-law audio from
Exotel over WebSocket, transcribes it (**ElevenLabs Scribe STT**), generates a
reply (**OpenAI `gpt-4o-mini`**), converts the reply to speech (**ElevenLabs
TTS**), and streams audio back — targeting under 1.5s of latency per
conversational turn. There are no fallback providers: on an STT/LLM/TTS
failure the turn is skipped (LLM failures speak a short apology instead).

This server is the **data plane**. The Lovable app at
[nexovaaii.lovable.app](https://nexovaaii.lovable.app) is the **control plane**:
it owns agent configuration, prompts, phone numbers, and call logs in Supabase.

## Architecture

```
[User's Phone]
     |  (PSTN call)
     v
[Exotel Number] -> [Exotel Flow: Greeting -> Voicebot Applet]
     |  (wss:// WebSocket, 8kHz mu-law PCM frames, 20ms each)
     v
[FastAPI Server on Railway]  <-- this repo
     |--> Fetch agent config from Lovable app (REST, cached 60s)
     |--> ElevenLabs Scribe    (STT, batch per utterance, model_id=scribe_v2)
     |--> OpenAI gpt-4o-mini   (LLM reply generation, streamed sentence-by-sentence)
     |--> ElevenLabs           (TTS streaming, eleven_multilingual_v2, mu-law 8kHz output)
     '--> Post call logs back to Lovable app (REST) on `stop`
```

### Per-turn pipeline

1. Exotel streams 20ms mu-law audio frames as `media` events.
2. Each frame is decoded to 16-bit PCM and fed into a WebRTC VAD
   (`app/vad.py`), which tracks speech vs. silence and reports
   `end_of_utterance` once ~500ms of trailing silence follows speech.
3. The buffered utterance is packaged as a WAV file and sent to
   ElevenLabs Scribe (`/v1/speech-to-text`, `model_id=scribe_v2`) for
   transcription.
4. The transcript + conversation history is sent to OpenAI's streaming
   Chat Completions endpoint (`gpt-4o-mini`), which streams its reply;
   the stream is split into sentences as they complete. On any LLM
   failure/timeout, the bot speaks "Sorry, ek moment ruko." and the turn
   ends — there is no fallback LLM.
5. Each sentence is sent to ElevenLabs TTS (streaming,
   `eleven_multilingual_v2`, `ulaw_8000` output), and the resulting
   mu-law audio is chunked into 20ms frames and sent back to Exotel as
   `media` events in real time. If ElevenLabs TTS yields no audio, the
   sentence is skipped (logged, no fallback voice).
6. If the caller starts talking while the bot is speaking (barge-in), the
   server sends a `clear` event, cancels the in-flight TTS/LLM task, and
   immediately starts buffering the new utterance.
7. On `stop`, the full transcript and call metadata are POSTed back to the
   Lovable app.

## Project layout

```
exotel-voicebot/
├── app/
│   ├── main.py                 # FastAPI app, WebSocket route, /health, /metrics
│   ├── config.py                # Pydantic settings, env vars
│   ├── websocket_handler.py     # Core per-call session loop
│   ├── audio_utils.py           # mu-law/PCM conversion, chunking, resampling
│   ├── vad.py                   # Voice activity / end-of-utterance detection
│   ├── models.py                # Pydantic models for events, configs, logs
│   └── services/
│       ├── elevenlabs_stt.py    # ElevenLabs Scribe batch STT
│       ├── openai_llm.py        # OpenAI streaming chat completion (no fallback)
│       ├── elevenlabs_tts.py    # ElevenLabs streaming TTS
│       └── lovable_api.py       # Agent config fetch + call log POST
├── tests/
│   ├── test_audio_utils.py
│   ├── test_utils.py            # retry_async unit tests
│   ├── test_services.py         # elevenlabs_stt / openai_llm unit tests
│   └── test_websocket.py        # VAD unit tests + WS integration test
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── railway.json
└── README.md
```

## Local development

### 1. Install dependencies

```bash
cd exotel-voicebot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt + pytest
```

### 2. Configure environment

```bash
cp .env.example .env
# then edit .env and fill in real API keys
```

At minimum you need `LOVABLE_API_SECRET`, `ELEVENLABS_API_KEY` (powers both
Scribe STT and TTS), and `OPENAI_API_KEY` for the full pipeline to work. The
server will still start without them (with a warning logged), which is
useful for testing the WebSocket protocol handling in isolation.

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Check it's alive:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 4. Run tests

```bash
pytest -v
```

## Testing the WebSocket locally with `wscat`

Install `wscat` if you don't have it: `npm install -g wscat`.

```bash
wscat -c ws://localhost:8000/exotel/voicebot
```

Then paste in the Exotel protocol messages by hand, in order:

```json
{"event": "connected", "protocol": "websocket", "version": "1.0.0"}
```

```json
{"event": "start", "sequence_number": "1", "start": {"stream_sid": "s1", "call_sid": "c1", "account_sid": "a1", "from": "+919999999999", "to": "+917971451588", "custom_parameters": {"agent_id": "YOUR_AGENT_UUID"}, "media_format": {"encoding": "audio/x-mulaw", "sample_rate": 8000, "channels": 1}}}
```

You should immediately see `media` (and `mark`) events come back — this is
the greeting (`first_message`) being synthesized and streamed to you as
base64 mu-law audio.

To simulate audio frames, send `media` events with base64-encoded mu-law
payloads (160 bytes of mu-law = 20ms at 8kHz):

```json
{"event": "media", "sequence_number": "2", "media": {"chunk": "1", "timestamp": "100", "payload": "<base64 mulaw bytes>"}}
```

End the call:

```json
{"event": "stop", "sequence_number": "999", "stop": {"call_sid": "c1"}}
```

For a scripted version of this flow with real audio and mocked services, see
`tests/test_websocket.py`.

## Railway deployment

1. Push this repository to GitHub.
2. In Railway, click **New Project -> Deploy from GitHub repo** and select it.
3. Railway will detect the `Dockerfile` automatically (via `railway.json`'s
   `"builder": "DOCKERFILE"`).
4. Under **Variables**, set every key from `.env.example` with real values
   (`LOVABLE_APP_URL`, `LOVABLE_API_SECRET`, `ELEVENLABS_API_KEY`,
   `OPENAI_API_KEY`, etc). Railway automatically injects `PORT`; the
   Dockerfile's `CMD` binds to `0.0.0.0:8000`, matching the exposed port
   Railway expects — if Railway assigns a different port, update the `CMD`
   or use `$PORT` in a startup script.
5. Deploy. Railway will hit `GET /health` per `railway.json`'s
   `healthcheckPath` to confirm the service is up before routing traffic.
6. Once deployed, your WebSocket URL will be:

   ```
   wss://your-app.up.railway.app/exotel/voicebot
   ```

## Configuring the Exotel Voicebot Applet

In your Exotel Flow, add a **Voicebot Applet** node (after a Greeting node if
you want a pre-connect message) and set its WebSocket URL to:

```
wss://your-app.up.railway.app/exotel/voicebot
```

Pass the `agent_id` for the call as a custom parameter so the server knows
which Lovable-configured agent/persona to use — this is read from
`start.custom_parameters.agent_id`.

## Troubleshooting

**Audio sounds garbled / robotic**
- Confirm Exotel's `media_format` is `audio/x-mulaw` at `8000`Hz — this
  server assumes 8kHz mu-law both ways. If ElevenLabs output format ever
  changes from `ulaw_8000`, audio sent to Exotel will be corrupted.
- Verify you are not double-encoding/decoding: `media.payload` from Exotel
  and the `media.payload` this server sends back must both be raw mu-law
  bytes, base64-encoded exactly once.

**High latency (> 1.5s per turn)**
- Check `/metrics` for `voicebot_avg_latency_ms`. If STT is slow, verify
  `ELEVENLABS_API_KEY`/network path to `api.elevenlabs.io` from your Railway
  region.
- OpenAI and ElevenLabs both have internal timeouts (4s and ~10s
  respectively); persistent timeouts usually indicate an upstream outage or
  rate limiting — check structured logs for `*.timeout` / `*.failed` events.
- Make sure you're running a single worker close to your Exotel region;
  cross-region round trips add material latency for a real-time voice loop.

**WebSocket disconnects mid-call**
- The handler never intentionally closes the socket except on `stop` or a
  fatal receive error; check Railway logs for `websocket.unhandled_error`.
- Confirm Railway's reverse proxy / load balancer WebSocket idle timeout is
  longer than your longest expected silence gap between Exotel keepalives.

**Bot doesn't respond / no greeting audio**
- Check `server.missing_config` in startup logs — if any of
  `LOVABLE_API_SECRET`, `ELEVENLABS_API_KEY`, `OPENAI_API_KEY` are unset,
  calls will still connect but STT/LLM/TTS calls will fail.
- Confirm the Lovable app's `/api/public/voicebot/agent/{agent_id}` route
  exists and returns a 200 with the expected JSON shape.

**Barge-in feels slow or doesn't interrupt**
- Barge-in relies on WebRTC VAD detecting speech energy over background
  noise; on very noisy lines, consider lowering VAD aggressiveness
  (`VADProcessor(mode=2)`) or tuning `start_speech_ms`.

## Performance targets

| Metric | Target |
|---|---|
| Time to first audio byte after user stops speaking | < 1200ms |
| Barge-in latency (detect + `clear`) | < 200ms |
| Concurrent calls per instance | 50+ |
| Memory per session | < 20MB |

## Bonus features included

- `/metrics` endpoint (Prometheus format): `voicebot_active_calls`,
  `voicebot_total_calls`, `voicebot_avg_latency_ms` (measured as real
  time-to-first-audio-frame-sent per turn, not just first LLM token).
- Sentence-level barge-in: the in-flight LLM/TTS task is cancelled the
  instant caller speech is detected, and Exotel is told to `clear` its
  playback buffer immediately. The VAD's speaking state is preserved across
  barge-in so the first ~60ms of the caller's new utterance isn't dropped.
- No fallback providers by design: if ElevenLabs STT/TTS or OpenAI fail, the
  turn is skipped (LLM failures speak a short apology) rather than silently
  switching providers mid-call.
- Retry with exponential backoff on transient (connection-level) failures
  for the Lovable config fetch/call-log POST, ElevenLabs Scribe STT, and the
  initial ElevenLabs TTS connection — without blowing the per-turn latency
  budget (only connection-level errors are retried, not full timeouts).
- Optional shared-secret WebSocket auth (`EXOTEL_WS_AUTH_TOKEN` + `?token=`
  query param) to stop random clients from connecting to the publicly
  reachable WS endpoint.
- Hardened `Dockerfile`: runs as a non-root user and defines a container
  `HEALTHCHECK` in addition to Railway's own health check.

## Production hardening notes

- **Config validation**: `AgentConfig.temperature` and `.max_tokens` are
  clamped (`0.0-2.0`, `1-8192`) via pydantic `Field` constraints, so a bad
  value from the Lovable app can't silently break OpenAI generation calls
  — validation failures fall back to the safe default `AgentConfig`.
- **Never trust a single upstream failure**: every external call (Lovable,
  ElevenLabs STT/TTS, OpenAI) is wrapped in try/except and degrades
  gracefully (default config, apology reply text, skipped turn) rather than
  propagating and killing the call. There are no fallback providers - a
  failed STT/TTS call simply skips that turn/sentence.
- **Backpressure-safe audio sending**: TTS audio is chunked into real
  20ms frames and paced with `asyncio.sleep` to match real-time playback,
  and every send loop checks `session.active` so a mid-call disconnect
  stops in-flight audio immediately instead of buffering indefinitely.
- **Graceful shutdown**: the FastAPI `lifespan` handler waits (bounded to
  30s) for in-flight WebSocket sessions to finish before the process exits,
  so a Railway redeploy doesn't hang up active calls mid-sentence.
