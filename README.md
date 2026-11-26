# Whisper Transcription Pipeline

Self-hosted, GPU-accelerated speech-to-text with speaker diarization and LLM post-correction. Runs as a Docker API — POST audio, get back transcripts. No audio leaves your machine.

## ⚠️ Vibe-coded

I'm terrible in python and I saw the potentional of <https://github.com/steemandavid/whisper-transcribe> but it ran via a bash script. I only needed it to run as a docker container exposing an API so I could finally get rid of the azure speech service.
My main problem with the Azure service was that it is expensive and the quality is not great for Dutch.

My use case is transcribing Dutch and english audio from 3 - 60 mins long.

| Implementation | Understands Dutch | Speaker Diarization | Word-level timestamps | Local |
| ---------------- | ----------------- | -------------------- | ---------------------- | ------- |
| [Azure Speech Service](https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/) | ✅ | ✅ | ✅ | ❌ |
| [OpenAI Whisper](https://openai.com/research/whisper) | ✅ | ❌ | ✅ | ❌ |
| [whisperx](https://github.com/m-bain/whisperX) | ✅ | ❌ | ✅ | ✅ |
| This project | ✅ | ✅ | ✅ | ✅ |

## Quick Start

```powershell
# 1. Set your HuggingFace token (required for speaker diarization)
cp .env.example .env
# Edit .env: add HF_TOKEN=hf_...

# 2. Build and start
docker compose up -d

# 3. Transcribe
curl -X POST http://localhost:8000/transcribe `
  -F "file=@recording.mp3" `
  -F "quality=good" `
  -F "speakers=3"
```

**Requirements:** NVIDIA GPU with driver >= 570, Docker with `--gpus` support.

## Features

- **REST API** — POST audio, get back JSON with segments, speakers, timestamps, confidence
- **OpenAI-compatible API** — `POST /v1/audio/transcriptions` (batch) and `WS /live-transcript` (Realtime-style live VAD + diarization + STT, nl/en)
- **Speaker diarization** — identifies who said what (pyannote 4.x)
- **Word-level timestamps** — precise alignment via wav2vec2
- **Multiple LLM backends** — Ollama, OpenAI, Azure OpenAI, any OpenAI-compatible endpoint
- **Auto-prompt** — LLM extracts domain vocabulary to improve recognition accuracy
- **Post-correction** — LLM fixes phonetic mishearings and brand names
- **Summaries** — LLM-generated summary, decisions, and action items
- **GPU-accelerated** — ~60s for a 20-minute recording on RTX 5070 Ti
- **Privacy-first** — all processing in Docker, only LLM calls leave the container

## API Reference

### `POST /transcribe`

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `file` | file | *(required)* | Audio file (mp3, m4a, wav, etc.) |
| `quality` | string | `custom` | Preset: `fast`, `medium`, `good`, `perfect` |
| `model` | string | `medium` | Override model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `language` | string | auto | Force: `nl`, `en`, `fr`, `de`, or leave empty for auto-detect |
| `speakers` | int | auto | Pin speaker count for diarization |
| `enhance` | bool | `false` | Enable audio preprocessing (loudnorm, highpass) |
| `denoise` | bool | `false` | Enable DeepFilterNet3 denoising |
| `ensemble` | bool | `false` | Multi-engine ASR with ROVER reconciliation |
| `correct` | bool | `false` | LLM post-correction |
| `auto_prompt` | bool | `false` | Auto-generate vocabulary via LLM |
| `summary` | bool | `false` | Append LLM summary to transcript |
| `prompt_text` | string | — | Manual vocabulary hints |
| `context` | string | — | Free-text context for prompt building |
| `glossary_text` | string | — | Glossary (wrong → right mapping) |
| `format` | string | — | Set to `azure` for Azure-compatible output format |
| `llm_base_url` | string | `http://host.docker.internal:11434` | LLM endpoint (Ollama, OpenAI, etc.) |
| `llm_model` | string | auto | LLM model name |
| `llm_api_key` | string | — | API key for OpenAI-compatible endpoints |
| `keep_intermediates` | bool | `false` | Preserve scratch files for debugging |

### Quality Presets

| Preset | ASR Model | Enhance | Denoise | Ensemble | Correct |
| -------- | ----------- | --------- | --------- | ---------- | --------- |
| `fast` | small | ❌ | ❌ | ❌ | ❌ |
| `medium` | medium | ❌ | ❌ | ❌ | ❌ |
| `good` | **large-v3** | ❌ | ❌ | ❌ | ❌ |
| `perfect` | **large-v3** | ✅ | ✅ | ✅ | ✅ |

### `GET /health`

```json
{"ok": true, "work_root": "/tmp/whisper-transcribe-api", "default_ollama_base_url": "http://host.docker.internal:11434"}
```

### `POST /jobs/{job_id}/cleanup`

Delete intermediate files for a completed job.

## Response Format

```json
{
  "job_id": "fcf6d8b4-...",
  "return_code": 0,
  "language": "nl",
  "verbatim_text": "[SPEAKER_01]\n  Goedemorgen.\n\n[SPEAKER_00]\n  Hallo.\n",
  "segments": [
    {"id": 0, "start": 0.0,  "end": 1.2,  "speaker": "SPEAKER_01", "lang": "nl", "text": "Goedemorgen.", "confidence": -0.34},
    {"id": 1, "start": 1.5,  "end": 2.1,  "speaker": "SPEAKER_00", "lang": "nl", "text": "Hallo.",      "confidence": -0.12}
  ]
}
```

With `format=azure`:

```json
{
  "azure": [
    {"Text": "Goedemorgen.", "Speaker": "1", "Duration": "00:00:01.20", "StartTime": "00:00:00.00", "Confidence": 0.712},
    {"Text": "Hallo.",       "Speaker": "2", "Duration": "00:00:00.60", "StartTime": "00:00:01.50", "Confidence": 0.887}
  ]
}
```

## LLM Configuration

The pipeline uses an LLM for three optional stages: auto-prompt, post-correction, and summary. Any OpenAI-compatible endpoint works:

**Ollama (local):**

```powershell
curl -F "file=@audio.mp3" -F "correct=true" `
  -F "llm_base_url=http://host.docker.internal:11434/v1" `
  -F "llm_model=qwen2.5:14b" `
  http://localhost:8000/transcribe
```

**OpenAI:**

```powershell
curl -F "file=@audio.mp3" -F "correct=true" `
  -F "llm_base_url=https://api.openai.com/v1" `
  -F "llm_api_key=sk-..." `
  -F "llm_model=gpt-4o" `
  http://localhost:8000/transcribe
```

**Azure OpenAI:**

```powershell
curl -F "file=@audio.mp3" -F "correct=true" `
  -F "llm_base_url=https://your-resource.openai.azure.com/openai/deployments/gpt-4o" `
  -F "llm_api_key=your-key" `
  http://localhost:8000/transcribe
```

URLs without `/v1` are treated as Ollama. URLs with `/v1` or an API key are treated as OpenAI-compatible.

## Environment Variables

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `HF_TOKEN` | — | HuggingFace token for pyannote diarization models |
| `LLM_BASE_URL` | — | Fallback LLM endpoint |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Fallback Ollama URL |
| `TRANSCRIBE_API_WORKDIR` | `/tmp/whisper-transcribe-api` | Job scratch directory |

## Docker Compose

```yaml
services:
  whisper:
    build: .
    ports: ["8000:8000"]
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - HF_TOKEN=hf_...
    volumes:
      - whisper-hf-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  whisper-hf-cache:
```

The `whisper-hf-cache` volume persists downloaded HuggingFace models across restarts. First run downloads ~3 GB of models (one-time cost).

## OpenAI-compatible API

The server speaks the OpenAI transcription surface — batch over HTTP, live over
WebSocket. Everything else (voice commands, per-sentence agent, entity
reactions) runs **client-side** on the transcript events.

### Batch: `POST /v1/audio/transcriptions`

Standard OpenAI multipart fields: `file` (required), `model` (default
`whisper-1`), `language`, `prompt`, `response_format`
(`json` | `text` | `verbose_json` | `srt` | `vtt`), `temperature`,
`timestamp_granularities[]` (`word` and/or `segment`).
Extensions: `quality` (`fast`/`medium`/`good`/`perfect`), `speakers` (pin
diarization count). `verbose_json` segments carry `speaker` and `avg_logprob`
extensions plus word-level timestamps when `timestamp_granularities[]=word`.

```bash
# POST http://localhost:8000/v1/audio/transcriptions with multipart form:
#   file=@recording.mp3, model=whisper-1, language=nl,
#   response_format=verbose_json, timestamp_granularities[]=word
```

`json` (default) returns `{"text": "..."}`; `verbose_json` returns
`{"task", "language", "duration", "text", "segments": [...], "words": [...]}`.

The original custom `POST /transcribe` endpoint is unchanged.

### Live: `WS /live-transcript`

OpenAI Realtime-style events. Audio in = base64 PCM16, 16 kHz, mono.

Client -> server (JSON text frames; binary frames are accepted as raw PCM16):

| Event | Payload |
| ----- | ------- |
| `session.update` | `session.input_audio_transcription` (`model`, `language`, `prompt`), `session.turn_detection.silence_duration_ms` (maps to VAD hangover) |
| `input_audio_buffer.append` | `audio`: base64 PCM16 chunk |
| `input_audio_buffer.commit` | finalize the current speech turn now |
| `input_audio_buffer.clear` | drop buffered, uncommitted audio |
| `ping` | `{"type":"ping","ping":{"value":123}}` -> `pong` |

Server -> client:

| Event | Payload |
| ----- | ------- |
| `session.created` / `session.updated` | `session` object (`id`, `model`, effective `input_audio_transcription`, `turn_detection`) |
| `input_audio_buffer.speech_started` | `audio_start_ms`, `item_id` |
| `input_audio_buffer.speech_stopped` | `audio_end_ms` |
| `input_audio_buffer.committed` / `cleared` | `item_id` |
| `conversation.item.input_audio_transcribed` | `item_id`, `transcript`, `language`, `segments`: `[{start, end, text, speaker, avg_logprob, words}]` |
| `pong` / `error` | `value` / `error.type`, `error.message` |

Every frame has `type` + `event_id`. Speaker labels (`SPEAKER_01`...) come from
online diarization; timestamps are seconds relative to stream start.

```csharp
var ws = new ClientWebSocket();
await ws.ConnectAsync(new Uri("ws://localhost:8000/live-transcript"), default);
await ws.SendAsync(Encoding.UTF8.GetBytes("""{"type":"session.update","session":{
    "input_audio_transcription":{"model":"whisper-1","language":"nl"},
    "turn_detection":{"type":"server_vad","silence_duration_ms":800}}}"""),
    WebSocketMessageType.Text, true, default);

// stream 16 kHz mono PCM16 (ffmpeg -i call.mp3 -ar 16000 -ac 1 -f s16le call.pcm)
var buf = new byte[960]; // 30 ms
var send = Task.Run(async () => {
    using var fs = File.OpenRead("call.pcm");
    int n;
    while ((n = await fs.ReadAsync(buf)) > 0)
        await ws.SendAsync(new ArraySegment<byte>(buf, 0, n),
            WebSocketMessageType.Binary, true, default);
});

var recv = new byte[64 * 1024];
while (ws.State == WebSocketState.Open)
{
    var sb = new StringBuilder();
    WebSocketReceiveResult r;
    do {
        r = await ws.ReceiveAsync(new ArraySegment<byte>(recv), default);
        if (r.MessageType == WebSocketMessageType.Close) break;
        sb.Append(Encoding.UTF8.GetString(recv, 0, r.Count));
    } while (!r.EndOfMessage);
    var ev = JsonDocument.Parse(sb.ToString()).RootElement;
    switch (ev.GetProperty("type").GetString())
    {
        case "conversation.item.input_audio_transcribed":
            var seg = ev.GetProperty("segments")[0];
            Console.WriteLine($"[{seg.GetProperty("speaker")}] {seg.GetProperty("text")}");
            // -> run your client-side agent / commands / entity extraction here
            break;
        case "input_audio_buffer.speech_started":
            Console.WriteLine($"speech @ {ev.GetProperty("audio_start_ms")} ms"); break;
    }
}
await send;
```

## CLI Mode

The container also supports direct CLI usage:

```powershell
docker compose run --rm whisper `
  pipeline.cli /audio/recording.mp3 --quality good --speakers 3 --summary
```

## Architecture

```
POST /transcribe (audio file)
     │
     ▼
Stage 1: Preprocessing (optional)
  - ffmpeg loudnorm, highpass, channel split
     │
     ▼
Stage 2: ASR (WhisperX)
  - Engine A: large-v3 / medium / etc.
  - Engine B: Dutch fine-tune (ensemble only)
  - Word-level timestamps
     │
     ▼
Stage 3: ROVER Reconciliation (ensemble only)
  - Merges Engine A + B, glossary-weighted vote
     │
     ▼
Stage 4: Speaker Diarization (pyannote 4.x)
  - Speaker turns from VAD + clustering
     │
     ▼
Stage 5: LLM Post-Correction (optional)
  - Fixes mishearings, brand names, jargon
     │
     ▼
Stage 6: Render
  - Assigns speakers to segments
  - Generates verbatim + cleaned transcripts
     │
     ▼
Stage 7: Summary (optional)
  - LLM summary, decisions, action items
     │
     ▼
JSON response with segments, speakers, confidence
```

## Credits

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [WhisperX](https://github.com/m-bain/whisperX), and [pyannote-audio](https://github.com/pyannote/pyannote-audio).
