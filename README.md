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
