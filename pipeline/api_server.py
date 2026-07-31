from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib import request

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import asr_engines, diarize, postcorrect, preprocess, prompt_builder, render, rover
from .glossary import Glossary
from .llm import LLMClient, create_client

app = FastAPI(title="Whisper Transcribe API", version="1.0")

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = Path(os.environ.get("TRANSCRIBE_API_WORKDIR", Path(tempfile.gettempdir()) / "whisper-transcribe-api"))
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "work_root": str(WORK_ROOT),
        "default_ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
    }


def _quality_preset(quality: str | None) -> tuple[str, str, int, int, int, bool, bool, bool, bool]:
    if quality == "fast":
        return ("base", "small", 2, 2, 16, False, False, False, False)
    if quality == "medium":
        return ("base", "medium", 5, 5, 12, False, False, False, False)
    if quality == "good":
        return ("medium", "large-v3", 10, 10, 8, False, False, False, False)
    if quality == "perfect":
        return ("large-v3", "large-v3", 10, 10, 8, True, True, True, True)
    return ("medium", "medium", 10, 10, 8, False, False, False, False)


def _resolve_hf_token() -> str | None:
    # Direct env var first
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_path = os.environ.get("HF_TOKEN_FILE")
    candidates = [
        Path(env_path) if env_path else None,
        Path("/run/secrets/hf-token"),
        Path.home() / ".config" / "whisper" / "hf-token",
    ]
    for p in candidates:
        if p and p.exists():
            return p.read_text(encoding="utf-8", errors="replace").strip()
    return None


def _map_model_path(model: str) -> str:
    if model == "large-v3":
        return "/opt/whisper-models/large-v3"
    return model


def _build_segments_output(rover_path: Path, diarize_path: Path, cleaned_path: Path | None) -> list[dict]:
    """Build structured JSON segments with speaker labels from pipeline artifacts."""
    import json as _json

    rover_data = _json.loads(rover_path.read_text()) if rover_path.exists() else {}
    diarize_data = _json.loads(diarize_path.read_text()) if diarize_path.exists() else {}
    cleaned_data = _json.loads(cleaned_path.read_text()) if cleaned_path and cleaned_path.exists() else None

    segments = rover_data.get("segments", [])
    turns = diarize_data.get("turns", [])

    # Assign speakers to segments (same logic as render._assign_speakers)
    if turns:
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", seg_start + 1.0)
            seg_center = (seg_start + seg_end) / 2
            best_turn = None
            best_overlap = -1.0
            for turn in turns:
                overlap_start = max(seg_start, turn["start"])
                overlap_end = min(seg_end, turn["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_turn = turn
            seg["speaker"] = best_turn["speaker"] if best_turn else "UNKNOWN"
    else:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"

    # Use cleaned segments if available
    source = cleaned_data.get("segments", segments) if cleaned_data else segments
    seg_map = {s["id"]: s for s in source}

    result = []
    for seg in segments:
        sid = seg.get("id", 0)
        src = seg_map.get(sid, seg)
        result.append({
            "id": sid,
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "lang": seg.get("lang", ""),
            "text": src.get("text", "").strip(),
            "confidence": round(seg.get("avg_logprob", 0), 3),
        })

    return result


def _to_azure_format(segments: list[dict]) -> list[dict]:
    """Convert internal segments to Azure-compatible format."""
    import math
    speaker_map: dict[str, str] = {}
    next_speaker = 1

    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def _map_speaker(label: str) -> str:
        nonlocal next_speaker
        if label not in speaker_map:
            speaker_map[label] = str(next_speaker)
            next_speaker += 1
        return speaker_map[label]

    result = []
    for seg in segments:
        confidence = seg.get("confidence", 0)
        # Convert log-prob to probability (0..1)
        if confidence < 0:
            confidence = round(math.exp(confidence), 7)
        duration = seg["end"] - seg["start"]
        result.append({
            "Text": seg.get("text", ""),
            "Speaker": _map_speaker(seg.get("speaker", "UNKNOWN")),
            "Duration": _format_time(duration),
            "StartTime": _format_time(seg["start"]),
            "Confidence": confidence,
        })
    return result





def _ensure_wav(input_audio: Path, output_wav: Path) -> Path:
    if input_audio.suffix.lower() in (".wav", ".wave"):
        return input_audio
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        check=True,
        capture_output=True,
    )
    return output_wav


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
    quality: str | None = Form(default=None),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt_text: str | None = Form(default=None),
    auto_prompt: bool = Form(default=False),
    summary: bool = Form(default=False),
    enhance: bool = Form(default=False),
    denoise: bool = Form(default=False),
    ensemble: bool = Form(default=False),
    correct: bool = Form(default=False),
    no_enhance: bool = Form(default=False),
    no_denoise: bool = Form(default=False),
    no_ensemble: bool = Form(default=False),
    no_correct: bool = Form(default=False),
    context: str | None = Form(default=None),
    speakers: int | None = Form(default=None),
    glossary_text: str | None = Form(default=None),
    diarize_model: str | None = Form(default=None),
    ollama_model: str | None = Form(default=None),
    ollama_base_url: str | None = Form(default=None),
    llm_model: str | None = Form(default=None),
    llm_base_url: str | None = Form(default=None),
    llm_api_key: str | None = Form(default=None),
    keep_intermediates: bool = Form(default=False),
    format: str | None = Form(default=None),
) -> dict:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    job_dir = WORK_ROOT / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "audio.bin").name
    input_path = input_dir / filename
    with input_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    extra: dict = {}
    if options:
        try:
            extra = json.loads(options)
            if not isinstance(extra, dict):
                raise ValueError("options must be a JSON object")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid options JSON: {e}") from e

    def pick(name: str, current):
        return extra[name] if name in extra else current

    quality = pick("quality", quality)
    model = pick("model", model)
    language = pick("language", language)
    prompt_text = pick("prompt_text", prompt_text)
    auto_prompt = bool(pick("auto_prompt", auto_prompt))
    summary = bool(pick("summary", summary))
    enhance = bool(pick("enhance", enhance))
    denoise = bool(pick("denoise", denoise))
    ensemble = bool(pick("ensemble", ensemble))
    correct = bool(pick("correct", correct))
    no_enhance = bool(pick("no_enhance", no_enhance))
    no_denoise = bool(pick("no_denoise", no_denoise))
    no_ensemble = bool(pick("no_ensemble", no_ensemble))
    no_correct = bool(pick("no_correct", no_correct))
    context = pick("context", context)
    speakers = pick("speakers", speakers)
    glossary_text = pick("glossary_text", glossary_text)
    diarize_model = pick("diarize_model", diarize_model)
    ollama_model = pick("ollama_model", ollama_model)
    ollama_base_url = pick("ollama_base_url", ollama_base_url) or DEFAULT_OLLAMA_BASE_URL
    llm_model = pick("llm_model", llm_model)
    llm_base_url = pick("llm_base_url", llm_base_url)
    llm_api_key = pick("llm_api_key", llm_api_key)
    # New names take priority, fall back to old ollama_ names
    effective_model = llm_model or ollama_model
    effective_base_url = llm_base_url or ollama_base_url
    effective_api_key = llm_api_key or None
    keep_intermediates = bool(pick("keep_intermediates", keep_intermediates))
    output_format = pick("format", format)

    base = input_path.stem
    verbatim_path = input_dir / f"{base}.txt"
    cleaned_path = input_dir / f"{base}.cleaned.txt"
    scratch = input_dir / f".whisper-run-{job_id}"
    scratch.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []

    try:
        scan_model, quality_model, beam_size, best_of, batch_size, q_enhance, q_denoise, q_ensemble, q_correct = _quality_preset(quality)

        if model:
            quality_model = model

        if quality == "" or quality is None:
            quality = "custom"

        if q_enhance:
            enhance = True
        if q_denoise:
            denoise = True
        if q_ensemble:
            ensemble = True
        if q_correct:
            correct = True

        if no_enhance:
            enhance = False
        if no_denoise:
            denoise = False
        if no_ensemble:
            ensemble = False
        if no_correct:
            correct = False

        hf_token = _resolve_hf_token()

        glossary_obj = Glossary()
        if glossary_text:
            glossary_path = job_dir / "glossary.txt"
            glossary_path.write_text(glossary_text, encoding="utf-8")
            glossary_obj = Glossary.load(glossary_path)
        else:
            glossary_obj = Glossary.resolve(None)

        prompt_text_final = ""
        if auto_prompt:
            client = create_client(effective_base_url, effective_model, effective_api_key)
            if not client.model:
                raise RuntimeError("auto-prompt requested but no LLM model available")

            scan_path = "/opt/whisper-models/large-v3" if scan_model == "large-v3" else scan_model
            rough = asr_engines.run(
                input_path, scratch,
                language=None if language in (None, "", "auto") else language,
                ensemble=False, model_path=scan_path, batch_size=8, hf_token=hf_token,
            )
            rough_text = " ".join(
                s.get("text", "").strip() for s in rough.get("segments", []) if s.get("text", "").strip()
            )
            if rough_text:
                extract_prompt = (
                    "Extract vocabulary terms from this speech-to-text transcript that would improve recognition accuracy. "
                    "Focus on proper nouns, domain-specific terminology, and technical jargon. "
                    "Correct obvious misrecognitions to their likely intended form. "
                    "Return ONLY a comma-separated list of terms (max 50), as a single line.\n"
                    "Transcript:\n---\n"
                    f"{rough_text}\n"
                    "---"
                )
                llm_vocab = client.chat(extract_prompt, system="/no_think")
                prompt_result = prompt_builder.build_prompt(
                    audio_path=input_path, glossary=glossary_obj,
                    llm_vocab=llm_vocab, user_context=context,
                )
                prompt_text_final = prompt_result.prompt

        if prompt_text and not auto_prompt:
            prompt_result = prompt_builder.build_prompt(
                audio_path=input_path,
                glossary=glossary_obj,
                llm_vocab=prompt_text,
                user_context=context,
            )
            prompt_text_final = prompt_result.prompt

        logs.append(
            f"Model={quality_model} quality={quality} beam={beam_size} best_of={best_of} batch={batch_size} "
            f"enhance={enhance} denoise={denoise} ensemble={ensemble} correct={correct}"
        )

        audio_input = input_path
        prep = None
        if enhance or denoise:
            prep = preprocess.run(input_path, scratch, denoise=denoise, enhance=enhance)
            audio_input = prep.mono_path

        diarize_input = _ensure_wav(audio_input, scratch / "converted.wav")

        model_path = _map_model_path(quality_model)
        asr_result = asr_engines.run(
            audio_input,
            scratch,
            prompt=prompt_text_final or None,
            language=None if language in (None, "", "auto") else language,
            ensemble=ensemble,
            model_path=model_path,
            batch_size=batch_size,
            hf_token=hf_token,
        )
        logs.append(f"ASR segments={len(asr_result.get('segments', []))}")

        rover_path = scratch / "rover.json"
        engine_a_path = scratch / "asr_engine_a.json"
        engine_b_path = scratch / "asr_engine_b.json"
        if ensemble and engine_b_path.exists():
            rover.reconcile(
                engine_a_path,
                engine_b_path,
                glossary_obj,
                rover_path,
                rover.RoverConfig(),
            )
            logs.append("ROVER: enabled")
        else:
            shutil.copyfile(engine_a_path, rover_path)
            logs.append("ROVER: skipped (engine A only)")

        diarize_path = scratch / "diarize.json"
        stereo_paths = None
        if prep and prep.left_path and prep.right_path:
            stereo_paths = (prep.left_path, prep.right_path)
        diarize_cfg = diarize.DiarizeConfig(
            model=f"pyannote/speaker-diarization-{diarize_model or '3.1'}",
            min_speakers=speakers,
            max_speakers=speakers,
        )
        diarize.run(
            diarize_input,
            diarize_path,
            stereo_paths=stereo_paths,
            hf_token=hf_token,
            config=diarize_cfg,
            filename_hint=base,
        )
        # Read back to log diarization result
        if diarize_path.exists():
            diarize_data = json.loads(diarize_path.read_text())
            logs.append(f"Diarization: {len(diarize_data.get('turns', []))} turns, {len(diarize_data.get('speakers', []))} speakers ({diarize_data.get('method', '?')})")

        cleaned_json = None
        if correct:
            cleaned_json = scratch / "cleaned.json"
            postcorrect.run(
                rover_path, cleaned_json,
                glossary_text=glossary_obj.to_prompt_block(),
                language=(language or "nl"),
                llm_base_url=effective_base_url,
                llm_model=effective_model,
                llm_api_key=effective_api_key,
            )

        render.run(rover_path, diarize_path, cleaned_json, verbatim_path, cleaned_path if cleaned_json else None)

        if summary:
            client = create_client(effective_base_url, effective_model, effective_api_key)
            if client.model:
                summary_prompt = (
                    "Analyze this meeting transcript and provide:\n"
                    "1. A concise summary (3-5 sentences)\n"
                    "2. Key decisions or conclusions\n"
                    "3. Action items, follow-ups or open questions\n\n"
                    "Use the same language as the transcript. Be specific with names, places, numbers.\n\n"
                    "Format:\n\n"
                    "## Samenvatting\n<summary>\n\n"
                    "## Beslissingen\n- <decisions>\n\n"
                    "## Actiepunten\n- [ ] <action items>\n\n"
                    "Transcript:\n---\n"
                    f"{verbatim_path.read_text(encoding='utf-8', errors='replace')}\n"
                    "---"
                )
                summary_text = client.chat(summary_prompt, system="/no_think")
                if summary_text:
                    with verbatim_path.open("a", encoding="utf-8") as fp:
                        fp.write("\n---\n\n")
                        fp.write(summary_text)
                        fp.write("\n")

    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        (job_dir / "run.log").write_text("\n".join(logs), encoding="utf-8", errors="replace")
        raise HTTPException(
            status_code=500,
            detail={
                "job_id": job_id,
                "error": f"{type(e).__name__}: {e}",
                "logs": "\n".join(logs)[-12000:],
            },
        ) from e

    # Build structured segments before cleanup (reads from scratch dir)
    segments_out = _build_segments_output(rover_path, diarize_path, cleaned_json)

    # Extract detected language from ASR output
    detected_language = None
    if (scratch / "asr_engine_a.json").exists():
        asr_data = json.loads((scratch / "asr_engine_a.json").read_text())
        detected_language = asr_data.get("language") or (
            asr_data.get("segments", [{}])[0].get("lang") if asr_data.get("segments") else None
        )

    if not keep_intermediates:
        shutil.rmtree(scratch, ignore_errors=True)

    verbatim_text = verbatim_path.read_text(encoding="utf-8", errors="replace") if verbatim_path.exists() else ""
    cleaned_text = cleaned_path.read_text(encoding="utf-8", errors="replace") if cleaned_path.exists() else ""
    final_logs = "\n".join(logs)
    (job_dir / "run.log").write_text(final_logs, encoding="utf-8", errors="replace")

    return {
        "job_id": job_id,
        "return_code": 0,
        "language": detected_language,
        "verbatim_path": str(verbatim_path),
        "cleaned_path": str(cleaned_path) if cleaned_path.exists() else None,
        "verbatim_text": verbatim_text,
        "cleaned_text": cleaned_text if cleaned_path.exists() else None,
        "segments": segments_out,
        "azure": _to_azure_format(segments_out) if output_format == "azure" else None,
        "logs_tail": final_logs[-4000:],
        "llm_base_url": effective_base_url,
    }


@app.post("/jobs/{job_id}/cleanup")
def cleanup_job(job_id: str) -> dict:
    job_dir = WORK_ROOT / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    shutil.rmtree(job_dir, ignore_errors=True)
    return {"ok": True, "job_id": job_id}
