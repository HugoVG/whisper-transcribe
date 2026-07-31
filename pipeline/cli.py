#!/usr/bin/env python3
"""Whisper transcription pipeline v3 — pure Python orchestrator.

Replaces the bash `transcribe` script. All stages run in-process.
Usage:
    python -m pipeline.cli <audio_file> [options]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import asr_engines, diarize, postcorrect, preprocess, prompt_builder, render, rover
from .glossary import Glossary
from .llm import LLMClient, create_client


# ── Quality presets ──────────────────────────────────────────────────────────
QUALITY_PRESETS = {
    "fast":   {"scan_model": "base",     "quality_model": "small",    "beam_size": 2,  "best_of": 2,  "batch_size": 16, "enhance": False, "denoise": False, "ensemble": False, "correct": False},
    "medium": {"scan_model": "base",     "quality_model": "medium",   "beam_size": 5,  "best_of": 5,  "batch_size": 12, "enhance": False, "denoise": False, "ensemble": False, "correct": False},
    "good":   {"scan_model": "medium",   "quality_model": "large-v3", "beam_size": 10, "best_of": 10, "batch_size": 8,  "enhance": False, "denoise": False, "ensemble": False, "correct": False},
    "perfect":{"scan_model": "large-v3", "quality_model": "large-v3", "beam_size": 10, "best_of": 10, "batch_size": 8,  "enhance": True,  "denoise": True,  "ensemble": True,  "correct": True},
}

LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Build argument parser ────────────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    candidates = [
        Path("/run/secrets/hf-token"),
        Path.home() / ".config" / "whisper" / "hf-token",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return os.environ.get("HF_TOKEN")


def _ensure_wav(input_audio: Path, output_wav: Path) -> Path:
    if input_audio.suffix.lower() in (".wav", ".wave"):
        return input_audio
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_audio),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output_wav),
    ], check=True, capture_output=True)
    return output_wav


def _map_model_path(model: str) -> str:
    if model == "large-v3":
        return "/opt/whisper-models/large-v3"
    return model


# ── Build argument parser ────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="whisper-transcribe",
        description="GPU-accelerated Whisper transcription with speaker diarization.",
    )
    ap.add_argument("audio", type=Path, help="Audio file to transcribe.")
    ap.add_argument("--quality", "-Q", choices=["fast", "medium", "good", "perfect"],
                    help="Quality preset.")
    ap.add_argument("--model", "-m", help="Override transcription model (tiny, base, small, medium, large-v3).")
    ap.add_argument("--language", "-l", default=None, help="Language code (en, nl, fr, de, auto).")
    ap.add_argument("--prompt", "-p", type=Path, help="Path to vocabulary/context hints file.")
    ap.add_argument("--auto-prompt", "-a", action="store_true", help="Auto-generate vocabulary via Ollama.")
    ap.add_argument("--summary", "-s", action="store_true", help="Append LLM-generated summary to transcript.")
    ap.add_argument("--enhance", action="store_true", help="Enable audio preprocessing.")
    ap.add_argument("--denoise", action="store_true", help="Enable DeepFilterNet3 denoising.")
    ap.add_argument("--ensemble", action="store_true", help="Run multi-engine ASR with ROVER reconciliation.")
    ap.add_argument("--correct", action="store_true", help="LLM post-correction via Ollama.")
    ap.add_argument("--no-enhance", action="store_true", help="Override quality preset: disable enhance.")
    ap.add_argument("--no-denoise", action="store_true", help="Override quality preset: disable denoise.")
    ap.add_argument("--no-ensemble", action="store_true", help="Override quality preset: disable ensemble.")
    ap.add_argument("--no-correct", action="store_true", help="Override quality preset: disable post-correction.")
    ap.add_argument("--context", help="Free-text context for prompt building.")
    ap.add_argument("--speakers", type=int, help="Pin speaker count for diarization.")
    ap.add_argument("--glossary", type=Path, help="Path to glossary file.")
    ap.add_argument("--diarize-model", default="3.1", choices=["3.1", "community-1"], help="Pyannote diarization model.")
    ap.add_argument("--llm-model", help="LLM model name (e.g., gpt-4o, qwen2.5:14b).")
    ap.add_argument("--llm-base-url", help="OpenAI-compatible or Ollama base URL.")
    ap.add_argument("--llm-api-key", help="API key for OpenAI-compatible endpoints.")
    ap.add_argument("--keep-intermediates", action="store_true", help="Preserve scratch dir for debugging.")
    ap.add_argument("--output-dir", type=Path, help="Directory for output files (default: same as audio file).")
    return ap


# ── Auto-prompt pipeline ─────────────────────────────────────────────────────

def _run_auto_prompt(
    audio_path: Path, scratch: Path, base: str,
    scan_model: str, language: str | None,
    client: LLMClient,
    glossary_obj: Glossary, context: str | None,
    hf_token: str | None,
) -> str | None:
    scan_path = "/opt/whisper-models/large-v3" if scan_model == "large-v3" else scan_model
    print(f"  Auto-prompt Step 1: Quick scan ({scan_model})...")
    rough = asr_engines.run(
        audio_path, scratch,
        language=None if language in (None, "", "auto") else language,
        ensemble=False, model_path=scan_path, batch_size=8, hf_token=hf_token,
    )
    rough_text = " ".join(
        s.get("text", "").strip() for s in rough.get("segments", []) if s.get("text", "").strip()
    )
    if not rough_text:
        print("  Warning: Quick scan empty. Skipping auto-prompt.", file=sys.stderr)
        return None
    print(f"  Quick scan: {len(rough_text.split())} words")

    print("  Auto-prompt Step 2: LLM keyword extraction...")
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
    if not llm_vocab:
        print("  Warning: LLM empty. Skipping auto-prompt.", file=sys.stderr)
        return None

    print(f"  Prompt v1: {llm_vocab[:120]}...")
    prompt_result = prompt_builder.build_prompt(
        audio_path=audio_path, glossary=glossary_obj,
        llm_vocab=llm_vocab, user_context=context,
    )
    print(f"  Final prompt: {prompt_result.prompt[:120]}...")
    return prompt_result.prompt


# ── Summary generation ───────────────────────────────────────────────────────

def _run_summary(transcript_path: Path, client: LLMClient) -> str | None:
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
        f"{transcript_path.read_text(encoding='utf-8', errors='replace')}\n"
        "---"
    )
    summary_text = client.chat(summary_prompt, system="/no_think")
    if summary_text:
        with transcript_path.open("a", encoding="utf-8") as fp:
            fp.write("\n---\n\n")
            fp.write(summary_text)
            fp.write("\n")
    return summary_text


# ── Main orchestration ───────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        return 1

    base = audio_path.stem
    output_dir = (args.output_dir or audio_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verbatim_path = output_dir / f"{base}.txt"
    cleaned_path = output_dir / f"{base}.cleaned.txt"

    # ── Resolve quality preset ───────────────────────────────────────────────
    scan_model = "medium"
    quality_model = args.model or "medium"
    beam_size = 10; best_of = 10; batch_size = 8
    enhance = args.enhance; denoise = args.denoise
    ensemble = args.ensemble; correct = args.correct

    if args.quality:
        preset = QUALITY_PRESETS[args.quality]
        scan_model = preset["scan_model"]
        quality_model = args.model or preset["quality_model"]
        beam_size = preset["beam_size"]; best_of = preset["best_of"]
        batch_size = preset["batch_size"]
        enhance = preset["enhance"]; denoise = preset["denoise"]
        ensemble = preset["ensemble"]; correct = preset["correct"]

    if args.no_enhance: enhance = False
    if args.no_denoise: denoise = False
    if args.no_ensemble: ensemble = False
    if args.no_correct: correct = False

    llm_model = args.llm_model
    llm_base_url = args.llm_base_url or LLM_BASE_URL
    llm_api_key = args.llm_api_key or None
    language = args.language
    if language in (None, "", "auto"):
        language = None
    hf_token = _resolve_hf_token()

    # ── Glossary ─────────────────────────────────────────────────────────────
    glossary_obj = Glossary.resolve(args.glossary)

    # ── Scratch directory ────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())[:8]
    scratch = output_dir / f".whisper-run-{job_id}"
    scratch.mkdir(parents=True, exist_ok=True)
    run_log: list[str] = []

    def _cleanup():
        if not args.keep_intermediates:
            shutil.rmtree(scratch, ignore_errors=True)

    def _log(msg: str):
        run_log.append(msg)
        print(f"  {msg}")

    try:
        # ── Prompt building ──────────────────────────────────────────────────
        prompt_text_final = ""
        if args.auto_prompt:
            client = create_client(llm_base_url, llm_model, llm_api_key)
            if not client.model:
                print("Error: auto-prompt requires an LLM model.", file=sys.stderr)
                return 1
            auto_result = _run_auto_prompt(
                audio_path, scratch, base, scan_model, language,
                client, glossary_obj, args.context, hf_token,
            )
            if auto_result:
                prompt_text_final = auto_result

        if args.prompt and not args.auto_prompt:
            raw_vocab = args.prompt.read_text(encoding="utf-8", errors="replace").strip()
            prompt_result = prompt_builder.build_prompt(
                audio_path=audio_path, glossary=glossary_obj,
                llm_vocab=raw_vocab, user_context=args.context,
            )
            prompt_text_final = prompt_result.prompt

        # ── Preprocessing ────────────────────────────────────────────────────
        audio_input = audio_path
        prep = None
        if enhance or denoise:
            print("--- Stage 1: Preprocessing ---")
            prep = preprocess.run(audio_input, scratch, denoise=denoise, enhance=enhance)
            audio_input = prep.mono_path
            _log(f"Enhance={enhance} denoise={denoise} stereo_split={prep.stereo_split}")

        # ── ASR ──────────────────────────────────────────────────────────────
        print(f"--- Stage 2: ASR ({quality_model}) ---")
        model_path = _map_model_path(quality_model)
        asr_result = asr_engines.run(
            audio_input, scratch,
            prompt=prompt_text_final or None,
            language=language,
            ensemble=ensemble,
            model_path=model_path,
            batch_size=batch_size,
            hf_token=hf_token,
        )
        _log(f"ASR segments={len(asr_result.get('segments', []))}")

        # ── ROVER ────────────────────────────────────────────────────────────
        rover_path = scratch / "rover.json"
        engine_a_path = scratch / "asr_engine_a.json"
        engine_b_path = scratch / "asr_engine_b.json"
        if ensemble and engine_b_path.exists():
            print("--- Stage 3: ROVER reconciliation ---")
            rover.reconcile(engine_a_path, engine_b_path, glossary_obj, rover_path, rover.RoverConfig())
            _log("ROVER: enabled")
        else:
            shutil.copyfile(engine_a_path, rover_path)
            _log("ROVER: skipped (engine A only)")

        # ── Diarization ──────────────────────────────────────────────────────
        print("--- Stage 4: Diarization ---")
        diarize_input = _ensure_wav(audio_input, scratch / "converted.wav")
        diarize_path = scratch / "diarize.json"
        stereo_paths = None
        if prep and prep.left_path and prep.right_path:
            stereo_paths = (prep.left_path, prep.right_path)
        diarize_cfg = diarize.DiarizeConfig(
            model=f"pyannote/speaker-diarization-{args.diarize_model}",
            min_speakers=args.speakers,
            max_speakers=args.speakers,
        )
        diarize.run(diarize_input, diarize_path, stereo_paths=stereo_paths,
                    hf_token=hf_token, config=diarize_cfg, filename_hint=base)
        _log(f"Diarization: model={args.diarize_model}")

        # ── Post-correction ──────────────────────────────────────────────────
        cleaned_json = None
        if correct:
            print("--- Stage 5: Post-correction ---")
            cleaned_json = scratch / "cleaned.json"
            postcorrect.run(
                rover_path, cleaned_json,
                glossary_text=glossary_obj.to_prompt_block(),
                language=(language or "nl"),
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
            )
            _log("Post-correction: done")

        # ── Render ───────────────────────────────────────────────────────────
        print("--- Rendering transcripts ---")
        render.run(rover_path, diarize_path, cleaned_json, verbatim_path,
                   cleaned_path if cleaned_json else None)
        _log(f"Verbatim: {verbatim_path}")
        if cleaned_json:
            _log(f"Cleaned:  {cleaned_path}")

        # ── Summary ──────────────────────────────────────────────────────────
        if args.summary:
            client = create_client(llm_base_url, llm_model, llm_api_key)
            if client.model:
                print("--- Summary ---")
                summary_text = _run_summary(verbatim_path, client)
                _log(f"Summary: {'ok' if summary_text else 'empty'}")

        print()
        print("=== Pipeline complete ===")
        print(f"Verbatim: {verbatim_path}")
        if cleaned_json:
            print(f"Cleaned:  {cleaned_path}")

    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_intermediates:
            _cleanup()

    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
