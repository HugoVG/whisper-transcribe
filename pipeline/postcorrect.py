"""LLM post-correction: fix phonetic mishearings in the ROVER output.

Uses an LLM via OpenAI-compatible or Ollama API. Applies only high-confidence
corrections (>= 0.8) and rejects segments with > 35% token churn via a diff guard.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .artifacts import write_json
from .llm import LLMClient, create_client


# --- Diff guard ---

def _token_edit_ratio(original: str, corrected: str) -> float:
    """Token-level edit distance as a fraction of max(len(a), len(b))."""
    a = original.split()
    b = corrected.split()
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return dist / max(la, lb, 1)


# --- LLM Corrector ---

class LLMCorrector:
    def __init__(self, client: LLMClient):
        self.client = client

    def correct(self, segments: list[dict], system_prompt: str, language: str) -> list[dict]:
        batch_text = _format_batch(segments)
        user_prompt = f"Transcript segments to correct:\n{batch_text}"
        content = self.client.chat(
            user_prompt,
            system=f"{system_prompt}\n\nRespond only in valid JSON. No markdown fences.",
            temperature=0.1,
        )
        return _parse_corrections(content, segments)


# --- Shared helpers ---

_SYSTEM_PROMPT_TEMPLATE = """You are correcting speech-to-text transcription errors.

Language: {language}
Rules:
- Fix ONLY obvious phonetic mishearings (Dutch/Flemish dialect, brand names, proper nouns).
- Words marked with [?] are uncertain — prioritise correcting those.
- Use the glossary below for correct forms.
- Do NOT paraphrase, add, or remove content.
- Do NOT change punctuation outside of fixed words.
- Preserve speaker labels and [?] markers when you are not confident.

Glossary:
{glossary}

Output strict JSON array:
[{{"seg_id": <int>, "original": "<text>", "corrected": "<text>", "changed_words": [{{"idx": <int>, "before": "<word>", "after": "<word>", "confidence": <float>}}]}}]
"""

_BATCH_SIZE = 40
_CONFIDENCE_THRESHOLD = 0.8
_MAX_EDIT_RATIO = 0.35


def _format_batch(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{seg['id']}] {seg['text']}")
    return "\n".join(lines)


def _parse_corrections(raw: str, original_segments: list[dict]) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        corrections = json.loads(text)
    except json.JSONDecodeError:
        return original_segments
    if not isinstance(corrections, list):
        return original_segments

    seg_map = {seg["id"]: seg for seg in original_segments}
    results = []
    for corr in corrections:
        seg_id = corr.get("seg_id")
        if seg_id not in seg_map:
            continue
        original = seg_map[seg_id]
        corrected_text = corr.get("corrected", original["text"])
        changed = corr.get("changed_words", [])
        high_conf = [w for w in changed if w.get("confidence", 0) >= _CONFIDENCE_THRESHOLD]
        if not high_conf:
            results.append(dict(original))
            continue
        edit_ratio = _token_edit_ratio(original["text"], corrected_text)
        if edit_ratio > _MAX_EDIT_RATIO:
            results.append({**original, "text": original["text"] + " [!unverified]"})
            continue
        results.append({
            **original, "text": corrected_text,
            "corrections": high_conf,
            "edit_ratio": round(edit_ratio, 3),
        })
    for seg in original_segments:
        if not any(r.get("id") == seg["id"] for r in results):
            results.append(dict(seg))
    results.sort(key=lambda s: s.get("id", 0))
    return results


def run(
    rover_path: Path,
    output_path: Path,
    *,
    glossary_text: str = "",
    language: str = "nl",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
) -> dict:
    from .artifacts import read_json

    rover = read_json(rover_path)
    segments = rover.get("segments", [])
    if not segments:
        write_json(output_path, {"segments": [], "method": "none"})
        return {"segments": [], "method": "none"}

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        language=language, glossary=glossary_text or "(none)",
    )

    base_url = llm_base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    client = create_client(base_url=base_url, model=llm_model, api_key=llm_api_key)
    corrector = LLMCorrector(client)

    all_corrected: list[dict] = []
    for i in range(0, len(segments), _BATCH_SIZE):
        batch = segments[i : i + _BATCH_SIZE]
        try:
            corrected = corrector.correct(batch, system_prompt, language)
            all_corrected.extend(corrected)
        except Exception as e:
            print(f"Post-correct batch {i // _BATCH_SIZE}: {type(e).__name__}: {e}", file=sys.stderr)
            all_corrected.extend(batch)

    result = {
        "method": "llm",
        "model": client.model,
        "segments": all_corrected,
    }
    write_json(output_path, result)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LLM post-correction stage.")
    ap.add_argument("--rover", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--glossary-file", type=Path, default=None)
    ap.add_argument("--language", default="nl")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-api-key", default=None)
    args = ap.parse_args()

    glossary_text = ""
    if args.glossary_file and args.glossary_file.exists():
        from .glossary import Glossary
        g = Glossary.load(args.glossary_file)
        glossary_text = g.to_prompt_block()

    result = run(
        args.rover, args.output,
        glossary_text=glossary_text,
        language=args.language,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )
    n_corr = sum(1 for s in result["segments"] if "corrections" in s)
    print(f"Post-correct: {len(result['segments'])} segments, {n_corr} corrected ({result['method']})")
