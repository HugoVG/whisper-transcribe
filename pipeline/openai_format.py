"""OpenAI /v1/audio/transcriptions response formatters — pure stdlib.

Maps the internal transcribe result (verbatim_text, segments, language) to
OpenAI response_format payloads: json, text, verbose_json, srt, vtt.
"""

from __future__ import annotations


def _ts_srt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _ts_vtt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _seg_lines(segments: list[dict], ts) -> list[str]:
    lines = []
    for i, s in enumerate(segments):
        text = s.get("text", "").strip()
        if not text:
            continue
        speaker = s.get("speaker")
        label = f"[{speaker}] " if speaker else ""
        lines.append(f"{i + 1}\n{ts(s['start'])} --> {ts(s['end'])}\n{label}{text}\n")
    return lines


def to_srt(segments: list[dict]) -> str:
    return "\n".join(_seg_lines(segments, _ts_srt))


def to_vtt(segments: list[dict]) -> str:
    return "WEBVTT\n\n" + "\n".join(_seg_lines(segments, _ts_vtt))


def verbose_json(result: dict, *, include_words: bool = False) -> dict:
    """OpenAI verbose_json response, with speaker labels as an extension."""
    segments = result.get("segments", []) or []
    duration = max((s.get("end", 0.0) for s in segments), default=0.0)
    out_segments = []
    all_words = []
    for s in segments:
        d = {
            "id": s.get("id", 0),
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "text": s.get("text", ""),
        }
        if s.get("speaker"):
            d["speaker"] = s["speaker"]
        if s.get("lang"):
            d["language"] = s["lang"]
        if s.get("confidence") is not None:
            d["avg_logprob"] = s["confidence"]
        words = s.get("words") or []
        if include_words and words:
            d["words"] = words
            all_words.extend(words)
        out_segments.append(d)
    resp = {
        "task": "transcribe",
        "language": result.get("language") or "",
        "duration": round(duration, 3),
        "text": result.get("verbatim_text", ""),
        "segments": out_segments,
    }
    if include_words and all_words:
        resp["words"] = all_words
    return resp


def format_response(result: dict, response_format: str, *, include_words: bool = False):
    """Return (body, media_type) for the requested OpenAI response_format."""
    if response_format == "text":
        return result.get("verbatim_text", ""), "text/plain; charset=utf-8"
    if response_format == "verbose_json":
        import json
        return json.dumps(verbose_json(result, include_words=include_words)), "application/json"
    if response_format == "srt":
        return to_srt(result.get("segments", [])), "text/srt; charset=utf-8"
    if response_format == "vtt":
        return to_vtt(result.get("segments", [])), "text/vtt; charset=utf-8"
    # default: json {"text": ...}
    import json
    return json.dumps({"text": result.get("verbatim_text", "")}), "application/json"
