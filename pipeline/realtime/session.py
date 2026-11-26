"""Real-time session state machine: PCM in -> events out.

    PCM16 chunk -> VAD -> speech segment buffer
        -> (every partial_interval_s) partial transcript event
        -> on speech end: STT (nl/en auto) + online speaker label
        -> entity extraction (ticket numbers, device numbers, dates, ...)
        -> segment + entity events

Synchronous and injectable (stt / vad / diarizer), so it is fully unit-testable
without GPU models. The WebSocket server drives it via asyncio.to_thread.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from ..render import render_text
from .entities import collapse_repeated_digits, extract_entities
from .pcm import SAMPLE_RATE, pcm16_to_floats

_CONTINUATION_RE = re.compile(r"(?:\d[\s-]*){2,}")


@dataclass
class RealtimeConfig:
    sample_rate: int = SAMPLE_RATE
    language: str | None = None          # None = auto-detect nl/en per segment
    diarize: bool = True
    entities: bool = True
    vad: str = "auto"                    # auto | silero | energy
    hangover_s: float = 0.8
    min_speech_s: float = 0.3
    max_speech_s: float = 30.0
    partial_interval_s: float = 2.5
    partial_window_s: float = 15.0
    prompt: str | None = None            # whisper initial_prompt (glossary/context)
    beam_size: int = 5
    speaker_threshold: float = 0.65
    min_speakers: int | None = None
    max_speakers: int | None = None
    ticket_window_s: float = 8.0        # spoken-digit groups merge within this window
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class RealtimeSession:
    def __init__(self, config: RealtimeConfig | None = None, *, stt=None, vad=None, diarizer=None) -> None:
        self.config = config or RealtimeConfig()
        if stt is None:
            from .stt import make_stt
            stt = make_stt(beam_size=self.config.beam_size)
        if vad is None:
            from .vad import VADConfig, make_vad
            vad, _method = make_vad(
                VADConfig(
                    sample_rate=self.config.sample_rate,
                    hangover_s=self.config.hangover_s,
                    min_speech_s=self.config.min_speech_s,
                    max_speech_s=self.config.max_speech_s,
                ),
                prefer=self.config.vad,
            )
        self.stt = stt
        self.vad = vad
        self.diarizer = diarizer
        if self.diarizer is None and self.config.diarize:
            from .diarize_stream import make_diarizer
            self.diarizer, _m = make_diarizer(
                threshold=self.config.speaker_threshold,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )

        self.segments: list[dict] = []
        self.entities: list[dict] = []
        self._collecting = False
        self._pending: list[float] = []
        self._speech_start = 0.0
        self._total_samples = 0
        self._last_partial = 0.0
        self._segment_seq = 0
        self._open_ticket: dict | None = None
        self._ended = False

    # ── streaming ────────────────────────────────────────────────────────

    def feed_audio(self, pcm: bytes) -> list[dict]:
        if self._ended:
            return [{"type": "error", "detail": "session already ended"}]
        samples = pcm16_to_floats(pcm)
        events: list[dict] = []
        collecting_at_entry = self._collecting
        if collecting_at_entry:
            self._pending.extend(samples)
        for ev in self.vad.feed_floats(samples):
            if ev.kind == "speech_start":
                if not collecting_at_entry:
                    self._pending = list(samples)
                self._collecting = True
                self._speech_start = ev.time
                self._last_partial = ev.time
                events.append({
                    "type": "speech_start",
                    "start": round(ev.time, 3),
                    "item_id": f"item_{self._segment_seq:04d}",
                })
            elif ev.kind == "speech_end":
                events.append({
                    "type": "speech_end",
                    "start": round(self._speech_start, 3),
                    "end": round(ev.time, 3),
                    "duration": round(ev.duration, 3),
                })
                events.extend(self._finalize_segment(self._pending, self._speech_start, ev.time, ev.duration))
                self._pending = []
                self._collecting = False
                collecting_at_entry = False
        self._total_samples += len(samples)
        if self._collecting:
            events.extend(self._maybe_partial())
        return events

    def commit(self) -> list[dict]:
        """Force-finalize any ongoing speech (OpenAI input_audio_buffer.commit)."""
        events: list[dict] = []
        for ev in self.vad.flush():
            if ev.kind == "speech_end" and self._collecting:
                self._collecting = False
                events.append({
                    "type": "speech_end",
                    "start": round(self._speech_start, 3),
                    "end": round(ev.time, 3),
                    "duration": round(ev.duration, 3),
                })
                events.extend(self._finalize_segment(self._pending, self._speech_start, ev.time, ev.duration))
                self._pending = []
        return events

    def clear_pending(self) -> None:
        """Drop buffered but uncommitted audio (input_audio_buffer.clear)."""
        self._pending = []
        if self._collecting:
            self.vad.flush()
            self._collecting = False

    def end(self) -> list[dict]:
        if self._ended:
            return []
        self._ended = True
        events = self.commit()
        duration = self._total_samples / self.config.sample_rate
        events.append({
            "type": "session_end",
            "session_id": self.config.session_id,
            "duration_s": round(duration, 3),
            "segments": self.segments,
            "entities": self.entities,
            "verbatim_text": self.verbatim_text(),
        })
        return events

    # ── internals ────────────────────────────────────────────────────────

    def _maybe_partial(self) -> list[dict]:
        now = self._total_samples / self.config.sample_rate
        if now - self._last_partial < self.config.partial_interval_s:
            return []
        if len(self._pending) < int(1.0 * self.config.sample_rate):
            return []
        self._last_partial = now
        window = self._pending[-int(self.config.partial_window_s * self.config.sample_rate):]
        res = self.stt.transcribe(window, self.config.sample_rate, self.config.language, self.config.prompt)
        text = (res.get("text") or "").strip()
        if not text:
            return []
        return [{
            "type": "partial",
            "start": round(self._speech_start, 3),
            "end": round(now, 3),
            "lang": res.get("language"),
            "text": text,
        }]

    def _finalize_segment(self, samples: list[float], start: float, end: float, duration: float | None = None) -> list[dict]:
        if duration is None:
            duration = end - start
        if duration < self.config.min_speech_s or not samples:
            return []
        res = self.stt.transcribe(samples, self.config.sample_rate, self.config.language, self.config.prompt)
        text = (res.get("text") or "").strip()
        if not text:
            return []

        speaker = "SPEAKER_00"
        if self.config.diarize and self.diarizer is not None:
            speaker = self.diarizer.label(samples, self.config.sample_rate) or "SPEAKER_00"

        seg_id = self._segment_seq
        self._segment_seq += 1
        words = [
            {
                "word": w.get("word", ""),
                "start": round(start + w.get("start", 0.0), 3),
                "end": round(start + w.get("end", 0.0), 3),
                "probability": round(w.get("probability", 0.0), 3),
            }
            for w in res.get("words", [])
        ]
        segment = {
            "id": seg_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "lang": res.get("language"),
            "text": text,
            "confidence": round(res.get("avg_logprob", 0.0), 3),
            "words": words,
        }
        self.segments.append(segment)
        events: list[dict] = [{"type": "segment", **{k: v for k, v in segment.items()}}]
        if self.config.entities:
            events.extend(self._extract_segment_entities(segment))
        return events

    def _extract_segment_entities(self, segment: dict) -> list[dict]:
        events: list[dict] = []
        ents = extract_entities(segment["text"])
        for e in ents:
            if e.type == "ticket_number":
                m = re.match(r"(?P<prefix>[A-Z]*)(?P<digits>\d+)$", e.value)
                prefix = m.group("prefix") if m else ""
                digits = collapse_repeated_digits(m.group("digits")) if m else e.value
                self._open_ticket = {
                    "prefix": prefix,
                    "digits": digits,
                    "last_group": digits,
                    "last_end": segment["end"],
                }
                events.append(self._entity_event(segment, e, value=prefix + digits))
            elif e.type == "number" and self._is_ticket_open(segment):
                self._merge_ticket_group(segment, e.value, e.raw, e.start, e.end, e.confidence, events)
                continue
            else:
                events.append(self._entity_event(segment, e))
            self.entities.append(events[-1])
        # merge short spoken digit groups ("0 1", "5674") into an open ticket
        if self._is_ticket_open(segment):
            for m in _CONTINUATION_RE.finditer(segment["text"]):
                digits = re.sub(r"\D", "", m.group(0))
                if not digits:
                    continue
                if any(not (m.end() <= e.start or m.start() >= e.end) for e in ents):
                    continue
                self._merge_ticket_group(segment, digits, m.group(0).strip(), m.start(), m.end(), 0.8, events)
        if self._open_ticket and segment["end"] - self._open_ticket["last_end"] > self.config.ticket_window_s:
            self._open_ticket = None
        return events

    def _merge_ticket_group(
        self, segment: dict, group: str, raw: str, start: int, end: int, conf: float, events: list[dict]
    ) -> None:
        """Append a spoken digit group to the open ticket, skipping immediate
        repeats (confirmation: the same group said twice in a row)."""
        ticket = self._open_ticket
        ticket["last_end"] = segment["end"]
        if group == ticket.get("last_group"):
            return
        ticket["digits"] = collapse_repeated_digits(ticket["digits"] + group)
        ticket["last_group"] = group
        ev = {
            "type": "entity",
            "segment_id": segment["id"],
            "entity_type": "ticket_number",
            "value": ticket["prefix"] + ticket["digits"],
            "raw": raw,
            "start": start,
            "end": end,
            "confidence": conf,
            "updated": True,
        }
        events.append(ev)
        self.entities.append(ev)

    def _is_ticket_open(self, segment: dict) -> bool:
        return (
            self._open_ticket is not None
            and segment["end"] - self._open_ticket["last_end"] <= self.config.ticket_window_s
        )

    def _entity_event(self, segment: dict, e, *, value: str | None = None, entity_type: str | None = None, updated: bool = False) -> dict:
        ev = {
            "type": "entity",
            "segment_id": segment["id"],
            "entity_type": entity_type or e.type,
            "value": value if value is not None else e.value,
            "raw": e.raw,
            "start": e.start,
            "end": e.end,
            "confidence": e.confidence,
        }
        if updated:
            ev["updated"] = True
        return ev

    def verbatim_text(self) -> str:
        if not self.segments:
            return ""
        turns = [
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
            for s in self.segments
        ]
        return render_text(self.segments, turns)
