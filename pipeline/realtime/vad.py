"""Streaming voice activity detection.

`EnergyVAD` is a pure-Python streaming VAD: adaptive noise floor, hysteresis
thresholds, hangover before speech-end, min-speech filtering and a max-speech
cut so long monologues are split into segments. An optional `prob_fn`
(e.g. silero-vad) can replace the energy heuristic while keeping the same
state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .pcm import SAMPLE_RATE, pcm16_to_floats, rms


@dataclass
class VADConfig:
    sample_rate: int = SAMPLE_RATE
    frame_ms: int = 30
    min_energy: float = 0.005          # absolute floor for the noise estimate
    speech_factor: float = 3.0         # speech if rms > noise_floor * factor
    min_speech_factor: float = 2.0     # hysteresis: stay in speech above this
    noise_alpha: float = 0.98          # EMA for the adaptive noise floor
    min_speech_s: float = 0.25
    hangover_s: float = 0.8
    max_speech_s: float = 30.0
    prob_speech: float = 0.5           # used when prob_fn is set
    prob_silence: float = 0.3


@dataclass
class VADEvent:
    kind: str        # "speech_start" | "speech_end"
    time: float      # seconds from stream start
    duration: float = 0.0  # speech duration, set on speech_end

    def as_dict(self) -> dict:
        d = {"type": self.kind, "time": round(self.time, 3)}
        if self.kind == "speech_end":
            d["duration"] = round(self.duration, 3)
        return d


class EnergyVAD:
    """Streaming VAD over PCM16 chunks.

    `prob_fn`, when given, maps a frame of float samples to a speech
    probability in [0, 1] and replaces the energy heuristic (silero etc.).
    """

    def __init__(
        self,
        config: VADConfig | None = None,
        prob_fn: Callable[[list[float]], float] | None = None,
    ) -> None:
        self.cfg = config or VADConfig()
        self.prob_fn = prob_fn
        self._reset()

    def _reset(self) -> None:
        self._total_samples = 0
        self._frame: list[float] = []
        self._noise_floor = self.cfg.min_energy
        self._in_speech = False
        self._speech_start = 0.0
        self._last_speech_t = 0.0
        self._silence_frames = 0
        self._speech_frames = 0
        self._hangover_frames = max(1, int(self.cfg.hangover_s * 1000 / self.cfg.frame_ms))
        self._max_speech_frames = max(1, int(self.cfg.max_speech_s * 1000 / self.cfg.frame_ms))

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def _frame_score(self, frame: list[float]) -> tuple[float, bool]:
        """Return (rms, is_speech_candidate) for one frame."""
        r = rms(frame)
        if self.prob_fn is not None:
            try:
                p = float(self.prob_fn(frame))
            except Exception:
                p = None
            if p is not None:
                return r, p >= self.cfg.prob_speech
        threshold = max(self.cfg.min_energy, self._noise_floor * self.cfg.speech_factor)
        return r, r >= threshold

    def _stay_speech(self, frame: list[float]) -> bool:
        if self.prob_fn is not None:
            try:
                return float(self.prob_fn(frame)) >= self.cfg.prob_silence
            except Exception:
                pass
        threshold = max(self.cfg.min_energy, self._noise_floor * self.cfg.min_speech_factor)
        return rms(frame) >= threshold

    def _update_noise(self, frame: list[float], r: float) -> None:
        if not self._in_speech:
            self._noise_floor = self.cfg.noise_alpha * self._noise_floor + (1 - self.cfg.noise_alpha) * max(r, self.cfg.min_energy)

    def feed(self, pcm: bytes) -> list[VADEvent]:
        return self.feed_floats(pcm16_to_floats(pcm))

    def feed_floats(self, samples: list[float]) -> list[VADEvent]:
        events: list[VADEvent] = []
        frame_len = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        self._frame.extend(samples)
        while len(self._frame) >= frame_len:
            frame = self._frame[:frame_len]
            self._frame = self._frame[frame_len:]
            t = self._total_samples / self.cfg.sample_rate
            self._total_samples += frame_len

            r, candidate = self._frame_score(frame)
            self._update_noise(frame, r)

            if not self._in_speech:
                if candidate:
                    self._in_speech = True
                    self._speech_start = t
                    self._last_speech_t = t
                    self._speech_frames = 1
                    self._silence_frames = 0
                    events.append(VADEvent("speech_start", t))
            else:
                self._speech_frames += 1
                if self._stay_speech(frame):
                    self._silence_frames = 0
                    self._last_speech_t = t
                else:
                    self._silence_frames += 1
                if self._speech_frames >= self._max_speech_frames:
                    dur = self._last_speech_t - self._speech_start + self.cfg.frame_ms / 1000.0
                    events.append(VADEvent("speech_end", t, duration=dur))
                    self._in_speech = False
                    self._speech_frames = 0
                    self._silence_frames = 0
                    events.append(VADEvent("speech_start", t))
                    self._in_speech = True
                    self._speech_start = t
                    self._last_speech_t = t
                    self._speech_frames = 1
                elif self._silence_frames >= self._hangover_frames:
                    dur = self._last_speech_t - self._speech_start + self.cfg.frame_ms / 1000.0
                    events.append(VADEvent("speech_end", t, duration=dur))
                    self._in_speech = False
                    self._speech_frames = 0
                    self._silence_frames = 0
        return events

    def flush(self) -> list[VADEvent]:
        """Force-end any ongoing speech (stream end)."""
        events: list[VADEvent] = []
        if self._in_speech:
            t = self._total_samples / self.cfg.sample_rate
            dur = self._last_speech_t - self._speech_start + self.cfg.frame_ms / 1000.0
            events.append(VADEvent("speech_end", t, duration=dur))
            self._in_speech = False
        self._frame.clear()
        return events

    def speech_duration_s(self) -> float:
        return self._last_speech_t - self._speech_start + self.cfg.frame_ms / 1000.0


def make_vad(config: VADConfig | None = None, prefer: str = "auto"):
    """Build a VAD. prefer: "auto" | "silero" | "energy".

    "auto"/"silero" try to load silero-vad via torch.hub and fall back to the
    pure-Python energy VAD when unavailable (no torch / no network / no model).
    """
    from .models import get_silero_prob_fn

    if prefer in ("auto", "silero"):
        try:
            prob_fn = get_silero_prob_fn()
            if prob_fn is not None:
                return EnergyVAD(config, prob_fn=prob_fn), "silero"
        except Exception:
            pass
    if prefer == "silero":
        return EnergyVAD(config), "energy-fallback"
    return EnergyVAD(config), "energy"
