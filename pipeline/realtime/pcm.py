"""PCM16 (16-bit little-endian, mono) helpers — pure stdlib so they stay testable."""

from __future__ import annotations

import array
import math

SAMPLE_RATE = 16000


def pcm16_to_floats(data: bytes) -> list[float]:
    """Decode PCM16 LE bytes to floats in [-1, 1]."""
    if len(data) % 2:
        data = data[:-1]
    a = array.array("h")
    a.frombytes(data)
    if a.itemsize != 2:  # big-endian host: swap
        a.byteswap()
    return [s / 32768.0 for s in a]


def floats_to_pcm16(samples: list[float]) -> bytes:
    a = array.array("h", (max(-32768, min(32767, int(s * 32767))) for s in samples))
    if a.itemsize != 2:
        a.byteswap()
    return a.tobytes()


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def duration_s(samples: list[float] | int, sample_rate: int = SAMPLE_RATE) -> float:
    n = samples if isinstance(samples, int) else len(samples)
    return n / sample_rate
