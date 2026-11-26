"""Tests for pipeline.realtime.vad — streaming energy VAD."""

import math
import unittest

from pipeline.realtime.pcm import floats_to_pcm16
from pipeline.realtime.vad import EnergyVAD, VADConfig


def sine(seconds: float, amp: float = 0.3, freq: float = 220.0, sr: int = 16000) -> bytes:
    n = int(seconds * sr)
    return floats_to_pcm16([amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)])


def silence(seconds: float, sr: int = 16000) -> bytes:
    return floats_to_pcm16([0.0] * int(seconds * sr))


def chunked(data: bytes, chunk_bytes: int = 960):  # 30 ms @16 kHz
    for i in range(0, len(data), chunk_bytes):
        yield data[i:i + chunk_bytes]


class TestEnergyVAD(unittest.TestCase):
    def _vad(self, **kw) -> EnergyVAD:
        cfg = VADConfig(hangover_s=0.2, min_speech_s=0.2, **kw)
        return EnergyVAD(cfg)

    def _feed(self, vad: EnergyVAD, data: bytes) -> list:
        events = []
        for chunk in chunked(data):
            events.extend(vad.feed(chunk))
        return events

    def test_speech_burst_detected(self):
        vad = self._vad()
        events = self._feed(vad, sine(1.0) + silence(0.6))
        kinds = [e.kind for e in events]
        self.assertEqual(kinds, ["speech_start", "speech_end"])
        self.assertAlmostEqual(events[0].time, 0.0, delta=0.06)
        self.assertAlmostEqual(events[1].duration, 1.0, delta=0.15)

    def test_silence_only_no_events(self):
        vad = self._vad()
        events = self._feed(vad, silence(2.0))
        self.assertEqual(events, [])

    def test_two_bursts(self):
        vad = self._vad()
        events = self._feed(vad, sine(0.5) + silence(0.6) + sine(0.5) + silence(0.6))
        kinds = [e.kind for e in events]
        self.assertEqual(kinds, ["speech_start", "speech_end", "speech_start", "speech_end"])
        self.assertGreater(events[2].time, events[1].time)

    def test_flush_ends_speech(self):
        vad = self._vad()
        events = self._feed(vad, sine(1.0))
        self.assertEqual([e.kind for e in events], ["speech_start"])
        events = vad.flush()
        self.assertEqual([e.kind for e in events], ["speech_end"])
        self.assertAlmostEqual(events[0].duration, 1.0, delta=0.1)

    def test_prob_fn_override(self):
        # model VAD path: prob_fn decides speech
        vad = self._vad()
        vad.prob_fn = lambda frame: 0.9 if sum(abs(s) for s in frame) > 0.01 else 0.0
        events = self._feed(vad, sine(0.5) + silence(0.6))
        self.assertEqual([e.kind for e in events], ["speech_start", "speech_end"])

    def test_max_speech_cut(self):
        vad = self._vad(max_speech_s=0.5)
        events = self._feed(vad, sine(1.2) + silence(0.6))
        kinds = [e.kind for e in events]
        self.assertIn("speech_end", kinds)
        self.assertIn("speech_start", kinds[1:])


if __name__ == "__main__":
    unittest.main()
