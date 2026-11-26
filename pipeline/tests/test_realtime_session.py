"""Tests for pipeline.realtime.session — the streaming state machine with fakes."""

import math
import unittest

from pipeline.realtime.pcm import floats_to_pcm16
from pipeline.realtime.session import RealtimeConfig, RealtimeSession
from pipeline.realtime.vad import EnergyVAD, VADConfig


def sine(seconds: float, amp: float = 0.3, freq: float = 220.0, sr: int = 16000) -> bytes:
    n = int(seconds * sr)
    return floats_to_pcm16([amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)])


def silence(seconds: float, sr: int = 16000) -> bytes:
    return floats_to_pcm16([0.0] * int(seconds * sr))


def chunked(data: bytes, chunk_bytes: int = 960):
    for i in range(0, len(data), chunk_bytes):
        yield data[i:i + chunk_bytes]


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, samples, sample_rate, language, prompt):
        self.calls += 1
        text = self.texts.pop(0) if self.texts else "..."
        return {
            "text": text,
            "language": "nl",
            "avg_logprob": -0.3,
            "words": [{"word": w, "start": 0.0, "end": 0.1, "probability": 0.9} for w in text.split()],
        }


class FakeDiarizer:
    def __init__(self):
        self.n = 0

    def label(self, samples, sample_rate=16000):
        self.n += 1
        return f"SPEAKER_{self.n:02d}"


def make_session(texts, **cfg_kw) -> RealtimeSession:
    defaults = {"hangover_s": 0.2, "min_speech_s": 0.2, "partial_interval_s": 999.0}
    defaults.update(cfg_kw)
    cfg = RealtimeConfig(**defaults)
    vad = EnergyVAD(VADConfig(hangover_s=0.2, min_speech_s=0.2))
    return RealtimeSession(cfg, stt=FakeSTT(texts), vad=vad, diarizer=FakeDiarizer())


def feed(session, data: bytes) -> list[dict]:
    events = []
    for chunk in chunked(data):
        events.extend(session.feed_audio(chunk))
    return events


class TestSessionFlow(unittest.TestCase):
    def test_segments_speakers_and_verbatim(self):
        s = make_session(["Hallo, goeden dag", "Hoi hoi, wat is het probleem?"])
        events = feed(s, sine(1.0) + silence(0.5))
        events += feed(s, sine(1.0) + silence(0.5))
        events += s.end()

        segments = [e for e in events if e["type"] == "segment"]
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_01")
        self.assertEqual(segments[1]["speaker"], "SPEAKER_02")
        self.assertEqual(segments[0]["text"], "Hallo, goeden dag")
        self.assertEqual(segments[0]["lang"], "nl")
        # gap = 0.5 s silence minus the 0.2 s hangover already inside speech_end
        self.assertAlmostEqual(segments[1]["start"] - segments[0]["end"], 0.3, delta=0.15)
        self.assertTrue(segments[0]["words"])

        kinds = [e["type"] for e in events]
        self.assertIn("speech_start", kinds)
        self.assertIn("speech_end", kinds)
        end = [e for e in events if e["type"] == "session_end"][0]
        self.assertIn("[SPEAKER_01]", end["verbatim_text"])
        self.assertIn("Hallo, goeden dag", end["verbatim_text"])
        self.assertIn("[SPEAKER_02]", end["verbatim_text"])
        self.assertEqual(len(end["segments"]), 2)

    def test_ticket_number_spoken_across_segments(self):
        s = make_session(["Ja hoor, TI 0 1", "5674", "5674"])
        events = feed(s, sine(0.8) + silence(0.4))
        events += feed(s, sine(0.6) + silence(0.4))
        events += feed(s, sine(0.6) + silence(0.4))
        events += s.end()

        tickets = [e for e in events if e["type"] == "entity" and e["entity_type"] == "ticket_number"]
        self.assertTrue(tickets)
        # "TI 0 1" + "5674" + repeated "5674" (confirmation collapsed)
        self.assertEqual(tickets[-1]["value"], "TI015674")
        self.assertTrue(any(t.get("updated") for t in tickets))

    def test_device_and_date_entities(self):
        s = make_session(["met device nummer NL012345", "op donderdag 5 december"])
        events = feed(s, sine(0.8) + silence(0.4))
        events += feed(s, sine(0.8) + silence(0.4))
        events += s.end()

        ents = [e for e in events if e["type"] == "entity"]
        types = {e["entity_type"]: e["value"] for e in ents}
        self.assertEqual(types.get("device_number"), "NL012345")
        self.assertTrue(any(e["entity_type"] == "date" for e in ents))

    def test_short_blip_filtered(self):
        s = make_session(["noise"])
        events = feed(s, sine(0.1) + silence(0.5))
        events += s.end()
        segments = [e for e in events if e["type"] == "segment"]
        self.assertEqual(segments, [])

    def test_partial_events(self):
        s = make_session(["dit is een langere zin die doordsurt"], partial_interval_s=0.5)
        events = feed(s, sine(2.0) + silence(0.5))
        events += s.end()
        partials = [e for e in events if e["type"] == "partial"]
        self.assertGreaterEqual(len(partials), 1)
        self.assertEqual(partials[0]["text"], "dit is een langere zin die doordsurt")

    def test_end_without_audio(self):
        s = make_session([])
        events = s.end()
        self.assertEqual([e["type"] for e in events], ["session_end"])
        self.assertEqual(events[0]["segments"], [])
        self.assertEqual(events[0]["verbatim_text"], "")

    def test_feed_after_end_errors(self):
        s = make_session([])
        s.end()
        events = s.feed_audio(silence(0.1))
        self.assertEqual(events[0]["type"], "error")

    def test_commit_finalizes_ongoing_speech(self):
        s = make_session(["hallo daar"])
        events = feed(s, sine(1.0))  # still speaking, no silence yet
        self.assertEqual([e for e in events if e["type"] == "segment"], [])
        events = s.commit()  # input_audio_buffer.commit
        segments = [e for e in events if e["type"] == "segment"]
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "hallo daar")
        self.assertIn("speech_end", [e["type"] for e in events])

    def test_clear_pending_drops_audio(self):
        s = make_session(["dit wordt gewist"])
        feed(s, sine(1.0))
        s.clear_pending()
        events = s.commit()
        self.assertEqual([e for e in events if e["type"] == "segment"], [])

    def test_speech_start_carries_item_id(self):
        s = make_session(["hoi"])
        events = feed(s, sine(0.8) + silence(0.4))
        starts = [e for e in events if e["type"] == "speech_start"]
        segments = [e for e in events if e["type"] == "segment"]
        self.assertTrue(starts and starts[0]["item_id"].startswith("item_"))
        self.assertEqual(starts[0]["item_id"], f"item_{segments[0]['id']:04d}")


if __name__ == "__main__":
    unittest.main()
