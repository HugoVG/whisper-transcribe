"""Tests for pipeline.openai_format — OpenAI /v1/audio/transcriptions payloads."""

import json
import unittest

from pipeline.openai_format import format_response, to_srt, to_vtt, verbose_json

SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 1.2, "speaker": "SPEAKER_01", "lang": "nl",
     "text": "Hallo, goeden dag", "confidence": -0.3,
     "words": [{"word": "Hallo", "start": 0.0, "end": 0.5, "probability": 0.9}]},
    {"id": 1, "start": 1.5, "end": 2.1, "speaker": "SPEAKER_02", "lang": "nl",
     "text": "Hoi hoi", "confidence": -0.12, "words": []},
]
RESULT = {"verbatim_text": "[SPEAKER_01]\n  Hallo, goeden dag\n", "segments": SEGMENTS, "language": "nl"}


class TestVerboseJson(unittest.TestCase):
    def test_shape(self):
        v = verbose_json(RESULT, include_words=True)
        self.assertEqual(v["task"], "transcribe")
        self.assertEqual(v["language"], "nl")
        self.assertEqual(v["duration"], 2.1)
        self.assertEqual(v["text"], RESULT["verbatim_text"])
        self.assertEqual(len(v["segments"]), 2)
        self.assertEqual(v["segments"][0]["speaker"], "SPEAKER_01")
        self.assertEqual(v["segments"][0]["avg_logprob"], -0.3)
        self.assertEqual(v["segments"][0]["words"][0]["word"], "Hallo")
        self.assertEqual(len(v["words"]), 1)

    def test_words_omitted_without_granularity(self):
        v = verbose_json(RESULT, include_words=False)
        self.assertNotIn("words", v["segments"][0])
        self.assertNotIn("words", v)


class TestSrtVtt(unittest.TestCase):
    def test_srt(self):
        srt = to_srt(SEGMENTS)
        self.assertIn("00:00:00,000 --> 00:00:01,200", srt)
        self.assertIn("[SPEAKER_01] Hallo, goeden dag", srt)
        self.assertIn("00:00:01,500 --> 00:00:02,100", srt)

    def test_vtt(self):
        vtt = to_vtt(SEGMENTS)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.200", vtt)


class TestFormatResponse(unittest.TestCase):
    def test_json_default(self):
        body, mt = format_response(RESULT, "json")
        self.assertEqual(mt, "application/json")
        self.assertEqual(json.loads(body), {"text": RESULT["verbatim_text"]})

    def test_text(self):
        body, mt = format_response(RESULT, "text")
        self.assertEqual(body, RESULT["verbatim_text"])
        self.assertIn("text/plain", mt)

    def test_srt_media_type(self):
        body, mt = format_response(RESULT, "srt")
        self.assertIn("text/srt", mt)
        self.assertIn("-->", body)


if __name__ == "__main__":
    unittest.main()
