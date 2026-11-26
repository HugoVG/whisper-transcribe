"""Tests for pipeline.realtime.diarize_stream — online speaker labeling."""

import unittest

from pipeline.realtime.diarize_stream import OnlineDiarizer


class TestOnlineDiarizer(unittest.TestCase):
    def test_same_embedding_same_label(self):
        vectors = iter([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]])
        d = OnlineDiarizer(lambda s, sr: next(vectors))
        a = d.label([0.1] * 100)
        b = d.label([0.2] * 100)
        self.assertEqual(a, b)
        self.assertEqual(a, "SPEAKER_01")

    def test_different_embedding_new_label(self):
        vectors = iter([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        d = OnlineDiarizer(lambda s, sr: next(vectors))
        a = d.label([0.1] * 100)
        b = d.label([0.2] * 100)
        self.assertEqual(a, "SPEAKER_01")
        self.assertEqual(b, "SPEAKER_02")
        self.assertEqual(sorted(d.speakers), ["SPEAKER_01", "SPEAKER_02"])

    def test_max_speakers_forces_assignment(self):
        vectors = iter([[1.0, 0.0], [0.0, 1.0]])
        d = OnlineDiarizer(lambda s, sr: next(vectors), threshold=0.9, max_speakers=1)
        a = d.label([0.1] * 100)
        b = d.label([0.2] * 100)
        self.assertEqual(a, b)

    def test_threshold_boundary(self):
        # cosine ~0.707 < 0.8 threshold -> new speaker
        vectors = iter([[1.0, 0.0], [1.0, 1.0]])
        d = OnlineDiarizer(lambda s, sr: next(vectors), threshold=0.8)
        a = d.label([0.1] * 100)
        b = d.label([0.2] * 100)
        self.assertNotEqual(a, b)

    def test_no_embed_fn_returns_none(self):
        d = OnlineDiarizer(None)
        self.assertIsNone(d.label([0.1] * 100))

    def test_reset(self):
        d = OnlineDiarizer(lambda s, sr: [1.0, 0.0])
        d.label([0.1] * 100)
        d.reset()
        self.assertEqual(d.speakers, [])
        self.assertEqual(d.label([0.1] * 100), "SPEAKER_01")


if __name__ == "__main__":
    unittest.main()
