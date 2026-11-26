"""Streaming STT: transcribe finished VAD segments with faster-whisper.

The model is loaded once (singleton in models.py) and reused across segments.
faster-whisper gives word-level timestamps natively, and language=None lets
it auto-detect Dutch vs English per segment.
"""

from __future__ import annotations


class StreamingSTT:
    def __init__(self, transcribe_fn=None, *, beam_size: int = 5) -> None:
        # transcribe_fn(samples, sample_rate, language, prompt) -> dict
        self._transcribe_fn = transcribe_fn
        self.beam_size = beam_size

    def transcribe(
        self,
        samples: list[float],
        sample_rate: int = 16000,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Return {text, language, avg_logprob, words, segments}."""
        if self._transcribe_fn is not None:
            return self._transcribe_fn(samples, sample_rate, language, prompt)
        from .models import get_asr_model

        model = get_asr_model()
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32)
        segments, info = model.transcribe(
            audio,
            language=language or None,
            beam_size=self.beam_size,
            vad_filter=False,  # we already VAD'd
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        seg_list = []
        words = []
        all_logprobs: list[float] = []
        for seg in segments:
            seg_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
            })
            all_logprobs.append(seg.avg_logprob or 0.0)
            for w in seg.words or []:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
        text = " ".join(s["text"] for s in seg_list).strip()
        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "avg_logprob": sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0,
            "segments": seg_list,
            "words": words,
        }


def make_stt(beam_size: int = 5) -> StreamingSTT:
    return StreamingSTT(beam_size=beam_size)
