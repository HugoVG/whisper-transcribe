"""Online (streaming) speaker diarization.

pyannote's batch pipeline needs the whole recording; for a live stream we
embed each finished speech segment and match it against running per-speaker
centroids with cosine similarity. New speaker when the best match is below
`threshold` (and under `max_speakers`).

`embed_fn` is injectable so the logic is unit-testable without pyannote.
"""

from __future__ import annotations

import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class OnlineDiarizer:
    def __init__(
        self,
        embed_fn=None,
        *,
        threshold: float = 0.65,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._centroids: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}
        self._next_id = 1

    @property
    def speakers(self) -> list[str]:
        return sorted(self._centroids)

    def reset(self) -> None:
        self._centroids.clear()
        self._counts.clear()
        self._next_id = 1

    def _new_label(self) -> str:
        label = f"SPEAKER_{self._next_id:02d}"
        self._next_id += 1
        return label

    def label(self, samples: list[float], sample_rate: int = 16000) -> str | None:
        """Assign a speaker label to one speech segment.

        Returns None when no embedding backend is available (caller falls
        back to a single-speaker label, same as the batch pipeline without
        an HF token).
        """
        if self.embed_fn is None:
            return None
        vec = self.embed_fn(samples, sample_rate)
        if not vec:
            return None
        vec = [float(v) for v in vec]

        best_label = None
        best_sim = -1.0
        for label, centroid in self._centroids.items():
            sim = _cosine(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_label is not None and (
            best_sim >= self.threshold
            or (self.max_speakers is not None and len(self._centroids) >= self.max_speakers)
        ):
            label = best_label
        else:
            label = self._new_label()

        # running-mean centroid update
        n = self._counts.get(label, 0)
        if label in self._centroids:
            self._centroids[label] = [
                (c * n + v) / (n + 1) for c, v in zip(self._centroids[label], vec)
            ]
        else:
            self._centroids[label] = list(vec)
        self._counts[label] = n + 1
        return label


def make_diarizer(
    hf_token: str | None = None,
    *,
    threshold: float = 0.65,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """Build an OnlineDiarizer backed by pyannote embeddings when possible.

    Returns (diarizer, method) where method is "pyannote" or "none".
    """
    from .models import get_embed_fn

    embed_fn = get_embed_fn(hf_token)
    if embed_fn is None:
        return OnlineDiarizer(None, threshold=threshold, min_speakers=min_speakers, max_speakers=max_speakers), "none"
    return OnlineDiarizer(embed_fn, threshold=threshold, min_speakers=min_speakers, max_speakers=max_speakers), "pyannote"
