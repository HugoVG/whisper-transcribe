"""Lazy singletons for the real-time models.

Everything heavy (torch, numpy, faster-whisper, pyannote) is imported inside
functions so importing this module is cheap and works without a GPU — the
pure-Python fallbacks (energy VAD, no-embedding diarization) stay available.

Config via env:
  REALTIME_VAD          auto | silero | energy   (default auto)
  REALTIME_ASR_MODEL    faster-whisper model path/name (default: $WHISPER_MODEL_DIR/large-v3 if present, else "large-v3")
  REALTIME_EMBED_MODEL  pyannote embedding model   (default pyannote/embedding)
  REALTIME_DEVICE       cuda | cpu                (default cuda)
  HF_TOKEN / HF_TOKEN_FILE / /run/secrets/hf-token   (same resolution as the batch API)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

_lock = threading.Lock()
_silero_prob_fn = None
_silero_tried = False
_embed_fn = None
_embed_tried = False
_asr_model = None
_asr_tried = False


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_path = os.environ.get("HF_TOKEN_FILE")
    candidates = [
        Path(env_path) if env_path else None,
        Path("/run/secrets/hf-token"),
        Path.home() / ".config" / "whisper" / "hf-token",
    ]
    for p in candidates:
        if p and p.exists():
            return p.read_text(encoding="utf-8", errors="replace").strip()
    return None


def _device() -> str:
    return os.environ.get("REALTIME_DEVICE", "cuda")


def get_silero_prob_fn():
    """Return prob_fn(frame_samples) -> float for silero-vad, or None."""
    global _silero_prob_fn, _silero_tried
    with _lock:
        if _silero_tried:
            return _silero_prob_fn
        _silero_tried = True
        try:
            import torch

            model, _utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True, force_reload=False
            )
            model.to(_device())
            (get_speech_timestamps, _, read_audio, _, _) = _utils

            def prob_fn(frame: list[float]) -> float:
                import numpy as np

                wav = torch.from_numpy(np.asarray(frame, dtype=np.float32))
                if wav.dim() > 1:
                    wav = wav.mean(dim=0)
                return float(model(wav, 16000).item())

            _silero_prob_fn = prob_fn
        except Exception:
            _silero_prob_fn = None
        return _silero_prob_fn


def get_embed_fn(token: str | None = None):
    """Return embed_fn(samples, sample_rate) -> list[float], or None."""
    global _embed_fn, _embed_tried
    with _lock:
        if _embed_tried:
            return _embed_fn
        _embed_tried = True
        try:
            import numpy as np
            import torch
            from pyannote.audio import Inference

            model_name = os.environ.get("REALTIME_EMBED_MODEL", "pyannote/embedding")
            tok = token or hf_token()
            if not tok:
                return None
            os.environ["HF_TOKEN"] = tok
            inference = Inference(model_name, device=torch.device(_device()), token=tok)

            def embed_fn(samples: list[float], sample_rate: int = 16000):
                if sample_rate != 16000:
                    import numpy as _np
                    x = _np.asarray(samples, dtype=_np.float32)
                    n_out = int(len(x) * 16000 / sample_rate)
                    samples = _np.interp(
                        _np.linspace(0, len(x), n_out, endpoint=False),
                        _np.arange(len(x)), x,
                    ).tolist()
                waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32))[None, :]
                with torch.no_grad():
                    emb = inference({"waveform": waveform, "sample_rate": 16000})
                return emb.squeeze(0).cpu().tolist()

            _embed_fn = embed_fn
        except Exception:
            _embed_fn = None
        return _embed_fn


def asr_model_path() -> str:
    env = os.environ.get("REALTIME_ASR_MODEL", "").strip()
    if env:
        return env
    default = Path(os.environ.get("WHISPER_MODEL_DIR", "/opt/whisper-models")) / "large-v3"
    if default.exists():
        return str(default)
    return "large-v3"


def get_asr_model():
    """Load (once) the faster-whisper model used for segment transcription."""
    global _asr_model, _asr_tried
    with _lock:
        if _asr_tried:
            return _asr_model
        _asr_tried = True
        from faster_whisper import WhisperModel

        compute_type = os.environ.get("REALTIME_COMPUTE_TYPE", "float16")
        _asr_model = WhisperModel(
            asr_model_path(), device=_device(), compute_type=compute_type
        )
        return _asr_model


def asr_model_loaded() -> bool:
    return _asr_tried and _asr_model is not None


def reset_models() -> None:
    """Drop cached models (tests / reconfiguration)."""
    global _silero_prob_fn, _silero_tried, _embed_fn, _embed_tried, _asr_model, _asr_tried
    with _lock:
        _silero_prob_fn = None
        _silero_tried = False
        _embed_fn = None
        _embed_tried = False
        _asr_model = None
        _asr_tried = False
