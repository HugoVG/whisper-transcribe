"""Real-time streaming transcription: VAD + diarization + STT over WebSocket.

Pure-stdlib at import time — heavy deps (numpy, torch, faster-whisper,
pyannote) are imported lazily inside the model-backed functions so the
pure-logic parts stay unit-testable without a GPU.
"""

from .entities import extract_entities
from .session import RealtimeConfig, RealtimeSession
from .vad import EnergyVAD, VADConfig

__all__ = [
    "EnergyVAD",
    "RealtimeConfig",
    "RealtimeSession",
    "VADConfig",
    "extract_entities",
]
