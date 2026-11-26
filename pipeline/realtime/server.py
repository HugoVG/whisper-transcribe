"""Real-time streaming API: WebSocket endpoint + health.

Protocol (ws://host:8000/ws/stream):

  client -> server:
    {"type": "start", "config": {...}}   optional; query params are defaults
    <binary>                             PCM16 LE, mono, 16 kHz chunks
    {"type": "end"}                      finish session (also on disconnect)
    {"type": "ping"}

  server -> client (JSON events):
    ready | speech_start | speech_end | partial | segment | entity | error | session_end

Query params: language, diarize, entities, vad, hangover_s, min_speech_s,
max_speech_s, partial_interval_s, prompt, beam_size, speaker_threshold,
min_speakers, max_speakers.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import models
from .models import asr_model_path, hf_token
from .session import RealtimeConfig, RealtimeSession

router = APIRouter()

_CONFIG_KEYS = {
    "language": str,
    "diarize": bool,
    "entities": bool,
    "vad": str,
    "hangover_s": float,
    "min_speech_s": float,
    "max_speech_s": float,
    "partial_interval_s": float,
    "partial_window_s": float,
    "prompt": str,
    "beam_size": int,
    "speaker_threshold": float,
    "min_speakers": int,
    "max_speakers": int,
    "ticket_window_s": float,
}


def _config_from(query, overrides: dict | None) -> RealtimeConfig:
    cfg = RealtimeConfig()
    raw = dict(query)
    if overrides:
        raw.update(overrides)
    for key, cast in _CONFIG_KEYS.items():
        if key not in raw or raw[key] in (None, ""):
            continue
        val = raw[key]
        if cast is bool:
            val = str(val).lower() in ("1", "true", "yes", "on")
        else:
            val = cast(val)
        setattr(cfg, key, val)
    return cfg


@router.get("/realtime/health")
def realtime_health() -> dict:
    return {
        "ok": True,
        "asr_model": asr_model_path(),
        "asr_model_loaded": models.asr_model_loaded(),
        "vad": "silero if available, else energy",
        "diarization": "pyannote" if hf_token() else "none (set HF_TOKEN)",
        "hf_token_present": hf_token() is not None,
    }


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    query = dict(websocket.query_params)
    session: RealtimeSession | None = None

    async def send_events(events: list[dict]) -> None:
        for ev in events:
            await websocket.send_json(ev)

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes") is not None:
                if session is None:
                    session = RealtimeSession(_config_from(query, None))
                    await websocket.send_json({
                        "type": "ready",
                        "session_id": session.config.session_id,
                        "config": {k: getattr(session.config, k) for k in _CONFIG_KEYS},
                    })
                events = await asyncio.to_thread(session.feed_audio, msg["bytes"])
                await send_events(events)
            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "invalid JSON message"})
                    continue
                mtype = data.get("type")
                if mtype == "start":
                    if session is not None:
                        await websocket.send_json({"type": "error", "detail": "session already started"})
                        continue
                    session = RealtimeSession(_config_from(query, data.get("config")))
                    await websocket.send_json({
                        "type": "ready",
                        "session_id": session.config.session_id,
                        "config": {k: getattr(session.config, k) for k in _CONFIG_KEYS},
                    })
                elif mtype == "end":
                    if session is not None:
                        events = await asyncio.to_thread(session.end)
                        await send_events(events)
                    break
                elif mtype == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    await websocket.send_json({"type": "error", "detail": f"unknown message type: {mtype}"})
    except WebSocketDisconnect:
        if session is not None:
            await asyncio.to_thread(session.end)
