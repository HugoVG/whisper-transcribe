"""OpenAI-style live transcription WebSocket: /live-transcript.

Speaks the OpenAI Realtime API event shape (audio in → transcript events out).
All agent logic (commands, entity extraction, per-sentence LLM) is client-side.

Client -> server (JSON text frames):
  {"type": "session.update", "session": {
      "input_audio_transcription": {"model": "whisper-1", "language": "nl", "prompt": "..."},
      "turn_detection": {"type": "server_vad", "silence_duration_ms": 800}}}
  {"type": "input_audio_buffer.append", "audio": "<base64 PCM16, 16 kHz mono>"}
  {"type": "input_audio_buffer.commit"}
  {"type": "input_audio_buffer.clear"}
  {"type": "ping", "ping": {"value": 123}}
  (binary frames are also accepted and treated as a raw PCM16 append)

Server -> client:
  session.created / session.updated
  input_audio_buffer.speech_started  {audio_start_ms, item_id}
  input_audio_buffer.speech_stopped  {audio_end_ms}
  input_audio_buffer.committed       {item_id}
  input_audio_buffer.cleared
  conversation.item.input_audio_transcribed
      {item_id, transcript, language, segments: [{start, end, text, speaker, avg_logprob, words}]}
  pong {value}
  error {error: {type, message}}

Query params (defaults, overridable via session.update): language, diarize,
hangover_s (= silence_duration_ms/1000), min_speech_s, max_speech_s, prompt,
beam_size, speaker_threshold, min_speakers, max_speakers.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import itertools
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

try:  # newer starlette raises WebSocketDisconnected from receive() after close
    from starlette.websockets import WebSocketDisconnected as _WSDisconnected
except ImportError:  # pragma: no cover
    _WSDisconnected = WebSocketDisconnect

from .session import RealtimeConfig, RealtimeSession

router = APIRouter()

_CONFIG_KEYS = {
    "language": str,
    "diarize": bool,
    "hangover_s": float,
    "min_speech_s": float,
    "max_speech_s": float,
    "prompt": str,
    "beam_size": int,
    "speaker_threshold": float,
    "min_speakers": int,
    "max_speakers": int,
}


def _config_from_query(query) -> RealtimeConfig:
    cfg = RealtimeConfig(entities=False)  # entities are extracted client-side
    for key, cast in _CONFIG_KEYS.items():
        raw = query.get(key)
        if raw in (None, ""):
            continue
        val = raw
        if cast is bool:
            val = str(val).lower() in ("1", "true", "yes", "on")
        else:
            val = cast(val)
        setattr(cfg, key, val)
    return cfg


def _apply_session_update(cfg: RealtimeConfig, session_payload: dict) -> None:
    iat = session_payload.get("input_audio_transcription") or {}
    if iat.get("language"):
        cfg.language = iat["language"] or None
    if iat.get("prompt"):
        cfg.prompt = iat["prompt"]
    td = session_payload.get("turn_detection") or {}
    if td.get("silence_duration_ms") is not None:
        cfg.hangover_s = float(td["silence_duration_ms"]) / 1000.0


def _session_object(cfg: RealtimeConfig, model: str) -> dict:
    return {
        "id": cfg.session_id,
        "object": "realtime.session",
        "model": model,
        "input_audio_transcription": {"model": model, "language": cfg.language},
        "turn_detection": {
            "type": "server_vad",
            "silence_duration_ms": int(cfg.hangover_s * 1000),
        },
    }


def _map_events(events: list[dict]) -> list[tuple[str, dict]]:
    """Map internal session events to OpenAI Realtime-style server events.

    Internal partial/entity/session_end events are intentionally not mapped:
    they are not part of the spec and are handled client-side.
    """
    out = []
    for e in events:
        t = e["type"]
        if t == "speech_start":
            out.append(("input_audio_buffer.speech_started", {
                "audio_start_ms": int(round(e["start"] * 1000)),
                "item_id": e.get("item_id", "item_0000"),
            }))
        elif t == "speech_end":
            out.append(("input_audio_buffer.speech_stopped", {
                "audio_end_ms": int(round(e["end"] * 1000)),
            }))
        elif t == "segment":
            out.append(("conversation.item.input_audio_transcribed", {
                "item_id": f"item_{e['id']:04d}",
                "transcript": e["text"],
                "language": e.get("lang"),
                "segments": [{
                    "start": e["start"],
                    "end": e["end"],
                    "text": e["text"],
                    "speaker": e["speaker"],
                    "avg_logprob": e["confidence"],
                    "words": e.get("words", []),
                }],
            }))
    return out


@router.websocket("/live-transcript")
async def live_transcript(websocket: WebSocket) -> None:
    await websocket.accept()
    query = dict(websocket.query_params)
    cfg = _config_from_query(query)
    session: RealtimeSession | None = None
    model = "whisper-1"
    counter = itertools.count(1)

    async def send_event(type_: str, **payload) -> None:
        await websocket.send_json({"type": type_, "event_id": f"event_{next(counter):05d}", **payload})

    def ensure_session() -> RealtimeSession:
        nonlocal session
        if session is None:
            session = RealtimeSession(cfg)
        return session

    await send_event("session.created", session=_session_object(cfg, model))

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes") is not None:
                events = await asyncio.to_thread(ensure_session().feed_audio, msg["bytes"])
                for type_, payload in _map_events(events):
                    await send_event(type_, **payload)
                continue
            text = msg.get("text")
            if text is None:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await send_event("error", error={"type": "invalid_request", "message": "invalid JSON frame"})
                continue
            mtype = data.get("type")

            if mtype == "session.update":
                _apply_session_update(cfg, data.get("session") or {})
                model = (data.get("session") or {}).get("input_audio_transcription", {}).get("model", model)
                await send_event("session.updated", session=_session_object(cfg, model))
            elif mtype == "input_audio_buffer.append":
                audio_b64 = data.get("audio", "")
                try:
                    pcm = base64.b64decode(audio_b64, validate=True)
                except (binascii.Error, ValueError):
                    await send_event("error", error={"type": "invalid_request", "message": "audio must be base64 PCM16"})
                    continue
                events = await asyncio.to_thread(ensure_session().feed_audio, pcm)
                for type_, payload in _map_events(events):
                    await send_event(type_, **payload)
            elif mtype == "input_audio_buffer.commit":
                if session is None:
                    await send_event("error", error={"type": "invalid_request", "message": "no audio buffered"})
                    continue
                events = await asyncio.to_thread(session.commit)
                mapped = _map_events(events)
                item_ids = {p.get("item_id") for _, p in mapped if p.get("item_id")}
                await send_event("input_audio_buffer.committed", item_id=(next(iter(item_ids)) if item_ids else "item_0000"))
                for type_, payload in mapped:
                    await send_event(type_, **payload)
            elif mtype == "input_audio_buffer.clear":
                if session is not None:
                    await asyncio.to_thread(session.clear_pending)
                await send_event("input_audio_buffer.cleared")
            elif mtype == "ping":
                await send_event("pong", value=(data.get("ping") or {}).get("value"))
            else:
                await send_event("error", error={"type": "invalid_request", "message": f"unsupported event type: {mtype}"})
    except (WebSocketDisconnect, _WSDisconnected):
        if session is not None:
            await asyncio.to_thread(session.commit)
