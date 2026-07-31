"""Shared LLM client — supports Ollama and any OpenAI-compatible endpoint.

Auto-detects endpoint type based on URL pattern:
- URLs without /v1/ path → Ollama native API (/api/chat, /api/tags)
- URLs with /v1/ path → OpenAI-compatible API (/v1/chat/completions, /v1/models)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class LLMClient:
    base_url: str
    model: str | None = None
    api_key: str | None = None
    timeout: int = 300

    @property
    def is_openai(self) -> bool:
        return "/v1" in self.base_url or self.api_key is not None

    # ── Chat ────────────────────────────────────────────────────────────────

    def chat(self, prompt: str, *, system: str = "", temperature: float = 0.1) -> str:
        if self.is_openai:
            return self._chat_openai(prompt, system, temperature)
        return self._chat_ollama(prompt, system, temperature)

    def _chat_openai(self, prompt: str, system: str, temperature: float) -> str:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/v1/chat/completions" if "/v1" not in url else "/chat/completions"
        payload = {
            "model": self.model or "gpt-4o",
            "messages": [],
            "temperature": temperature,
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()

    def _chat_ollama(self, prompt: str, system: str, temperature: float) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model or "llama3",
            "messages": [
                {"role": "system", "content": system or "/no_think"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": 4096, "temperature": temperature},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return (data.get("message", {}) or {}).get("content", "").strip()

    # ── List models ─────────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        if self.is_openai:
            return self._list_models_openai()
        return self._list_models_ollama()

    def _list_models_openai(self) -> list[str]:
        url = self.base_url.rstrip("/")
        if not url.endswith("/models"):
            url += "/v1/models" if "/v1" not in url else "/models"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def _list_models_ollama(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Auto-select best model ──────────────────────────────────────────────

    def auto_select_model(self) -> str | None:
        models = self.list_models()
        if not models:
            return None
        # Filter out embedding/specialist models
        filtered = [m for m in models if "embed" not in m.lower()]
        if not filtered:
            filtered = models
        if self.is_openai:
            # Prefer gpt-4o, then gpt-4, then first model
            for pref in ("gpt-4o", "gpt-4", "gpt-4-turbo"):
                for m in filtered:
                    if pref in m.lower():
                        return m
            return filtered[0]
        # Ollama: prefer instruct, penalize coder/qwen
        scored = []
        import re
        for name in filtered:
            s = 1.0
            lower = name.lower()
            if "instruct" in lower:
                s *= 1.3
            if "coder" in lower:
                s *= 0.7
            if "qwen3" in lower:
                s *= 0.5
            size_m = re.search(r":(\d+)b", lower)
            if size_m:
                params = int(size_m.group(1))
                if params >= 32:
                    s *= 1.5
                elif params >= 14:
                    s *= 1.2
                elif params < 8:
                    s *= 0.6
            scored.append((name, s))
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]


def create_client(
    base_url: str,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Factory that resolves model if not provided."""
    client = LLMClient(base_url=base_url, model=model, api_key=api_key)
    if not client.model:
        client.model = client.auto_select_model()
    return client
