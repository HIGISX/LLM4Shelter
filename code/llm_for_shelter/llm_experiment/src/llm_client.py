"""Reproducible local Ollama client; no paid API is used."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass(frozen=True)
class InferenceConfig:
    model: str = "qwen3.5:9b"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 512
    think: bool = False
    endpoint: str = "http://127.0.0.1:11434/api/chat"


class OllamaClient:
    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()

    def metadata(self) -> dict[str, Any]:
        return asdict(self.config)

    def chat(self, system_prompt: str, request_text: str, *, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "think": self.config.think,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_text},
            ],
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "seed": self.config.seed,
                "num_predict": self.config.max_tokens,
            },
        }
        if schema is not None:
            body["format"] = schema
        started = datetime.now(timezone.utc).isoformat()
        response = requests.post(self.config.endpoint, json=body, timeout=180)
        response.raise_for_status()
        payload = response.json()
        return {
            "timestamp_utc": started,
            "model": payload.get("model", self.config.model),
            "configuration": self.metadata(),
            "raw_request": request_text,
            "raw_response": payload.get("message", {}).get("content", ""),
            "ollama_response": payload,
        }


def save_raw_record(path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
