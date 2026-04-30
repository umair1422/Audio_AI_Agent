from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    timeout_s: float


def default_ollama_config() -> OllamaConfig:
    return OllamaConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "phi3:mini"),
        timeout_s=float(os.getenv("OLLAMA_TIMEOUT_S", "20")),
    )


class OllamaError(RuntimeError):
    pass


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort JSON extraction for models that may add extra text.
    """
    text = text.strip()
    if not text:
        raise OllamaError("LLM returned empty response.")

    # Fast-path: strict JSON
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # Fallback: scan for first '{' ... matching '}'.
    start = text.find("{")
    if start == -1:
        raise OllamaError("LLM response did not contain JSON object.")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    val = json.loads(candidate)
                    if isinstance(val, dict):
                        return val
                except Exception:
                    break
    raise OllamaError("Failed to parse JSON from LLM response.")


def ollama_generate_json(prompt: str, *, config: OllamaConfig | None = None) -> dict[str, Any]:
    cfg = config or default_ollama_config()
    url = f"{cfg.base_url.rstrip('/')}/api/generate"
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=cfg.timeout_s)
    except requests.RequestException as e:
        raise OllamaError(f"Failed to reach Ollama at '{cfg.base_url}': {e}") from e

    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    raw = (data.get("response") or "").strip()
    return _extract_first_json_object(raw)

