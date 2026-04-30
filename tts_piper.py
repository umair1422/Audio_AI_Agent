from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import wave

from piper.voice import PiperVoice  # type: ignore

@dataclass(frozen=True)
class PiperConfig:
    model_path: str
    model_json_path: str | None = None


def default_piper_config() -> PiperConfig:
    return PiperConfig(
        model_path=os.getenv("PIPER_MODEL", "models/piper/en_US-lessac-medium.onnx"),
        model_json_path=os.getenv("PIPER_MODEL_JSON") or None,
    )


class PiperTtsError(RuntimeError):
    pass


def synthesize_wav_bytes(text: str, *, config: PiperConfig | None = None) -> bytes:
    cfg = config or default_piper_config()

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise PiperTtsError(
            f"Piper model not found at '{model_path}'. Download a piper .onnx model and set PIPER_MODEL."
        )
    model_json_path = Path(cfg.model_json_path) if cfg.model_json_path else None
    if model_json_path and not model_json_path.exists():
        raise PiperTtsError(f"Piper model json not found at '{model_json_path}'. Set PIPER_MODEL_JSON correctly.")

    safe_text = (text or "").strip()
    if not safe_text:
        raise PiperTtsError("TTS text is empty.")

    try:
        voice = PiperVoice.load(str(model_path), str(model_json_path) if model_json_path else None)
    except Exception as e:
        raise PiperTtsError(f"Failed to load piper voice model: {e}") from e

    buf = BytesIO()
    try:
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(safe_text, wav_file)
    except Exception as e:
        raise PiperTtsError(f"Piper synthesis failed: {e}") from e

    return buf.getvalue()

