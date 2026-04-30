from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import wave


@dataclass(frozen=True)
class WhisperCppConfig:
    whisper_bin: str
    model_path: str
    ffmpeg_bin: str
    language: str | None = None


def default_whispercpp_config() -> WhisperCppConfig:
    return WhisperCppConfig(
        # Homebrew's whisper.cpp installs `whisper-cli` (not `whisper`) on macOS.
        whisper_bin=os.getenv("WHISPER_CPP_BIN", "whisper-cli"),
        # For low latency, default to the faster `base` model.
        # You can switch to `small` for higher accuracy by setting WHISPER_CPP_MODEL.
        model_path=os.getenv("WHISPER_CPP_MODEL", "models/ggml-base.bin"),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
        # Default to English for higher accuracy + less hallucination.
        # You can still override via WHISPER_CPP_LANG if needed.
        language=os.getenv("WHISPER_CPP_LANG") or "en",
    )


class WhisperCppError(RuntimeError):
    pass


def _require_binary(bin_name: str) -> str:
    resolved = shutil.which(bin_name)
    if resolved:
        return resolved
    if Path(bin_name).exists():
        return bin_name
    raise WhisperCppError(
        f"Required binary '{bin_name}' not found in PATH. "
        f"Set env var (e.g. WHISPER_CPP_BIN/FFMPEG_BIN) to an absolute path."
    )


def transcribe_wav_bytes(wav_bytes: bytes, *, config: WhisperCppConfig | None = None) -> str:
    """
    Transcribe a WAV file payload (must be a valid wav container) using whisper.cpp.
    This path avoids ffmpeg and is ideal for real-time PCM streaming.
    """
    cfg = config or default_whispercpp_config()
    whisper_bin = _require_binary(cfg.whisper_bin)

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise WhisperCppError(
            f"Whisper model not found at '{model_path}'. "
            "Download a ggml model (e.g. ggml-small.bin) and set WHISPER_CPP_MODEL."
        )

    with tempfile.TemporaryDirectory(prefix="voice_stt_wav_") as td:
        td_path = Path(td)
        wav_path = td_path / "audio.wav"
        out_prefix = td_path / "out"
        out_txt = td_path / "out.txt"

        wav_path.write_bytes(wav_bytes)

        whisper_cmd = [
            whisper_bin,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-otxt",
            "-of",
            str(out_prefix),
        ]
        if cfg.language:
            whisper_cmd += ["-l", cfg.language]

        whisper_res = subprocess.run(whisper_cmd, capture_output=True, text=True)
        if whisper_res.returncode != 0:
            raise WhisperCppError(
                "whisper.cpp failed: "
                + (whisper_res.stderr.strip() or whisper_res.stdout.strip() or "unknown error")
            )

        if not out_txt.exists():
            raise WhisperCppError("whisper.cpp did not produce expected .txt output.")

        return out_txt.read_text(encoding="utf-8", errors="ignore").strip()


def pcm16le_to_wav_bytes(pcm16le: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    if not pcm16le:
        raise WhisperCppError("PCM payload is empty.")
    if channels != 1:
        raise WhisperCppError("Only mono PCM is supported in streaming mode.")
    if sample_rate <= 0:
        raise WhisperCppError("Invalid sample_rate.")

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16le)
    return buf.getvalue()


def transcribe_audio_bytes(audio_bytes: bytes, *, config: WhisperCppConfig | None = None) -> str:
    """
    Transcribe an audio blob (e.g. browser-recorded webm/ogg/wav) using:
    - ffmpeg (decode -> 16k mono wav)
    - whisper.cpp (model = ggml-small by default)
    """
    cfg = config or default_whispercpp_config()
    whisper_bin = _require_binary(cfg.whisper_bin)
    ffmpeg_bin = _require_binary(cfg.ffmpeg_bin)

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise WhisperCppError(
            f"Whisper model not found at '{model_path}'. "
            "Download a ggml model (e.g. ggml-small.bin) and set WHISPER_CPP_MODEL."
        )

    with tempfile.TemporaryDirectory(prefix="voice_stt_") as td:
        td_path = Path(td)
        in_path = td_path / "input_audio.bin"
        wav_path = td_path / "audio.wav"
        out_prefix = td_path / "out"
        out_txt = td_path / "out.txt"

        in_path.write_bytes(audio_bytes)

        # Decode to 16kHz mono PCM wav for whisper.cpp
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(wav_path),
        ]
        ffmpeg_res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if ffmpeg_res.returncode != 0 or not wav_path.exists():
            raise WhisperCppError(f"ffmpeg decode failed: {ffmpeg_res.stderr.strip() or ffmpeg_res.stdout.strip()}")

        whisper_cmd = [
            whisper_bin,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-otxt",
            "-of",
            str(out_prefix),
        ]
        if cfg.language:
            whisper_cmd += ["-l", cfg.language]

        whisper_res = subprocess.run(whisper_cmd, capture_output=True, text=True)
        if whisper_res.returncode != 0:
            raise WhisperCppError(
                "whisper.cpp failed: "
                + (whisper_res.stderr.strip() or whisper_res.stdout.strip() or "unknown error")
            )

        if not out_txt.exists():
            raise WhisperCppError("whisper.cpp did not produce expected .txt output.")

        transcript = out_txt.read_text(encoding="utf-8", errors="ignore").strip()
        return transcript

