from __future__ import annotations

import asyncio
import base64
import os
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `numpy`. Install dependencies in your active venv:\n"
        "  pip install -r requirements.txt"
    ) from e

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `python-dotenv`. Install dependencies in your active venv:\n"
        "  pip install -r requirements.txt"
    ) from e
try:
    # `livekit` is provided by `pip install livekit`
    from livekit import rtc  # type: ignore[import-not-found]
    from livekit import api as lkapi  # type: ignore[import-not-found]
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "LiveKit SDK not found. Install dependencies in your active venv:\n"
        "  pip install -r requirements.txt\n"
        "If you're in Cursor/VSCode, also ensure the interpreter is set to your `.venv`."
    ) from e

from agent_logic import build_agent_reply
from ExpertServiceSearch import build_search_engine
from stt_whispercpp import pcm16le_to_wav_bytes, transcribe_wav_bytes
from tts_piper import synthesize_wav_bytes


@dataclass(frozen=True)
class AgentConfig:
    room_name: str
    identity: str = "expert-service-agent"
    # LiveKit connection
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_token: str | None = None
    # Audio / VAD
    input_sample_rate: int = 48000  # LiveKit audio frames are typically 48kHz
    target_sample_rate: int = 16000
    window_sec: float = 4.0
    step_sec: float = 0.5
    min_rms_int16: float = 350.0
    silence_finalize_sec: float = 1.2
    # Reply publishing
    out_sample_rate: int = 48000
    out_channels: int = 1
    out_frame_ms: int = 20


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resample_int16_mono(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """
    Simple resampler for int16 mono PCM. Good enough for voice.
    Prefer exact decimation when possible (48k -> 16k).
    """
    if in_sr == out_sr:
        return x.astype(np.int16, copy=False)
    if in_sr == 48000 and out_sr == 16000:
        # Decimate by 3 with simple averaging to reduce aliasing.
        n = (len(x) // 3) * 3
        if n <= 0:
            return np.zeros((0,), dtype=np.int16)
        y = x[:n].reshape(-1, 3).mean(axis=1)
        return np.clip(y, -32768, 32767).astype(np.int16)

    # Generic linear interpolation
    ratio = out_sr / in_sr
    out_len = int(len(x) * ratio)
    if out_len <= 0:
        return np.zeros((0,), dtype=np.int16)
    idx = np.linspace(0, len(x) - 1, out_len)
    y = np.interp(idx, np.arange(len(x)), x.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)


def _wav_bytes_to_int16_mono(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2:
            raise RuntimeError(f"Unsupported sample width: {sw}")
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch)[:, 0]
    return audio, sr


def _rms_int16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((x.astype(np.float32) ** 2))))


def _make_token(cfg: AgentConfig) -> str:
    if cfg.livekit_token:
        return cfg.livekit_token
    if not (cfg.livekit_api_key and cfg.livekit_api_secret):
        raise RuntimeError("Provide LIVEKIT_TOKEN or LIVEKIT_API_KEY + LIVEKIT_API_SECRET.")
    t = lkapi.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
    t.with_identity(cfg.identity).with_name(cfg.identity).with_grants(
        lkapi.VideoGrants(room_join=True, room=cfg.room_name)
    )
    return t.to_jwt()


async def _publish_wav_as_audio_track(room: rtc.Room, wav_bytes: bytes, *, cfg: AgentConfig) -> None:
    audio_i16, sr = _wav_bytes_to_int16_mono(wav_bytes)
    audio_i16 = _resample_int16_mono(audio_i16, sr, cfg.out_sample_rate)

    source = rtc.AudioSource(cfg.out_sample_rate, cfg.out_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-tts", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    samples_per_frame = int(cfg.out_sample_rate * (cfg.out_frame_ms / 1000.0))
    if samples_per_frame <= 0:
        samples_per_frame = 960

    # Stream frames
    idx = 0
    while idx < len(audio_i16):
        chunk = audio_i16[idx : idx + samples_per_frame]
        if chunk.size < samples_per_frame:
            pad = np.zeros((samples_per_frame - chunk.size,), dtype=np.int16)
            chunk = np.concatenate([chunk, pad])
        frame = rtc.AudioFrame.create(cfg.out_sample_rate, cfg.out_channels, samples_per_frame)
        # frame.data expects bytes
        frame.data[:] = chunk.tobytes()
        await source.capture_frame(frame)
        idx += samples_per_frame
        await asyncio.sleep(cfg.out_frame_ms / 1000.0)


async def run_agent(cfg: AgentConfig, *, engine) -> None:
    token = _make_token(cfg)
    room = rtc.Room()
    print("Connecting to LiveKit…", flush=True)

    async def handle_track_subscribed(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity == cfg.identity:
            return

        # Stream audio frames from remote participant
        stream = rtc.AudioStream(track, sample_rate=cfg.input_sample_rate, num_channels=1)

        rolling: list[np.ndarray] = []
        rolling_samples_max = int(cfg.input_sample_rate * cfg.window_sec)
        step_samples = int(cfg.input_sample_rate * cfg.step_sec)
        buf = np.zeros((0,), dtype=np.int16)

        last_voice_ts = time.time()
        in_turn = False

        async for ev in stream:
            pcm = np.frombuffer(ev.frame.data, dtype=np.int16)
            if pcm.size == 0:
                continue

            buf = np.concatenate([buf, pcm])
            while buf.size >= step_samples:
                step = buf[:step_samples]
                buf = buf[step_samples:]

                rolling.append(step)
                total = sum(x.size for x in rolling)
                while total > rolling_samples_max and rolling:
                    total -= rolling[0].size
                    rolling.pop(0)

                win = np.concatenate(rolling) if rolling else np.zeros((0,), dtype=np.int16)
                # VAD
                if _rms_int16(win) >= cfg.min_rms_int16:
                    last_voice_ts = time.time()
                    in_turn = True
                else:
                    if in_turn and (time.time() - last_voice_ts) >= cfg.silence_finalize_sec:
                        # finalize turn
                        in_turn = False
                        # Transcribe rolling window
                        win16 = _resample_int16_mono(win, cfg.input_sample_rate, cfg.target_sample_rate)
                        wav = pcm16le_to_wav_bytes(win16.tobytes(), sample_rate=cfg.target_sample_rate, channels=1)
                        text = transcribe_wav_bytes(wav).strip()

                        agent = build_agent_reply(text=text, engine=engine)
                        reply_wav = synthesize_wav_bytes(agent.reply_text)
                        await _publish_wav_as_audio_track(room, reply_wav, cfg=cfg)
                    continue

                # Optional: you can send partial transcripts here, but keep it fast.

    # Attach handler
    def on_track_subscribed(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        # LiveKit requires sync callbacks; spawn async work.
        asyncio.create_task(handle_track_subscribed(track, publication, participant))

    room.on("track_subscribed", on_track_subscribed)

    # Avoid silent hangs; surface connect issues quickly.
    await asyncio.wait_for(room.connect(cfg.livekit_url, token), timeout=20)
    print(f"Connected to LiveKit room={cfg.room_name} as {cfg.identity}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    # Load env
    load_dotenv()

    # Build the BM25 search engine from your JSON file.
    json_path = _env("EXPERT_SERVICES_JSON", "ExpertServices.json")
    if not os.path.exists(json_path):
        raise SystemExit(f"Expert services JSON not found at '{json_path}'.")
    engine = build_search_engine(json_path)

    cfg = AgentConfig(
        room_name=_env("LIVEKIT_ROOM", "demo"),
        identity=_env("LIVEKIT_IDENTITY", "expert-service-agent"),
        livekit_url=_env("LIVEKIT_URL"),
        livekit_api_key=_env("LIVEKIT_API_KEY"),
        livekit_api_secret=_env("LIVEKIT_API_SECRET"),
        livekit_token=_env("LIVEKIT_TOKEN") or None,
    )
    asyncio.run(run_agent(cfg, engine=engine))

