from typing import Iterator
from dataclasses import dataclass
import random

import numpy as np
import soundfile as sf

from .types import AudioChunk


@dataclass
class MockStreamConfig:
    emotion: str = "neutral"
    intensity: float = 1.0
    chunk_sec_min: float = 0.10
    chunk_sec_max: float = 0.30
    seed: int = 1234


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    x /= 32768.0
    return x


def float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def interleaved_to_mono(x: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return x.astype(np.float32)
    if len(x) % channels != 0:
        valid = (len(x) // channels) * channels
        x = x[:valid]
    x = x.reshape(-1, channels)
    return x.mean(axis=1).astype(np.float32)


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    if src_sr == dst_sr:
        return x.copy()
    src_len = len(x)
    dst_len = int(round(src_len * float(dst_sr) / float(src_sr)))
    if dst_len <= 1:
        return np.zeros(max(1, dst_len), dtype=np.float32)
    src_idx = np.linspace(0.0, src_len - 1.0, num=src_len, dtype=np.float32)
    dst_idx = np.linspace(0.0, src_len - 1.0, num=dst_len, dtype=np.float32)
    y = np.interp(dst_idx, src_idx, x).astype(np.float32)
    return y


def preprocess_stream_chunk_to_16k_mono(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
    target_sr: int,
    bits_per_sample: int = 16,
) -> np.ndarray:
    if bits_per_sample != 16:
        raise ValueError(f"Only PCM16 is supported in this demo, got {bits_per_sample}")
    x = pcm16_bytes_to_float32(pcm_bytes)
    x = interleaved_to_mono(x, channels=channels)
    x = resample_linear(x, src_sr=sample_rate, dst_sr=target_sr)
    return x.astype(np.float32)


def iter_mock_tts_stream_from_wav(
    wav_path: str,
    session_id: str,
    sample_rate_override: int = 0,
    config: MockStreamConfig = MockStreamConfig(),
) -> Iterator[AudioChunk]:
    rng = random.Random(config.seed)
    wav, sr = sf.read(wav_path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 1:
        channels = 1
        base = wav
        total_frames = len(wav)
    else:
        channels = wav.shape[1]
        base = wav
        total_frames = wav.shape[0]

    source_sr = sample_rate_override or sr
    start_frame = 0
    chunk_id = 0
    while start_frame < total_frames:
        chunk_sec = rng.uniform(config.chunk_sec_min, config.chunk_sec_max)
        chunk_n = max(1, int(round(chunk_sec * source_sr)))
        end_frame = min(total_frames, start_frame + chunk_n)
        pts_ms = int(round(start_frame * 1000.0 / source_sr))
        if channels == 1:
            chunk_f32 = base[start_frame:end_frame]
        else:
            chunk_f32 = base[start_frame:end_frame, :].reshape(-1)
        yield AudioChunk(
            session_id=session_id,
            chunk_id=chunk_id,
            pts_ms=pts_ms,
            pcm_bytes=float32_to_pcm16_bytes(chunk_f32),
            sample_rate=source_sr,
            channels=channels,
            bits_per_sample=16,
            emotion=config.emotion,
            intensity=config.intensity,
        )
        chunk_id += 1
        start_frame = end_frame
