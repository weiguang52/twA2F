from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionConfig:
    session_id: str
    emotion: str = "neutral"
    intensity: float = 1.0
    output_fps: float = 30.0
    save_npy: bool = True
    npy_save_dir: Optional[str] = None
    request_id: str = ""
    speaker_id: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class AudioChunk:
    session_id: str
    chunk_id: int
    pts_ms: int
    pcm_bytes: bytes
    sample_rate: int
    channels: int = 1
    bits_per_sample: int = 16
    emotion: Optional[str] = None
    intensity: Optional[float] = None
    emotion_meta: Dict[str, str] = field(default_factory=dict)
    emotion_mix: Optional[Dict[str, float]] = None


@dataclass
class MotorFrame:
    frame_id: int
    time_code_ms: int
    motor_values: List[float]
    debug_features: List[float]
    speech_gate: float = 0.


@dataclass
class SessionSummaryData:
    session_id: str
    total_audio_ms: int
    total_frames: int
    actual_output_fps: float
    npy_path: str = ""
    preprocess_ms_avg: float = 0.0
    infer_ms_avg: float = 0.0
    retarget_ms_avg: float = 0.0
