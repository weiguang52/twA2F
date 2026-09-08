import io
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..motion_core.settings import (
    EXPORT_FPS,
    HOP,
    MOTOR_CFG,
    NUM_FEATURES,
    TARGET_SR,
    WINDOW,
)
from ..motion_core.motion_recorder import SessionRecorder
from .segment_timeline import build_segment_times, resample_to_times

if TYPE_CHECKING:
    from ..motion_core.motion_session import MotorStreamSession


@dataclass
class ActionSegment:
    session_id: str
    segment_index: int
    start_ms: int
    end_ms: int
    is_final: bool
    frame_count: int
    npy_bytes: bytes
    build_ms: float = 0.0


_EPS = 1e-6


def _recorder_array(
    items: List[Any],
    frame_count: int,
    *,
    width: Optional[int] = None,
) -> np.ndarray:
    if not items:
        if frame_count > 0 and width != 0:
            raise ValueError(
                f"recorder series is empty for {frame_count} native frames"
            )
        shape = (frame_count, width) if width is not None else (frame_count,)
        return np.zeros(shape, dtype=np.float32)

    arr = np.asarray(items, dtype=np.float32)
    if width is None:
        arr = arr.reshape(-1)
    else:
        arr = arr.reshape(-1, width)
    if len(arr) != frame_count:
        raise ValueError(
            "recorder series length mismatch: "
            f"expected={frame_count} actual={len(arr)}"
        )
    return arr


def _validate_export_lengths(payload: Dict[str, Any], expected: int) -> None:
    keys = (
        "frame_times_30fps",
        "weights_30fps",
        "weights_30fps_3d",
        "features_30fps",
        "arkit52_30fps",
        "motor_values_30fps",
        "audio_rms_30fps",
        "speech_gate_30fps",
    )
    mismatched = {
        key: len(payload[key])
        for key in keys
        if len(payload[key]) != expected
    }
    if mismatched:
        raise ValueError(
            f"segment export length mismatch: expected={expected} actual={mismatched}"
        )


def build_payload_from_recorder(
    recorder: SessionRecorder,
    start_ms: int,
    end_ms: int,
    *,
    session_id: str,
    segment_index: int,
    emotion: str,
    intensity: float,
    is_final: bool,
    export_fps: float = EXPORT_FPS,
) -> Tuple[Dict[str, Any], int]:
    start_s = max(0.0, float(start_ms) / 1000.0)
    end_s = max(start_s, float(end_ms) / 1000.0)

    frame_times_all = np.asarray(recorder.frame_times, dtype=np.float32)
    native_count = len(frame_times_all)
    weights_all = _recorder_array(recorder.weights, native_count, width=NUM_FEATURES)
    features_all = _recorder_array(recorder.features, native_count, width=7)
    motor_width = len(recorder.motor_names)
    motor_values_all = _recorder_array(
        recorder.motor_values, native_count, width=motor_width
    )
    audio_rms_all = _recorder_array(recorder.audio_rms, native_count)
    speech_gate_all = _recorder_array(recorder.speech_gate, native_count)
    if recorder.arkit52_names:
        arkit52_all = _recorder_array(
            recorder.arkit52,
            native_count,
            width=len(recorder.arkit52_names),
        )
    else:
        arkit52_all = np.zeros((native_count, 0), dtype=np.float32)

    if len(frame_times_all) > 0:
        if is_final:
            mask = (frame_times_all >= start_s - _EPS) & (frame_times_all <= end_s + _EPS)
        else:
            mask = (frame_times_all >= start_s - _EPS) & (frame_times_all < end_s - _EPS)
    else:
        mask = np.zeros(0, dtype=bool)

    frame_times = frame_times_all[mask] if len(frame_times_all) > 0 else np.zeros(0, dtype=np.float32)
    weights = weights_all[mask]
    features = features_all[mask]
    motor_values = motor_values_all[mask]
    audio_rms = audio_rms_all[mask]
    speech_gate = speech_gate_all[mask]
    arkit52 = arkit52_all[mask]

    frame_times_30 = build_segment_times(start_ms, end_ms, export_fps)
    weights_30 = resample_to_times(
        frame_times_all,
        weights_all,
        frame_times_30,
        empty_value=np.zeros(NUM_FEATURES, dtype=np.float32),
    )
    features_30 = resample_to_times(
        frame_times_all,
        features_all,
        frame_times_30,
        empty_value=np.zeros(7, dtype=np.float32),
    )
    motor_neutral = np.asarray(
        [MOTOR_CFG[name]["neutral"] for name in recorder.motor_names],
        dtype=np.float32,
    )
    motor_values_30 = resample_to_times(
        frame_times_all,
        motor_values_all,
        frame_times_30,
        empty_value=motor_neutral,
    )
    audio_rms_30 = resample_to_times(
        frame_times_all,
        audio_rms_all,
        frame_times_30,
        empty_value=0.0,
    )
    speech_gate_30 = resample_to_times(
        frame_times_all,
        speech_gate_all,
        frame_times_30,
        empty_value=0.0,
    )
    arkit52_30 = resample_to_times(
        frame_times_all,
        arkit52_all,
        frame_times_30,
        empty_value=np.zeros(len(recorder.arkit52_names), dtype=np.float32),
    )

    audio_all = np.concatenate(recorder.audio_16k_all, axis=0) if recorder.audio_16k_all else np.zeros(0, dtype=np.float32)
    audio_start = max(0, int(round(start_s * TARGET_SR)))
    audio_end = max(audio_start, int(round(end_s * TARGET_SR)))
    audio_16k = audio_all[audio_start:audio_end].astype(np.float32)

    meta: Dict[str, Any] = {
        "target_sr": TARGET_SR,
        "window": WINDOW,
        "hop": HOP,
        "num_features": NUM_FEATURES,
        "native_fps_est": float(TARGET_SR / float(HOP)),
        "export_fps": float(export_fps),
        "feature_names": [
            "jaw_open",
            "mouth_round",
            "mouth_wide",
            "mouth_left_right",
            "upper_face_activity",
            "eye_activity",
            "blink_like",
        ],
        "motor_names": recorder.motor_names,
        "arkit52_names": recorder.arkit52_names,
        "session_id": session_id,
        "segment_index": int(segment_index),
        "segment_start_ms": int(start_ms),
        "segment_end_ms": int(end_ms),
        "segment_duration_ms": int(max(0, end_ms - start_ms)),
        "segment_is_final": bool(is_final),
        "emotion": emotion,
        "intensity": float(intensity),
    }
    meta.update(recorder.meta_extra)

    payload = {
        "meta": meta,
        "frame_times_native": frame_times,
        "weights_native": weights,
        "weights_native_3d": weights[:, None, :] if len(weights) > 0 else np.zeros((0, 1, NUM_FEATURES), dtype=np.float32),
        "features_native": features,
        "arkit52_native": arkit52,
        "motor_values_native": motor_values,
        "audio_rms_native": audio_rms,
        "speech_gate_native": speech_gate,
        "emotion_vectors_native": np.asarray(recorder.emotion_vectors, dtype=np.float32).reshape(-1, 26)[mask],
        "emotion_bias_scales_native": np.asarray(recorder.emotion_bias_scales, dtype=np.float32)[mask],
        "frame_times_30fps": frame_times_30,
        "weights_30fps": weights_30,
        "weights_30fps_3d": weights_30[:, None, :] if len(weights_30) > 0 else np.zeros((0, 1, NUM_FEATURES), dtype=np.float32),
        "features_30fps": features_30,
        "arkit52_30fps": arkit52_30,
        "motor_values_30fps": motor_values_30,
        "audio_rms_30fps": audio_rms_30,
        "speech_gate_30fps": speech_gate_30,
        "audio_16k_mono": audio_16k,
    }
    frame_count = int(len(frame_times_30))
    _validate_export_lengths(payload, frame_count)
    return payload, frame_count


def payload_to_npy_bytes(payload: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    np.save(buf, payload, allow_pickle=True)
    return buf.getvalue()


class RollingNpySegmentExporter:
    def __init__(
        self,
        session: "MotorStreamSession",
        *,
        segment_ms: int = 640,
        first_segment_ms: Optional[int] = None,
        emit_cover_slack_ms: int = 80,
    ):
        self.session = session
        self.segment_ms = int(segment_ms)
        if self.segment_ms <= 0:
            raise ValueError(f"segment_ms must be positive, got {segment_ms}")
        self.first_segment_ms = int(first_segment_ms) if first_segment_ms is not None else self.segment_ms
        if self.first_segment_ms <= 0:
            raise ValueError(f"first_segment_ms must be positive, got {first_segment_ms}")
        self.emit_cover_slack_ms = int(emit_cover_slack_ms)
        self.next_segment_start_ms = 0
        self.next_segment_end_ms = self.first_segment_ms
        self._first_segment_emitted = False
        self.segment_index = 0
        self.current_emotion = session.config.emotion
        self.current_intensity = float(session.config.intensity)

    def update_emotion(self, emotion: Optional[str], intensity: Optional[float]) -> None:
        if emotion:
            self.current_emotion = emotion
        if intensity is not None:
            self.current_intensity = float(intensity)

    def _latest_frame_ms(self) -> int:
        if not self.session.recorder.frame_times:
            return -1
        return int(round(float(self.session.recorder.frame_times[-1]) * 1000.0))

    def _build_segment(self, start_ms: int, end_ms: int, is_final: bool) -> ActionSegment:
        build_started = time.perf_counter()
        payload, frame_count = build_payload_from_recorder(
            self.session.recorder,
            start_ms,
            end_ms,
            session_id=self.session.config.session_id,
            segment_index=self.segment_index,
            emotion=self.current_emotion,
            intensity=self.current_intensity,
            is_final=is_final,
            export_fps=self.session.config.output_fps or EXPORT_FPS,
        )
        return ActionSegment(
            session_id=self.session.config.session_id,
            segment_index=self.segment_index,
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            is_final=bool(is_final),
            frame_count=frame_count,
            npy_bytes=payload_to_npy_bytes(payload),
            build_ms=(time.perf_counter() - build_started) * 1000.0,
        )

    def _advance_segment_window(self) -> None:
        """Advance after one emitted segment without leaving a startup gap.

        A short first packet is only an early part of the normal first
        ``segment_ms`` interval.  Complete that interval next (for example,
        ``0-320``, then ``320-640``) before returning to the regular 640 ms
        cadence.  Advancing directly by ``segment_ms`` would incorrectly make
        the next window ``320-960``.
        """

        self.segment_index += 1
        self.next_segment_start_ms = self.next_segment_end_ms
        if (
            not self._first_segment_emitted
            and self.next_segment_start_ms < self.segment_ms
        ):
            self.next_segment_end_ms = self.segment_ms
        else:
            self.next_segment_end_ms = self.next_segment_start_ms + self.segment_ms
        self._first_segment_emitted = True

    def collect_ready_segments(self, audio_received_ms: int) -> List[ActionSegment]:
        ready: List[ActionSegment] = []
        latest_frame_ms = self._latest_frame_ms()
        # Keep the segment ending exactly at the received-audio boundary until
        # EOS.  Otherwise it can be emitted as non-final and force a synthetic
        # zero-duration final marker later.
        while self.next_segment_end_ms < audio_received_ms:
            if latest_frame_ms + self.emit_cover_slack_ms < self.next_segment_end_ms:
                break
            ready.append(self._build_segment(self.next_segment_start_ms, self.next_segment_end_ms, is_final=False))
            self._advance_segment_window()
        return ready

    def flush_final_segments(self, final_audio_ms: int) -> List[ActionSegment]:
        ready: List[ActionSegment] = []
        while self.next_segment_end_ms < final_audio_ms:
            ready.append(self._build_segment(self.next_segment_start_ms, self.next_segment_end_ms, is_final=False))
            self._advance_segment_window()
        if final_audio_ms > self.next_segment_start_ms:
            ready.append(self._build_segment(self.next_segment_start_ms, final_audio_ms, is_final=True))
            self.segment_index += 1
            self.next_segment_start_ms = final_audio_ms
            self.next_segment_end_ms = self.next_segment_start_ms + self.segment_ms
        return ready
