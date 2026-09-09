import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .settings import EXPORT_FPS, MOTOR_CFG, NUM_FEATURES, WINDOW, HOP, TARGET_SR


class SessionRecorder:
    def __init__(self):
        self.frame_times = []
        self.weights = []
        self.features = []
        self.arkit52 = []
        self.arkit52_names = []
        self.motor_names = list(MOTOR_CFG.keys())
        self.motor_values = []
        self.base_motor_values = []
        self.behavior_layer = None
        self.audio_16k_all = []
        self.audio_rms = []
        self.speech_gate = []
        self.emotion_vectors = []
        self.emotion_bias_scales = []
        self.meta_extra: Dict[str, object] = {}

    def set_meta_extra(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.meta_extra[k] = v

    def append_audio_16k(self, x16k: np.ndarray):
        if len(x16k) > 0:
            self.audio_16k_all.append(np.asarray(x16k, dtype=np.float32))

    def append_frame(
        self,
        time_code_s: float,
        weights169: np.ndarray,
        feats: Dict[str, float],
        motors: Dict[str, float],
        arkit52: Optional[Dict[str, float]] = None,
        audio_rms: Optional[float] = None,
        debug_extra: Optional[Dict[str, float]] = None,
        base_motors: Optional[Dict[str, float]] = None,
    ):
        self.frame_times.append(float(time_code_s))
        self.weights.append(np.asarray(weights169, dtype=np.float32))
        self.features.append([
            float(feats["jaw_open"]),
            float(feats["mouth_round"]),
            float(feats["mouth_wide"]),
            float(feats["mouth_left_right"]),
            float(feats["upper_face_activity"]),
            float(feats["eye_activity"]),
            float(feats["blink_like"]),
        ])
        self.motor_values.append([float(motors[k]) for k in self.motor_names])
        base_motors = base_motors if isinstance(base_motors, dict) else motors
        self.base_motor_values.append([float(base_motors[k]) for k in self.motor_names])
        self.audio_rms.append(float(audio_rms) if audio_rms is not None else 0.0)
        debug_extra = debug_extra or {}
        for key in ('calibration_sha256', 'robot_id'):
            if isinstance(debug_extra.get(key), str) and debug_extra[key]:
                self.meta_extra[key] = debug_extra[key]
        self.emotion_vectors.append(debug_extra.get('emotion_vector', [0.0] * 26))
        self.emotion_bias_scales.append(float(debug_extra.get('emotion_bias_scale', 0.0)))
        self.speech_gate.append(float(debug_extra.get("speech_gate", 0.0)))
        if arkit52 is not None:
            if not self.arkit52_names:
                self.arkit52_names = list(arkit52.keys())
            self.arkit52.append([float(arkit52[k]) for k in self.arkit52_names])
        elif self.arkit52_names:
            self.arkit52.append([0.0 for _ in self.arkit52_names])

    @staticmethod
    def _interp_to_fps(times: np.ndarray, values: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
        if len(times) == 0:
            tail_shape = values.shape[1:] if hasattr(values, "shape") and values.ndim >= 1 else ()
            return np.zeros(0, dtype=np.float32), np.zeros((0,) + tail_shape, dtype=np.float32)
        if len(times) == 1:
            return times.astype(np.float32), values.astype(np.float32)
        t0 = float(times[0])
        t1 = float(times[-1])
        dt = 1.0 / float(fps)
        new_times = np.arange(t0, t1 + 1e-6, dt, dtype=np.float32)
        if values.ndim == 1:
            new_values = np.interp(new_times, times, values).astype(np.float32)
            return new_times, new_values
        out = np.zeros((len(new_times), values.shape[1]), dtype=np.float32)
        for i in range(values.shape[1]):
            out[:, i] = np.interp(new_times, times, values[:, i]).astype(np.float32)
        return new_times, out

    def save(self, out_npy: str) -> str:
        os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
        frame_times = np.asarray(self.frame_times, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        features = np.asarray(self.features, dtype=np.float32)
        motor_values = np.asarray(self.motor_values, dtype=np.float32)
        base_motor_values = np.asarray(self.base_motor_values, dtype=np.float32)
        if self.arkit52_names and self.arkit52:
            arkit52 = np.asarray(self.arkit52, dtype=np.float32)
        else:
            arkit52 = np.zeros((len(frame_times), 0), dtype=np.float32)
        audio_16k = np.concatenate(self.audio_16k_all, axis=0) if self.audio_16k_all else np.zeros(0, dtype=np.float32)
        audio_rms = np.asarray(self.audio_rms, dtype=np.float32)
        speech_gate = np.asarray(self.speech_gate, dtype=np.float32)

        if len(frame_times) >= 2:
            frame_times_30, weights_30 = self._interp_to_fps(frame_times, weights, EXPORT_FPS)
            _, features_30 = self._interp_to_fps(frame_times, features, EXPORT_FPS)
            _, motor_values_30 = self._interp_to_fps(frame_times, motor_values, EXPORT_FPS)
            _, audio_rms_30 = self._interp_to_fps(frame_times, audio_rms, EXPORT_FPS)
            _, speech_gate_30 = self._interp_to_fps(frame_times, speech_gate, EXPORT_FPS)
            if arkit52.shape[1] > 0:
                _, arkit52_30 = self._interp_to_fps(frame_times, arkit52, EXPORT_FPS)
            else:
                arkit52_30 = np.zeros((len(frame_times_30), 0), dtype=np.float32)
        else:
            frame_times_30 = frame_times.copy()
            weights_30 = weights.copy()
            features_30 = features.copy()
            motor_values_30 = motor_values.copy()
            audio_rms_30 = audio_rms.copy()
            speech_gate_30 = speech_gate.copy()
            arkit52_30 = arkit52.copy()

        meta = {
            **(self.behavior_layer.metadata() if self.behavior_layer is not None else {}),
            "target_sr": TARGET_SR,
            "window": WINDOW,
            "hop": HOP,
            "num_features": NUM_FEATURES,
            "native_fps_est": float(TARGET_SR / float(HOP)),
            "export_fps": EXPORT_FPS,
            "feature_names": [
                "jaw_open", "mouth_round", "mouth_wide", "mouth_left_right",
                "upper_face_activity", "eye_activity", "blink_like"
            ],
            "motor_names": self.motor_names,
            "arkit52_names": self.arkit52_names,
        }
        meta.update(self.meta_extra)
        if self.behavior_layer is not None:
            _, base_30 = self._interp_to_fps(frame_times, base_motor_values, EXPORT_FPS)
            behavior_speech = np.maximum(speech_gate_30, np.clip(audio_rms_30/.035,0,1))
            motor_values_30 = self.behavior_layer.apply_array(frame_times_30, base_30, behavior_speech, self.motor_names)
            if len(features_30): features_30[:,0] = motor_values_30[:,-1]

        payload = {
            "meta": meta,
            "frame_times_native": frame_times,
            "weights_native": weights,
            "weights_native_3d": weights[:, None, :] if len(weights) > 0 else np.zeros((0, 1, NUM_FEATURES), dtype=np.float32),
            "features_native": features,
            "arkit52_native": arkit52,
            "motor_values_native": motor_values,
            "motor_base_values_native": base_motor_values,
            "audio_rms_native": audio_rms,
            "speech_gate_native": speech_gate,
            "emotion_vectors_native": np.asarray(self.emotion_vectors, dtype=np.float32).reshape(-1, 26),
            "emotion_bias_scales_native": np.asarray(self.emotion_bias_scales, dtype=np.float32),
            "frame_times_30fps": frame_times_30,
            "weights_30fps": weights_30,
            "weights_30fps_3d": weights_30[:, None, :] if len(weights_30) > 0 else np.zeros((0, 1, NUM_FEATURES), dtype=np.float32),
            "features_30fps": features_30,
            "arkit52_30fps": arkit52_30,
            "motor_values_30fps": motor_values_30,
            "audio_rms_30fps": audio_rms_30,
            "speech_gate_30fps": speech_gate_30,
            "audio_16k_mono": audio_16k.astype(np.float32),
        }
        np.save(out_npy, payload, allow_pickle=True)
        return out_npy
