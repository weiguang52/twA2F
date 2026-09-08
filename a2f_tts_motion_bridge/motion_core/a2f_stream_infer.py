import glob
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .settings import EMOTION_LABELS, HOP, NUM_FEATURES, WINDOW


def now() -> float:
    return time.perf_counter()


def find_onnx_model(search_dirs):
    candidates = []
    for base in search_dirs:
        if os.path.exists(base):
            candidates.extend(glob.glob(os.path.join(base, "**", "*.onnx"), recursive=True))
    if not candidates:
        raise FileNotFoundError("No .onnx model found")
    scored = []
    for p in candidates:
        name = os.path.basename(p).lower()
        score = 0
        if "a2f" in name:
            score += 5
        if "face" in name:
            score += 5
        if "network" in name:
            score += 4
        if "regress" in name:
            score += 4
        if "diffusion" in name:
            score -= 2
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def get_providers(ort_module, force_cpu=False):
    avail = ort_module.get_available_providers()
    if force_cpu:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(model_path, force_cpu=False):
    import onnxruntime as ort

    t0 = now()
    providers = get_providers(ort, force_cpu=force_cpu)
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    t1 = now()
    return session, providers, (t1 - t0)


def _safe_dim_to_int(d):
    return d if isinstance(d, int) else None


def _infer_input_array(inp, audio_chunk, target_emotion="neutral", emotion_intensity=1.0):
    name = inp.name.lower()
    shape = inp.shape
    rank = len(shape)
    dims = [_safe_dim_to_int(x) for x in shape]
    batch = audio_chunk.shape[0]

    if (
        "audio" in name or "wav" in name or "input" in name
        or (rank == 3 and (dims[-1] in (WINDOW, None)))
        or (rank == 2 and (dims[-1] in (WINDOW, None)))
    ):
        if rank == 3:
            if dims[1] in (1, None) and dims[2] in (WINDOW, None):
                return audio_chunk.astype(np.float32)
            if dims[0] in (1, None) and dims[2] in (WINDOW, None):
                return np.transpose(audio_chunk, (1, 0, 2)).astype(np.float32)
        if rank == 2:
            return audio_chunk[:, 0, :].astype(np.float32)

    if "emotion" in name or "cond" in name:
        emo_dim = dims[-1] if dims[-1] is not None else len(EMOTION_LABELS)
        emo_array = np.zeros((batch, emo_dim), dtype=np.float32)
        idx = 0
        if target_emotion in EMOTION_LABELS:
            idx = EMOTION_LABELS.index(target_emotion)
        if idx < emo_dim:
            emo_array[:, idx] = float(emotion_intensity)
        if rank == 3:
            emo_array = np.expand_dims(emo_array, axis=1)
        return emo_array

    fixed_dims = []
    for i, d in enumerate(dims):
        if d is None:
            fixed_dims.append(batch if i == 0 else 1)
        else:
            fixed_dims.append(max(1, int(d)))
    return np.zeros(fixed_dims, dtype=np.float32)


def _pick_output(outputs):
    best = None
    best_score = -1
    for out in outputs:
        arr = np.asarray(out)
        score = 0
        if arr.ndim == 3 and arr.shape[-1] == NUM_FEATURES:
            score += 10
        if arr.ndim == 2 and arr.shape[-1] == NUM_FEATURES:
            score += 9
        if arr.ndim >= 2 and NUM_FEATURES in arr.shape:
            score += 5
        if arr.ndim == 3:
            score += 2
        if arr.ndim == 2:
            score += 1
        if score > best_score:
            best_score = score
            best = arr
    if best is None:
        raise RuntimeError("No valid model output")
    return best


def _normalize_output_shape(out_arr, batch_size):
    arr = np.asarray(out_arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == NUM_FEATURES:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[1] == batch_size:
            return np.transpose(arr, (1, 0, 2))
    if arr.ndim == 2 and arr.shape[-1] == NUM_FEATURES:
        return arr[:, None, :]
    if arr.ndim == 1 and arr.shape[0] == NUM_FEATURES:
        return arr[None, None, :]
    if arr.ndim == 3 and arr.shape[1] == NUM_FEATURES and arr.shape[2] == 1:
        return np.transpose(arr, (0, 2, 1))
    raise RuntimeError(f"Cannot normalize output shape {arr.shape} -> [B,1,169]")


def _compute_center_rms(window: np.ndarray, target_sr: int, center_ms: float = 80.0) -> float:
    """
    Short centered RMS aligned to the model time-code.

    Why not use the full model window?
    - the inference window is much longer than the articulatory event we care about
    - gating on the full window increases silence leakage and offset lag
    - centered short RMS better matches the output time_code_s (= window center)
    """
    window = np.asarray(window, dtype=np.float32)
    if len(window) == 0:
        return 0.0

    half = max(1, int(round(0.5 * center_ms * 1e-3 * float(target_sr))))
    center = len(window) // 2
    start = max(0, center - half)
    end = min(len(window), center + half)
    seg = window[start:end]
    if len(seg) == 0:
        seg = window
    return float(np.sqrt(np.mean(np.square(seg)) + 1e-12))


class A2FModelRuntime:
    """Process-wide ONNX model runtime shared by independent stream engines."""

    def __init__(self, model_path: str, force_cpu: bool = False, warmup: int = 1):
        self.model_path = model_path
        self.session, self.providers, self.session_init_sec = make_session(
            model_path,
            force_cpu=force_cpu,
        )
        self.inputs_meta = self.session.get_inputs()
        self.warmup_sec = 0.0
        self.warmup_run_sec: List[float] = []
        if warmup > 0:
            self._warmup(warmup)

    def _warmup(self, warmup: int) -> None:
        dummy = np.zeros((1, 1, WINDOW), dtype=np.float32)
        t0 = now()
        for _ in range(warmup):
            ort_inputs = {
                inp.name: _infer_input_array(inp, dummy, "neutral", 1.0)
                for inp in self.inputs_meta
            }
            run_t0 = now()
            _ = self.session.run(None, ort_inputs)
            self.warmup_run_sec.append(now() - run_t0)
        self.warmup_sec = now() - t0


class StreamingA2FEngine:
    def __init__(
        self,
        model_path: str,
        target_sr: int,
        force_cpu: bool = False,
        warmup: int = 1,
        runtime: Optional[A2FModelRuntime] = None,
    ):
        self.target_sr = int(target_sr)
        self.runtime = runtime or A2FModelRuntime(
            model_path,
            force_cpu=force_cpu,
            warmup=warmup,
        )
        self.session = self.runtime.session
        self.providers = self.runtime.providers
        self.session_init_sec = self.runtime.session_init_sec
        self.inputs_meta = self.runtime.inputs_meta
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start_idx = 0
        self.total_received = 0
        self.next_infer_end_idx = WINDOW
        self.current_emotion = "neutral"
        self.current_intensity = 1.0
        self.warmup_sec = self.runtime.warmup_sec
        self.input_pack_sec = 0.0
        self.pure_infer_sec = 0.0
        self.post_sec = 0.0
        self.infer_frames = 0
        self.first_infer_sec: Optional[float] = None

    def _extract_window(self, end_idx: int) -> np.ndarray:
        local_end = end_idx - self.buffer_start_idx
        local_start = local_end - WINDOW
        if local_start < 0 or local_end > len(self.buffer):
            raise RuntimeError(
                f"Window extraction failed end_idx={end_idx}, buffer_start_idx={self.buffer_start_idx}, len={len(self.buffer)}"
            )
        return self.buffer[local_start:local_end].astype(np.float32)

    def _trim_buffer(self):
        keep_from = max(0, self.next_infer_end_idx - WINDOW)
        drop = keep_from - self.buffer_start_idx
        if drop > 0:
            self.buffer = self.buffer[drop:]
            self.buffer_start_idx = keep_from
        max_keep = WINDOW + 8 * HOP
        if len(self.buffer) > max_keep:
            extra = len(self.buffer) - max_keep
            self.buffer = self.buffer[extra:]
            self.buffer_start_idx += extra

    def push_audio_chunk(self, samples_16k_mono: np.ndarray, emotion: str, intensity: float) -> List[Dict]:
        samples_16k_mono = np.asarray(samples_16k_mono, dtype=np.float32)
        if len(samples_16k_mono) == 0:
            return []
        self.current_emotion = emotion
        self.current_intensity = float(intensity)
        self.buffer = np.concatenate([self.buffer, samples_16k_mono], axis=0)
        self.total_received += len(samples_16k_mono)

        out_items = []
        while self.total_received >= self.next_infer_end_idx:
            win = self._extract_window(self.next_infer_end_idx)
            x = win.reshape(1, 1, WINDOW).astype(np.float32)
            aligned_audio_rms = _compute_center_rms(win, self.target_sr, center_ms=80.0)

            t0 = now()
            ort_inputs = {
                inp.name: _infer_input_array(inp, x, self.current_emotion, self.current_intensity)
                for inp in self.inputs_meta
            }
            t1 = now()
            self.input_pack_sec += (t1 - t0)

            t2 = now()
            out_list = self.session.run(None, ort_inputs)
            t3 = now()
            infer_sec = t3 - t2
            self.pure_infer_sec += infer_sec
            if self.first_infer_sec is None:
                self.first_infer_sec = infer_sec

            t4 = now()
            picked = _pick_output(out_list)
            picked = _normalize_output_shape(picked, batch_size=1)
            weights169 = picked[0, 0].astype(np.float32)
            time_code_s = (self.next_infer_end_idx - WINDOW / 2.0) / float(self.target_sr)
            out_items.append({
                "time_code_s": float(time_code_s),
                "weights169": weights169,
                "audio_rms": float(aligned_audio_rms),
                "emotion": self.current_emotion,
                "intensity": float(self.current_intensity),
            })
            self.infer_frames += 1
            t5 = now()
            self.post_sec += (t5 - t4)
            self.next_infer_end_idx += HOP
        self._trim_buffer()
        return out_items

    def finalize(self) -> List[Dict]:
        outputs = []
        if self.total_received < WINDOW:
            need = WINDOW - self.total_received
        else:
            rem = (self.total_received - WINDOW) % HOP
            need = 0 if rem == 0 else (HOP - rem)
        if need > 0:
            silence = np.zeros(need, dtype=np.float32)
            outputs.extend(self.push_audio_chunk(silence, self.current_emotion, self.current_intensity))
        return outputs
