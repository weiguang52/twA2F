# -*- coding: utf-8 -*-
"""
A2F raw-169 -> ARKit 52 blendshapes.

Patched v4: jaw tail preservation by decoupling jaw normalization from speech-gate dropouts.

Integration-oriented hybrid mapper:
    169 raw -> official geometry solve + semantic fallback -> ARKit 52

Why hybrid:
- The model bundle gives us official skin decoder and official ARKit basis files.
- In practice, a pure online solve from the exposed bundle can be numerically very weak
  without the rest of NVIDIA's internal runtime details.
- This mapper therefore keeps the official path, but blends it with a robust semantic
  fallback derived from the raw 169-D streams. The project architecture still becomes:

      raw-169 -> ARKit 52 -> robot motors

This is designed for engineering use:
- downstream projects can consume ARKit 52 directly
- future model swaps only need to update this file
- robot retargeting stays isolated in a second file
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ARKIT_52_NAMES: List[str] = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
    "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight", "jawForward",
    "jawLeft", "jawRight", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "browDownLeft",
    "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _clip(x, lo, hi):
    return max(lo, min(hi, float(x)))


class AdaptiveRange01:
    def __init__(
        self,
        init_span: float = 0.15,
        silence_center_alpha: float = 0.18,
        active_center_alpha: float = 0.02,
        peak_rise_alpha: float = 0.18,
        peak_decay: float = 0.004,
        eps: float = 1e-6,
    ):
        self.center: Optional[float] = None
        self.peak: Optional[float] = None
        self.init_span = float(init_span)
        self.silence_center_alpha = float(silence_center_alpha)
        self.active_center_alpha = float(active_center_alpha)
        self.peak_rise_alpha = float(peak_rise_alpha)
        self.peak_decay = float(peak_decay)
        self.eps = float(eps)

    def update(self, x: float, silence_like: bool = False) -> float:
        x = float(x)
        if not math.isfinite(x):
            x = 0.0
        if self.center is None:
            self.center = x
            self.peak = x + self.init_span
            return 0.0
        a = self.silence_center_alpha if silence_like else self.active_center_alpha
        self.center = (1.0 - a) * self.center + a * x
        assert self.peak is not None
        if x >= self.peak:
            self.peak = (1.0 - self.peak_rise_alpha) * self.peak + self.peak_rise_alpha * x
        else:
            self.peak = (1.0 - self.peak_decay) * self.peak + self.peak_decay * x
        self.peak = max(self.peak, self.center + max(self.init_span * 0.5, self.eps))
        y = (x - self.center) / max(self.peak - self.center, self.eps)
        return float(np.clip(y, 0.0, 1.0))


class SignedAdaptiveRange:
    def __init__(self, peak_decay: float = 0.01, min_peak: float = 0.10):
        self.peak = float(min_peak)
        self.peak_decay = float(peak_decay)
        self.min_peak = float(min_peak)

    def update(self, x: float) -> float:
        x = float(x)
        ax = abs(x)
        if ax > self.peak:
            self.peak = 0.85 * self.peak + 0.15 * ax
        else:
            self.peak = (1.0 - self.peak_decay) * self.peak + self.peak_decay * ax
        self.peak = max(self.peak, self.min_peak)
        return float(np.clip(x / self.peak, -1.0, 1.0))


class SmallEMA:
    def __init__(self):
        self.state: Dict[str, float] = {}

    def update(self, name: str, x: float, rise_alpha: float, fall_alpha: float) -> float:
        x = float(x)
        if name not in self.state:
            self.state[name] = x
            return x
        prev = self.state[name]
        a = rise_alpha if x >= prev else fall_alpha
        y = (1.0 - a) * prev + a * x
        self.state[name] = y
        return y


class AudioGate:
    def __init__(self, on_th: float = 0.16, off_th: float = 0.06):
        self.norm = AdaptiveRange01(init_span=0.04, silence_center_alpha=0.20, active_center_alpha=0.02,
                                    peak_rise_alpha=0.22, peak_decay=0.01)
        self.on_th = float(on_th)
        self.off_th = float(off_th)
        self.state = 0.0

    def update(self, audio_rms: Optional[float], fallback: float = 0.0) -> float:
        if audio_rms is None:
            e = float(np.clip(fallback, 0.0, 1.0))
        else:
            e = self.norm.update(float(audio_rms), silence_like=self.state < 0.10)
        if e >= self.on_th:
            target = np.clip((e - self.off_th) / max(self.on_th - self.off_th, 1e-6), 0.0, 1.0)
        elif e >= self.off_th:
            target = 0.25 * np.clip((e - self.off_th) / max(self.on_th - self.off_th, 1e-6), 0.0, 1.0)
        else:
            target = 0.0
        a = 0.40 if target >= self.state else 0.85
        self.state = float((1.0 - a) * self.state + a * target)
        return self.state


class A2F169ToARKit52:
    _bundle_cache: Dict[str, Dict[str, Any]] = {}
    _bundle_cache_lock = threading.Lock()

    def __init__(self, model_dir: Optional[str | os.PathLike[str]] = None, calib_frames: int = 20):
        self.model_dir = self._find_model_dir(model_dir)
        self._load_bundle(self.model_dir)
        self.calib_frames = max(1, int(calib_frames))

        self.audio_gate = AudioGate()
        self.ema = SmallEMA()

        self.jaw_norm = AdaptiveRange01(
            init_span=0.08,
            silence_center_alpha=0.10,
            active_center_alpha=0.008,
            peak_rise_alpha=0.16,
            peak_decay=0.018,
        )
        self.round_norm = AdaptiveRange01(init_span=0.08, silence_center_alpha=0.18, active_center_alpha=0.03)
        self.wide_norm = AdaptiveRange01(init_span=0.08, silence_center_alpha=0.12, active_center_alpha=0.03)
        self.upper_norm = AdaptiveRange01(init_span=0.10, silence_center_alpha=0.10, active_center_alpha=0.03)
        self.blink_norm = AdaptiveRange01(init_span=0.08, silence_center_alpha=0.18, active_center_alpha=0.03)
        self.eye_norm = AdaptiveRange01(init_span=0.08, silence_center_alpha=0.12, active_center_alpha=0.03)
        self.lr_norm = SignedAdaptiveRange(min_peak=0.08)

        self.jaw_baseline: Optional[np.ndarray] = None
        self.eye_baseline: Optional[np.ndarray] = None
        self.jaw_mag_ema = 0.0
        self.prev_active = np.zeros(len(self.active_indices), dtype=np.float32)
        self.last_bs = {name: 0.0 for name in ARKIT_52_NAMES}

        self._jaw_hist: List[np.ndarray] = []
        self._skin_hist: List[np.ndarray] = []
        self._eyes_hist: List[np.ndarray] = []
        self._calibrated = False
        self.jaw_idx = np.array([13, 10, 11, 7], dtype=np.int64)
        self.skin_idx = np.array([0, 1, 6, 3, 2, 4, 5, 9], dtype=np.int64)
        self.blink_idx = 0

    @staticmethod
    def _find_model_dir(model_dir: Optional[str | os.PathLike[str]]) -> Path:
        candidates: List[Path] = []
        if model_dir:
            p = Path(model_dir)
            candidates.append(p if p.is_dir() else p.parent)
        env_dir = os.environ.get("A2F_MODEL_DIR", "").strip()
        if env_dir:
            p = Path(env_dir)
            candidates.append(p if p.is_dir() else p.parent)
        candidates.extend([
            Path("/home/cxm2/data/a2f/audio2face-3d-model"),
            Path(__file__).resolve().parent.parent / "audio2face-3d-model",
        ])
        for p in candidates:
            if p and p.exists() and (p / "model_data.npz").exists() and (p / "bs_skin.npz").exists():
                return p.resolve()
        tried = "\n".join(str(p) for p in candidates if str(p))
        raise FileNotFoundError(
            "Cannot locate audio2face-3d model bundle. Set A2F_MODEL_DIR or pass model_dir.\n"
            f"Tried:\n{tried}"
        )

    @classmethod
    def preload_bundle(cls, model_dir: Optional[str | os.PathLike[str]] = None) -> bool:
        """Load immutable retarget assets once; return True on a cache hit."""
        resolved = cls._find_model_dir(model_dir)
        _, cache_hit = cls._get_cached_bundle(resolved)
        return cache_hit

    @classmethod
    def _get_cached_bundle(cls, model_dir: Path) -> Tuple[Dict[str, Any], bool]:
        cache_key = str(model_dir.resolve())
        with cls._bundle_cache_lock:
            cached = cls._bundle_cache.get(cache_key)
            if cached is not None:
                return cached, True
            bundle = cls._read_bundle(model_dir)
            cls._bundle_cache[cache_key] = bundle
            return bundle, False

    def _load_bundle(self, model_dir: Path) -> None:
        bundle, _ = self._get_cached_bundle(model_dir)
        for name, value in bundle.items():
            setattr(self, name, value)

    @staticmethod
    def _read_bundle(model_dir: Path) -> Dict[str, Any]:
        with np.load(model_dir / "model_data.npz", allow_pickle=True) as model_data:
            shapes_matrix_skin = np.asarray(model_data["shapes_matrix_skin"], dtype=np.float32)
            lip_open_pose_delta = np.asarray(model_data["lip_open_pose_delta"], dtype=np.float32)
            eye_close_pose_delta = np.asarray(model_data["eye_close_pose_delta"], dtype=np.float32)

        with open(model_dir / "model_config.json", "r", encoding="utf-8") as f:
            model_cfg = json.load(f)["config"]
        with open(model_dir / "bs_skin_config.json", "r", encoding="utf-8") as f:
            bs_cfg = json.load(f)["blendshape_params"]
        with open(model_dir / "network_info.json", "r", encoding="utf-8") as f:
            net_info = json.load(f)

        with np.load(model_dir / "bs_skin.npz", allow_pickle=True) as bs_skin:
            pose_names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in bs_skin["poseNames"]]
            pose_names_53 = pose_names
            pose_names_52 = pose_names[1:]
            arkit_index = {name: i for i, name in enumerate(pose_names_52)}

            frontal_mask = np.asarray(bs_skin["frontalMask"], dtype=np.int64)
            mask = frontal_mask
            neutral = np.asarray(bs_skin["neutral"], dtype=np.float32)[mask]

            bs_deltas_masked = []
            for name in pose_names_52:
                pose = np.asarray(bs_skin[name], dtype=np.float32)[mask]
                bs_deltas_masked.append((pose - neutral).reshape(-1))
        bs_deltas_masked = np.stack(bs_deltas_masked, axis=1).astype(np.float32)

        # Advanced indexing creates a strided layout. tensordot otherwise
        # copies/reorders this large constant matrix for EVERY native frame.
        # Normalize once in the shared model bundle; keep all solver math intact.
        shapes_matrix_skin_masked = np.ascontiguousarray(
            shapes_matrix_skin[:, mask, :], dtype=np.float32)
        lip_open_pose_delta_masked = np.asarray(lip_open_pose_delta[mask], dtype=np.float32).reshape(-1)
        eye_close_pose_delta_masked = np.asarray(eye_close_pose_delta[mask], dtype=np.float32).reshape(-1)

        active_mask = np.asarray(bs_cfg["bsSolveActivePoses"], dtype=np.int32).astype(bool)
        active_indices = np.where(active_mask)[0].astype(np.int64)
        active_names = [pose_names_52[i] for i in active_indices]

        A = bs_deltas_masked[:, active_indices]
        l2 = float(bs_cfg.get("strengthL2regularization", 0.5))
        G = (A.T @ A + (l2 * np.eye(A.shape[1], dtype=np.float32))).astype(np.float32)
        A_T = A.T.astype(np.float32)
        eigvals = np.linalg.eigvalsh(G.astype(np.float64))
        step = float(1.0 / max(float(np.max(eigvals)), 1e-6))

        sym_groups = np.asarray(bs_cfg.get("bsSolveSymmetryPoses", [-1] * len(pose_names_52)), dtype=np.int32)
        weight_multipliers = np.asarray(bs_cfg.get("bsWeightMultipliers", [1.0] * len(pose_names_52)), dtype=np.float32)
        weight_offsets = np.asarray(bs_cfg.get("bsWeightOffsets", [0.0] * len(pose_names_52)), dtype=np.float32)
        template_bb_size = float(bs_cfg.get("templateBBSize", 1.0))

        return {
            "shapes_matrix_skin": shapes_matrix_skin,
            "lip_open_pose_delta": lip_open_pose_delta,
            "eye_close_pose_delta": eye_close_pose_delta,
            "model_cfg": model_cfg,
            "bs_cfg": bs_cfg,
            "net_info": net_info,
            "pose_names_53": pose_names_53,
            "pose_names_52": pose_names_52,
            "arkit_index": arkit_index,
            "mask": mask,
            "bs_deltas_masked": bs_deltas_masked,
            "shapes_matrix_skin_masked": shapes_matrix_skin_masked,
            "lip_open_pose_delta_masked": lip_open_pose_delta_masked,
            "eye_close_pose_delta_masked": eye_close_pose_delta_masked,
            "active_indices": active_indices,
            "active_names": active_names,
            "G": G,
            "A_T": A_T,
            "step": step,
            "sym_groups": sym_groups,
            "weight_multipliers": weight_multipliers,
            "weight_offsets": weight_offsets,
            "template_bb_size": template_bb_size,
        }

    @staticmethod
    def _split_weights(w169: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w169 = np.asarray(w169, dtype=np.float32).reshape(-1)
        if w169.shape[0] != 169:
            raise ValueError(f"Expected 169-D vector, got shape {w169.shape}")
        skin = w169[0:140]
        tongue = w169[140:150]
        jaw = w169[150:165]
        eyes = w169[165:169]
        return skin, tongue, jaw, eyes

    def _update_calibration(self, skin: np.ndarray, jaw: np.ndarray, eyes: np.ndarray) -> None:
        if self._calibrated:
            return
        self._skin_hist.append(skin.astype(np.float32).copy())
        self._jaw_hist.append(jaw.astype(np.float32).copy())
        self._eyes_hist.append(eyes.astype(np.float32).copy())
        if len(self._jaw_hist) < self.calib_frames:
            return
        jaw_std = np.std(np.stack(self._jaw_hist, axis=0), axis=0)
        skin_std = np.std(np.stack(self._skin_hist, axis=0), axis=0)
        eyes_std = np.std(np.stack(self._eyes_hist, axis=0), axis=0)
        self.jaw_idx = np.argsort(jaw_std)[::-1][: min(4, len(jaw_std))].astype(np.int64)
        self.skin_idx = np.argsort(skin_std)[::-1][: min(8, len(skin_std))].astype(np.int64)
        self.blink_idx = int(np.argmax(eyes_std)) if len(eyes_std) else 0
        self._calibrated = True

    def _jaw_open_scalar(self, jaw_vec: np.ndarray, speech_gate: float) -> float:
        if self.jaw_baseline is None:
            self.jaw_baseline = jaw_vec.astype(np.float32).copy()

        raw_signed = float(np.mean(-jaw_vec[self.jaw_idx])) if len(self.jaw_idx) else 0.0
        baseline_signed_pre = float(np.mean(-self.jaw_baseline[self.jaw_idx])) if len(self.jaw_idx) else 0.0
        pre_mag = max(0.0, raw_signed - baseline_signed_pre)

        # Do not let a transient speech-gate dip aggressively re-center the jaw while
        # quiet-but-still-voiced audio is present. Only update the baseline quickly when
        # both audio and jaw activity look truly silent/closed.
        silence_like = (speech_gate < 0.035 and pre_mag < 0.045 and self.jaw_mag_ema < 0.060)
        baseline_alpha = 0.14 if silence_like else 0.0025
        self.jaw_baseline = (1.0 - baseline_alpha) * self.jaw_baseline + baseline_alpha * jaw_vec

        baseline_signed = float(np.mean(-self.jaw_baseline[self.jaw_idx])) if len(self.jaw_idx) else 0.0
        mag = max(0.0, raw_signed - baseline_signed)
        ema_alpha = 0.18 if mag >= self.jaw_mag_ema else 0.08
        self.jaw_mag_ema = float((1.0 - ema_alpha) * self.jaw_mag_ema + ema_alpha * mag)

        y = self.jaw_norm.update(mag, silence_like=silence_like)
        if speech_gate > 0.10:
            # Preserve low-energy voiced tail openings without globally boosting the jaw.
            y = max(float(y), float(np.clip(0.35 * mag / max(self.jaw_norm.init_span, 1e-6), 0.0, 1.0)))
        return float(np.clip(y, 0.0, 1.0))

    def _blink_scalar(self, eye_vec: np.ndarray, speech_gate: float) -> float:
        if self.eye_baseline is None:
            self.eye_baseline = eye_vec.astype(np.float32).copy()
        a = 0.12 if speech_gate < 0.08 else 0.02
        self.eye_baseline = (1.0 - a) * self.eye_baseline + a * eye_vec
        raw_signed = float(-(eye_vec[self.blink_idx])) if len(eye_vec) else 0.0
        base_signed = float(-(self.eye_baseline[self.blink_idx])) if len(eye_vec) else 0.0
        mag = max(0.0, raw_signed - base_signed)
        return self.blink_norm.update(mag, silence_like=speech_gate < 0.08)

    def _semantic_features(self, skin: np.ndarray, tongue: np.ndarray, jaw: np.ndarray, eyes: np.ndarray, speech_gate: float) -> Dict[str, float]:
        self._update_calibration(skin, jaw, eyes)

        jaw_open = self._jaw_open_scalar(jaw, speech_gate)
        tongue0 = float(tongue[0]) if len(tongue) > 0 else 0.0
        tongue1 = float(tongue[1]) if len(tongue) > 1 else 0.0
        jaw_signed = float(np.mean(-jaw[self.jaw_idx])) if len(self.jaw_idx) else 0.0
        round_raw = tongue0 + 0.50 * tongue1 - 0.25 * jaw_signed
        mouth_round = self.round_norm.update(round_raw, silence_like=speech_gate < 0.08)

        wide_raw = float(np.mean(skin[self.skin_idx])) if len(self.skin_idx) else 0.0
        mouth_wide = self.wide_norm.update(wide_raw, silence_like=False)

        lr_raw = float(skin[self.skin_idx[0]] - skin[self.skin_idx[1]]) if len(self.skin_idx) >= 2 else 0.0
        mouth_lr = self.lr_norm.update(lr_raw)

        upper_raw = float(np.std(skin[self.skin_idx])) if len(self.skin_idx) else float(np.std(skin))
        upper_face = self.upper_norm.update(upper_raw, silence_like=False)

        blink_like = self._blink_scalar(eyes, speech_gate)
        eye_act_raw = float(np.linalg.norm(eyes))
        eye_activity = self.eye_norm.update(eye_act_raw, silence_like=False)

        return {
            "jaw_open": float(np.clip(jaw_open, 0.0, 1.0)),
            "mouth_round": float(np.clip(mouth_round, 0.0, 1.0)),
            "mouth_wide": float(np.clip(mouth_wide, 0.0, 1.0)),
            "mouth_left_right": float(np.clip(mouth_lr, -1.0, 1.0)),
            "upper_face_activity": float(np.clip(upper_face, 0.0, 1.0)),
            "blink_like": float(np.clip(blink_like, 0.0, 1.0)),
            "eye_activity": float(np.clip(eye_activity, 0.0, 1.0)),
        }

    def _build_target_delta(self, skin: np.ndarray, jaw_vec: np.ndarray, eye_vec: np.ndarray, jaw_open_sem: float, blink_sem: float) -> np.ndarray:
        skin_delta = np.tensordot(skin.astype(np.float32), self.shapes_matrix_skin_masked, axes=(0, 0)).reshape(-1)
        lower_face_strength = float(self.model_cfg.get("lower_face_strength", 1.25))
        blink_strength = float(self.model_cfg.get("blink_strength", 1.0))
        skin_strength = float(self.model_cfg.get("skin_strength", 1.0))
        target = (
            skin_strength * skin_delta
            + lower_face_strength * jaw_open_sem * self.lip_open_pose_delta_masked
            + blink_strength * blink_sem * self.eye_close_pose_delta_masked
        )
        return target.astype(np.float32)

    def _solve_active_weights(self, target_delta: np.ndarray) -> np.ndarray:
        b = self.A_T @ target_delta.astype(np.float32)
        w = self.prev_active.copy()
        for _ in range(20):
            grad = self.G @ w - b
            w = np.maximum(0.0, w - self.step * grad)
        self.prev_active = w.astype(np.float32)
        return self.prev_active

    def _apply_symmetry(self, full_w: np.ndarray) -> np.ndarray:
        out = full_w.astype(np.float32).copy()
        group_to_indices: Dict[int, List[int]] = {}
        for i, grp in enumerate(self.sym_groups):
            if grp >= 0:
                group_to_indices.setdefault(int(grp), []).append(i)
        for inds in group_to_indices.values():
            if len(inds) == 2:
                avg = 0.5 * (out[inds[0]] + out[inds[1]])
                out[inds[0]] = avg
                out[inds[1]] = avg
        return out

    def _semantic_to_arkit(self, s: Dict[str, float], speech_gate: float) -> np.ndarray:
        out = np.zeros(52, dtype=np.float32)
        idx = self.arkit_index

        jaw = s["jaw_open"]
        rnd = s["mouth_round"]
        wide = s["mouth_wide"]
        lr = s["mouth_left_right"]
        upper = s["upper_face_activity"]
        blink = s["blink_like"]
        eye_act = s["eye_activity"]

        expr_mask = np.clip(1.0 - 0.68 * speech_gate, 0.20, 1.0)
        smile = expr_mask * np.clip(0.70 * wide - 0.12 * rnd, 0.0, 1.0)
        stretch = np.clip((0.24 + 0.62 * speech_gate) * wide, 0.0, 1.0)
        dimple = expr_mask * 0.35 * smile
        mouth_l = float(np.clip(max(0.0, -lr), 0.0, 1.0))
        mouth_r = float(np.clip(max(0.0, lr), 0.0, 1.0))
        active_open = np.clip(max(speech_gate, jaw), 0.0, 1.0)
        mouth_close = np.clip((1.0 - 0.82 * active_open) * (0.12 + 0.24 * (1.0 - jaw)), 0.0, 0.65)
        funnel = np.clip(rnd * (0.38 + 0.42 * speech_gate), 0.0, 1.0)
        pucker = np.clip(rnd * (0.52 + 0.18 * (1.0 - wide)), 0.0, 1.0)
        press = np.clip((1.0 - 0.65 * active_open) * (0.05 + 0.15 * mouth_close), 0.0, 0.50)
        lower_down = np.clip(0.30 * jaw + 0.12 * wide, 0.0, 1.0)
        upper_up = np.clip(0.18 * jaw + 0.10 * wide + 0.08 * rnd, 0.0, 1.0)
        shrug_lower = np.clip(0.18 * press + 0.12 * mouth_close, 0.0, 1.0)
        shrug_upper = np.clip(0.14 * upper_up + 0.12 * rnd, 0.0, 1.0)
        roll_lower = np.clip(0.10 * press + 0.08 * mouth_close, 0.0, 1.0)
        roll_upper = np.clip(0.10 * press + 0.06 * mouth_close, 0.0, 1.0)

        eye_blink = np.clip(blink, 0.0, 1.0)
        eye_squint = np.clip(0.58 * blink + 0.22 * eye_act, 0.0, 1.0)
        eye_wide = np.clip(0.22 * (1.0 - blink) * eye_act, 0.0, 1.0)

        # Unsigned activity cannot identify brow direction. Preserve the
        # geometry-derived brow channels instead of inventing bilateral lifts
        # (or treating every blink as a frown).
        brow_inner = brow_outer = brow_down = 0.0
        cheek_sq = np.clip(0.12 * smile + 0.16 * eye_squint, 0.0, 1.0)
        nose_sneer = np.clip(0.10 * smile + 0.06 * upper, 0.0, 1.0)
        cheek_puff = np.clip(0.18 * pucker, 0.0, 1.0)

        pairs = {
            "eyeBlinkLeft": eye_blink,
            "eyeBlinkRight": eye_blink,
            "eyeSquintLeft": eye_squint,
            "eyeSquintRight": eye_squint,
            "eyeWideLeft": eye_wide,
            "eyeWideRight": eye_wide,
            "jawOpen": jaw,
            "mouthClose": mouth_close,
            "mouthFunnel": funnel,
            "mouthPucker": pucker,
            "mouthLeft": mouth_l,
            "mouthRight": mouth_r,
            "mouthSmileLeft": np.clip(smile * (1.0 - 0.30 * mouth_l), 0.0, 1.0),
            "mouthSmileRight": np.clip(smile * (1.0 - 0.30 * mouth_r), 0.0, 1.0),
            "mouthFrownLeft": 0.0,
            "mouthFrownRight": 0.0,
            "mouthDimpleLeft": np.clip(dimple * (1.0 - 0.25 * mouth_l), 0.0, 1.0),
            "mouthDimpleRight": np.clip(dimple * (1.0 - 0.25 * mouth_r), 0.0, 1.0),
            "mouthStretchLeft": np.clip(stretch * (1.0 + 0.22 * mouth_l), 0.0, 1.0),
            "mouthStretchRight": np.clip(stretch * (1.0 + 0.22 * mouth_r), 0.0, 1.0),
            "mouthRollLower": roll_lower,
            "mouthRollUpper": roll_upper,
            "mouthShrugLower": shrug_lower,
            "mouthShrugUpper": shrug_upper,
            "mouthPressLeft": press,
            "mouthPressRight": press,
            "mouthLowerDownLeft": lower_down,
            "mouthLowerDownRight": lower_down,
            "mouthUpperUpLeft": upper_up,
            "mouthUpperUpRight": upper_up,
            "browDownLeft": brow_down,
            "browDownRight": brow_down,
            "browInnerUp": brow_inner,
            "browOuterUpLeft": brow_outer,
            "browOuterUpRight": brow_outer,
            "cheekPuff": cheek_puff,
            "cheekSquintLeft": cheek_sq,
            "cheekSquintRight": cheek_sq,
            "noseSneerLeft": nose_sneer,
            "noseSneerRight": nose_sneer,
        }
        for name, val in pairs.items():
            out[idx[name]] = float(np.clip(val, 0.0, 1.0))
        return out

    def _blend_geo_and_semantic(self, geo: np.ndarray, sem: np.ndarray, speech_gate: float) -> np.ndarray:
        geo = np.asarray(geo, dtype=np.float32)
        sem = np.asarray(sem, dtype=np.float32)
        out = np.zeros_like(sem)
        geo_gain = 180.0
        boosted_geo = np.clip(geo_gain * geo, 0.0, 1.0)

        strong_semantic = {
            "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
            "mouthSmileLeft", "mouthSmileRight", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthPressLeft", "mouthPressRight",
            "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
            "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight",
            "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
            "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "cheekPuff",
        }
        for i, name in enumerate(self.pose_names_52):
            if name in strong_semantic:
                # Use the stronger of semantic and boosted geometry, then light smoothing.
                v = max(float(sem[i]), float(boosted_geo[i]))
            else:
                v = max(float(0.5 * sem[i]), float(boosted_geo[i]))
            if name.startswith("eye"):
                out[i] = self.ema.update(name, v, rise_alpha=0.30, fall_alpha=0.40)
            elif name in {"jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight"}:
                out[i] = self.ema.update(name, v, rise_alpha=0.52, fall_alpha=0.62)
            else:
                out[i] = self.ema.update(name, v, rise_alpha=0.24, fall_alpha=0.34)

        # A2F commonly does not drive these from the bundle output path.
        for name in [
            "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
            "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
            "tongueOut", "jawForward", "jawLeft", "jawRight",
        ]:
            if name in self.arkit_index:
                out[self.arkit_index[name]] = 0.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def project_array(self, w169: np.ndarray, audio_rms: Optional[float] = None) -> np.ndarray:
        skin, tongue, jaw, eyes = self._split_weights(w169)

        fallback_energy = 0.0
        if len(jaw):
            fallback_energy = float(np.clip(np.linalg.norm(jaw) / (np.linalg.norm(jaw) + 10.0), 0.0, 1.0))
        speech_gate = self.audio_gate.update(audio_rms, fallback=fallback_energy)

        sem = self._semantic_features(skin, tongue, jaw, eyes, speech_gate)
        target_delta = self._build_target_delta(skin, jaw, eyes, sem["jaw_open"], sem["blink_like"])
        active_w = self._solve_active_weights(target_delta)

        full_w = np.zeros(52, dtype=np.float32)
        full_w[self.active_indices] = active_w
        full_w = self._apply_symmetry(full_w)
        full_w = full_w * self.weight_multipliers + self.weight_offsets
        full_w = np.clip(full_w, 0.0, 1.0)

        sem_arr = self._semantic_to_arkit(sem, speech_gate=speech_gate)
        return self._blend_geo_and_semantic(full_w, sem_arr, speech_gate=speech_gate)

    def project(self, w169: np.ndarray, audio_rms: Optional[float] = None) -> Dict[str, float]:
        arr = self.project_array(w169, audio_rms=audio_rms)
        out = {name: float(arr[i]) for i, name in enumerate(self.pose_names_52)}
        self.last_bs = out
        return out

    def neutral_dict(self) -> Dict[str, float]:
        return {name: 0.0 for name in ARKIT_52_NAMES}
