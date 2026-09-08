# -*- coding: utf-8 -*-
"""
Current-head retargeter v2: A2F raw-169 -> ARKit52 -> current normalized head DOFs.

Purpose
-------
Use this file as:
    a2f_tts_motion_bridge/motion_core/arkit52_to_motor.py

This stage outputs only normalized logical mechanism targets for preview and later IK:
- Four-bar modules: vertical targets in [0, 1], where 0.5 is neutral.
- Mouth-corner five-bars: 2D local mouth-corner targets in [0, 1]^2, where 0.5 is neutral.
- Jaw: direct vertical open target in [0, 1], where 0.0 is closed/rest and 1.0 is open.

It does NOT output final hardware motor angles. Final angle conversion / five-bar inverse
kinematics should be implemented after this stage.

Public API remains compatible with MotionStreamSession:
    feats, motors = OnlineRetargeter().update(w169, emotion, intensity, audio_rms)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from .a2f169_to_arkit52 import A2F169ToARKit52
except ImportError:  # pragma: no cover
    from data.a2f.a2f_tts_motion_bridge.motion_core.a2f169_to_arkit52 import A2F169ToARKit52


RETARGET_VERSION = "current_head_arkit52_to_norm_dof_v2_13dof"

# Current logical DOFs for the new head mechanism.
# 8 four-bar vertical modules + 2D right mouth corner + 2D left mouth corner + direct jaw = 13.
CURRENT_HEAD_DOF_NAMES = [
    # Four-bar vertical DOFs: 0.0 lower limit, 0.5 neutral, 1.0 upper limit.
    "right_outer_brow_y",
    "left_inner_brow_y",
    "right_inner_brow_y",
    "left_outer_brow_y",
    "right_upper_lid_y",
    "left_upper_lid_y",
    "right_lower_lid_y",
    "left_lower_lid_y",
    # Five-bar mouth-corner logical Cartesian targets in each side's local coordinates.
    # x: 0.0 inward, 0.5 neutral, 1.0 outward.
    # y: 0.0 down,   0.5 neutral, 1.0 up.
    "right_mouth_x",
    "right_mouth_y",
    "left_mouth_x",
    "left_mouth_y",
    # Direct jaw vertical open target: 0.0 closed/rest, 1.0 open/down.
    "jaw_y",
]

# Keep this symbol for existing code that imports MOTOR_CFG from this module.
MOTOR_CFG = {}
for name in CURRENT_HEAD_DOF_NAMES:
    neutral = 0.0 if name == "jaw_y" else 0.5
    MOTOR_CFG[name] = {"neutral": neutral, "min": 0.0, "max": 1.0}


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def center_clip(x: float, span: float = 0.5) -> float:
    """Clip centered logical coordinates to [0, 1]."""
    return clip(0.5 + float(x), 0.5 - span, 0.5 + span)


class AsymmetricEMA:
    def __init__(self):
        self.state: Dict[str, float] = {}

    def update(self, name: str, x: float, rise_alpha: float, fall_alpha: float) -> float:
        x = float(x)
        if name not in self.state:
            self.state[name] = x
            return x
        prev = self.state[name]
        # rise/fall here means numerical increase/decrease, not anatomical rise/fall.
        alpha = rise_alpha if x >= prev else fall_alpha
        y = (1.0 - alpha) * prev + alpha * x
        self.state[name] = y
        return y


class RateLimiter:
    def __init__(self):
        self.prev: Dict[str, float] = {
            name: (0.0 if name == "jaw_y" else 0.5) for name in CURRENT_HEAD_DOF_NAMES
        }

    def step(self, name: str, target: float, max_rise: float, max_fall: float) -> float:
        target = clip(target)
        prev = self.prev.get(name, 0.0 if name == "jaw_y" else 0.5)
        delta = target - prev
        if delta >= 0.0:
            y = prev + min(delta, max_rise)
        else:
            y = prev + max(delta, -max_fall)
        y = clip(y)
        self.prev[name] = y
        return y


class SpeechGate:
    """Lightweight audio gate used only to suppress silent-mouth leakage."""

    def __init__(self):
        self.state = 0.0
        self.rms_floor: Optional[float] = None
        self.rms_peak: Optional[float] = None

    def update(self, audio_rms: Optional[float], fallback: float) -> float:
        if audio_rms is None:
            target = clip(fallback)
        else:
            x = float(audio_rms)
            if self.rms_floor is None:
                self.rms_floor = x
                self.rms_peak = x + 0.03
            assert self.rms_peak is not None
            if x < self.rms_floor:
                self.rms_floor = 0.85 * self.rms_floor + 0.15 * x
            else:
                self.rms_floor = 0.98 * self.rms_floor + 0.02 * x
            if x > self.rms_peak:
                self.rms_peak = 0.82 * self.rms_peak + 0.18 * x
            else:
                self.rms_peak = 0.995 * self.rms_peak + 0.005 * x
            env = clip((x - self.rms_floor) / max(self.rms_peak - self.rms_floor, 1e-6))
            if env >= 0.16:
                target = clip((env - 0.06) / 0.10)
            elif env >= 0.06:
                target = 0.20 * clip((env - 0.06) / 0.10)
            else:
                target = 0.0
        alpha = 0.42 if target >= self.state else 0.86
        self.state = clip((1.0 - alpha) * self.state + alpha * target)
        return self.state


def _get(bs: Dict[str, float], key: str) -> float:
    return float(bs.get(key, 0.0))


def _emotion_bias(emotion: str, intensity: float) -> Dict[str, float]:
    """Small retarget-space bias so preview emotions are visibly different.

    These biases are intentionally conservative. They change only logical targets,
    not final hardware angles.
    """
    e = (emotion or "neutral").lower().strip()
    s = clip(float(intensity), 0.0, 1.5)
    z = {name: 0.0 for name in CURRENT_HEAD_DOF_NAMES}

    if e in {"happy", "smile", "joy"}:
        # Mouth corners outward/up, eyelids slightly squint, brows mildly raised.
        z["left_mouth_x"] += 0.18 * s
        z["right_mouth_x"] += 0.18 * s
        z["left_mouth_y"] += 0.22 * s
        z["right_mouth_y"] += 0.22 * s
        z["left_lower_lid_y"] += 0.08 * s
        z["right_lower_lid_y"] += 0.08 * s
        z["left_outer_brow_y"] += 0.04 * s
        z["right_outer_brow_y"] += 0.04 * s
    elif e in {"sad", "sorrow"}:
        # Mouth corners down/in, inner brows raised, outer brows slightly low.
        z["left_mouth_x"] -= 0.06 * s
        z["right_mouth_x"] -= 0.06 * s
        z["left_mouth_y"] -= 0.20 * s
        z["right_mouth_y"] -= 0.20 * s
        z["left_inner_brow_y"] += 0.14 * s
        z["right_inner_brow_y"] += 0.14 * s
        z["left_outer_brow_y"] -= 0.08 * s
        z["right_outer_brow_y"] -= 0.08 * s
    elif e in {"angry", "anger"}:
        # Brows down/in, tighter lower lids, smaller mouth. Four-bar y uses vertical only.
        z["left_inner_brow_y"] -= 0.18 * s
        z["right_inner_brow_y"] -= 0.18 * s
        z["left_outer_brow_y"] -= 0.10 * s
        z["right_outer_brow_y"] -= 0.10 * s
        z["left_lower_lid_y"] += 0.10 * s
        z["right_lower_lid_y"] += 0.10 * s
        z["left_upper_lid_y"] -= 0.06 * s
        z["right_upper_lid_y"] -= 0.06 * s
        z["left_mouth_x"] -= 0.04 * s
        z["right_mouth_x"] -= 0.04 * s
        z["left_mouth_y"] -= 0.06 * s
        z["right_mouth_y"] -= 0.06 * s
    elif e in {"surprise", "surprised"}:
        # Brows up, upper lids up/open, jaw open bias, corners slightly outward.
        for k in ["left_inner_brow_y", "right_inner_brow_y", "left_outer_brow_y", "right_outer_brow_y"]:
            z[k] += 0.16 * s
        z["left_upper_lid_y"] += 0.18 * s
        z["right_upper_lid_y"] += 0.18 * s
        z["left_lower_lid_y"] -= 0.08 * s
        z["right_lower_lid_y"] -= 0.08 * s
        z["left_mouth_x"] += 0.06 * s
        z["right_mouth_x"] += 0.06 * s
        z["jaw_y"] += 0.15 * s
    return z


def arkit52_to_current_head_dofs(
    bs: Dict[str, float],
    speech_gate: float = 1.0,
    emotion: str = "neutral",
    intensity: float = 1.0,
) -> Dict[str, float]:
    """Stateless ARKit52 -> current normalized mechanism DOFs.

    Definition:
    - Four-bar modules output vertical position y_norm: 0 lower, 0.5 neutral, 1 upper.
    - Mouth corners output local x/y in [0, 1]: x inward/outward, y down/up, 0.5 neutral.
    - Jaw outputs jaw_y in [0, 1]: 0 closed/rest, 1 open/down.
    """
    # Eyes.
    blink_l = _get(bs, "eyeBlinkLeft")
    blink_r = _get(bs, "eyeBlinkRight")
    squint_l = _get(bs, "eyeSquintLeft")
    squint_r = _get(bs, "eyeSquintRight")
    wide_l = _get(bs, "eyeWideLeft")
    wide_r = _get(bs, "eyeWideRight")
    cheek_l = _get(bs, "cheekSquintLeft")
    cheek_r = _get(bs, "cheekSquintRight")

    # Brows.
    inner_up = _get(bs, "browInnerUp")
    outer_up_l = _get(bs, "browOuterUpLeft")
    outer_up_r = _get(bs, "browOuterUpRight")
    brow_down_l = _get(bs, "browDownLeft")
    brow_down_r = _get(bs, "browDownRight")

    # Mouth.
    smile_l = _get(bs, "mouthSmileLeft")
    smile_r = _get(bs, "mouthSmileRight")
    dimple_l = _get(bs, "mouthDimpleLeft")
    dimple_r = _get(bs, "mouthDimpleRight")
    stretch_l = _get(bs, "mouthStretchLeft")
    stretch_r = _get(bs, "mouthStretchRight")
    frown_l = _get(bs, "mouthFrownLeft")
    frown_r = _get(bs, "mouthFrownRight")
    press_l = _get(bs, "mouthPressLeft")
    press_r = _get(bs, "mouthPressRight")
    pucker = _get(bs, "mouthPucker")
    funnel = _get(bs, "mouthFunnel")
    jaw_open_bs = _get(bs, "jawOpen")
    mouth_close = _get(bs, "mouthClose")

    speech = clip(speech_gate)

    # Jaw: suppress small jaw leakage in silence, keep speech-driven opening.
    jaw_close_cancel = (0.015 + 0.070 * (1.0 - speech)) * mouth_close
    jaw_y = clip(1.10 * max(jaw_open_bs - jaw_close_cancel, 0.0) + 0.08 * max(pucker, funnel))

    # Four-bar y targets. 0.5 is neutral. Positive displacement means up.
    left_inner_brow_y = center_clip(0.34 * inner_up + 0.05 * outer_up_l - 0.24 * brow_down_l)
    right_inner_brow_y = center_clip(0.34 * inner_up + 0.05 * outer_up_r - 0.24 * brow_down_r)
    left_outer_brow_y = center_clip(0.38 * outer_up_l + 0.06 * inner_up - 0.22 * brow_down_l)
    right_outer_brow_y = center_clip(0.38 * outer_up_r + 0.06 * inner_up - 0.22 * brow_down_r)

    # Upper lid y: higher = more open/up; blink closes downward.
    left_upper_lid_y = center_clip(0.30 * wide_l - 0.42 * blink_l - 0.10 * squint_l)
    right_upper_lid_y = center_clip(0.30 * wide_r - 0.42 * blink_r - 0.10 * squint_r)

    # Lower lid y: higher = lower eyelid raises upward.
    left_lower_lid_y = center_clip(0.22 * blink_l + 0.34 * squint_l + 0.18 * cheek_l - 0.08 * wide_l)
    right_lower_lid_y = center_clip(0.22 * blink_r + 0.34 * squint_r + 0.18 * cheek_r - 0.08 * wide_r)

    # Mouth-corner five-bar local 2D targets.
    # x: outward positive; pucker/funnel pulls inward.
    # y: up positive; frown pulls downward.
    left_mouth_x = center_clip(0.24 * smile_l + 0.18 * stretch_l + 0.08 * dimple_l - 0.20 * max(pucker, funnel) - 0.06 * press_l)
    right_mouth_x = center_clip(0.24 * smile_r + 0.18 * stretch_r + 0.08 * dimple_r - 0.20 * max(pucker, funnel) - 0.06 * press_r)
    left_mouth_y = center_clip(0.32 * smile_l + 0.10 * dimple_l - 0.30 * frown_l - 0.04 * press_l)
    right_mouth_y = center_clip(0.32 * smile_r + 0.10 * dimple_r - 0.30 * frown_r - 0.04 * press_r)

    dofs = {
        "right_outer_brow_y": right_outer_brow_y,
        "left_inner_brow_y": left_inner_brow_y,
        "right_inner_brow_y": right_inner_brow_y,
        "left_outer_brow_y": left_outer_brow_y,
        "right_upper_lid_y": right_upper_lid_y,
        "left_upper_lid_y": left_upper_lid_y,
        "right_lower_lid_y": right_lower_lid_y,
        "left_lower_lid_y": left_lower_lid_y,
        "right_mouth_x": right_mouth_x,
        "right_mouth_y": right_mouth_y,
        "left_mouth_x": left_mouth_x,
        "left_mouth_y": left_mouth_y,
        "jaw_y": jaw_y,
    }

    # Add conservative emotion-space offsets after ARKit mapping.
    bias = _emotion_bias(emotion, intensity)
    for k in CURRENT_HEAD_DOF_NAMES:
        dofs[k] = clip(dofs[k] + bias.get(k, 0.0))
    return dofs


def debug_features_from_arkit52(bs: Dict[str, float], dofs: Dict[str, float], speech_gate: float) -> Dict[str, float]:
    # Convert new DOFs to the old seven abstract features used by the preview renderer.
    jaw_open = float(dofs["jaw_y"])
    mouth_wide = clip(0.5 * ((dofs["left_mouth_x"] - 0.5) + (dofs["right_mouth_x"] - 0.5)) / 0.5)
    mouth_corner_up = clip(0.5 + 0.5 * ((dofs["left_mouth_y"] - 0.5) + (dofs["right_mouth_y"] - 0.5)) / 0.5)
    mouth_lr = clip((dofs["right_mouth_y"] - dofs["left_mouth_y"]) + 0.5 * (dofs["right_mouth_x"] - dofs["left_mouth_x"]), -1.0, 1.0)
    upper_face = clip(np.mean([
        abs(dofs["left_inner_brow_y"] - 0.5),
        abs(dofs["right_inner_brow_y"] - 0.5),
        abs(dofs["left_outer_brow_y"] - 0.5),
        abs(dofs["right_outer_brow_y"] - 0.5),
    ]) * 2.0)
    eye_activity = clip(np.mean([
        abs(dofs["left_upper_lid_y"] - 0.5),
        abs(dofs["right_upper_lid_y"] - 0.5),
        abs(dofs["left_lower_lid_y"] - 0.5),
        abs(dofs["right_lower_lid_y"] - 0.5),
    ]) * 2.0)
    blink_like = clip(0.5 * ((0.5 - dofs["left_upper_lid_y"]) + (0.5 - dofs["right_upper_lid_y"])) / 0.5)
    mouth_round = clip(max(_get(bs, "mouthFunnel"), _get(bs, "mouthPucker")))
    return {
        "jaw_open": jaw_open,
        "mouth_round": float(mouth_round),
        "mouth_wide": float(max(mouth_wide, mouth_corner_up * 0.25)),
        "mouth_left_right": float(mouth_lr),
        "upper_face_activity": float(upper_face),
        "eye_activity": float(eye_activity),
        "blink_like": float(blink_like),
        "speech_gate": float(speech_gate),
    }


class OnlineRetargeter:
    @staticmethod
    def preload_model_bundle(model_dir: Optional[str] = None) -> bool:
        return A2F169ToARKit52.preload_bundle(model_dir=model_dir)

    def __init__(self, model_dir: Optional[str] = None, calib_frames: int = 20, **_ignored):
        self.mapper = A2F169ToARKit52(model_dir=model_dir, calib_frames=calib_frames)
        self.speech_gate = SpeechGate()
        self.ema = AsymmetricEMA()
        self.rate = RateLimiter()
        self.last_blendshapes = self.mapper.neutral_dict()
        self.last_debug: Dict[str, float] = {}
        self.retarget_version = RETARGET_VERSION

    def _smooth_dofs(self, raw: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in CURRENT_HEAD_DOF_NAMES:
            x = clip(raw.get(name, 0.0 if name == "jaw_y" else 0.5))
            if name == "jaw_y":
                x = self.ema.update(name, x, rise_alpha=0.62, fall_alpha=0.84)
                y = self.rate.step(name, x, max_rise=0.12, max_fall=0.22)
            elif "mouth" in name:
                x = self.ema.update(name, x, rise_alpha=0.24, fall_alpha=0.45)
                y = self.rate.step(name, x, max_rise=0.045, max_fall=0.060)
            elif "lid" in name:
                x = self.ema.update(name, x, rise_alpha=0.28, fall_alpha=0.40)
                y = self.rate.step(name, x, max_rise=0.050, max_fall=0.065)
            else:
                x = self.ema.update(name, x, rise_alpha=0.18, fall_alpha=0.28)
                y = self.rate.step(name, x, max_rise=0.035, max_fall=0.045)
            out[name] = float(y)
        return out

    def project_blendshapes(self, w169, audio_rms: Optional[float] = None) -> Dict[str, float]:
        bs = self.mapper.project(w169, audio_rms=audio_rms)
        self.last_blendshapes = bs
        return bs

    def update(self, w169, emotion: str = "neutral", intensity: float = 1.0, audio_rms: Optional[float] = None):
        bs = self.project_blendshapes(w169, audio_rms=audio_rms)
        speech_gate = self.speech_gate.update(
            audio_rms,
            fallback=max(_get(bs, "jawOpen"), _get(bs, "mouthFunnel"), _get(bs, "mouthPucker")),
        )
        raw_dofs = arkit52_to_current_head_dofs(
            bs,
            speech_gate=speech_gate,
            emotion=emotion,
            intensity=intensity,
        )
        dofs = self._smooth_dofs(raw_dofs)
        feats = debug_features_from_arkit52(bs, dofs, speech_gate)
        self.last_debug = {
            "speech_gate": feats["speech_gate"],
            "arkit_jawOpen": _get(bs, "jawOpen"),
            "arkit_mouthClose": _get(bs, "mouthClose"),
            "retarget_version": RETARGET_VERSION,
        }
        feats_out = {
            "jaw_open": feats["jaw_open"],
            "mouth_round": feats["mouth_round"],
            "mouth_wide": feats["mouth_wide"],
            "mouth_left_right": feats["mouth_left_right"],
            "upper_face_activity": feats["upper_face_activity"],
            "eye_activity": feats["eye_activity"],
            "blink_like": feats["blink_like"],
        }
        return feats_out, dofs
