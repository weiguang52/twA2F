#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 motorstream 项目生成的 .npy 渲染为带音频的预览视频。

修正版：
- 严格以音频总时长作为视频主时间轴
- 对动作/特征按音频全长补齐
- 动作范围外：特征默认回中性，motor 默认回 neutral

推荐：
python visualize/render_motor_npy.py --input_npy visualize/npy/happy_1.0.npy
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
if __package__:
    from .render_current_head_13dof_npy import DOF_NAMES, draw_face as draw_dof_face, dof_row_to_dict
else:
    from render_current_head_13dof_npy import DOF_NAMES, draw_face as draw_dof_face, dof_row_to_dict

FPS = 30
W = 1280
H = 1100

BG = (245, 245, 245)
FG = (30, 30, 30)
ACCENT = (40, 80, 220)
WAVE = (80, 80, 80)
CURSOR = (220, 80, 80)
OK_BAR = (80, 170, 90)

SMOOTH_K = 7

FEATURE_NAMES = [
    "jaw_open",
    "mouth_round",
    "mouth_wide",
    "mouth_left_right",
    "upper_face_activity",
    "eye_activity",
    "blink_like",
]

DEFAULT_MOTOR_NEUTRAL = {
    # Retain support for recordings made with older local motor names.
    "right_outer_brow": 0.0,
    "left_inner_brow": 0.0,
    "right_inner_brow": 0.0,
    "left_outer_brow": 0.0,
    "right_upper_lid_close": 0.0,
    "left_upper_lid_close": 0.0,
    "right_lower_lid_raise": 0.0,
    "left_lower_lid_raise": 0.0,
    "right_mouth_corner": 0.0,
    "left_mouth_corner": 0.0,
    "jaw_open": 0.0,
    "head_nod": 0.0,
    # Current-head logical DOFs: jaw closed, other DOFs at midpoint.
    "right_outer_brow_y": 0.5,
    "left_inner_brow_y": 0.5,
    "right_inner_brow_y": 0.5,
    "left_outer_brow_y": 0.5,
    "right_upper_lid_y": 0.5,
    "left_upper_lid_y": 0.5,
    "right_lower_lid_y": 0.5,
    "left_lower_lid_y": 0.5,
    "right_mouth_x": 0.5,
    "right_mouth_y": 0.5,
    "left_mouth_x": 0.5,
    "left_mouth_y": 0.5,
    "jaw_y": 0.0,
    "jaw": 0.50,
    "mouth_left": 0.50,
    "mouth_right": 0.50,
    "upper_lid_left": 0.30,
    "upper_lid_right": 0.30,
    "lower_lid_left": 0.20,
    "lower_lid_right": 0.20,
    "inner_brow_left": 0.50,
    "inner_brow_right": 0.50,
    "outer_brow_left": 0.50,
    "outer_brow_right": 0.50,
}


def moving_average(x, k=5):
    x = np.asarray(x, dtype=np.float32)
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="valid")


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x)) if len(x) else 0.0
    xmax = float(np.max(x)) if len(x) else 0.0
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def signed_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m < 1e-8:
        return np.zeros_like(x)
    return x / m


def extract_features_from_weights(weights):
    skin = weights[:, 0:140]
    tongue = weights[:, 140:150]
    jaw = weights[:, 150:165]
    eyes = weights[:, 165:169]

    skin_energy = moving_average(normalize01(np.linalg.norm(skin, axis=1)), SMOOTH_K)
    tongue_energy = moving_average(normalize01(np.linalg.norm(tongue, axis=1)), SMOOTH_K)
    jaw_energy = moving_average(normalize01(np.linalg.norm(jaw, axis=1)), SMOOTH_K)
    eyes_energy = moving_average(normalize01(np.linalg.norm(eyes, axis=1)), SMOOTH_K)

    jaw_std = jaw.std(axis=0)
    jaw_top = np.argsort(jaw_std)[::-1][:4]
    jaw_open_raw = jaw[:, jaw_top].mean(axis=1)
    jaw_open = moving_average(normalize01(-jaw_open_raw), SMOOTH_K)

    tongue0 = tongue[:, 0] if tongue.shape[1] > 0 else np.zeros(len(weights))
    tongue1 = tongue[:, 1] if tongue.shape[1] > 1 else np.zeros(len(weights))
    mouth_round_raw = tongue0 + 0.5 * tongue1 - 0.3 * jaw_open_raw
    mouth_round = moving_average(normalize01(mouth_round_raw), SMOOTH_K)

    skin_std = skin.std(axis=0)
    skin_top = np.argsort(skin_std)[::-1][:8]
    mouth_wide_raw = skin[:, skin_top].mean(axis=1)
    mouth_wide = moving_average(normalize01(mouth_wide_raw), SMOOTH_K)

    if len(skin_top) >= 2:
        mouth_lr_raw = skin[:, skin_top[0]] - skin[:, skin_top[1]]
    else:
        mouth_lr_raw = np.zeros(len(weights))
    mouth_left_right = moving_average(signed_normalize(mouth_lr_raw), SMOOTH_K)

    upper_face_activity = moving_average(normalize01(skin.std(axis=1)), SMOOTH_K)
    eye_activity = moving_average(normalize01(np.linalg.norm(eyes, axis=1)), SMOOTH_K)

    eye_top = int(np.argmax(eyes.std(axis=0))) if eyes.shape[1] > 0 else 0
    blink_like = (
        moving_average(normalize01(-eyes[:, eye_top]), SMOOTH_K)
        if eyes.shape[1] > 0 else np.zeros(len(weights))
    )

    return {
        "skin_energy": skin_energy,
        "tongue_energy": tongue_energy,
        "jaw_energy": jaw_energy,
        "eyes_energy": eyes_energy,
        "jaw_open": jaw_open,
        "mouth_round": mouth_round,
        "mouth_wide": mouth_wide,
        "mouth_left_right": mouth_left_right,
        "upper_face_activity": upper_face_activity,
        "eye_activity": eye_activity,
        "blink_like": blink_like,
    }


def feature_dict_from_array(features_2d: np.ndarray):
    out = {}
    for idx, name in enumerate(FEATURE_NAMES):
        if features_2d.shape[1] > idx:
            out[name] = features_2d[:, idx].astype(np.float32)
        else:
            out[name] = np.zeros(features_2d.shape[0], dtype=np.float32)
    return out


def prepare_waveform_image(wav, width, height):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1e-8:
        wav = wav / peak

    img = np.full((height, width, 3), 252, dtype=np.uint8)
    mid_y = height // 2
    cv2.line(img, (0, mid_y), (width - 1, mid_y), (210, 210, 210), 1, cv2.LINE_AA)

    if len(wav) == 0:
        return img

    samples_per_col = max(1, len(wav) // width)
    for x in range(width):
        start = x * samples_per_col
        end = min(len(wav), start + samples_per_col)
        seg = wav[start:end]
        if len(seg) == 0:
            continue
        ymin = float(np.min(seg))
        ymax = float(np.max(seg))
        y1 = int(mid_y - ymax * (height * 0.42))
        y2 = int(mid_y - ymin * (height * 0.42))
        cv2.line(img, (x, y1), (x, y2), WAVE, 1, cv2.LINE_AA)
    return img


def draw_face_outline(img):
    center = (W // 2 - 120, 440)
    axes = (220, 300)
    cv2.ellipse(img, center, axes, 0, 0, 360, (210, 210, 210), 2, cv2.LINE_AA)


def draw_brow(img, cx, cy, length, tilt, lift, side=1):
    y = int(cy - 35 * lift)
    x1 = int(cx - length // 2)
    x2 = int(cx + length // 2)
    dy = int(18 * tilt * side)
    cv2.line(img, (x1, y - dy), (x2, y + dy), FG, 5, cv2.LINE_AA)


def draw_eye(img, cx, cy, eye_open, lower_raise):
    eye_h = int(6 + 26 * eye_open)
    eye_w = 58
    cv2.ellipse(img, (cx, cy), (eye_w, max(eye_h, 1)), 0, 0, 360, FG, 2, cv2.LINE_AA)
    upper_y = int(cy - eye_h)
    lower_y = int(cy + eye_h - 10 * lower_raise)
    cv2.line(img, (cx - eye_w, upper_y), (cx + eye_w, upper_y), FG, 3, cv2.LINE_AA)
    cv2.line(img, (cx - eye_w + 6, lower_y), (cx + eye_w - 6, lower_y), FG, 2, cv2.LINE_AA)
    if eye_open > 0.12:
        cv2.circle(img, (cx, cy), 8, ACCENT, -1, cv2.LINE_AA)


def draw_mouth(img, cx, cy, jaw_open, mouth_wide, mouth_round, mouth_lr):
    width = int(70 + 110 * mouth_wide)
    open_h = int(8 + 95 * jaw_open)
    round_boost = int(40 * mouth_round)

    left_x = int(cx - width // 2 - 18 * mouth_lr)
    right_x = int(cx + width // 2 - 18 * mouth_lr)
    upper_y = cy - max(4, round_boost // 3)
    lower_y = cy + open_h + round_boost // 2

    upper_pts = np.array([
        [left_x, cy],
        [cx - width // 4, upper_y],
        [cx + width // 4, upper_y],
        [right_x, cy],
    ], dtype=np.int32).reshape(-1, 1, 2)

    lower_pts = np.array([
        [left_x, cy],
        [cx - width // 4, lower_y],
        [cx + width // 4, lower_y],
        [right_x, cy],
    ], dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(img, [upper_pts], False, FG, 4, cv2.LINE_AA)
    cv2.polylines(img, [lower_pts], False, FG, 4, cv2.LINE_AA)

    if open_h > 10:
        inner_top = cy + 4
        inner_bottom = cy + max(8, open_h)
        cv2.ellipse(
            img,
            (cx - int(8 * mouth_lr), (inner_top + inner_bottom) // 2),
            (max(18, width // 4), max(4, (inner_bottom - inner_top) // 2)),
            0, 0, 360, (40, 40, 60), -1, cv2.LINE_AA,
        )


def draw_chin(img, cx, base_y, jaw_open):
    chin_drop = int(15 + 95 * jaw_open)
    cv2.circle(img, (cx, base_y + chin_drop), 7, FG, -1, cv2.LINE_AA)


def draw_energy_bar(img, x, y, w, h, value, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 1)
    fill = int(w * float(np.clip(value, 0.0, 1.0)))
    cv2.rectangle(img, (x, y), (x + fill, y + h), ACCENT, -1)
    cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FG, 1, cv2.LINE_AA)


def draw_motor_panel(img, motor_names, motor_values_row):
    panel_x = 900
    panel_y = 180
    bar_w = 280
    bar_h = 18
    gap = 42

    cv2.putText(img, "robot motor preview", (panel_x, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FG, 2, cv2.LINE_AA)
    show_names = list(motor_names)
    motor_map = {name: float(motor_values_row[idx]) for idx, name in enumerate(motor_names)}

    for i, name in enumerate(show_names):
        if name not in motor_map:
            continue
        y = panel_y + i * gap
        val = float(np.clip(motor_map[name], 0.0, 1.0))
        cv2.rectangle(img, (panel_x, y), (panel_x + bar_w, y + bar_h), (180, 180, 180), 1)
        fill = int(bar_w * val)
        cv2.rectangle(img, (panel_x, y), (panel_x + fill, y + bar_h), OK_BAR, -1)
        cv2.putText(img, f"{name}: {val:.3f}", (panel_x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FG, 1, cv2.LINE_AA)


def parse_emotion_and_intensity_from_name(stem: str):
    if "_" not in stem:
        return stem, "?"
    emotion, intensity = stem.rsplit("_", 1)
    return emotion, intensity


def _interp_track(src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray, fill_mode: str = "edge", fill_value: float = 0.0) -> np.ndarray:
    src_t = np.asarray(src_t, dtype=np.float32)
    src_v = np.asarray(src_v, dtype=np.float32)
    dst_t = np.asarray(dst_t, dtype=np.float32)
    if len(dst_t) == 0:
        return np.zeros((0,) + src_v.shape[1:], dtype=np.float32)
    if len(src_t) == 0 or len(src_v) == 0:
        if src_v.ndim == 1:
            return np.full(len(dst_t), fill_value, dtype=np.float32)
        return np.full((len(dst_t), src_v.shape[1]), fill_value, dtype=np.float32)

    if src_v.ndim == 1:
        left = float(src_v[0]) if fill_mode == "edge" else float(fill_value)
        right = float(src_v[-1]) if fill_mode == "edge" else float(fill_value)
        return np.interp(dst_t, src_t, src_v, left=left, right=right).astype(np.float32)

    out = np.zeros((len(dst_t), src_v.shape[1]), dtype=np.float32)
    for i in range(src_v.shape[1]):
        left = float(src_v[0, i]) if fill_mode == "edge" else float(fill_value)
        right = float(src_v[-1, i]) if fill_mode == "edge" else float(fill_value)
        out[:, i] = np.interp(dst_t, src_t, src_v[:, i], left=left, right=right).astype(np.float32)
    return out


def _neutral_feature_tracks(dst_t: np.ndarray):
    z = np.zeros(len(dst_t), dtype=np.float32)
    return {
        "jaw_open": z.copy(),
        "mouth_round": z.copy(),
        "mouth_wide": z.copy(),
        "mouth_left_right": z.copy(),
        "upper_face_activity": z.copy(),
        "eye_activity": z.copy(),
        "blink_like": z.copy(),
        "skin_energy": z.copy(),
        "tongue_energy": z.copy(),
        "jaw_energy": z.copy(),
        "eyes_energy": z.copy(),
    }


def load_payload(path: Path):
    data = np.load(path, allow_pickle=True).item()
    meta = data.get("meta", {})
    audio_16k = np.asarray(data.get("audio_16k_mono", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    audio_sr = int(meta.get("target_sr", 16000))
    duration = len(audio_16k) / float(audio_sr) if len(audio_16k) > 0 else 0.0
    total_frames = max(1, int(round(duration * FPS))) if duration > 0 else 1
    video_times = (np.arange(total_frames, dtype=np.float32) / float(FPS))

    # source tracks
    if "frame_times_30fps" in data and len(data["frame_times_30fps"]) > 0:
        src_t = np.asarray(data["frame_times_30fps"], dtype=np.float32)
        weights_src = np.asarray(data.get("weights_30fps", np.zeros((0, 169), dtype=np.float32)), dtype=np.float32)
        features_src = np.asarray(data.get("features_30fps", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        motors_src = np.asarray(data.get("motor_values_30fps", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    else:
        src_t = np.asarray(data.get("frame_times_native", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        weights_src = np.asarray(data.get("weights_native", np.zeros((0, 169), dtype=np.float32)), dtype=np.float32)
        features_src = np.asarray(data.get("features_native", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        motors_src = np.asarray(data.get("motor_values_native", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)

    if features_src.size == 0 and weights_src.size > 0:
        base_features = extract_features_from_weights(weights_src)
    elif features_src.size > 0:
        base_features = feature_dict_from_array(features_src)
        if weights_src.size > 0:
            extras = extract_features_from_weights(weights_src)
            for key in ["skin_energy", "tongue_energy", "jaw_energy", "eyes_energy"]:
                base_features[key] = extras[key]
        else:
            z = np.zeros(len(src_t), dtype=np.float32)
            for key in ["skin_energy", "tongue_energy", "jaw_energy", "eyes_energy"]:
                base_features[key] = z
    else:
        base_features = _neutral_feature_tracks(src_t)

    # interpolate to full audio duration; outside source range, features -> neutral zero
    features = {}
    for key, src_v in base_features.items():
        features[key] = _interp_track(src_t, np.asarray(src_v, dtype=np.float32), video_times, fill_mode="constant", fill_value=0.0)

    weights = _interp_track(src_t, weights_src, video_times, fill_mode="constant", fill_value=0.0) if weights_src.size > 0 else np.zeros((len(video_times), 169), dtype=np.float32)

    motor_names = list(meta.get("motor_names", []))
    if len(motor_names) > 0 and motors_src.size > 0:
        neutral = np.array([DEFAULT_MOTOR_NEUTRAL.get(name, 0.5) for name in motor_names], dtype=np.float32)
        motors = np.zeros((len(video_times), len(motor_names)), dtype=np.float32)
        for i in range(len(motor_names)):
            motors[:, i] = _interp_track(src_t, motors_src[:, i], video_times, fill_mode="constant", fill_value=float(neutral[i]))
    else:
        motors = np.zeros((len(video_times), 0), dtype=np.float32)

    return data, meta, duration, video_times, weights, features, motor_names, motors, audio_16k, audio_sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy", required=True, help="motorstream 输出的 .npy")
    parser.add_argument("--output_video", default=None, help="输出 mp4 路径；默认写到 visualize/videos/<same_name>.mp4")
    parser.add_argument("--transcript", default="", help="可选文本展示")
    parser.add_argument("--ffmpeg_bin", default="ffmpeg")
    parser.add_argument("--blind_id", default="", help="Hide emotion/intensity labels; show this anonymous trial ID")
    args = parser.parse_args()

    input_npy = Path(args.input_npy).resolve()
    if not input_npy.exists():
        raise FileNotFoundError(f"找不到输入 npy: {input_npy}")

    script_dir = Path(__file__).resolve().parent
    default_video_dir = script_dir / "videos"
    default_video_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path(args.output_video).resolve() if args.output_video else (default_video_dir / f"{input_npy.stem}.mp4")

    data, meta, duration, video_times, weights, f, motor_names, motors, wav, wav_sr = load_payload(input_npy)

    wave_x = 70
    wave_y = 900
    wave_w = W - 140
    wave_h = 140
    wave_img = prepare_waveform_image(wav, wave_w, wave_h)

    emotion, intensity = parse_emotion_and_intensity_from_name(input_npy.stem)
    retarget_version = str(meta.get("retarget_version", "unknown"))
    transcript = args.transcript or f"emotion={emotion}  intensity={intensity}  retarget={retarget_version}"
    if args.blind_id:
        transcript = f"Trial {args.blind_id}"

    output_video.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="motorviz_") as tmp_dir:
        temp_video_path = Path(tmp_dir) / "temp_silent_video.mp4"
        temp_audio_path = Path(tmp_dir) / "temp_audio_track.wav"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video_path), fourcc, FPS, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {temp_video_path}")
        face_cx = W // 2 - 120

        total_frames = len(video_times)
        print(f"[INFO] render {input_npy.name} -> {output_video}")
        print(f"[INFO] frames={total_frames}, duration={duration:.3f}s, wav_sr={wav_sr}")

        for i, t in enumerate(video_times):
            img = np.full((H, W, 3), BG, dtype=np.uint8)
            draw_face_outline(img)

            jaw_open = float(f["jaw_open"][i])
            mouth_wide = float(f["mouth_wide"][i])
            mouth_round = float(f["mouth_round"][i])
            mouth_lr = float(f["mouth_left_right"][i])
            upper_face = float(f["upper_face_activity"][i])
            blink = float(f["blink_like"][i])
            eye_act = float(f["eye_activity"][i])

            brow_lift = np.clip(0.25 + 0.85 * upper_face, 0.0, 1.0)
            brow_tilt = 0.15 * mouth_lr
            eye_open = np.clip(1.0 - 0.9 * blink + 0.1 * eye_act, 0.03, 1.0)
            lower_raise = np.clip(0.35 * blink + 0.2 * eye_act, 0.0, 1.0)

            if set(DOF_NAMES).issubset(motor_names):
                draw_dof_face(img, dof_row_to_dict(motor_names, motors[i]))
            else:
                draw_brow(img, face_cx - 130, 280, 120, brow_tilt, brow_lift, side=-1)
                draw_brow(img, face_cx + 130, 280, 120, brow_tilt, brow_lift, side=1)
                draw_eye(img, face_cx - 130, 390, eye_open, lower_raise)
                draw_eye(img, face_cx + 130, 390, eye_open, lower_raise)
                draw_mouth(img, face_cx, 620, jaw_open, mouth_wide, mouth_round, mouth_lr)
                draw_chin(img, face_cx, 700, jaw_open)

            draw_energy_bar(img, 60, 90, 180, 18, f["skin_energy"][i], "skin")
            draw_energy_bar(img, 60, 130, 180, 18, f["jaw_energy"][i], "jaw")
            draw_energy_bar(img, 60, 170, 180, 18, f["eyes_energy"][i], "eyes")
            draw_energy_bar(img, 60, 210, 180, 18, f["tongue_energy"][i], "tongue")

            cv2.putText(img, "A2F -> Robot expression preview", (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2, cv2.LINE_AA)
            cv2.putText(img, f"time: {t:0.2f}s / {duration:0.2f}s", (900, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FG, 2, cv2.LINE_AA)
            if args.blind_id:
                cv2.putText(img, f"Trial {args.blind_id}", (900, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT, 2, cv2.LINE_AA)
            else:
                cv2.putText(img, f"emotion: {emotion}", (900, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT, 2, cv2.LINE_AA)
                cv2.putText(img, f"intensity: {intensity}", (900, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT, 2, cv2.LINE_AA)

            cv2.rectangle(img, (250, 80), (860, 150), (235, 235, 235), -1)
            cv2.rectangle(img, (250, 80), (860, 150), (210, 210, 210), 1)
            cv2.putText(img, transcript, (280, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, FG, 2, cv2.LINE_AA)

            if len(motor_names) > 0 and len(motors) > i:
                draw_motor_panel(img, motor_names, motors[i])

            img[wave_y:wave_y + wave_h, wave_x:wave_x + wave_w] = wave_img
            cv2.rectangle(img, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (190, 190, 190), 1)
            cv2.putText(img, "audio waveform", (wave_x, wave_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)

            cursor_x = wave_x + int(np.clip(t / max(duration, 1e-6), 0.0, 1.0) * (wave_w - 1))
            cv2.line(img, (cursor_x, wave_y), (cursor_x, wave_y + wave_h - 1), CURSOR, 2, cv2.LINE_AA)
            writer.write(img)

            if i % 30 == 0 or i == total_frames - 1:
                print(f"[INFO] progress: {i+1}/{total_frames}")

        writer.release()
        sf.write(str(temp_audio_path), wav, wav_sr)

        cmd = [
            args.ffmpeg_bin,
            "-y",
            "-i", str(temp_video_path),
            "-i", str(temp_audio_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(output_video),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"[OK] saved video: {output_video}")


if __name__ == "__main__":
    main()
