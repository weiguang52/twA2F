#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render current-head 13-DOF normalized retarget .npy to an mp4 preview.

This renderer uses the logical current-head DOFs directly:
- four-bar y modules: 0 lower, 0.5 neutral, 1 upper
- mouth corner x/y: 0.5 neutral, x outward positive, y upward positive
- jaw_y: 0 closed, 1 open/down

It does not require final motor-angle conversion or linkage IK.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

FPS = 30
W, H = 1280, 1100
BG = (245, 245, 245)
FG = (30, 30, 30)
GRID = (210, 210, 210)
ACCENT = (40, 80, 220)
OK_BAR = (80, 170, 90)
WARN_BAR = (220, 150, 60)
WAVE = (80, 80, 80)
CURSOR = (220, 80, 80)

DOF_NAMES = [
    "right_outer_brow_y",
    "left_inner_brow_y",
    "right_inner_brow_y",
    "left_outer_brow_y",
    "right_upper_lid_y",
    "left_upper_lid_y",
    "right_lower_lid_y",
    "left_lower_lid_y",
    "right_mouth_x",
    "right_mouth_y",
    "left_mouth_x",
    "left_mouth_y",
    "jaw_y",
]

DEFAULT_NEUTRAL = {name: 0.5 for name in DOF_NAMES}
DEFAULT_NEUTRAL["jaw_y"] = 0.0


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def interp_track(src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray, fill_value=0.0) -> np.ndarray:
    src_t = np.asarray(src_t, dtype=np.float32)
    src_v = np.asarray(src_v, dtype=np.float32)
    dst_t = np.asarray(dst_t, dtype=np.float32)
    if len(dst_t) == 0:
        return np.zeros((0,) + src_v.shape[1:], dtype=np.float32)
    if len(src_t) == 0 or src_v.size == 0:
        if src_v.ndim <= 1:
            return np.full((len(dst_t),), fill_value, dtype=np.float32)
        return np.full((len(dst_t), src_v.shape[1]), fill_value, dtype=np.float32)
    if src_v.ndim == 1:
        return np.interp(dst_t, src_t, src_v, left=fill_value, right=fill_value).astype(np.float32)
    out = np.zeros((len(dst_t), src_v.shape[1]), dtype=np.float32)
    for j in range(src_v.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, src_v[:, j], left=fill_value, right=fill_value).astype(np.float32)
    return out


def prepare_waveform_image(wav, width, height):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1e-8:
        wav = wav / peak
    img = np.full((height, width, 3), 252, dtype=np.uint8)
    mid = height // 2
    cv2.line(img, (0, mid), (width - 1, mid), (210, 210, 210), 1, cv2.LINE_AA)
    if len(wav) == 0:
        return img
    samples_per_col = max(1, len(wav) // width)
    for x in range(width):
        seg = wav[x * samples_per_col: min(len(wav), (x + 1) * samples_per_col)]
        if len(seg) == 0:
            continue
        y1 = int(mid - float(np.max(seg)) * height * 0.42)
        y2 = int(mid - float(np.min(seg)) * height * 0.42)
        cv2.line(img, (x, y1), (x, y2), WAVE, 1, cv2.LINE_AA)
    return img


def load_payload(path: Path):
    data = np.load(path, allow_pickle=True).item()
    meta = dict(data.get("meta", {}))
    wav = np.asarray(data.get("audio_16k_mono", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    sr = int(meta.get("target_sr", 16000))
    duration = len(wav) / float(sr) if len(wav) else 0.0
    total_frames = max(1, int(round(duration * FPS))) if duration > 0 else 1
    video_t = np.arange(total_frames, dtype=np.float32) / float(FPS)

    if "frame_times_30fps" in data and len(data["frame_times_30fps"]) > 0:
        src_t = np.asarray(data["frame_times_30fps"], dtype=np.float32)
        src_m = np.asarray(data.get("motor_values_30fps", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    else:
        src_t = np.asarray(data.get("frame_times_native", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        src_m = np.asarray(data.get("motor_values_native", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)

    names = list(meta.get("motor_names", []))
    if not names and src_m.shape[1] == len(DOF_NAMES):
        names = DOF_NAMES.copy()

    if len(names) == 0 or src_m.size == 0:
        motors = np.zeros((len(video_t), 0), dtype=np.float32)
    else:
        # Fill outside A2F track with logical neutral values.
        motors = np.zeros((len(video_t), len(names)), dtype=np.float32)
        for j, name in enumerate(names):
            fill = DEFAULT_NEUTRAL.get(name, 0.5)
            motors[:, j] = interp_track(src_t, src_m[:, j], video_t, fill_value=fill)
    return data, meta, duration, video_t, names, motors, wav, sr


def dof_row_to_dict(names, row):
    d = DEFAULT_NEUTRAL.copy()
    for i, name in enumerate(names):
        if i < len(row):
            d[name] = clip01(float(row[i]))
    return d


def front_view_dofs(d):
    """Screen-left is robot-right. Only transform drawing, never motor data."""
    return {key: d[('right_'+key[5:] if key.startswith('left_') else
                   'left_'+key[6:] if key.startswith('right_') else key)]
            for key in d}


def draw_face(img, d):
    d = front_view_dofs(d)
    cx, cy = W // 2 - 130, 450
    cv2.ellipse(img, (cx, cy), (225, 305), 0, 0, 360, GRID, 2, cv2.LINE_AA)

    # Four-bar vertical modules.
    # Bigger y_norm means visually higher except upper eyelid, where y_norm also means anatomically higher/open.
    lb_o = d["left_outer_brow_y"]
    lb_i = d["left_inner_brow_y"]
    rb_i = d["right_inner_brow_y"]
    rb_o = d["right_outer_brow_y"]
    lu = d["left_upper_lid_y"]
    ru = d["right_upper_lid_y"]
    ll = d["left_lower_lid_y"]
    rl = d["right_lower_lid_y"]

    lx, rx = cx - 135, cx + 135
    eye_y = cy - 55
    brow_base_y = cy - 185

    def draw_brow_segment(x_center, outer_y, inner_y, side):
        # side=-1 left, side=+1 right. outer is farther from center.
        x_outer = x_center - 65 if side < 0 else x_center + 65
        x_inner = x_center + 65 if side < 0 else x_center - 65
        y_outer = int(brow_base_y - (outer_y - 0.5) * 180)
        y_inner = int(brow_base_y - (inner_y - 0.5) * 180)
        cv2.line(img, (x_outer, y_outer), (x_inner, y_inner), FG, 5, cv2.LINE_AA)

    draw_brow_segment(lx, lb_o, lb_i, side=-1)
    draw_brow_segment(rx, rb_o, rb_i, side=+1)

    def draw_eye(x, upper_y_norm, lower_y_norm):
        # upper_y_norm higher => upper lid higher/open; lower_y_norm higher => lower lid moves up/closed.
        upper_y = int(eye_y - 34 - (upper_y_norm - 0.5) * 70)
        lower_y = int(eye_y + 34 - (lower_y_norm - 0.5) * 55)
        lower_y = max(lower_y, upper_y + 4)
        eye_w = 62
        cv2.ellipse(img, (x, (upper_y + lower_y) // 2), (eye_w, max(3, (lower_y - upper_y) // 2)), 0, 0, 360, FG, 2, cv2.LINE_AA)
        cv2.line(img, (x - eye_w, upper_y), (x + eye_w, upper_y), FG, 3, cv2.LINE_AA)
        cv2.line(img, (x - eye_w + 6, lower_y), (x + eye_w - 6, lower_y), FG, 2, cv2.LINE_AA)
        if lower_y - upper_y > 14:
            cv2.circle(img, (x, eye_y), 8, ACCENT, -1, cv2.LINE_AA)

    draw_eye(lx, lu, ll)
    draw_eye(rx, ru, rl)

    # Mouth five-bar logical corner targets.
    jaw = d["jaw_y"]
    lmx, lmy = d["left_mouth_x"], d["left_mouth_y"]
    rmx, rmy = d["right_mouth_x"], d["right_mouth_y"]
    mouth_cy = cy + 185
    half_w = 48 + 30 * jaw
    # left/right local x: outward from center.
    left = np.array([cx - half_w - (lmx - 0.5) * 90, mouth_cy - (lmy - 0.5) * 90], dtype=np.float32)
    right = np.array([cx + half_w + (rmx - 0.5) * 90, mouth_cy - (rmy - 0.5) * 90], dtype=np.float32)
    top_mid = np.array([cx, mouth_cy - 8 - 20 * (0.5 * ((lmy - 0.5) + (rmy - 0.5)))], dtype=np.float32)
    bottom_mid = np.array([cx, mouth_cy + 8 + jaw * 95], dtype=np.float32)
    upper_pts = np.array([left, top_mid, right], dtype=np.int32).reshape(-1, 1, 2)
    lower_pts = np.array([left, bottom_mid, right], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [upper_pts], False, FG, 4, cv2.LINE_AA)
    cv2.polylines(img, [lower_pts], False, FG, 4, cv2.LINE_AA)
    if jaw > 0.08:
        cv2.ellipse(img, (cx, int(mouth_cy + 16 + jaw * 38)), (max(16, int(36 + jaw * 30)), max(5, int(jaw * 45))), 0, 0, 360, (40, 40, 60), -1, cv2.LINE_AA)

    # Jaw/chin direct vertical.
    cv2.circle(img, (cx, int(cy + 260 + jaw * 85)), 7, FG, -1, cv2.LINE_AA)


def draw_bar_panel(img, names, row):
    x, y0 = 900, 180
    w, h, gap = 285, 16, 37
    cv2.putText(img, "current head 13-DOF preview", (x, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, FG, 2, cv2.LINE_AA)
    d = dof_row_to_dict(names, row)
    order = DOF_NAMES
    for i, name in enumerate(order):
        y = y0 + i * gap
        val = d[name]
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 1)
        # Draw neutral marker for centered coordinates.
        neutral = DEFAULT_NEUTRAL.get(name, 0.5)
        nx = x + int(w * neutral)
        cv2.line(img, (nx, y - 2), (nx, y + h + 2), WARN_BAR, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + int(w * val), y + h), OK_BAR, -1)
        cv2.putText(img, f"{name}: {val:.3f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, FG, 1, cv2.LINE_AA)


def parse_name(stem):
    if "_" not in stem:
        return stem, "?"
    a, b = stem.rsplit("_", 1)
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--ffmpeg_bin", default="ffmpeg")
    args = parser.parse_args()

    input_path = Path(args.input_npy).resolve()
    output_path = Path(args.output_video).resolve()
    data, meta, duration, video_t, names, motors, wav, sr = load_payload(input_path)
    if motors.shape[1] != len(DOF_NAMES):
        print(f"[WARN] expected 13 DOFs; got motors shape {motors.shape}, names={names}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wave_x, wave_y, wave_w, wave_h = 70, 900, W - 140, 140
    wave_img = prepare_waveform_image(wav, wave_w, wave_h)
    emotion, intensity = parse_name(input_path.stem)
    retarget_version = str(meta.get("retarget_version", "unknown"))
    transcript = args.transcript or f"{input_path.stem} | {retarget_version}"

    with tempfile.TemporaryDirectory(prefix="current_head_viz_") as tmp:
        temp_video = Path(tmp) / "silent.mp4"
        temp_audio = Path(tmp) / "audio.wav"
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        print(f"[INFO] render {input_path.name} -> {output_path}")
        print(f"[INFO] frames={len(video_t)}, duration={duration:.3f}s, wav_sr={sr}, dofs={motors.shape[1]}")
        for i, t in enumerate(video_t):
            img = np.full((H, W, 3), BG, dtype=np.uint8)
            row = motors[i] if len(motors) > i else np.zeros(len(names), dtype=np.float32)
            d = dof_row_to_dict(names, row)
            draw_face(img, d)
            draw_bar_panel(img, names, row)
            cv2.putText(img, "A2F -> current head retarget preview", (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2, cv2.LINE_AA)
            cv2.putText(img, f"time: {t:0.2f}s / {duration:0.2f}s", (900, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FG, 2, cv2.LINE_AA)
            cv2.putText(img, f"source: {input_path.name}", (40, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, FG, 1, cv2.LINE_AA)
            cv2.putText(img, f"retarget: {retarget_version}", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FG, 1, cv2.LINE_AA)
            cv2.rectangle(img, (250, 80), (860, 150), (235, 235, 235), -1)
            cv2.rectangle(img, (250, 80), (860, 150), GRID, 1)
            cv2.putText(img, transcript, (280, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.78, FG, 2, cv2.LINE_AA)
            img[wave_y:wave_y + wave_h, wave_x:wave_x + wave_w] = wave_img
            cv2.rectangle(img, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (190, 190, 190), 1)
            cv2.putText(img, "audio waveform", (wave_x, wave_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 2, cv2.LINE_AA)
            cursor_x = wave_x + int(np.clip(t / max(duration, 1e-6), 0.0, 1.0) * (wave_w - 1))
            cv2.line(img, (cursor_x, wave_y), (cursor_x, wave_y + wave_h - 1), CURSOR, 2, cv2.LINE_AA)
            writer.write(img)
            if i % 30 == 0 or i == len(video_t) - 1:
                print(f"[INFO] progress: {i + 1}/{len(video_t)}")
        writer.release()
        sf.write(str(temp_audio), wav, sr)
        cmd = [
            args.ffmpeg_bin, "-y", "-i", str(temp_video), "-i", str(temp_audio),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"[OK] saved video: {output_path}")


if __name__ == "__main__":
    main()
