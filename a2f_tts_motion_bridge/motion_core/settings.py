# -*- coding: utf-8 -*-
"""Settings for current-head 13-DOF normalized retarget preview.

Use this file as:
    a2f_tts_motion_bridge/motion_core/settings.py

The output named MOTOR_CFG is kept for compatibility with existing recorder/session code.
These are logical mechanism DOFs, not final motor angles.
"""

TARGET_SR = 16000
WINDOW = 8320
HOP = 1600
NUM_FEATURES = 169
EXPORT_FPS = 30.0

# Claire 2.3 network_info.json order. Explicit block occupies absolute dims 16:26.
EXPLICIT_EMOTION_LABELS = [
    "amazement", "anger", "cheekiness", "disgust", "fear",
    "grief", "joy", "outofbreath", "pain", "sadness",
]
EMOTION_LABELS = ["neutral", *EXPLICIT_EMOTION_LABELS]
EMOTION_ALIASES = {
    "happy": "joy", "smile": "joy", "laugh": "joy",
    "sad": "sadness", "sorrow": "sadness",
    "surprise": "amazement", "surprised": "amazement", "angry": "anger",
    "out_of_breath": "outofbreath", "out-of-breath": "outofbreath",
}
EMOTION_BIAS_SCALE = 0.35

# 8 four-bar vertical modules + 2D right mouth corner + 2D left mouth corner + direct jaw = 13.
# Four-bar y and mouth x/y use neutral=0.5. Jaw uses closed/rest=0.0.
MOTOR_CFG = {
    "right_outer_brow_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "left_inner_brow_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "right_inner_brow_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "left_outer_brow_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "right_upper_lid_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "left_upper_lid_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "right_lower_lid_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "left_lower_lid_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "right_mouth_x": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "right_mouth_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "left_mouth_x": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "left_mouth_y": {"neutral": 0.5, "min": 0.0, "max": 1.0},
    "jaw_y": {"neutral": 0.0, "min": 0.0, "max": 1.0},
}
