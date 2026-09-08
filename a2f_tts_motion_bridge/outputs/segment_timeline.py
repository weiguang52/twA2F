import math
from typing import Optional

import numpy as np


def build_segment_times(start_ms: int, end_ms: int, fps: float) -> np.ndarray:
    """Build the fixed-rate, half-open timeline declared by one segment."""

    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    if end_ms < start_ms:
        raise ValueError(
            f"segment end_ms must be >= start_ms, got {start_ms}-{end_ms}"
        )

    duration_ms = int(end_ms) - int(start_ms)
    if duration_ms == 0:
        return np.zeros(0, dtype=np.float32)

    # Round halves up instead of relying on Python's banker rounding.  This is
    # the wire contract: each segment independently owns duration * fps frames.
    frame_count = max(1, int(math.floor(duration_ms * fps / 1000.0 + 0.5)))
    start_s = float(start_ms) / 1000.0
    offsets = np.arange(frame_count, dtype=np.float64) / fps
    return (start_s + offsets).astype(np.float32)


def resample_to_times(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
    *,
    empty_value: np.ndarray | float,
    left_value: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Interpolate values onto an explicit timeline with deterministic edges.

    Values before the first native frame use ``left_value`` (or
    ``empty_value``).  Values after the last native frame hold that frame.
    When no native frame exists, the complete result uses ``empty_value``.
    """

    source_times = np.asarray(source_times, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    source_values = np.asarray(source_values, dtype=np.float32)

    if source_values.ndim == 0:
        raise ValueError("source_values must have a frame dimension")
    if len(source_values) != len(source_times):
        raise ValueError(
            "source time/value length mismatch: "
            f"times={len(source_times)} values={len(source_values)}"
        )
    if len(source_times) > 1 and np.any(np.diff(source_times) <= 0.0):
        raise ValueError("source_times must be strictly increasing")
    if not np.all(np.isfinite(source_times)) or not np.all(np.isfinite(target_times)):
        raise ValueError("source_times and target_times must be finite")

    tail_shape = source_values.shape[1:]
    empty = np.asarray(empty_value, dtype=np.float32)
    try:
        empty = np.broadcast_to(empty, tail_shape).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"empty_value shape {empty.shape} cannot fill value shape {tail_shape}"
        ) from exc

    if len(target_times) == 0:
        return np.zeros((0,) + tail_shape, dtype=np.float32)
    if len(source_times) == 0:
        return np.broadcast_to(empty, (len(target_times),) + tail_shape).copy()

    left = empty if left_value is None else np.asarray(left_value, dtype=np.float32)
    try:
        left = np.broadcast_to(left, tail_shape).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"left_value shape {left.shape} cannot fill value shape {tail_shape}"
        ) from exc

    flat_source = source_values.reshape(len(source_times), -1)
    flat_left = left.reshape(-1)
    flat_result = np.empty((len(target_times), flat_source.shape[1]), dtype=np.float32)
    for column in range(flat_source.shape[1]):
        flat_result[:, column] = np.interp(
            target_times,
            source_times,
            flat_source[:, column],
            left=float(flat_left[column]),
            right=float(flat_source[-1, column]),
        ).astype(np.float32)
    return flat_result.reshape((len(target_times),) + tail_shape)
