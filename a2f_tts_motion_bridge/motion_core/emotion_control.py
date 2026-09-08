"""Validated Claire 2.3 emotion controls shared by protocol, model and retargeting."""
from dataclasses import dataclass, replace
import json
import math
from typing import Mapping, Optional, Tuple

import numpy as np

from .settings import EXPLICIT_EMOTION_LABELS, EMOTION_ALIASES, EMOTION_BIAS_SCALE

EXPLICIT_KEY = 'emotion_explicit_weights'
IMPLICIT_KEY = 'emotion_implicit_vector'
BIAS_KEY = 'emotion_bias_scale'


def canonical_emotion(value: str) -> str:
    name = str(value or 'neutral').strip().lower()
    name = EMOTION_ALIASES.get(name, name)
    if name not in ('neutral', *EXPLICIT_EMOTION_LABELS):
        raise ValueError(f'Unknown emotion: {value!r}')
    return name


def mixture_weights(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError('emotion_mix must be a JSON object, e.g. {"joy":0.6,"amazement":0.4}') from exc
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('emotion_mix must be an emotion-to-weight object')
    result = np.zeros(10, dtype=np.float64)
    for name, weight in value.items():
        label = canonical_emotion(name)
        weight = finite_nonnegative(weight, 'emotion_mix weight')
        if label == 'neutral':
            if weight:
                raise ValueError('neutral has no explicit anchor; use an empty mix to release')
        else:
            result[EXPLICIT_EMOTION_LABELS.index(label)] += weight
    return _vector(result, 10, 'emotion_mix', True)


def finite_nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite nonnegative number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite nonnegative number') from exc
    if not math.isfinite(result) or result < 0 or result > np.finfo(np.float32).max:
        raise ValueError(f'{name} must be finite, nonnegative and representable as float32')
    return result


def _vector(value, length, name, nonnegative=False):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name} must be a JSON array or null') from exc
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'{name} must contain exactly {length} numbers')
    arr = np.asarray(value)
    if arr.shape != (length,) or arr.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must contain exactly {length} numbers')
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all() or np.any(np.abs(arr) > np.finfo(np.float32).max):
        raise ValueError(f'{name} must contain finite float32 values')
    if nonnegative and np.any(arr < 0):
        raise ValueError(f'{name} weights must be nonnegative')
    return tuple(float(x) for x in arr)


@dataclass(frozen=True)
class EmotionControl:
    emotion: str = 'neutral'
    intensity: float = 1.0
    explicit_weights: Optional[Tuple[float, ...]] = None
    implicit_vector: Optional[Tuple[float, ...]] = None
    bias_scale: float = EMOTION_BIAS_SCALE

    def __post_init__(self):
        object.__setattr__(self, 'emotion', canonical_emotion(self.emotion))
        object.__setattr__(self, 'intensity', finite_nonnegative(self.intensity, 'intensity'))
        object.__setattr__(self, 'bias_scale', finite_nonnegative(self.bias_scale, BIAS_KEY))
        object.__setattr__(self, 'explicit_weights', _vector(self.explicit_weights, 10, EXPLICIT_KEY, True))
        object.__setattr__(self, 'implicit_vector', _vector(self.implicit_vector, 16, IMPLICIT_KEY))
        if self.explicit_weights and any(x * self.intensity > np.finfo(np.float32).max for x in self.explicit_weights):
            raise ValueError('scaled explicit emotion weights overflow float32')

    def explicit(self) -> np.ndarray:
        if self.explicit_weights is not None:
            values = np.asarray(self.explicit_weights, dtype=np.float64)
        else:
            values = np.zeros(10, dtype=np.float64)
            # A supplied latent vector alone means latent-only conditioning.
            if self.implicit_vector is None and self.emotion != 'neutral':
                values[EXPLICIT_EMOTION_LABELS.index(self.emotion)] = 1.0
        return (values * self.intensity).astype(np.float32)

    def vector(self) -> np.ndarray:
        result = np.zeros(26, dtype=np.float32)
        if self.implicit_vector is not None:
            result[:16] = self.implicit_vector
        result[16:] = self.explicit()
        return result

    def patch(self, meta: Mapping[str, str]) -> 'EmotionControl':
        selectors = [k for k in ('emotion_mix', 'emotion_va', EXPLICIT_KEY) if k in meta]
        if len(selectors) > 1:
            raise ValueError('Use only one of emotion_mix, emotion_va, emotion_explicit_weights per update')
        meta = dict(meta)
        if 'emotion_mix' in meta:
            meta[EXPLICIT_KEY] = mixture_weights(meta.pop('emotion_mix'))
            if IMPLICIT_KEY not in meta:
                meta[IMPLICIT_KEY] = None
        if 'emotion_va' in meta:
            from .emotion_latent import va_weights
            meta[EXPLICIT_KEY] = va_weights(meta.pop('emotion_va'))
            if IMPLICIT_KEY not in meta:
                meta[IMPLICIT_KEY] = None
        values = {}
        # A new label is a new control selection; do not retain stale vectors.
        if 'emotion' in meta:
            values.update(emotion=meta['emotion'], explicit_weights=None, implicit_vector=None)
        if 'intensity' in meta:
            values['intensity'] = meta['intensity']
        for key, field in ((EXPLICIT_KEY, 'explicit_weights'), (IMPLICIT_KEY, 'implicit_vector'), (BIAS_KEY, 'bias_scale')):
            if key in meta:
                values[field] = meta[key]
        return replace(self, **values)

    def metadata(self):
        return dict(emotion=self.emotion, intensity=self.intensity,
                    emotion_vector=self.vector().tolist(), emotion_bias_scale=self.bias_scale,
                    emotion_encoding='claire26_implicit16_explicit10_v1')
