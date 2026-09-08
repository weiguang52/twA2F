"""Read-only actor latent library. No pickle, per-session RNG, or network access."""
from dataclasses import dataclass, replace
import json
import math
from types import MappingProxyType

import numpy as np

from .emotion_control import finite_nonnegative
from .settings import EXPLICIT_EMOTION_LABELS

# Engineering coordinates, NOT actor annotations or a learned psychological model.
VA_COORDINATES = np.asarray([
    [.3, .9], [-.8, .8], [.7, .4], [-.8, .2], [-.8, .9],
    [-.9, -.4], [.9, .5], [-.1, .8], [-.9, .5], [-.7, -.7],
], dtype=np.float64)


def va_weights(value):
    if isinstance(value, str):
        value = json.loads(value)
    a = np.asarray(value)
    if a.shape != (2,) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
        raise ValueError('emotion_va must be [valence, arousal] in [-1,1]')
    # Smooth normalized radial basis interpolation in the ten-anchor convex hull.
    logits = -np.sum((VA_COORDINATES - a) ** 2, axis=1) / (2 * .45 ** 2)
    weights = np.exp(logits - logits.max())
    return tuple(weights / weights.sum())


@dataclass(frozen=True)
class LatentOptions:
    mode: str = 'off'
    gain: float = .5
    speech_scale: float = .25
    fps: float = 30.0
    seed: int = 0

    def patch(self, meta):
        values = {}
        for field in ('mode', 'gain', 'speech_scale', 'fps', 'seed'):
            key = 'emotion_latent_' + field
            if key in meta:
                values[field] = meta[key]
        result = replace(self, **values)
        if result.mode not in ('off', 'centroid', 'sequence', 'random'):
            raise ValueError('emotion_latent_mode must be off/centroid/sequence/random')
        numbers = {}
        for name, maximum in (('gain', 2), ('speech_scale', 1), ('fps', 120)):
            x = finite_nonnegative(getattr(result, name), 'emotion_latent_' + name)
            if x > maximum or (name == 'fps' and x == 0):
                raise ValueError('emotion_latent_' + name + ' outside supported range')
            numbers[name] = x
        seed = finite_nonnegative(result.seed, 'emotion_latent_seed')
        if seed != int(seed) or seed > 2**32 - 1:
            raise ValueError('emotion_latent_seed must be a uint32 integer')
        return replace(result, **numbers, seed=int(seed))


class EmotionLatentLibrary:
    def __init__(self, path):
        self.path = str(path)
        with np.load(path, allow_pickle=False) as db:
            vectors = np.asarray(db['emo_db'], dtype=np.float32)
            names = [x.decode('utf8') if isinstance(x, bytes) else str(x) for x in db['emo_spec_names']]
            starts, sizes = db['emo_spec_start'], db['emo_spec_size']
            if vectors.ndim != 2 or vectors.shape[1] != 16 or not np.isfinite(vectors).all():
                raise ValueError('Invalid implicit emo_db: expected finite [N,16]')
            if starts.shape != (len(names),) or sizes.shape != starts.shape or len(set(names)) != len(names):
                raise ValueError('Invalid latent spec index')
            if starts.dtype.kind not in 'iu' or sizes.dtype.kind not in 'iu':
                raise ValueError('Latent spec offsets must be integers')
            specs = {}
            for name, start, size in zip(names, starts, sizes):
                if start < 0 or size <= 0 or int(start) + int(size) > len(vectors):
                    raise ValueError('Invalid latent spec slice: ' + name)
                segment = vectors[int(start):int(start)+int(size)].copy()
                segment.flags.writeable = False
                specs[name] = segment
        self.specs = MappingProxyType(specs)
        self.sequences = tuple(specs['mute_' + e] for e in EXPLICIT_EMOTION_LABELS)
        self.anchors = np.stack([s.mean(axis=0, dtype=np.float64) for s in self.sequences]).astype(np.float32)
        self.anchors.flags.writeable = False

    def vector(self, weights, time_s, options):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (10,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('Latent weights must be finite nonnegative [10]')
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Latent time must be finite and nonnegative')
        total = weights.sum()
        if total == 0 or options.mode == 'off':
            return np.zeros(16, dtype=np.float32)
        # Preserve sub-unit mixture strength, normalize only sums > 1 to avoid
        # silently leaving the actor anchor hull. Gain is a separate calibration.
        weights = weights / max(1., total)
        if options.mode == 'centroid':
            points = self.anchors
        else:
            position = time_s * options.fps
            frame, alpha = int(position), position % 1
            points = []
            for index, segment in enumerate(self.sequences):
                if options.mode == 'sequence':
                    a, b = frame % len(segment), (frame + 1) % len(segment)
                else:
                    # Counter-based seeded samples: deterministic across chunking,
                    # interleaved sessions and seeks; interpolate between samples.
                    a = int(np.random.default_rng(np.random.SeedSequence([options.seed, index, frame])).integers(len(segment)))
                    b = int(np.random.default_rng(np.random.SeedSequence([options.seed, index, frame + 1])).integers(len(segment)))
                points.append(segment[a] * (1-alpha) + segment[b] * alpha)
            points = np.asarray(points)
        return np.asarray(weights @ points * options.gain, dtype=np.float32)
