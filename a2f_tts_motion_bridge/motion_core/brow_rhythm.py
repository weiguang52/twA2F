"""Deterministic, aperiodic brow tension envelopes on the output clock.

This is a procedural residual, not recovered actor performance. Scaling the
existing signed pose preserves anger/sadness direction, asymmetry and neutral.
"""
import math


def brow_rhythm_offsets(base, time_s, seed=0, gain=.35):
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError('Brow rhythm time must be finite and nonnegative')
    if not math.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError('Brow rhythm gain must be in [0,1]')
    # Independent hashed windows: no mutable RNG and no per-chunk phase reset.
    window = int(time_s // 4.)
    def uniform(index, salt):
        x = ((int(seed) ^ ((index+1)*0x9e3779b1) ^ salt) & 0xffffffff)
        x ^= x >> 16; x = (x * 0x85ebca6b) & 0xffffffff
        x ^= x >> 13; x = (x * 0xc2b2ae35) & 0xffffffff
        return ((x ^ (x >> 16)) & 0xffffffff) / 4294967296.
    envelope = 0.
    for index in (window-1, window):
        if index < 0: continue
        start = 4.*index + .3 + .8*uniform(index, 11)
        duration = 1.8 + 1.3*uniform(index, 23)
        phase = (time_s-start)/duration
        if 0 < phase < 1:
            # Broad relaxation, followed by a smaller tension rebound.
            envelope += (-1. if phase < .65 else .6) * math.sin(
                math.pi*(phase/.65 if phase < .65 else (phase-.65)/.35))**2
    return {key: max(-.06, min(.06, (float(value)-.5)*gain*envelope))
            for key,value in base.items() if 'brow' in key}
