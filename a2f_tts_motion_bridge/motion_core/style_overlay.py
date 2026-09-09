"""Analytic style/micro-event curves, evaluated at OUTPUT rate (>=30 Hz)."""
import math


STYLE_DURATIONS = {'smile': 1.2, 'laugh': .6, 'sigh': 2., 'sob': 1.2,
                   'brow_flash': .12, 'wink': .24, 'smile_suppress': .5}


def pulse(age, duration):
    if age <= 0 or age >= duration:
        return 0.
    return math.sin(math.pi * age / duration) ** 2


def style_offsets(kind, age, duration, weight=1., side='left', seed=0):
    """Return normalized-DOF deltas and unilateral closure envelopes."""
    if not 0 <= age < duration:
        return {}, {}
    env = pulse(age, duration) * weight
    delta, closure = {}, {}
    if kind in ('laugh', 'smile'):
        for s in ('left', 'right'):
            delta[s+'_mouth_y'] = .16 * env
            delta[s+'_mouth_x'] = .07 * env
            delta[s+'_upper_lid_y'] = -.08 * env
            delta[s+'_lower_lid_y'] = .10 * env
        if kind == 'laugh':
            # Short attack/release, 4.5 Hz carrier inside a 0.3--0.8s burst.
            burst = min(1., age/.05, (duration-age)/.08) * weight
            delta['jaw_y'] = .24 * burst * (.5-.5*math.cos(2*math.pi*4.5*age))
    elif kind == 'sigh':
        delta['jaw_y'] = .18 * env
        for s in ('left','right'): delta[s+'_mouth_x'] = -.03 * env
    elif kind == 'sob':
        # Instantaneous frequency 2.1--2.9 Hz; seeded phase, not frame RNG.
        phase = (seed % 997) / 997 * 2*math.pi
        carrier = 2*math.pi*2.5*age + (.4/.7)*math.sin(2*math.pi*.7*age+phase)
        delta['jaw_y'] = .14 * env * (.5-.5*math.cos(carrier))**2
        for s in ('left','right'):
            delta[s+'_inner_brow_y'] = .07*env
            delta[s+'_mouth_y'] = -.04*env
    elif kind == 'brow_flash':
        for s in ('left','right'):
            delta[s+'_outer_brow_y'] = .12*env
            delta[s+'_inner_brow_y'] = .08*env
    elif kind == 'wink':
        closure[side] = env
    elif kind == 'smile_suppress':
        phase = age/duration
        amount = (.10*pulse(phase,.45) if phase < .45 else -.05*pulse(phase-.45,.55))*weight
        for s in ('left','right'): delta[s+'_mouth_y'] = amount
    else:
        raise ValueError('Unknown style event: '+str(kind))
    return delta, closure
