"""Sample-clock emotion envelopes and crossfades, independent of chunk arrival."""
from dataclasses import asdict, dataclass, replace

import numpy as np

from .emotion_control import EmotionControl, finite_nonnegative
from .emotion_latent import LatentOptions
from .arkit52_to_motor import SpeechGate
from .settings import EXPLICIT_EMOTION_LABELS

# attack, sustain (None means hold), release, residual fraction; seconds.
TIMINGS = {
    'amazement': (.12, .25, .9, .15), 'anger': (.15, None, .6, .35),
    'cheekiness': (.25, None, .7, .2), 'disgust': (.25, None, .6, .15),
    'fear': (.15, None, .8, .2), 'grief': (.7, None, 1.8, .2),
    'joy': (.35, None, .8, .2), 'outofbreath': (.3, None, .8, .2),
    'pain': (.15, None, .8, .2), 'sadness': (.6, None, 1.5, .25),
    'neutral': (.3, None, .5, 0.),
}


@dataclass(frozen=True)
class TrackOptions:
    enabled: bool = False
    attack: object = None
    sustain: object = None
    release: object = None
    crossfade: float = .4
    ramp: float = .25

    def patch(self, meta, latent):
        values = {}
        if any(k in meta for k in ('emotion_mix', 'emotion_va')) or ('emotion_latent_mode' in meta and latent.mode != 'off'):
            values['enabled'] = True
        if 'emotion_track' in meta:
            v = meta['emotion_track']
            if not isinstance(v, (str, bool)) or v not in (True, False, 'true', 'false'):
                raise ValueError('emotion_track must be true/false')
            values['enabled'] = v is True or v == 'true'
        for name in ('attack', 'sustain', 'release', 'crossfade', 'ramp'):
            key = 'emotion_' + name + '_ms'
            if key in meta:
                value = meta[key]
                if value is None or value == 'null':
                    if name in ('crossfade', 'ramp'):
                        raise ValueError(key + ' cannot be null')
                    values[name] = None
                else:
                    value = finite_nonnegative(value, key) / 1000
                    if value > 60:
                        raise ValueError(key + ' must be <= 60000')
                    values[name] = value
        return replace(self, **values)


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


class EmotionTrack:
    def __init__(self, library=None):
        self.library = library
        self.latent = LatentOptions()
        self.options = TrackOptions()
        self.events = []
        self.gate = SpeechGate()
        self.gate.rms_floor, self.gate.rms_peak = 0., .03
        self.last_time = -1.
        self.ever_enabled = False

    def schedule(self, control, meta, time_s):
        latent = self.latent.patch(meta)
        options = self.options.patch(meta, latent)
        if latent.mode != 'off':
            if self.library is None:
                raise ValueError('implicit_emo_db.npz is required for latent mode')
            if control.implicit_vector is not None:
                raise ValueError('Raw implicit vector and latent library mode are mutually exclusive; set emotion_latent_mode=off')
        if self.events and time_s < self.events[-1]['time']:
            raise ValueError('Emotion events must be chronological')
        if self.events and (control, latent, options) == self.events[-1]['request']:
            return
        start, generated, old_bias = self._value(time_s) if self.events else (np.zeros(26), np.zeros(16), control.bias_scale)
        previous = self.events[-1] if self.events else None
        weights = control.explicit()
        label = EXPLICIT_EMOTION_LABELS[int(np.argmax(weights))] if weights.any() else 'neutral'
        attack, sustain, release, residual = TIMINGS[label]
        attack = attack if options.attack is None else options.attack
        sustain = sustain if options.sustain is None else options.sustain
        release = release if options.release is None else options.release
        if previous is None or not np.any(start):
            transition = attack
        elif not np.any(control.vector()):
            transition = previous['release']
        elif (previous['control'].implicit_vector == control.implicit_vector
              and previous['latent'] == latent
              and np.allclose(previous['control'].explicit() / max(previous['control'].intensity, 1e-30),
                              weights / max(control.intensity, 1e-30))):
            transition = options.ramp
        else:
            transition = options.crossfade
        event = dict(time=float(time_s), control=control, latent=latent, options=options,
                     request=(control, latent, options), start=start, generated=generated,
                     old_bias=old_bias, transition=transition, sustain=sustain,
                     release=release, residual=residual)
        # Same-sample updates replace an event, not add unbounded duplicate entries.
        if previous is not None and previous['time'] == time_s:
            self.events[-1] = event
        else:
            self.events.append(event)
        self.latent, self.options = latent, options
        self.ever_enabled |= options.enabled or latent.mode != 'off'

    def metadata(self):
        return dict(emotion_track_options=asdict(self.options),
                    emotion_latent_options=asdict(self.latent),
                    emotion_timeline_encoding='p16_sample_clock_v1')

    def _event(self, time_s):
        index = max(0, int(np.searchsorted([e['time'] for e in self.events], time_s, side='right')) - 1)
        return index, self.events[index]

    def _value(self, time_s):
        _, event = self._event(time_s)
        control, opt = event['control'], event['options']
        age = max(0., time_s - event['time'])
        vector = control.vector().astype(np.float64)
        generated = np.zeros(16)
        if event['latent'].mode != 'off':
            base_weights = control.explicit().astype(np.float64) / max(control.intensity, 1e-30)
            generated = self.library.vector(base_weights, age, event['latent']).astype(np.float64) * control.intensity
            # Speech suppression is applied AFTER temporal blending, to generated
            # actor latents only. P06 raw vectors remain exact passthrough.
        if not opt.enabled:
            vector[:16] += generated
            return vector, generated * (1-event['latent'].speech_scale), control.bias_scale
        transition = event['transition']
        alpha = smooth(age / transition) if transition > 0 else 1.
        envelope = 1.
        if event['sustain'] is not None:
            tail = age - transition - event['sustain']
            if tail > 0:
                envelope = event['residual'] + (1-event['residual']) * (1-smooth(tail/max(event['release'], 1e-9)))
        vector[:16] += generated
        vector *= envelope
        suppressible = generated * (1-event['latent'].speech_scale) * envelope
        return (event['start']*(1-alpha) + vector*alpha,
                event['generated']*(1-alpha) + suppressible*alpha,
                event['old_bias']*(1-alpha) + control.bias_scale*alpha)

    def sample(self, time_s, audio_rms):
        if not self.events:
            return EmotionControl()
        if not self.ever_enabled:
            # Preserve P06's immediate chunk control semantics unless P16 enabled.
            self.events[:] = self.events[-1:]
            return self.events[-1]['control']
        if time_s < self.last_time:
            raise ValueError('EmotionTrack.sample requires monotonic audio timestamps')
        self.last_time = time_s
        index, event = self._event(time_s)
        vector, suppressible, bias = self._value(time_s)
        gate = self.gate.update(audio_rms, fallback=0.)
        vector[:16] -= suppressible * gate
        # Keep only the event needed for past-window lookback and queued future ones.
        if index:
            del self.events[:index]
        return EmotionControl(event['control'].emotion, 1., tuple(vector[16:]), tuple(vector[:16]), bias)
