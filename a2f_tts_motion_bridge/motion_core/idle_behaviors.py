"""Seeded refractory-Poisson blinks and tiny DOF-only listening/idle motion."""
import math
import numpy as np


def blink_envelope(age, slow=False):
    close, hold, release = (.16,.06,.32) if slow else (.045,.035,.14)
    if age < 0 or age >= close+hold+release:
        return 0.
    if age < close:
        x=age/close
    elif age < close+hold:
        return 1.
    else:
        x=1-(age-close-hold)/release
    return x*x*(3-2*x)


class IdleBehaviors:
    def __init__(self, seed=0):
        self.rng=np.random.default_rng(seed)
        self.blinks=[]
        self.next_blink=1.5+float(self.rng.exponential(1.5))

    def ensure_until(self,time_s):
        while self.next_blink <= time_s:
            at=self.next_blink
            self.blinks.append(at)
            if self.rng.random()<.15:
                self.blinks.append(at+.32)
            self.next_blink=at+1.5+float(self.rng.exponential(1.5))

    def sample(self,time_s,mode='off',auto_blink=False,gain=1.):
        self.ensure_until(time_s)
        closure=0.
        if auto_blink:
            pos=int(np.searchsorted(self.blinks,time_s,side='right'))
            for at in self.blinks[max(0,pos-2):pos]:
                closure=max(closure,blink_envelope(time_s-at))
        delta={}
        if mode in ('idle','listening'):
            breath=.5-.5*math.cos(2*math.pi*.23*time_s)
            delta['jaw_y']=.008*breath*gain
            for side in ('left','right'):
                delta[side+'_mouth_y']=(.010+.005*breath)*gain
            if mode=='listening':
                for side in ('left','right'):
                    delta[side+'_outer_brow_y']=.025*gain
                    delta[side+'_inner_brow_y']=.015*gain
                if auto_blink:
                    closure=max(closure,blink_envelope(time_s%6,slow=True))
        return delta,{'left':closure*gain,'right':closure*gain}
