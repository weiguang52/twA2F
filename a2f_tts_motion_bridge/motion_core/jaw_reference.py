"""Silence-referenced model jaw activity and sample-clock audio gating.

Latent distance is an activity proxy, not anatomical jaw angle calibration.
It avoids treating the first spoken frame as neutral or discarding negative
latent displacements. Audio only attenuates activity; it never forces opening.
"""
import numpy as np


def jaw_activity(vector, reference):
    vector=np.asarray(vector,dtype=np.float32)
    reference=np.asarray(reference,dtype=np.float32)
    if vector.shape!=(15,) or reference.shape!=(15,) or not np.isfinite(vector).all() or not np.isfinite(reference).all():
        raise ValueError('Jaw vectors must contain 15 finite values')
    distance=float(np.linalg.norm(vector-reference))
    return min(.85,distance/(distance+1.5))


class AudioJawGate:
    """20 ms centered RMS; shared chunk list, no audio copies or random state."""
    def __init__(self,chunks,sr=16000):
        self.chunks=chunks;self.sr=sr;self.ends=[];self.total=0

    def sample(self,time_s):
        if not np.isfinite(time_s) or time_s<0:raise ValueError('Invalid audio time')
        while len(self.ends)<len(self.chunks):
            self.total+=len(self.chunks[len(self.ends)]);self.ends.append(self.total)
        center=int(time_s*self.sr);radius=int(.01*self.sr)
        lo=max(0,center-radius);hi=min(self.total,center+radius)
        if hi<=lo:return 0.
        i=int(np.searchsorted(self.ends,lo,side='right'));energy=0.;count=0
        while i<len(self.ends):
            start=self.ends[i-1] if i else 0
            a=max(lo,start)-start;b=min(hi,self.ends[i])-start
            if b>a:
                segment=np.asarray(self.chunks[i][a:b],dtype=np.float64)
                energy+=float(np.dot(segment,segment));count+=b-a
            if self.ends[i]>=hi:break
            i+=1
        rms=np.sqrt(energy/max(count,1))
        x=float(np.clip((rms-.003)/.012,0.,1.))
        return x*x*(3-2*x)
