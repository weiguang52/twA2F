"""Validated, precomputed text-face keyframes on the session output clock.

No LLM or file paths from a client are executed in the streaming loop.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .a2f169_to_arkit52 import ARKIT_52_NAMES

SCHEMA = 'a2f_text_face_v1'
POSE_NAMES = ('HeadYaw','HeadPitch','HeadRoll','LeftEyeYaw','LeftEyePitch',
              'LeftEyeRoll','RightEyeYaw','RightEyePitch','RightEyeRoll')
REGIONS = {
    'brows': tuple(s+'_'+p+'_brow_y' for s in ('left','right') for p in ('inner','outer')),
    'eyes': tuple(s+'_'+p+'_lid_y' for s in ('left','right') for p in ('upper','lower')),
}
REGIONS['upper_face'] = REGIONS['brows'] + REGIONS['eyes']


def finite(value, low, high, name):
    if isinstance(value, bool): raise ValueError(name+' must be numeric, not boolean')
    try: value = float(value)
    except (ValueError, TypeError): raise ValueError(name+' must be numeric') from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return value


def normalize_arkit(data):
    """Accept upstream capitalized names, reject missing/unknown/duplicate keys."""
    if not isinstance(data, dict): raise ValueError('ARKit coefficients must be an object')
    names = {n.lower(): n for n in ARKIT_52_NAMES}
    poses = {n.lower() for n in POSE_NAMES}
    result = {}; seen = set()
    for key, value in data.items():
        if not isinstance(key,str): raise ValueError('ARKit names must be strings')
        lower = key.lower()
        if lower in seen: raise ValueError('Duplicate ARKit channel: '+key)
        seen.add(lower)
        if lower in names:
            result[names[lower]] = finite(value,0,1,key)
        elif lower in poses:
            finite(value,-1,1,key)  # Valid but deliberately not sent to this robot.
        else: raise ValueError('Unknown ARKit channel: '+key)
    if set(result) != set(ARKIT_52_NAMES): raise ValueError('All 52 ARKit channels are required')
    return result


def smooth(x):
    x = float(np.clip(x,0,1))
    return x*x*(3-2*x)


class TextFaceClip:
    def __init__(self, document, sha256=''):
        if not isinstance(document,dict) or document.get('schema') != SCHEMA:
            raise ValueError('Unsupported text-face schema')
        self.duration = finite(document.get('duration_s'),.1,60,'duration_s')
        provenance = document.get('provenance')
        if not isinstance(provenance,dict) or not provenance.get('model'):
            raise ValueError('Text-face model provenance is required')
        self.provenance = json.loads(json.dumps(provenance,allow_nan=False))
        self.sha256 = sha256
        keys = document.get('keyframes')
        if not isinstance(keys,list) or not 1 <= len(keys) <= 64:
            raise ValueError('Expected 1..64 text-face keyframes')
        times=[]; poses=[]
        # Lazy import avoids behavior_layer <-> retargeter import cycles.
        from .arkit52_to_motor import arkit52_to_current_head_dofs
        for frame in keys:
            if not isinstance(frame,dict): raise ValueError('Invalid keyframe')
            t = finite(frame.get('time_s'),0,self.duration,'time_s')
            if times and t <= times[-1]: raise ValueError('Keyframe times must increase strictly')
            arkit = normalize_arkit(frame.get('arkit'))
            pose = arkit52_to_current_head_dofs(arkit, emotion='neutral', bias_scale=0.)
            times.append(t); poses.append([pose[k] for k in REGIONS['upper_face']])
        if times[0] != 0: raise ValueError('First keyframe must start at zero')
        self.times=np.array(times);self.poses=np.array(poses)
        self.times.flags.writeable=False;self.poses.flags.writeable=False

    def sample(self, age):
        if age < 0 or age >= self.duration:return None
        i=max(0,int(np.searchsorted(self.times,age,side='right'))-1)
        v=self.poses[i]
        if i+1<len(self.times):
            alpha=smooth((age-self.times[i])/(self.times[i+1]-self.times[i]))
            v=(1-alpha)*v+alpha*self.poses[i+1]
        return dict(zip(REGIONS['upper_face'],v))


class TextFaceTrack:
    def __init__(self):
        self.entries=[]
        self.cache={}

    def _load(self, ident):
        if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',ident):
            raise ValueError('text_face_clip must be a simple asset ID')
        if ident in self.cache:return self.cache[ident]
        directory=os.environ.get('A2F_TEXT_FACE_DIR')
        if not directory:raise ValueError('Operator must configure A2F_TEXT_FACE_DIR')
        root=Path(directory).resolve();path=(root/(ident+'.json')).resolve()
        if path.parent != root:raise ValueError('Text-face asset escapes configured directory')
        if path.stat().st_size > 1024*1024:raise ValueError('Text-face asset exceeds 1 MiB')
        raw=path.read_bytes()
        if len(raw)>1024*1024:raise ValueError('Text-face asset exceeds 1 MiB')
        return TextFaceClip(json.loads(raw),hashlib.sha256(raw).hexdigest())

    def prepare(self, meta, time_s):
        time_s=finite(time_s,0,1e9,'text-face clock')
        if self.entries and time_s < self.entries[-1]['time']:
            raise ValueError('Text-face updates must be chronological')
        keys={k for k in meta if k.startswith('text_face_')}
        allowed={'text_face_clip','text_face_region','text_face_weight','text_face_fade_ms'}
        if keys-allowed:raise ValueError('Unknown text-face option: '+','.join(sorted(keys-allowed)))
        if not keys:return None
        if 'text_face_clip' not in keys:raise ValueError('Include text_face_clip when changing text-face options')
        ident=meta['text_face_clip']
        region=meta.get('text_face_region','upper_face')
        if not isinstance(region,str) or region not in REGIONS:raise ValueError('Invalid text-face region')
        weight=finite(meta.get('text_face_weight',1),0,1,'text_face_weight')
        fade=finite(meta.get('text_face_fade_ms',250),50,2000,'text_face_fade_ms')/1000
        if not isinstance(ident,str):raise ValueError('text_face_clip must be an asset ID or none')
        signature=(ident,region,weight,fade)
        if self.entries and self.entries[-1]['signature']==signature:return None
        if len(self.entries)>=128:raise ValueError('Text-face schedule budget exceeded')
        clip=None if ident=='none' else self._load(ident)
        return dict(time=float(time_s),id=ident,region=region,weight=weight,fade=fade,
                    clip=clip,signature=signature)

    def commit(self, entry):
        if entry is not None:
            self.entries.append(entry)
            if entry['clip'] is not None:self.cache[entry['id']]=entry['clip']

    def _apply_entry(self, index, base, t):
        if index<0:return dict(base)
        e=self.entries[index];clip=e['clip'];age=t-e['time'];target=dict(base)
        if clip is not None:
            pose=clip.sample(age)
            if pose is not None:
                fade=min(e['fade'],clip.duration/2)
                weight=e['weight']*smooth((clip.duration-age)/fade)
                for k in REGIONS[e['region']]:target[k]=(1-weight)*base[k]+weight*pose[k]
        fade=e['fade'] if clip is None else min(e['fade'],clip.duration/2)
        alpha=smooth(age/fade)
        if alpha>=1:return target
        previous=self._apply_entry(index-1,base,t)
        # Always constrain crossfade to supported upper-face channels.
        for k in REGIONS['upper_face']:target[k]=(1-alpha)*previous[k]+alpha*target[k]
        return target

    def apply(self, base, time_s):
        index=int(np.searchsorted([e['time'] for e in self.entries],time_s,side='right'))-1
        return self._apply_entry(index,base,time_s)

    def metadata(self):
        return {'text_face_version':SCHEMA,'text_face_schedule':[
            {k:e[k] for k in ('time','id','region','weight','fade')} | {
                'sha256':e['clip'].sha256 if e['clip'] else None,
                'provenance':e['clip'].provenance if e['clip'] else None}
            for e in self.entries]}
