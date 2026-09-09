"""Per-session timestamped behavior/event bus; deterministic export and live ticks."""
from dataclasses import asdict, dataclass, replace
import json
import math

import numpy as np

from .emotion_control import finite_nonnegative
from .idle_behaviors import IdleBehaviors
from .style_overlay import STYLE_DURATIONS, style_offsets
from .settings import MOTOR_CFG
from .brow_rhythm import brow_rhythm_offsets


def number(value,name,low=0.,high=1.):
    x=finite_nonnegative(value,name)
    if not low<=x<=high: raise ValueError(f'{name} must be in [{low},{high}]')
    return x


def boolean(value,name):
    if not isinstance(value,(bool,str)) or value not in (True,False,'true','false'):
        raise ValueError(name+' must be true/false')
    return value is True or value=='true'


@dataclass(frozen=True)
class BehaviorConfig:
    idle_mode: str='off'
    auto_blink: bool=False
    auto_brow: bool=False
    brow_rhythm: bool=True
    brow_rhythm_gain: float=.35
    idle_gain: float=1.
    style_gain: float=1.
    behavior_speech_scale: float=.2
    bias_left_scale: float=1.
    bias_right_scale: float=1.


class BehaviorLayer:
    def __init__(self,seed=0):
        self.seed=seed
        self.idle=IdleBehaviors(seed)
        self.configs=[(0.,BehaviorConfig())]
        self.events=[]
        self.event_ids={}
        self.cancels=[]
        self.last_style='none'
        self.energy=.003
        self.previous_rms=0.
        self.last_auto=-1.
        self.last_observed=-1.

    def prepare(self,meta,time_s):
        if not math.isfinite(time_s) or time_s<0:raise ValueError('Invalid behavior clock')
        if time_s<self.configs[-1][0]:raise ValueError('Behavior updates must be chronological')
        config=self.configs[-1][1];values={};seed=self.seed
        if 'behavior_seed' in meta:
            seed=number(meta['behavior_seed'],'behavior_seed',0,2**32-1)
            if seed!=int(seed) or (time_s>0 and seed!=self.seed):
                raise ValueError('behavior_seed is a uint32 set at session start only')
            seed=int(seed)
        if 'idle_mode' in meta:
            mode=meta['idle_mode']
            if mode not in ('off','idle','listening'):raise ValueError('idle_mode must be off/idle/listening')
            values.update(idle_mode=mode,auto_blink=mode!='off')
        for key in ('auto_blink','auto_brow','brow_rhythm'):
            if key in meta:values[key]=boolean(meta[key],key)
        for key in ('idle_gain','style_gain','behavior_speech_scale','bias_left_scale','bias_right_scale','brow_rhythm_gain'):
            if key in meta:values[key]=number(meta[key],key,0,2 if key.startswith('bias_') else 1)
        config=replace(config,**values)
        actions=[];style=self.last_style;cancel=False
        if 'style' in meta:
            style=meta['style']
            if style not in ('none','smile','laugh','sigh','sob'):raise ValueError('Unknown style')
            cancel=style=='none'
            if style!='none' and style!=self.last_style:
                actions.append(dict(id=f'style:{time_s}:{style}',type=style,
                    weight=meta.get('style_weight',1),duration_ms=meta.get('style_duration_ms',STYLE_DURATIONS[style]*1000)))
        # Validate knobs even for repeated style packets (no hidden invalid values).
        if 'style_weight' in meta:number(meta['style_weight'],'style_weight')
        if 'style_duration_ms' in meta:number(meta['style_duration_ms'],'style_duration_ms',50,10000)
        if 'behavior_events' in meta:
            extra=meta['behavior_events']
            if isinstance(extra,str):
                try:extra=json.loads(extra)
                except ValueError as exc:raise ValueError('behavior_events must be a JSON list') from exc
            if not isinstance(extra,list) or len(extra)>64:raise ValueError('behavior_events must contain at most 64 events')
            actions.extend(extra)
        parsed=[];ids=dict(self.event_ids)
        for action in actions:
            if not isinstance(action,dict):raise ValueError('Event must be an object')
            if set(action)-{'id','type','weight','duration_ms','offset_ms','side'}:raise ValueError('Unknown behavior event fields')
            ident=action.get('id');kind=action.get('type')
            if not isinstance(ident,str) or not 1<=len(ident)<=128:raise ValueError('Event id required, max 128 characters')
            if ident.startswith('auto-brow:'):raise ValueError('Reserved automatic-event ID prefix')
            if kind not in STYLE_DURATIONS:raise ValueError('Unknown behavior event type')
            duration=number(action.get('duration_ms',STYLE_DURATIONS[kind]*1000),'event duration',50,10000)/1000
            if kind=='laugh' and not .3<=duration<=.8:raise ValueError('laugh duration must be 300..800 ms')
            if kind=='brow_flash' and not .08<=duration<=.15:raise ValueError('brow_flash duration must be 80..150 ms')
            side=action.get('side','left')
            if side not in ('left','right'):raise ValueError('event side must be left/right')
            weight=number(action.get('weight',1),'event weight')
            offset=number(action.get('offset_ms',0),'event offset',0,60000)/1000
            signature=(kind,duration,weight,side,offset)
            if ident in ids:
                if ids[ident]!=signature:raise ValueError('Event id reused with different payload')
                continue
            ids[ident]=signature
            parsed.append(dict(id=ident,type=kind,time=time_s+offset,created=time_s,duration=duration,weight=weight,side=side))
        if len(ids)>10000:raise ValueError('Session event budget exceeded (10000)')
        return config,seed,parsed,ids,style,cancel,float(time_s)

    def commit(self,prepared):
        config,seed,events,ids,style,cancel,time_s=prepared
        if seed!=self.seed:self.seed=seed;self.idle=IdleBehaviors(seed)
        if config!=self.configs[-1][1]:self.configs.append((time_s,config))
        self.events.extend(events);self.events.sort(key=lambda e:e['time'])
        self.event_ids=ids;self.last_style=style
        if cancel:self.cancels.append(time_s)

    def schedule(self,meta,time_s):self.commit(self.prepare(meta,time_s))

    def config_at(self,time_s):
        i=max(0,int(np.searchsorted([x[0] for x in self.configs],time_s,side='right'))-1)
        return self.configs[i][1]

    def observe_audio(self,time_s,rms):
        if time_s<=self.last_observed:return
        self.last_observed=time_s
        x=max(0.,float(rms or 0.));threshold=max(.02,1.8*self.energy)
        if self.config_at(time_s).auto_brow and x>threshold and self.previous_rms<=threshold and time_s-self.last_auto>=.6:
            # Internal IDs cannot collide with upstream IDs.
            self.events.append(dict(id='auto-brow:'+str(time_s),type='brow_flash',time=time_s,
                                    duration=.12,weight=.5,side='left'))
            self.last_auto=time_s
        self.energy=.9*self.energy+.1*x;self.previous_rms=x

    def apply(self,base,time_s,speech=0.):
        if not math.isfinite(time_s) or time_s<0 or not math.isfinite(speech):
            raise ValueError('Behavior time/gate must be finite and time nonnegative')
        if any(k not in base or not math.isfinite(float(base[k])) for k in MOTOR_CFG):
            raise ValueError('Behavior base must contain all finite 13 DOFs')
        config=self.config_at(time_s)
        delta,closure=self.idle.sample(time_s,config.idle_mode,config.auto_blink,config.idle_gain)
        gate=float(np.clip(speech,0,1));mouth_scale=1-gate*(1-config.behavior_speech_scale)
        for event in self.events:
            age=time_s-event['time']
            if not 0<=age<event['duration']:continue
            if any(event.get('created',event['time'])<c<=time_s for c in self.cancels):continue
            d,c=style_offsets(event['type'],age,event['duration'],event['weight']*config.style_gain,event['side'],self.seed)
            for k,v in d.items():delta[k]=delta.get(k,0.)+v
            for k,v in c.items():closure[k]=max(closure.get(k,0.),v)
        out=dict(base)
        if config.brow_rhythm:
            for key,value in brow_rhythm_offsets(base,time_s,self.seed,config.brow_rhythm_gain).items():
                delta[key]=delta.get(key,0.)+value
        for name in MOTOR_CFG:
            amount=np.clip(delta.get(name,0.),-.3,.3)
            if 'mouth' in name or name=='jaw_y':amount*=mouth_scale
            out[name]=float(np.clip(out[name]+amount,0.,1.))
        for side,value in closure.items():
            strength=float(np.clip(value,0,1))
            up,low=side+'_upper_lid_y',side+'_lower_lid_y'
            out[up]*=1-strength
            out[low]+=(1-out[low])*strength
        return out

    def apply_array(self,times,values,speech,names=None):
        names=list(names or MOTOR_CFG)
        if len(values)!=len(times) or len(speech)!=len(times):
            raise ValueError('Behavior dense series length mismatch')
        if not len(times):return np.asarray(values,dtype=np.float32).copy()
        return np.asarray([[row[k] for k in names] for row in
            (self.apply(dict(zip(names,v)),float(t),float(s)) for t,v,s in zip(times,values,speech))],dtype=np.float32)

    def tick(self,time_s,base=None):
        """Audio-independent host timer entry point (call at 30/90 Hz)."""
        base=base or {k:v['neutral'] for k,v in MOTOR_CFG.items()}
        return self.apply(base,time_s,0.)

    def metadata(self):
        return dict(behavior_config=asdict(self.configs[-1][1]),behavior_seed=self.seed,
                    behavior_config_timeline=[dict(time=t,config=asdict(c)) for t,c in self.configs],
                    behavior_cancel_times=list(self.cancels),
                    behavior_events=list(self.events),behavior_version='p26_output_clock_v1',
                    unsupported_behavior_dofs=['eye_gaze','head_tilt','shoulder'])
