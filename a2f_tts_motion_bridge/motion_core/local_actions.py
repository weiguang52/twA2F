"""Deterministic upper-face actions, separate from neural text-face generation.

Normalized targets are software poses, NOT calibrated motor angles. Local actions
own only their channels; latest started action wins without resurrecting old ones.
"""
import json
import math
import re

from .text_face import finite, smooth

ACTIONS=('raise_brow','frown','close_eye','open_eye','squint')
SIDES=('left','right','both')
FIELDS={'id','action','side','strength','attack_ms','hold_ms','release_ms','offset_ms'}


def strict_json(value):
    if not isinstance(value,str):return value
    if len(value)>65536:raise ValueError('Local action payload exceeds 64 KiB')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate JSON key: '+k)
            result[k]=v
        return result
    return json.loads(value,object_pairs_hook=pairs)


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value):
        raise ValueError('Action id must be 1..64 ASCII letters, digits, _ or -')
    return value


def validate_action(value):
    if not isinstance(value,dict) or set(value)-FIELDS:raise ValueError('Unknown local action fields')
    ident=identifier(value.get('id'))
    action=value.get('action');side=value.get('side','both')
    if action not in ACTIONS:raise ValueError('Unsupported local action')
    if side not in SIDES:raise ValueError('side must be robot left/right/both')
    result=dict(id=ident,action=action,side=side)
    for k,default,low,high in [('strength',.6,0,1),('attack_ms',200,50,2000),
            ('hold_ms',1000,0,10000),('release_ms',300,50,3000),('offset_ms',0,0,60000)]:
        result[k]=finite(value.get(k,default),low,high,k)
    return result


def targets(action):
    """Absolute neutral-relative poses, never touch mouth/jaw or phantom DOFs."""
    strength=action['strength'];kind=action['action'];out={}
    for side in ('left','right') if action['side']=='both' else (action['side'],):
        if kind=='raise_brow':values={'inner_brow_y':.5+.25*strength,'outer_brow_y':.5+.30*strength}
        elif kind=='frown':values={'inner_brow_y':.5-.24*strength,'outer_brow_y':.5-.12*strength}
        elif kind=='close_eye':values={'upper_lid_y':.5*(1-strength),'lower_lid_y':.5+.5*strength}
        elif kind=='open_eye':values={'upper_lid_y':.5+.35*strength,'lower_lid_y':.5-.10*strength}
        else:values={'upper_lid_y':.5-.22*strength,'lower_lid_y':.5+.22*strength}
        out.update({side+'_'+k:v for k,v in values.items()})
    return out


class LocalActionTrack:
    def __init__(self):
        self.actions=[];self.ids={};self.cancel_ids={};self.cancels={}
        self.channels={};self.last_time=0.

    def prepare(self,meta,time_s):
        allowed={'local_actions','local_action_cancels'}
        keys={k for k in meta if k.startswith('local_action')}
        if keys-allowed:raise ValueError('Unknown local-action meta fields')
        if not keys:return None
        time_s=finite(time_s,0,1e9,'local-action clock')
        if time_s<self.last_time:raise ValueError('Local action updates must be chronological')
        incoming=strict_json(meta.get('local_actions',[]))
        cancels=strict_json(meta.get('local_action_cancels',[]))
        if not isinstance(incoming,list) or len(incoming)>32:raise ValueError('At most 32 actions per packet')
        if not isinstance(cancels,list) or len(cancels)>32:raise ValueError('At most 32 cancellations per packet')
        ids=dict(self.ids);cancel_ids=dict(self.cancel_ids);stop=dict(self.cancels);new=[]
        for raw in incoming:
            action=validate_action(raw);ident=action['id']
            if ident in ids:
                if ids[ident]!=action:raise ValueError('Action id reused with different payload')
                continue
            ids[ident]=action
            new.append(dict(action,start=time_s+action['offset_ms']/1000,
                            order=len(self.actions)+len(new),target=targets(action)))
        if len(ids)>256:raise ValueError('Session local-action budget exceeded (256)')
        for raw in cancels:
            if not isinstance(raw,dict) or set(raw)-{'id','targets','release_ms'}:raise ValueError('Invalid cancellation')
            ident=identifier(raw.get('id'));selected=raw.get('targets')
            release=finite(raw.get('release_ms',200),50,3000,'cancel release_ms')
            if selected!='all':
                if not isinstance(selected,list) or not selected or len(selected)>256:raise ValueError('Cancellation targets must be IDs or all')
                selected=sorted(set(identifier(x) for x in selected))
            signature=dict(targets=selected,release_ms=release)
            if ident in cancel_ids:
                if cancel_ids[ident]!=signature:raise ValueError('Cancellation id reused with different payload')
                continue
            resolved=list(ids) if selected=='all' else selected
            if any(x not in ids for x in resolved):raise ValueError('Cannot cancel an unknown action id')
            cancel_ids[ident]=signature
            for x in resolved:stop.setdefault(x,(time_s,release/1000))
        if len(cancel_ids)>256:raise ValueError('Session cancellation budget exceeded (256)')
        return new,ids,cancel_ids,stop,time_s

    def commit(self,prepared):
        if prepared is None:return
        new,self.ids,self.cancel_ids,self.cancels,self.last_time=prepared
        self.actions.extend(new)
        self.channels={}
        for action in sorted(self.actions,key=lambda x:(x['start'],x['order'])):
            # Zero strength is a no-op, not a command to reset an existing expression.
            if not action['strength']:continue
            for key in action['target']:self.channels.setdefault(key,[]).append(action)

    def _eligible(self,channel,time_s):
        return [a for a in self.channels[channel] if a['start']<=time_s and
                not (a['id'] in self.cancels and self.cancels[a['id']][0]<=a['start'])]

    def _value(self,events,index,key,base,t,ignore_cancel=False):
        if index<0:return base
        a=events[index];age=t-a['start']
        attack=a['attack_ms']/1000;hold=a['hold_ms']/1000;release=a['release_ms']/1000
        stop=self.cancels.get(a['id'])
        if not ignore_cancel and stop is not None and t>=stop[0]:
            at_stop=self._value(events,index,key,base,stop[0],True)
            return base+(at_stop-base)*(1-smooth((t-stop[0])/stop[1]))
        if age<attack:
            # Same-start predecessors are still at attack=0. Skip the chain,
            # preserving its exact initial pose without recursive per-frame cost.
            previous_index=index-1
            while previous_index>=0 and events[previous_index]['start']==a['start']:
                previous_index-=1
            previous=self._value(events,previous_index,key,base,a['start'])
            return previous+(a['target'][key]-previous)*smooth(age/attack)
        if age<attack+hold:return a['target'][key]
        return base+(a['target'][key]-base)*(1-smooth((age-attack-hold)/release))

    def apply(self,base,time_s):
        if not math.isfinite(time_s) or time_s<0:raise ValueError('Invalid local-action sample time')
        out=dict(base)
        for key in self.channels:
            events=self._eligible(key,time_s)
            if events:out[key]=float(self._value(events,len(events)-1,key,float(base[key]),time_s))
        return out

    def metadata(self):
        return {'local_action_version':'v1_deterministic_upper_face',
                'local_action_source':'structured_plan_not_keyframeface',
                'local_actions':[dict(a) for a in self.actions],
                'local_action_cancellations':{k:dict(time=v[0],release_s=v[1]) for k,v in self.cancels.items()}}
