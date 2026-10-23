import copy
import json
import unittest
from unittest import mock
import numpy as np

from a2f_tts_motion_bridge.motion_core.behavior_layer import BehaviorLayer
from a2f_tts_motion_bridge.motion_core.local_actions import LocalActionTrack,validate_action
from a2f_tts_motion_bridge.motion_core.local_action_planner import validate_plan,plan_to_meta
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG
from a2f_tts_motion_bridge.preview.local_action_cases import cases,action,plan

BASE={k:v['neutral'] for k,v in MOTOR_CFG.items()}


def send(layer,actions,t=0):layer.schedule({'local_actions':actions},t)


def oracle_targets(a):
    # Deliberately independent of production targets().
    coefficients={'raise_brow':('inner_brow_y','outer_brow_y',.25,.30),
        'frown':('inner_brow_y','outer_brow_y',-.24,-.12),
        'close_eye':('upper_lid_y','lower_lid_y',-.5,.5),
        'open_eye':('upper_lid_y','lower_lid_y',.35,-.1),
        'squint':('upper_lid_y','lower_lid_y',-.22,.22)}
    k1,k2,c1,c2=coefficients[a['action']];out={}
    for side in ('left','right') if a['side']=='both' else (a['side'],):
        out[side+'_'+k1]=.5+c1*a['strength'];out[side+'_'+k2]=.5+c2*a['strength']
    return out


def evaluate_case(case):
    expected=case['expected'];layer=BehaviorLayer();layer.schedule({'brow_rhythm':False},0)
    if expected['unsupported']:
        try:plan_to_meta(expected,case['id'])
        except ValueError:return {'id':case['id'],'status':'passed','kind':'unsupported_rejected'}
        raise AssertionError('Unsupported request was executed')
    if expected['cancel_all']:
        send(layer,[dict(action(hold_ms=5000),id='existing')]);layer.schedule(plan_to_meta(expected,case['id']),.5)
        np.testing.assert_allclose(list(layer.tick(.7).values()),list(BASE.values()),atol=1e-14)
        return {'id':case['id'],'status':'passed','kind':'cancel_all'}
    layer.schedule(plan_to_meta(expected,case['id']),0)
    end=max((a['offset_ms']+a['attack_ms']+a['hold_ms']+a['release_ms'])/1000 for a in expected['actions'])+.2
    times=np.arange(0,end,1/90);values=[];largest=0.
    for t in times:
        wanted=dict(BASE)
        for a in expected['actions']:
            age=t-a['offset_ms']/1000
            if age<0:continue
            attack,hold,release=(a[k]/1000 for k in ('attack_ms','hold_ms','release_ms'))
            q=max(0,min(1,age/attack)) if age<attack else max(0,min(1,1-(age-attack-hold)/release))
            envelope=q*q*(3-2*q)
            for key,target in oracle_targets(a).items():wanted[key]=BASE[key]+(target-BASE[key])*envelope
        actual=layer.tick(float(t));values.append(list(actual.values()))
        error=max(abs(actual[k]-wanted[k]) for k in BASE);largest=max(largest,error)
        if error>1e-12:raise AssertionError((case['id'],t,error))
    array=np.asarray(values)
    if not np.isfinite(array).all() or array.min()<0 or array.max()>1:raise AssertionError('Unsafe values')
    return {'id':case['id'],'status':'passed','kind':'action','max_dof_error':largest,
            'times':times,'values':array,'meta':layer.metadata()}


class Acceptance50(unittest.TestCase):pass


def make_case(case):
    def test(self):self.assertEqual(evaluate_case(case)['status'],'passed')
    return test


for case in cases():setattr(Acceptance50,'test_'+case['id'].replace('-','_'),make_case(case))


class LocalActionTests(unittest.TestCase):
    def layer(self):
        layer=BehaviorLayer();layer.schedule({'brow_rhythm':False},0);return layer

    def test_switch_continuity_no_resurrection(self):
        layer=self.layer();send(layer,[dict(action(hold_ms=5000),id='old')])
        before=layer.tick(1)
        send(layer,[dict(action('frown',hold_ms=100),id='new')],1)
        self.assertEqual(layer.tick(1),before)
        self.assertLess(layer.tick(1.2)['left_inner_brow_y'],.5)
        self.assertEqual(layer.tick(1.7),BASE)
        self.assertEqual(layer.tick(2),BASE)

    def test_cancel_mid_attack_is_continuous(self):
        layer=self.layer();send(layer,[dict(action(),id='a')]);before=layer.tick(.1)
        layer.schedule({'local_action_cancels':[{'id':'c','targets':['a'],'release_ms':200}]},.1)
        self.assertEqual(layer.tick(.1),before)
        self.assertEqual(layer.tick(.31),BASE)

    def test_cancel_future_action_does_not_take_over(self):
        layer=self.layer();send(layer,[dict(action(hold_ms=5000),id='a'),dict(action('frown',offset_ms=1000),id='b')])
        layer.schedule({'local_action_cancels':[{'id':'c','targets':['b']}]},.5)
        self.assertEqual(layer.tick(2)['left_inner_brow_y'],.65)

    def test_repeated_ids_do_not_restart_or_recancel_new_work(self):
        layer=self.layer();a=dict(action(),id='a');send(layer,[a]);send(layer,[a],1)
        self.assertEqual(len(layer.local_actions.actions),1)
        cancel={'local_action_cancels':[{'id':'c','targets':'all'}]}
        layer.schedule(cancel,1);send(layer,[dict(action(side='right'),id='b')],1.2)
        layer.schedule(cancel,1.3)
        self.assertGreater(layer.tick(1.6)['right_inner_brow_y'],.5)
        self.assertEqual(layer.tick(2)['left_inner_brow_y'],.5)

    def test_invalid_meta_atomic_across_all_layers(self):
        layer=self.layer();snapshot=copy.deepcopy(layer.metadata())
        invalid=[{'local_actions':[dict(action(),id='bad',strength=float('nan'))]},
                 {'local_actions':[dict(action(),id='bad',side='screen_left')]},
                 {'local_actions':'[{"id":"a","id":"b"}]'},
                 {'local_action_prompt':'raise eyebrow'},
                 {'local_action_cancels':[{'id':'bad','targets':['unknown']}]},
                 {'local_actions':[dict(action(),id='bad',hold_ms=10001)]}]
        for meta in invalid:
            with self.assertRaises(ValueError):layer.schedule(dict(meta,style='laugh'),.5)
            self.assertEqual(layer.metadata(),snapshot)

    def test_conflicting_id_payload_rejected(self):
        layer=self.layer();send(layer,[dict(action(),id='a')])
        with self.assertRaises(ValueError):send(layer,[dict(action(side='right'),id='a')],1)

    def test_out_of_order_rejected(self):
        layer=self.layer();send(layer,[dict(action(),id='a')],1)
        with self.assertRaises(ValueError):send(layer,[dict(action(),id='b')],.5)

    def test_local_overrides_text_only_in_target_region(self):
        layer=self.layer();layer.text_face.entries=[{'time':0.}]
        def text(base,t):return dict(base,left_inner_brow_y=.2,left_outer_brow_y=.2,right_inner_brow_y=.3)
        with mock.patch.object(layer.text_face,'apply',side_effect=text):
            send(layer,[dict(action(),id='a')]);pose=layer.tick(.4)
            self.assertEqual(pose['left_inner_brow_y'],.65)
            self.assertEqual(pose['right_inner_brow_y'],.3)

    def test_blink_wins_over_open_eye(self):
        layer=self.layer();send(layer,[dict(action('open_eye'),id='a')])
        layer.schedule({'behavior_events':[{'id':'w','type':'wink','side':'left'}]},.5)
        self.assertLess(layer.tick(.62)['left_upper_lid_y'],.05)

    def test_exact_mouth_and_opposite_side_isolation(self):
        plain,local=self.layer(),self.layer()
        for l in (plain,local):l.schedule({'style':'laugh','idle_mode':'listening','behavior_seed':3},0)
        send(local,[dict(action('frown'),id='a')])
        for t in np.arange(0,3,1/90):
            base={k:float(.4+.1*np.sin(t)) for k in BASE};a,b=plain.apply(base,t,1),local.apply(base,t,1)
            for k in BASE:
                if k not in ('left_inner_brow_y','left_outer_brow_y'):self.assertEqual(a[k],b[k])

    def test_sampling_order_and_rates_are_deterministic(self):
        layer=self.layer();send(layer,[dict(action(),id='a')]);expected=layer.tick(.6)
        layer.tick(50);self.assertEqual(layer.tick(.6),expected)
        times=np.arange(0,3,1/90);base=np.tile(list(BASE.values()),(len(times),1))
        dense=layer.apply_array(times,base,np.zeros(len(times)))
        coarse=layer.apply_array(times[::3],base[::3],np.zeros(len(times[::3])))
        np.testing.assert_array_equal(dense[::3],coarse)

    def test_zero_strength_is_noop(self):
        layer=self.layer();base=dict(BASE,left_inner_brow_y=.3)
        send(layer,[dict(action(strength=0),id='a')])
        self.assertEqual(layer.apply(base,.4),base)

    def test_plan_validation_and_provenance(self):
        for bad in ({},dict(plan([action()]),unsupported=True),dict(plan(),cancel_all='true')):
            with self.assertRaises(ValueError):validate_plan(bad)
        layer=self.layer();send(layer,[dict(action(),id='a')])
        self.assertEqual(layer.metadata()['local_action_source'],'structured_plan_not_keyframeface')

    def test_many_same_start_actions_match_last_owner_and_budget_is_atomic(self):
        many,one=self.layer(),self.layer()
        for start in range(0,256,32):
            send(many,[dict(action(strength=i/255),id='a'+str(i)) for i in range(start,start+32)])
        send(one,[dict(action(strength=1),id='last')])
        for t in (0,.01,.1,.2,.8,1.4,2):self.assertEqual(many.tick(t),one.tick(t))
        before=copy.deepcopy(many.metadata())
        with self.assertRaises(ValueError):send(many,[dict(action(),id='over-budget')])
        self.assertEqual(many.metadata(),before)


if __name__=='__main__':unittest.main()
