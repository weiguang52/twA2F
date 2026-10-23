import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from a2f_tts_motion_bridge.motion_core.text_face import (
    SCHEMA, REGIONS, TextFaceClip, TextFaceTrack, normalize_arkit)
from a2f_tts_motion_bridge.motion_core.keyframeface_backend import parse_response, validate_script
from a2f_tts_motion_bridge.motion_core.a2f169_to_arkit52 import ARKIT_52_NAMES
from a2f_tts_motion_bridge.motion_core.behavior_layer import BehaviorLayer
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG

BASE={k:v['neutral'] for k,v in MOTOR_CFG.items()}


def document(**overrides):
    arkit=dict.fromkeys(ARKIT_52_NAMES,0.)
    arkit.update(overrides)
    return {'schema':SCHEMA,'duration_s':4.,'provenance':{'model':'synthetic-unit-test'},
            'keyframes':[{'time_s':0.,'arkit':arkit}]}


class ValidationTests(unittest.TestCase):
    def test_complete_capitalized_model_output(self):
        arkit=document()['keyframes'][0]['arkit']
        result=parse_response(json.dumps({'Keyframe_One':{k[0].upper()+k[1:]:v for k,v in arkit.items()}}))
        self.assertEqual(result,arkit)

    def test_missing_unknown_invalid_coefficients_rejected(self):
        arkit=document()['keyframes'][0]['arkit']
        for changed in ({},dict(arkit,jawOpen=float('nan')),dict(arkit,jawOpen=2),
                        dict(arkit,jawOpen=True),dict(arkit,arbitrary=0),dict(arkit,JawOpen=0)):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):normalize_arkit(changed)
        for response in ('{}','{"Keyframe_One":{}}','{"x":{},"x":{}}','<think>unfinished'):
            with self.assertRaises(ValueError):parse_response(response)

    def test_clip_timing_validation(self):
        doc=document()
        for value in (-1,float('nan'),61,True):
            changed=copy.deepcopy(doc);changed['duration_s']=value
            with self.assertRaises(ValueError):TextFaceClip(changed)
        changed=copy.deepcopy(doc);changed['keyframes']*=2
        with self.assertRaises(ValueError):TextFaceClip(changed)
        changed=copy.deepcopy(doc);changed['keyframes'][0]['time_s']=1
        with self.assertRaises(ValueError):TextFaceClip(changed)

    def test_explicit_timed_prompt_script(self):
        script={'duration_s':4,'keyframes':[{'time_s':0,'description':'Frown.'},{'time_s':2,'description':'Raise eyebrows.'}]}
        self.assertEqual(validate_script(script)[0],4.)
        script['keyframes'][1]['time_s']=0
        with self.assertRaises(ValueError):validate_script(script)

    def test_interpolation_and_no_extrapolation(self):
        doc=document(browDownLeft=1)
        doc['keyframes'].append({'time_s':2,'arkit':document(browOuterUpLeft=1)['keyframes'][0]['arkit']})
        clip=TextFaceClip(doc)
        a,b,c=clip.sample(0),clip.sample(1),clip.sample(2)
        for k in a:self.assertAlmostEqual(b[k],(a[k]+c[k])/2)
        self.assertIsNone(clip.sample(-1));self.assertIsNone(clip.sample(4))


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.env=mock.patch.dict(os.environ,{'A2F_TEXT_FACE_DIR':str(self.root)})
        self.env.start()
        for ident,doc in [('frown',document(browDownLeft=1,browDownRight=1)),
                          ('raise',document(browOuterUpLeft=1,browInnerUp=1,eyeWideLeft=1))]:
            (self.root/(ident+'.json')).write_text(json.dumps(doc))

    def tearDown(self):self.env.stop();self.tmp.cleanup()

    def layer(self,ident='frown',region='upper_face'):
        layer=BehaviorLayer()
        layer.schedule({'text_face_clip':ident,'text_face_region':region,'brow_rhythm':False},0.)
        return layer

    def test_unconfigured_default_is_unchanged(self):
        layer=BehaviorLayer();layer.schedule({'brow_rhythm':False},0)
        for t in (0,.5,9):self.assertEqual(layer.apply(BASE,t),BASE)

    def test_mouth_jaw_bit_exact_with_all_layers_enabled(self):
        a,b=BehaviorLayer(3),BehaviorLayer(3)
        meta={'style':'laugh','idle_mode':'listening'}
        a.schedule(meta,0);b.schedule(dict(meta,text_face_clip='raise'),0)
        for t in np.arange(0,5,1/90):
            base={k:float(.3+.1*np.sin(t)) for k in BASE}
            x,y=a.apply(base,t,1),b.apply(base,t,1)
            for k in BASE:
                if k not in REGIONS['upper_face']:self.assertEqual(x[k],y[k])

    def test_brow_mask_preserves_eye_motion(self):
        layer=self.layer('raise','brows');actual=layer.tick(.5)
        for k in REGIONS['eyes']:self.assertEqual(actual[k],BASE[k])
        self.assertGreater(actual['left_outer_brow_y'],BASE['left_outer_brow_y'])

    def test_blink_has_priority_over_text_eye_pose(self):
        layer=self.layer('raise')
        layer.schedule({'behavior_events':[{'id':'wink','type':'wink','side':'left'}]},.5)
        self.assertLess(layer.tick(.62)['left_upper_lid_y'],.05)

    def test_repeat_chunks_dont_restart_and_cache_is_stable(self):
        layer=self.layer();first=layer.tick(1)
        layer.schedule({'text_face_clip':'frown'},1)
        self.assertEqual(len(layer.text_face.entries),1)
        (self.root/'frown.json').write_text('{}')
        layer.schedule({'text_face_clip':'frown'},2)
        self.assertEqual(layer.tick(1),first)
        self.assertEqual(layer.tick(4),BASE)

    def test_cancel_fades_and_old_clip_never_resurrects(self):
        layer=self.layer();before=layer.tick(1)
        layer.schedule({'text_face_clip':'none'},1.)
        self.assertEqual(layer.tick(1),before)
        self.assertEqual(layer.tick(1.3),BASE)
        self.assertEqual(layer.tick(2),BASE)

    def test_switch_continuity_and_dense_rate_equivalence(self):
        layer=self.layer();before=layer.tick(1)
        layer.schedule({'text_face_clip':'raise'},1)
        self.assertEqual(layer.tick(1),before)
        names=list(BASE);times=np.arange(0,4,1/90)
        dense=layer.apply_array(times,np.tile(list(BASE.values()),(len(times),1)),np.zeros(len(times)))
        for i,t in enumerate(times[::3]):
            np.testing.assert_array_equal(dense[3*i],np.array([layer.tick(t)[k] for k in names],np.float32))
        layer.tick(20)
        self.assertEqual(layer.tick(1),before)

    def test_invalid_packet_is_atomic(self):
        layer=self.layer();snapshot=copy.deepcopy(layer.metadata())
        with self.assertRaises(ValueError):
            layer.schedule({'style':'laugh','text_face_clip':'../escape'},1)
        self.assertEqual(snapshot,layer.metadata())
        for meta in ({'text_face_prompt':'hello'},{'text_face_weight':1},
                     {'text_face_clip':'raise','text_face_weight':'NaN'},
                     {'text_face_clip':'raise','text_face_region':'mouth'}):
            with self.assertRaises(ValueError):layer.schedule(meta,1)
        self.assertEqual(snapshot,layer.metadata())

    def test_path_and_time_bounds(self):
        track=TextFaceTrack()
        for ident in ('../a','/tmp/a','a.json','a/b','a'*65):
            with self.assertRaises(ValueError):track.prepare({'text_face_clip':ident},0)
        track.commit(track.prepare({'text_face_clip':'raise'},1))
        with self.assertRaises(ValueError):track.prepare({'text_face_clip':'frown'},0)
        with self.assertRaises(ValueError):track.prepare({},float('nan'))

    def test_metadata_provenance(self):
        entry=self.layer().metadata()['text_face_schedule'][0]
        self.assertEqual(len(entry['sha256']),64)
        self.assertEqual(entry['provenance']['model'],'synthetic-unit-test')

    def test_symlink_escape_and_oversize_asset_rejected(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            target=Path(elsewhere)/'asset.json';target.write_text(json.dumps(document()))
            (self.root/'escape.json').symlink_to(target)
            with self.assertRaises(ValueError):TextFaceTrack().prepare({'text_face_clip':'escape'},0)
        (self.root/'huge.json').write_text(' '* (1024*1024+1))
        with self.assertRaises(ValueError):TextFaceTrack().prepare({'text_face_clip':'huge'},0)

    def test_weight_zero_and_expiry_restore_base(self):
        layer=BehaviorLayer();layer.schedule({'text_face_clip':'raise','text_face_weight':0,'brow_rhythm':False},0)
        for t in (0,.1,1,3.99,4,5):self.assertEqual(layer.tick(t),BASE)
        layer=self.layer('raise')
        final=layer.tick(4-1e-6)
        for k in BASE:self.assertAlmostEqual(final[k],BASE[k],places=8)


if __name__=='__main__':unittest.main()
