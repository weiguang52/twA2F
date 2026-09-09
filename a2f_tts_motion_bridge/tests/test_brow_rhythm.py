import unittest
import numpy as np
from a2f_tts_motion_bridge.motion_core.behavior_layer import BehaviorLayer
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG

class BrowRhythmTests(unittest.TestCase):
    def test_pose_direction_motion_and_other_channels(self):
        base={k:v['neutral'] for k,v in MOTOR_CFG.items()}
        base.update(left_inner_brow_y=.4,right_inner_brow_y=.4)
        layer=BehaviorLayer(seed=42)
        frames=[layer.apply(base,t,1.) for t in np.arange(0,12,1/90)]
        y=np.array([f['left_inner_brow_y'] for f in frames])
        self.assertGreater(np.ptp(y),.04)
        self.assertTrue(np.all(y<.5))
        self.assertLess(np.max(np.abs(np.diff(y))),.004)
        for f in frames:
            for k in base:
                if 'brow' not in k:self.assertEqual(f[k],base[k])

    def test_neutral_disable_validation_and_determinism(self):
        base={k:v['neutral'] for k,v in MOTOR_CFG.items()}
        layer=BehaviorLayer(seed=42)
        self.assertEqual(layer.apply(base,2.),base)
        base['left_inner_brow_y']=.6
        first=layer.apply(base,2.);layer.apply(base,20.)
        self.assertEqual(layer.apply(base,2.),first)
        layer.schedule({'brow_rhythm':'false'},3.)
        self.assertEqual(layer.apply(base,4.),base)
        self.assertEqual(layer.apply(base,2.),first)
        with self.assertRaises(ValueError):layer.schedule({'brow_rhythm_gain':'nan'},5.)

    def test_dense_and_individual_match(self):
        layer=BehaviorLayer(seed=4);names=list(MOTOR_CFG)
        base={k:v['neutral'] for k,v in MOTOR_CFG.items()};base['left_inner_brow_y']=.4
        times=np.arange(0,8,1/90);values=np.array([[base[k] for k in names]]*len(times))
        dense=layer.apply_array(times,values,np.zeros(len(times)),names)
        expected=[[layer.apply(base,float(t))[k] for k in names] for t in times]
        np.testing.assert_allclose(dense,expected,atol=1e-7)
