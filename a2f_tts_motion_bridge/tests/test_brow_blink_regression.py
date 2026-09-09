import unittest
import numpy as np
from a2f_tts_motion_bridge.motion_core.arkit52_to_motor import arkit52_to_current_head_dofs
from a2f_tts_motion_bridge.motion_core.behavior_layer import BehaviorLayer
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG


class BrowBlinkRegression(unittest.TestCase):
    def test_anger_can_correct_weak_opposing_lift(self):
        d = arkit52_to_current_head_dofs({'browInnerUp': .05}, emotion='anger')
        self.assertLess(d['left_inner_brow_y'], .5)
        self.assertLess(d['left_inner_brow_y'], d['left_outer_brow_y'])

    def test_sad_brow_tilt_and_model_only(self):
        d = arkit52_to_current_head_dofs({}, emotion='sadness')
        self.assertGreater(d['left_inner_brow_y'], .5)
        self.assertLess(d['left_outer_brow_y'], .5)
        bs = {'browInnerUp': .6}
        self.assertEqual(arkit52_to_current_head_dofs(bs, emotion='anger', bias_scale=0),
                         arkit52_to_current_head_dofs(bs, emotion='neutral', bias_scale=0))

    def test_enabled_blink_closes_both_eyes_and_recovers(self):
        layer = BehaviorLayer(seed=42)
        layer.schedule({'auto_blink': 'true'}, 0.)
        base = {k: v['neutral'] for k, v in MOTOR_CFG.items()}
        frames = [layer.apply(base, float(t), 1.) for t in np.arange(0, 10, 1/90)]
        self.assertTrue(any(all(f[s+'_upper_lid_y'] < .05 and f[s+'_lower_lid_y'] > .95
                                for s in ('left', 'right')) for f in frames))
        self.assertTrue(any(f['left_upper_lid_y'] == .5 for f in frames))


if __name__ == '__main__':
    unittest.main()
