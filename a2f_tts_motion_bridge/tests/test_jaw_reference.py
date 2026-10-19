import unittest
import numpy as np
from a2f_tts_motion_bridge.motion_core.jaw_reference import jaw_activity,AudioJawGate
from a2f_tts_motion_bridge.motion_core.behavior_layer import BehaviorLayer
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG

class JawReferenceTests(unittest.TestCase):
    def test_reference_and_both_displacement_directions(self):
        ref=np.ones(15,dtype=np.float32)*.2
        self.assertEqual(jaw_activity(ref,ref),0.)
        self.assertGreater(jaw_activity(ref+.1,ref),0.)
        self.assertGreater(jaw_activity(ref-.1,ref),0.)
        self.assertLessEqual(jaw_activity(ref+100,ref),.85)
        with self.assertRaises(ValueError):jaw_activity(np.ones(14),ref)
        with self.assertRaises(ValueError):jaw_activity(np.full(15,np.nan),ref)

    def test_chunks_append_and_replay_equal_full_audio(self):
        audio=np.r_[np.zeros(1000),np.ones(1000)*.1,np.zeros(1000)].astype(np.float32)
        chunks=[audio[:1200]];gate=AudioJawGate(chunks)
        gate.sample(.01);chunks.extend([audio[1200:1900],audio[1900:]])
        full=AudioJawGate([audio])
        for t in np.arange(0,.3,1/90):self.assertAlmostEqual(gate.sample(t),full.sample(t))
        self.assertEqual(gate.sample(.01),0.)
        self.assertEqual(gate.sample(.09),1.)
        self.assertEqual(gate.sample(.3),0.)

    def test_no_forced_open_and_preserve_other_dofs(self):
        base={k:v['neutral'] for k,v in MOTOR_CFG.items()};layer=BehaviorLayer()
        layer.jaw_audio_gate=AudioJawGate([np.ones(16000,dtype=np.float32)*.1])
        self.assertEqual(layer.apply(base,.5)['jaw_y'],0.)
        base['jaw_y']=.3
        layer.jaw_audio_gate=AudioJawGate([np.zeros(16000,dtype=np.float32)])
        out=layer.apply(base,.5);self.assertEqual(out['jaw_y'],0.)
        for k in base:
            if k!='jaw_y':self.assertEqual(out[k],base[k])

    def test_short_silent_gap_survives(self):
        audio=np.ones(16000,dtype=np.float32)*.1;audio[7000:8000]=0
        gate=AudioJawGate([audio]);self.assertEqual(gate.sample(.47),0.)
        self.assertEqual(gate.sample(.4),1.)

    def test_explicit_sigh_still_works_in_silence(self):
        layer=BehaviorLayer();layer.schedule({'style':'sigh'},0.)
        layer.jaw_audio_gate=AudioJawGate([np.zeros(48000,np.float32)])
        base={k:v['neutral'] for k,v in MOTOR_CFG.items()}
        self.assertGreater(layer.apply(base,1.)['jaw_y'],0.)
