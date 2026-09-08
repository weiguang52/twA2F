import io
import json
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from a2f_tts_motion_bridge.motion_core.emotion_control import EmotionControl
from a2f_tts_motion_bridge.motion_core.emotion_latent import EmotionLatentLibrary, LatentOptions, va_weights
from a2f_tts_motion_bridge.motion_core.emotion_track import EmotionTrack
from a2f_tts_motion_bridge.motion_core.settings import EXPLICIT_EMOTION_LABELS
from a2f_tts_motion_bridge.motion_core.a2f_stream_infer import _infer_input_array, StreamingA2FEngine, A2FModelRuntime

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'audio2face-3d-model/implicit_emo_db.npz'


class MixtureTests(unittest.TestCase):
    def test_aliases_sum_and_intensity(self):
        c = EmotionControl().patch({'emotion_mix': '{"happy":0.3,"joy":0.3,"surprise":0.4}', 'intensity':'2'})
        self.assertAlmostEqual(c.vector()[22], 1.2, places=6)
        self.assertAlmostEqual(c.vector()[16], .8, places=6)
        self.assertFalse(c.vector()[:16].any())

    def test_validation_reset_sticky(self):
        c = EmotionControl('joy', implicit_vector=[.1]*16).patch({'emotion_mix': {'fear':.4}})
        self.assertIsNone(c.implicit_vector)
        np.testing.assert_allclose(c.patch({'intensity': '.5'}).explicit()[4], .2)
        self.assertEqual(c.patch({'emotion':'sad'}).vector()[25], 1)
        for meta in ({'emotion_mix':'[]'}, {'emotion_mix':{'joy':-1}}, {'emotion_mix':{'bad':1}},
                     {'emotion_mix':{'joy':True}}, {'emotion_mix':'{"joy":NaN}'},
                     {'emotion_mix':{},'emotion_explicit_weights':'[0]'}, {'emotion_va':'[2,0]'}):
            with self.subTest(meta=meta), self.assertRaises(ValueError): c.patch(meta)

    def test_va_is_smooth_normalized_positive(self):
        a, b = np.array(va_weights([.3,.5])), np.array(va_weights([.301,.5]))
        self.assertAlmostEqual(a.sum(),1)
        self.assertTrue(np.all(a>0))
        self.assertLess(np.linalg.norm(a-b),.01)

    def test_ready26_no_double_scaling(self):
        v = np.r_[np.linspace(-.1,.1,16), np.arange(10)/10].astype(np.float32)
        for shape in ([2,26],[2,1,26]):
            out = _infer_input_array(SimpleNamespace(name='emotion',shape=shape),np.zeros((2,1,8320)),emotion_intensity=99,emotion_vector=v)
            np.testing.assert_allclose(out.reshape(2,26),np.tile(v,(2,1)))
        for bad in ([0]*25, [float('nan')]*26, [-1]*26):
            with self.assertRaises(ValueError):
                _infer_input_array(SimpleNamespace(name='emotion',shape=[1,26]),np.zeros((1,1,8320)),emotion_vector=bad)


@unittest.skipUnless(DB.exists(), 'actor db not installed')
class LatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.db=EmotionLatentLibrary(DB)

    def test_index_centroids_and_readonly(self):
        self.assertEqual(len(self.db.specs),84)
        for i,e in enumerate(EXPLICIT_EMOTION_LABELS):
            self.assertEqual(self.db.specs['mute_'+e].shape,(121,16))
            np.testing.assert_allclose(self.db.anchors[i],self.db.specs['mute_'+e].mean(0),atol=1e-7)
        with self.assertRaises(ValueError): self.db.anchors[0,0]=2

    def test_centroid_mixture_and_scale(self):
        w=np.zeros(10);w[6]=.6;w[0]=.4
        options=LatentOptions(mode='centroid',gain=1)
        np.testing.assert_allclose(self.db.vector(w,0,options),self.db.anchors[6]*.6+self.db.anchors[0]*.4,atol=1e-8)
        np.testing.assert_allclose(self.db.vector(w*2,0,options),self.db.vector(w,0,options),atol=1e-8)

    def test_sequence_loop_random_reproducible(self):
        w=np.eye(10)[6]
        opt=LatentOptions(mode='sequence',gain=1)
        np.testing.assert_allclose(self.db.vector(w,0,opt),self.db.sequences[6][0])
        np.testing.assert_allclose(self.db.vector(w,121/30,opt),self.db.vector(w,0,opt))
        r=LatentOptions(mode='random',seed=42)
        a=self.db.vector(w,.123,r)
        self.db.vector(w,90,r)
        np.testing.assert_array_equal(a,self.db.vector(w,.123,r))
        self.assertGreater(np.linalg.norm(a-self.db.vector(w,.8,r)),1e-6)

    def test_gate_and_raw_vector_passthrough(self):
        meta={'emotion_latent_mode':'centroid','emotion_track':'false','emotion_latent_gain':'1'}
        quiet, speech = EmotionTrack(self.db),EmotionTrack(self.db)
        for t in (quiet,speech): t.schedule(EmotionControl('joy'),meta,0)
        for n in range(12):
            q=quiet.sample(n*.1,0).vector();s=speech.sample(n*.1,.1).vector()
        self.assertLess(np.linalg.norm(s[:16]),np.linalg.norm(q[:16])*.5)
        np.testing.assert_allclose(s[16:],q[16:])
        np.testing.assert_allclose(q[:16],self.db.anchors[6],atol=1e-7)
        raw=EmotionTrack(self.db); raw.schedule(EmotionControl(implicit_vector=[.1]*16),{},0)
        np.testing.assert_allclose(raw.sample(.5,1).vector()[:16],.1)

    def test_missing_db_and_atomic_invalid_options(self):
        t=EmotionTrack();t.schedule(EmotionControl('joy'),{},0)
        previous=t.events[-1]
        for meta in ({'emotion_latent_mode':'centroid'}, {'emotion_latent_gain':'NaN'},
                     {'emotion_latent_fps':'0'},{'emotion_crossfade_ms':'-1'},
                     {'emotion_track':'yes'},{'emotion_latent_seed':'1.5'}):
            with self.assertRaises(ValueError):t.schedule(EmotionControl('fear'),meta,1)
            self.assertIs(t.events[-1],previous)


class TrackTests(unittest.TestCase):
    def test_attack_sustain_release_residual(self):
        t=EmotionTrack();t.schedule(EmotionControl('surprise'),{'emotion_track':'true'},0)
        values=[t.sample(x,0).vector()[16] for x in (0,.06,.12,.37,.82,1.27)]
        np.testing.assert_allclose(values,[0,.5,1,1,.575,.15],atol=1e-6)
        sad=EmotionTrack();sad.schedule(EmotionControl('sadness'),{'emotion_track':'true'},0)
        self.assertLess(sad.sample(.12,0).vector()[25],.2)
        self.assertEqual(sad.sample(2,0).vector()[25],1)

    def test_switch_continuity_intensity_ramp_and_neutral_tail(self):
        t=EmotionTrack();t.schedule(EmotionControl('joy'),{'emotion_track':'true'},0)
        before=t.sample(1,0).vector()
        t.schedule(EmotionControl('fear'),{},1)
        np.testing.assert_allclose(t.sample(1,0).vector(),before)
        mid=t.sample(1.2,0).vector()
        self.assertAlmostEqual(mid[22],.5);self.assertAlmostEqual(mid[20],.5)
        t.schedule(EmotionControl('fear',.2),{},1.5)
        self.assertAlmostEqual(t.sample(1.625,0).vector()[20],.6,places=6)
        t.schedule(EmotionControl('neutral'),{},2)
        self.assertGreater(t.sample(2.4,0).vector()[20],0)
        self.assertFalse(t.sample(3,0).vector().any())

    def test_future_chunk_event_does_not_leak_to_past_frame(self):
        t=EmotionTrack();t.schedule(EmotionControl('joy'),{'emotion_track':'true'},0)
        t.schedule(EmotionControl('fear'),{},1)
        self.assertEqual(t.sample(.96,0).vector()[20],0)
        self.assertGreater(t.sample(1.16,0).vector()[20],0)

    def test_override_and_no_repeat_restart(self):
        t=EmotionTrack();meta={'emotion_track':'true','emotion_attack_ms':'1000'}
        t.schedule(EmotionControl('joy'),meta,0)
        t.schedule(EmotionControl('joy'),{},.2)
        self.assertEqual(len(t.events),1)
        self.assertAlmostEqual(t.sample(.5,0).vector()[22],.5)

    def test_raw_latent_switch_uses_crossfade_not_intensity_ramp(self):
        t=EmotionTrack()
        t.schedule(EmotionControl(implicit_vector=[.1]*16),{'emotion_track':'true'},0)
        t.sample(1,0)
        t.schedule(EmotionControl(implicit_vector=[-.1]*16),{},1)
        np.testing.assert_allclose(t.sample(1.2,0).vector()[:16],0,atol=1e-7)

    def test_mid_transition_retarget_is_continuous(self):
        t=EmotionTrack();t.schedule(EmotionControl('joy'),{'emotion_track':'true'},0)
        t.schedule(EmotionControl('fear'),{},1)
        before=t.sample(1.2,0).vector()
        t.schedule(EmotionControl('sadness'),{},1.2)
        np.testing.assert_allclose(t.sample(1.2,0).vector(),before,atol=1e-7)

    def test_chunk_partition_invariance_and_tail(self):
        inputs=[SimpleNamespace(name='input',shape=[1,1,8320]),SimpleNamespace(name='emotion',shape=[1,1,26])]
        def run(sizes):
            network=mock.Mock();network.run.return_value=[np.zeros((1,1,169),np.float32)]
            runtime=SimpleNamespace(session=network,providers=[],session_init_sec=0,inputs_meta=inputs,warmup_sec=0)
            engine=StreamingA2FEngine('fake',16000,runtime=runtime)
            track=EmotionTrack();track.schedule(EmotionControl('joy'),{'emotion_track':'true'},0)
            engine.emotion_provider=track.sample
            audio=np.sin(np.arange(sum(sizes))*.08).astype(np.float32)*.1
            start=0;out=[]
            for size in sizes:
                out+=engine.push_audio_chunk(audio[start:start+size],'joy',1)
                start+=size
            out+=engine.finalize()
            return np.array([i['emotion_control'].vector() for i in out])
        np.testing.assert_array_equal(run([32101]),run([3200]*10+[101]))


@unittest.skipUnless(os.environ.get('A2F_TEST_MODEL'), 'Set A2F_TEST_MODEL')
class RealP16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runtime=A2FModelRuntime(os.environ['A2F_TEST_MODEL'],warmup=1)

    def test_sessions_share_database_but_not_temporal_state(self):
        from a2f_tts_motion_bridge.motion_core.motion_session import MotorStreamSession
        from a2f_tts_motion_bridge.motion_core.types import SessionConfig
        a=MotorStreamSession(os.environ['A2F_TEST_MODEL'],SessionConfig('a'),runtime=self.runtime)
        b=MotorStreamSession(os.environ['A2F_TEST_MODEL'],SessionConfig('b'),runtime=self.runtime)
        self.assertIs(a.emotion_track.library,b.emotion_track.library)
        self.assertIsNot(a.emotion_track.gate,b.emotion_track.gate)
        a.update_emotion_meta({'emotion_mix':'{"joy":1}','emotion_latent_mode':'random'})
        self.assertEqual(b.emotion_track.latent.mode,'off')
        previous=a.emotion_control;events=list(a.emotion_track.events)
        with self.assertRaises(ValueError):a.update_emotion_meta({'emotion_latent_gain':'-1'})
        self.assertIs(a.emotion_control,previous);self.assertEqual(a.emotion_track.events,events)

    def test_latent_gain_real_model_finite_and_effective(self):
        runtime=self.runtime;outputs=[]
        for gain in (0,.25,.5,1):
            v=np.zeros(26,np.float32)
            v[:16]=runtime.emotion_latent.anchors[6]*gain
            inputs={i.name:_infer_input_array(i,np.zeros((1,1,8320),np.float32),emotion_vector=v) for i in runtime.inputs_meta}
            y=runtime.session.run(None,inputs)[0];self.assertTrue(np.isfinite(y).all());outputs.append(y)
        deltas=[float(np.linalg.norm(y-outputs[0])) for y in outputs[1:]]
        print('P16 latent gain .25/.5/1 raw169 L2 vs zero:',deltas)
        self.assertTrue(all(x>1e-6 for x in deltas))

    def test_grpc_mixed_sequence_va_release_recording(self):
        from a2f_tts_motion_bridge.entrypoints.tts_action_npy_server import VoiceAgentCompat,vapb2,PCM16,ERROR
        service=VoiceAgentCompat(os.environ['A2F_TEST_MODEL'],runtime=self.runtime)
        context=mock.Mock();context.peer.return_value='p16-test'
        metas=[{'emotion_mix':'{"joy":0.6,"amazement":0.4}','emotion_latent_mode':'sequence','save_server_npy':'false'},
               {'emotion_mix':'{"fear":0.7,"sadness":0.3}'}, {'emotion_va':'[0.7,0.4]'}, {'emotion':'neutral'}]
        requests=[vapb2.AudioRequest(session_id='p16-real',request_id='p16-real',codec=PCM16,sample_rate=16000,channels=1,
                    audio_chunk=(np.sin(np.arange(16000)*.08)*2500).astype(np.int16).tobytes(),meta=m,end_of_stream=i==3)
                  for i,m in enumerate(metas)]
        replies=list(service.StreamSession(iter(requests),context))
        self.assertFalse([r.error_msg for r in replies if r.type==ERROR]);self.assertTrue(replies[-1].end_of_response)
        data=[np.load(io.BytesIO(r.action_npz),allow_pickle=True).item() for r in replies if r.action_npz]
        vectors=np.concatenate([p['emotion_vectors_native'] for p in data])
        self.assertTrue(np.isfinite(vectors).all());self.assertGreater(np.linalg.norm(vectors[:,:16]),1e-5)
        self.assertGreater(np.std(vectors[:,22]),.01)
        self.assertLess(np.linalg.norm(vectors[-1]),np.linalg.norm(vectors[-10]))
        for p in data:self.assertTrue(np.isfinite(p['motor_values_30fps']).all())


if __name__=='__main__':unittest.main()
