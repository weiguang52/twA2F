import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from a2f_tts_motion_bridge.motion_core.emotion_control import EmotionControl, canonical_emotion
from a2f_tts_motion_bridge.motion_core.a2f_stream_infer import _infer_input_array, StreamingA2FEngine
from a2f_tts_motion_bridge.motion_core.arkit52_to_motor import _emotion_bias, arkit52_to_current_head_dofs
from a2f_tts_motion_bridge.motion_core.settings import EXPLICIT_EMOTION_LABELS, MOTOR_CFG
from a2f_tts_motion_bridge.motion_core.motion_session import MotorStreamSession
from a2f_tts_motion_bridge.motion_core.motion_recorder import SessionRecorder
from a2f_tts_motion_bridge.motion_core.types import SessionConfig, AudioChunk
from a2f_tts_motion_bridge.entrypoints.tts_action_npy_server import VoiceAgentCompat, vapb2


class EncodingTests(unittest.TestCase):
    def test_business_alias_absolute_dimensions(self):
        for label, dim in dict(happy=22, sad=25, surprise=16, angry=17, disgust=19, fear=20).items():
            with self.subTest(label=label):
                expected = np.zeros(26, np.float32); expected[dim] = 1.25
                np.testing.assert_array_equal(EmotionControl(label, 1.25).vector(), expected)

    def test_all_native_labels_and_neutral(self):
        for i, label in enumerate(EXPLICIT_EMOTION_LABELS):
            vector = EmotionControl(label).vector()
            self.assertEqual(np.flatnonzero(vector).tolist(), [16 + i])
        self.assertFalse(EmotionControl('neutral', 10).vector().any())
        self.assertFalse(EmotionControl('happy', 0).vector().any())

    def test_explicit_overrides_label_and_scales_without_normalizing(self):
        values = [0., 0., 0., 0., 0., 0., 0.7, 0., 0., 0.3]
        control = EmotionControl('angry', 2, explicit_weights=values)
        np.testing.assert_allclose(control.vector()[16:], np.asarray(values)*2)
        self.assertFalse(control.vector()[:16].any())

    def test_latent_signed_passthrough_and_combined_blocks(self):
        latent = np.linspace(-1, 1, 16)
        control = EmotionControl('happy', 0, implicit_vector=latent)
        np.testing.assert_allclose(control.vector()[:16], latent)
        self.assertFalse(control.vector()[16:].any())
        control = EmotionControl('happy', 2, [0.1]*10, latent)
        np.testing.assert_allclose(control.vector()[16:], 0.2)

    def test_invalid_controls_rejected(self):
        invalid = [dict(emotion='typo'), dict(intensity=-1), dict(intensity=float('nan')),
                   dict(intensity=float('inf')), dict(explicit_weights=[0]*9),
                   dict(explicit_weights=[-0.1]*10), dict(explicit_weights=[[0]*10]),
                   dict(implicit_vector=[float('inf')]*16), dict(implicit_vector=[0]*15),
                   dict(explicit_weights='not json'), dict(bias_scale=-1)]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                EmotionControl(**values)

    def test_dynamic_emotion_name_precedes_audio_heuristic(self):
        x = np.zeros((2,1,8320), np.float32)
        for shape in (['batch',26], ['batch',1,26], ['batch',1,'n']):
            result = _infer_input_array(SimpleNamespace(name='emotion', shape=shape),x,'happy',0.5)
            self.assertEqual(result.shape, (2,26) if len(shape)==2 else (2,1,26))
            self.assertTrue(np.all(result[...,22]==0.5))
        with self.assertRaises(ValueError):
            _infer_input_array(SimpleNamespace(name='emotion',shape=[1,7]),x)
        np.testing.assert_array_equal(_infer_input_array(SimpleNamespace(name='input',shape=['b',1,8320]),x),x)

    def test_tail_flush_retains_explicit_and_latent_vectors(self):
        inputs = [SimpleNamespace(name='input',shape=['b',1,8320]), SimpleNamespace(name='emotion',shape=['b',1,26])]
        session = mock.Mock(); session.run.return_value=[np.zeros((1,1,169),np.float32)]
        runtime=SimpleNamespace(session=session,providers=['CPU'],session_init_sec=0,inputs_meta=inputs,warmup_sec=0)
        engine=StreamingA2FEngine('fake',16000,runtime=runtime)
        c=EmotionControl('happy',1.25,[0.2]*10,[0.3]*16)
        engine.push_audio_chunk(np.zeros(8400,np.float32),'happy',1.25,control=c)
        tail=engine.finalize()
        self.assertTrue(tail)
        for call in session.run.call_args_list:
            np.testing.assert_allclose(call.args[1]['emotion'][0,0],c.vector())


class ProtocolTests(unittest.TestCase):
    def test_session_delivers_control_to_retarget_and_records_it(self):
        control=EmotionControl('joy',0.5,[0.1]*10)
        item=dict(time_code_s=.26,weights169=np.zeros(169,np.float32),audio_rms=.1,
                  emotion='joy',intensity=.5,emotion_control=control)
        prefix='a2f_tts_motion_bridge.motion_core.motion_session.'
        with mock.patch(prefix+'StreamingA2FEngine') as engine_cls, mock.patch(prefix+'OnlineRetargeter') as retarget_cls:
            engine_cls.return_value.push_audio_chunk.return_value=[item]
            engine_cls.return_value.finalize.return_value=[dict(item,time_code_s=.36)]
            engine_cls.return_value.current_emotion='joy'
            engine_cls.return_value.current_intensity=.5
            retarget=retarget_cls.return_value
            features={k:0. for k in ['jaw_open','mouth_round','mouth_wide','mouth_left_right','upper_face_activity','eye_activity','blink_like']}
            retarget.update.return_value=(features,{n:c['neutral'] for n,c in MOTOR_CFG.items()})
            retarget.last_blendshapes={}; retarget.last_debug=control.metadata()
            session=MotorStreamSession('fake',SessionConfig('test',emotion='happy',intensity=.5,
                                         extra={'emotion_explicit_weights':json.dumps([.1]*10)}))
            session.process_chunk(AudioChunk('test',0,0,b'\0'*6400,16000))
            session.finalize()
            self.assertEqual(retarget.update.call_count,2)
            for call in retarget.update.call_args_list: self.assertIs(call.kwargs['control'],control)
            np.testing.assert_allclose(session.recorder.emotion_vectors[0],control.vector())

    def test_meta_sticky_reset_and_atomic_validation(self):
        session=MotorStreamSession.__new__(MotorStreamSession)
        session.config=SessionConfig('test')
        session.emotion_control=EmotionControl()
        session.recorder=SessionRecorder()
        exporter=mock.Mock(); state=dict(emotion='neutral',intensity=1.,sample_rate=16000,channels=1)
        service=VoiceAgentCompat('fake')
        def update(meta):
            service._update_control_state(vapb2.AudioRequest(meta=meta),session,exporter,state)
        update({'emotion_explicit_weights':json.dumps([0.1]*10),'emotion_implicit_vector':json.dumps([0.2]*16)})
        update({'intensity':'0.5'})
        np.testing.assert_allclose(session.emotion_control.vector()[16:],0.05)
        previous=session.emotion_control
        with self.assertRaises(ValueError): update({'emotion_explicit_weights':'[1]'})
        self.assertIs(session.emotion_control,previous)
        update({'emotion':'fear'})
        self.assertIsNone(session.emotion_control.implicit_vector)
        self.assertIsNone(session.emotion_control.explicit_weights)
        self.assertEqual(session.emotion_control.vector()[20],0.5)
        update({'emotion_explicit_weights':json.dumps([0.1]*10)})
        update({'emotion_explicit_weights':'null'})
        self.assertEqual(session.emotion_control.vector()[20],0.5)


class BiasTests(unittest.TestCase):
    def test_all_ten_have_distinct_bounded_biases(self):
        profiles=[]
        for e in EXPLICIT_EMOTION_LABELS:
            bias=_emotion_bias(e,1)
            self.assertTrue(any(bias.values()))
            profiles.append(tuple(bias.values()))
            self.assertFalse(any(_emotion_bias(e,0).values()))
            dofs=arkit52_to_current_head_dofs({},emotion=e,intensity=10)
            self.assertTrue(all(0<=x<=1 for x in dofs.values()))
        self.assertEqual(len(set(profiles)),10)

    def test_model_only_and_no_opposing_bias(self):
        bs={'mouthFrownLeft':0.8,'eyeWideLeft':0.8}
        model=arkit52_to_current_head_dofs(bs,emotion='neutral',bias_scale=0)
        self.assertEqual(model,arkit52_to_current_head_dofs(bs,emotion='happy',bias_scale=0))
        combined=arkit52_to_current_head_dofs(bs,emotion='happy')
        self.assertEqual(combined['left_mouth_y'],model['left_mouth_y'])

    def test_mixed_explicit_not_stale_label_controls_bias(self):
        weights=[0.]*10; weights[EXPLICIT_EMOTION_LABELS.index('fear')]=1
        c=EmotionControl('happy',1,weights)
        self.assertEqual(arkit52_to_current_head_dofs({},control=c),
                         arkit52_to_current_head_dofs({},emotion='fear'))
        latent=EmotionControl('happy',1,implicit_vector=[0.2]*16)
        self.assertEqual(arkit52_to_current_head_dofs({},control=latent),
                         arkit52_to_current_head_dofs({},emotion='neutral'))


if __name__=='__main__': unittest.main()
