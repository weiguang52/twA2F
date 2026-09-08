"""Opt-in real ONNX/protocol tests: A2F_TEST_MODEL=/path/network.onnx."""
import io
import json
import os
import unittest
from unittest import mock

import numpy as np

from a2f_tts_motion_bridge.motion_core.a2f_stream_infer import A2FModelRuntime, _infer_input_array
from a2f_tts_motion_bridge.motion_core.emotion_control import EmotionControl
from a2f_tts_motion_bridge.motion_core.settings import EXPLICIT_EMOTION_LABELS
from a2f_tts_motion_bridge.entrypoints.tts_action_npy_server import VoiceAgentCompat, vapb2, PCM16, ERROR


@unittest.skipUnless(os.environ.get('A2F_TEST_MODEL'), 'Set A2F_TEST_MODEL to enable real model tests')
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=os.environ['A2F_TEST_MODEL']
        cls.runtime=A2FModelRuntime(cls.model,warmup=1)

    def test_all_ten_condition_vectors_affect_real_network(self):
        x=np.zeros((1,1,8320),np.float32); results=[]
        for emotion in EXPLICIT_EMOTION_LABELS:
            inputs={i.name:_infer_input_array(i,x,emotion,1.) for i in self.runtime.inputs_meta}
            y=self.runtime.session.run(None,inputs)[0]
            self.assertTrue(np.isfinite(y).all())
            results.append(y)
        for i in range(10):
            for j in range(i):
                self.assertGreater(float(np.linalg.norm(results[i]-results[j])),1e-6)

    def test_protocol_stream_switches_weights_to_latent_and_preserves_tail(self):
        service=VoiceAgentCompat(self.model,runtime=self.runtime)
        context=mock.Mock(); context.peer.return_value='local-test'
        weights=[0.]*10; weights[4]=.8
        latent=np.linspace(-.1,.1,16).tolist()
        def req(n,meta,end=False):
            samples=(np.sin(np.arange(n)*.08)*2000).astype(np.int16).tobytes()
            return vapb2.AudioRequest(session_id='p06-real',request_id='p06-real',codec=PCM16,
                       sample_rate=16000,channels=1,audio_chunk=samples,meta=meta,end_of_stream=end)
        requests=[req(9600,{'emotion':'happy','intensity':'0.5','save_server_npy':'false'}),
                  req(6400,{'emotion_explicit_weights':json.dumps(weights)}),
                  req(6400,{'emotion_explicit_weights':'null','emotion_implicit_vector':json.dumps(latent)},True)]
        responses=list(service.StreamSession(iter(requests),context))
        self.assertTrue(responses[-1].end_of_response)
        self.assertFalse([r.error_msg for r in responses if r.type==ERROR])
        payloads=[np.load(io.BytesIO(r.action_npz),allow_pickle=True).item() for r in responses if r.action_npz]
        vectors=np.concatenate([p['emotion_vectors_native'] for p in payloads])
        expected=[EmotionControl('happy',.5).vector(),EmotionControl('happy',.5,weights).vector(),
                  EmotionControl('happy',.5,implicit_vector=latent).vector()]
        for vector in expected:
            self.assertTrue(any(np.allclose(v,vector) for v in vectors))
        np.testing.assert_allclose(vectors[-1],expected[-1])
        for p in payloads:
            self.assertEqual(p['motor_values_30fps'].shape[1],13)
            self.assertTrue(np.isfinite(p['motor_values_30fps']).all())

    def test_malformed_meta_returns_error_response(self):
        service=VoiceAgentCompat(self.model,runtime=self.runtime)
        req=vapb2.AudioRequest(session_id='invalid',codec=PCM16,sample_rate=16000,channels=1,
                              audio_chunk=b'\0'*100,end_of_stream=True,meta={'emotion_explicit_weights':'[1]'})
        context=mock.Mock(); context.peer.return_value='local-test'
        responses=list(service.StreamSession(iter([req]),context))
        self.assertEqual(responses[-1].type,ERROR)
        self.assertTrue(responses[-1].end_of_response)


if __name__=='__main__': unittest.main()
