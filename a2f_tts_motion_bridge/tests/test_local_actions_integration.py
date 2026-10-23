"""Opt-in real ONNX and gRPC tests for deterministic local actions."""
import io
import json
import os
import unittest
from unittest import mock
import numpy as np
from a2f_tts_motion_bridge.motion_core.a2f_stream_infer import A2FModelRuntime
from a2f_tts_motion_bridge.motion_core.motion_session import MotorStreamSession
from a2f_tts_motion_bridge.motion_core.local_actions import LocalActionTrack
from a2f_tts_motion_bridge.motion_core.types import SessionConfig,AudioChunk
from a2f_tts_motion_bridge.outputs.npy_segment_exporter import build_payload_from_recorder
from a2f_tts_motion_bridge.entrypoints.tts_action_npy_server import VoiceAgentCompat,vapb2,PCM16,ERROR


@unittest.skipUnless(os.environ.get('A2F_TEST_MODEL'),'Set A2F_TEST_MODEL for real local-action tests')
class RealLocalActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=os.environ['A2F_TEST_MODEL'];cls.runtime=A2FModelRuntime(cls.model,warmup=1)

    def test_real_paired_exports_preserve_non_target_dofs(self):
        action=dict(id='left',action='raise_brow',side='left',strength=.6,hold_ms=2000)
        session=MotorStreamSession(self.model,SessionConfig(session_id='local-pair',save_npy=False,
            extra={'local_actions':json.dumps([action]),'auto_blink':'true'}),runtime=self.runtime)
        pcm=(np.sin(np.arange(32000)*.08)*2000).astype('<i2')
        for i,start in enumerate(range(0,len(pcm),6400)):
            session.process_chunk(AudioChunk('local-pair',i,start//16,pcm[start:start+6400].tobytes(),16000))
        session.finalize();track=session.behavior_layer.local_actions
        for fps in (30,90):
            def export():return build_payload_from_recorder(session.recorder,0,2000,session_id='local-pair',
                segment_index=0,emotion='neutral',intensity=1.,is_final=True,export_fps=fps)[0]
            actual=export()
            session.behavior_layer.local_actions=LocalActionTrack()
            try:baseline=export()
            finally:session.behavior_layer.local_actions=track
            names=session.motor_names
            keep=[i for i,k in enumerate(names) if k not in ('left_inner_brow_y','left_outer_brow_y')]
            np.testing.assert_array_equal(actual['motor_values_30fps'][:,keep],baseline['motor_values_30fps'][:,keep])
            t=actual['frame_times_30fps'];steady=actual['motor_values_30fps'][t>=.21]
            np.testing.assert_allclose(steady[:,names.index('left_inner_brow_y')],.65,atol=1e-7)

    def test_grpc_repeated_chunks_switch_and_cancel(self):
        service=VoiceAgentCompat(self.model,runtime=self.runtime);context=mock.Mock();context.peer.return_value='local-action-test'
        action=dict(id='a',action='raise_brow',side='left',hold_ms=2000)
        meta={'local_actions':json.dumps([action]),'save_server_npy':'false'}
        requests=[]
        for i in range(5):
            packet=meta if i<3 else {'local_action_cancels':json.dumps([dict(id='c',targets='all')])}
            requests.append(vapb2.AudioRequest(session_id='local-proto',codec=PCM16,sample_rate=16000,
                channels=1,audio_chunk=b'\0'*12800,meta=packet,end_of_stream=i==4))
        responses=list(service.StreamSession(iter(requests),context))
        self.assertFalse([r.error_msg for r in responses if r.type==ERROR]);self.assertTrue(responses[-1].end_of_response)
        payloads=[np.load(io.BytesIO(r.action_npz),allow_pickle=True).item() for r in responses if r.action_npz]
        meta=payloads[-1]['meta'];self.assertEqual(len(meta['local_actions']),1)
        self.assertEqual(set(meta['local_action_cancellations']),{'a'})


if __name__=='__main__':unittest.main()
