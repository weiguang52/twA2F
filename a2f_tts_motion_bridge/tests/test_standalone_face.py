import io
import json
import subprocess
import sys
import unittest
import numpy as np
from a2f_tts_motion_bridge.motion_core.standalone_face import PromptFaceSession, ACTIONS, TONES
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG
from a2f_tts_motion_bridge.entrypoints.prompt_face_server import PromptFaceService
from a2f_tts_motion_bridge.protocol.voiceagent_generated import ActionImageMediaStream_pb2 as pb


def plan(action='smile', **kwargs):
    a = dict(action=action, side='both', strength=.6, tone='neutral', attack_ms=200,
             hold_ms=300, release_ms=300, offset_ms=0)
    a.update(kwargs)
    return dict(actions=[a], unsupported=False)


class StandaloneFaceTests(unittest.TestCase):
    def test_no_inference_dependency(self):
        result = subprocess.run([sys.executable, '-c',
            "from a2f_tts_motion_bridge.motion_core.standalone_face import PromptFaceSession; import sys; assert 'onnxruntime' not in sys.modules; assert 'torch' not in sys.modules; assert not any(k.endswith('.motion_session') for k in sys.modules)"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_actions_tones_sides(self):
        for action in ACTIONS:
            for tone in TONES:
                for side in ('left', 'right', 'both'):
                    if action == 'wink' and side == 'both': continue
                    s = PromptFaceSession(plan(action, tone=tone, side=side))
                    for t in (0, .1, .3, .6, .8, 1):
                        values = s.sample(t)
                        self.assertEqual(list(values), list(MOTOR_CFG))
                        self.assertTrue(all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()))
                    self.assertEqual(s.sample(0), s.neutral)
                    self.assertEqual(s.sample(1), s.neutral)

    def test_degree_and_tone(self):
        weak = PromptFaceSession(plan(strength=.25)).sample(.3)
        strong = PromptFaceSession(plan(strength=.85)).sample(.3)
        self.assertGreater(strong['left_mouth_y'], weak['left_mouth_y'])
        sad = PromptFaceSession(plan('frown', tone='sad')).sample(.3)
        angry = PromptFaceSession(plan('frown', tone='angry')).sample(.3)
        self.assertGreater(sad['left_inner_brow_y'], .5)
        self.assertLess(angry['left_inner_brow_y'], .5)

    def test_unilateral_and_blink(self):
        s = PromptFaceSession(plan('blink', side='left', strength=1, attack_ms=80, hold_ms=40, release_ms=140))
        pose = s.sample(.1)
        self.assertEqual(pose['left_upper_lid_y'], 0)
        self.assertEqual(pose['left_lower_lid_y'], 1)
        self.assertEqual(pose['right_upper_lid_y'], .5)
        self.assertEqual(pose['jaw_y'], 0)

    def test_export_contract(self):
        for fps in (30, 90):
            s = PromptFaceSession(plan(), 'contract')
            segments = list(s.segments(fps, 320))
            self.assertEqual([x.segment_index for x in segments], list(range(len(segments))))
            self.assertEqual(sum(x.is_final for x in segments), 1)
            self.assertEqual(segments[-1].end_ms, s.duration_ms)
            for i, seg in enumerate(segments):
                p = np.load(io.BytesIO(seg.npy_bytes), allow_pickle=True).item()
                self.assertEqual(p['meta']['motor_names'], list(MOTOR_CFG))
                self.assertEqual(p['motor_values_30fps'].shape, (seg.frame_count, 13))
                self.assertEqual(p['motor_values_30fps'].dtype, np.float32)
                self.assertEqual(p['arkit52_30fps'].shape, (seg.frame_count, 52))
                self.assertEqual(p['meta']['export_fps'], fps)
                self.assertEqual(len(p['audio_16k_mono']), 0)
                expected = [[s.sample(float(t))[n] for n in MOTOR_CFG] for t in p['frame_times_30fps']]
                np.testing.assert_allclose(p['motor_values_30fps'], expected, atol=3e-8)
                if i: self.assertEqual(seg.start_ms, segments[i-1].end_ms)

    def test_invalid(self):
        for p in (plan(strength=float('nan')), plan('wink'), plan(tone='unknown'),
                  plan(attack_ms=True), dict(actions=[], unsupported=True), plan('head_tilt')):
            with self.assertRaises(ValueError): PromptFaceSession(p)

    def test_neutral_tail(self):
        for fps in (30, 90):
            last = list(PromptFaceSession(plan()).segments(fps))[-1]
            payload = np.load(io.BytesIO(last.npy_bytes), allow_pickle=True).item()
            np.testing.assert_array_equal(payload['motor_values_30fps'][-1],
                np.asarray([v['neutral'] for v in MOTOR_CFG.values()], dtype=np.float32))

    def test_real_grpc_transport(self):
        import grpc
        from concurrent.futures import ThreadPoolExecutor
        from a2f_tts_motion_bridge.protocol.voiceagent_generated import ActionImageMediaStream_pb2_grpc as rpc
        server = grpc.server(ThreadPoolExecutor(max_workers=1))
        rpc.add_VoiceAgentServicer_to_server(PromptFaceService(), server)
        port = server.add_insecure_port('127.0.0.1:0')
        server.start()
        try:
            with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                stub = rpc.VoiceAgentStub(channel)
                req = pb.AudioRequest(request_id='transport', end_of_stream=True,
                    meta={'face_plan': json.dumps(plan('blink')), 'export_fps': '90'})
                replies = list(stub.StreamSession(iter([req]), timeout=10))
                self.assertTrue(replies[-1].end_of_response)
                self.assertTrue(all(r.type == pb.ACTION for r in replies[:-1]))
                self.assertGreater(len(replies), 1)
        finally:
            server.stop(0).wait()

    def test_service_wire(self):
        class Context:
            def is_active(self): return True
        req = pb.AudioRequest(request_id='wire', end_of_stream=True,
                              meta={'face_plan': json.dumps(plan())})
        responses = list(PromptFaceService().StreamSession(iter([req]), Context()))
        self.assertTrue(responses[-1].end_of_response)
        for response in responses[:-1]:
            decoded = pb.AudioResponse.FromString(response.SerializeToString())
            self.assertEqual(decoded.type, pb.ACTION)
            self.assertEqual(np.load(io.BytesIO(decoded.action_npz), allow_pickle=True).item()['meta']['motor_names'], list(MOTOR_CFG))
        req.audio_chunk = b'bad'
        rejected = list(PromptFaceService().StreamSession(iter([req]), Context()))
        self.assertEqual(rejected[0].type, pb.ERROR)
        self.assertTrue(rejected[0].end_of_response)


if __name__ == '__main__': unittest.main()
