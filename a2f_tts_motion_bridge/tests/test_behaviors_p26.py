import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from a2f_tts_motion_bridge.motion_core.behavior_layer import BehaviorLayer
from a2f_tts_motion_bridge.motion_core.style_overlay import style_offsets
from a2f_tts_motion_bridge.motion_core.idle_behaviors import IdleBehaviors,blink_envelope
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG
from a2f_tts_motion_bridge.motion_core.arkit52_to_motor import _emotion_bias,arkit52_to_current_head_dofs,OnlineRetargeter
from a2f_tts_motion_bridge.motion_core.retarget_calibration import fit_calibration,RetargetCalibration,NEUTRAL,DOFS
from a2f_tts_motion_bridge.motion_core.motion_recorder import SessionRecorder
from a2f_tts_motion_bridge.outputs.npy_segment_exporter import build_payload_from_recorder
from a2f_tts_motion_bridge.motion_core.motion_session import MotorStreamSession
from a2f_tts_motion_bridge.motion_core.types import SessionConfig,AudioChunk

BASE={k:v['neutral'] for k,v in MOTOR_CFG.items()}


class StyleTests(unittest.TestCase):
    def test_preview_swaps_screen_sides_without_changing_motor_data(self):
        from a2f_tts_motion_bridge.preview.render_current_head_13dof_npy import front_view_dofs
        actual=dict(BASE,left_upper_lid_y=0.,left_lower_lid_y=1.)
        view=front_view_dofs(actual)
        self.assertEqual(view['right_upper_lid_y'],0.)
        self.assertEqual(view['left_upper_lid_y'],.5)
        self.assertEqual(actual['left_upper_lid_y'],0.)
        self.assertEqual(actual['right_upper_lid_y'],.5)

    def test_laugh_frequency_and_return_to_rest(self):
        ts=np.arange(0,.8,1/90)
        jaw=np.array([style_offsets('laugh',t,.8)[0].get('jaw_y',0) for t in ts])
        peaks=np.where((jaw[1:-1]>jaw[:-2])&(jaw[1:-1]>jaw[2:]))[0]+1
        self.assertGreaterEqual(len(peaks),3)
        self.assertAlmostEqual(float(np.mean(np.diff(ts[peaks[:3]]))),1/4.5,delta=.02)
        self.assertEqual(style_offsets('laugh',1,.8),({},{}))

    def test_style_patterns_distinct_finite(self):
        curves=[]
        for kind in ('smile','laugh','sigh','sob'):
            values=np.array([[style_offsets(kind,t,2,seed=4)[0].get(k,0) for k in BASE] for t in np.arange(0,2,1/90)])
            self.assertTrue(np.isfinite(values).all());self.assertLessEqual(np.abs(values).max(),.3)
            curves.append(values)
        for a in range(4):
            for b in range(a):self.assertGreater(np.linalg.norm(curves[a]-curves[b]),.1)

    def test_wink_unilateral_brow_flash_and_suppress(self):
        layer=BehaviorLayer()
        layer.schedule({'behavior_events':[{'id':'w','type':'wink','side':'right'}]},0)
        pose=layer.tick(.12)
        self.assertLess(pose['right_upper_lid_y'],.05)
        self.assertEqual(pose['left_upper_lid_y'],.5)
        brow,_=style_offsets('brow_flash',.06,.12)
        self.assertAlmostEqual(brow['left_outer_brow_y'],.12)
        self.assertGreater(style_offsets('smile_suppress',.1,.5)[0]['left_mouth_y'],0)
        self.assertLess(style_offsets('smile_suppress',.35,.5)[0]['left_mouth_y'],0)

    def test_speech_yields_mouth_not_eyes(self):
        layer=BehaviorLayer();layer.schedule({'style':'laugh'},0)
        quiet=layer.apply(BASE,.1,0);speech=layer.apply(BASE,.1,1)
        self.assertAlmostEqual(speech['jaw_y'],quiet['jaw_y']*.2)
        self.assertEqual(speech['left_lower_lid_y'],quiet['left_lower_lid_y'])

    def test_headroom_all_overlays_bounded(self):
        layer=BehaviorLayer();layer.schedule({'idle_mode':'listening','behavior_events':[
            {'id':str(i),'type':'laugh'} for i in range(30)]},0)
        for t in np.arange(0,1,1/90):
            out=layer.apply({k:.99 for k in BASE},t)
            self.assertTrue(all(0<=v<=1 for v in out.values()))


class BehaviorTests(unittest.TestCase):
    def test_seed_replay_and_sampling_partition_independent(self):
        a,b=BehaviorLayer(22),BehaviorLayer(22)
        for layer in (a,b):layer.schedule({'idle_mode':'listening'},0)
        a.tick(100)  # Future inspection must not alter earlier samples.
        for t in np.arange(0,15,.033):self.assertEqual(a.tick(t),b.tick(t))

    def test_blinks_double_and_no_audio_clock(self):
        idle=IdleBehaviors(1);idle.ensure_until(1000)
        gaps=np.diff(idle.blinks)
        self.assertTrue(np.any(np.isclose(gaps,.32)))
        self.assertGreater(len(idle.blinks),250);self.assertLess(len(idle.blinks),500)
        self.assertGreater(blink_envelope(.06),.99)
        self.assertEqual(blink_envelope(.4),0)
        layer=BehaviorLayer();layer.schedule({'idle_mode':'listening'},0)
        self.assertNotEqual(layer.tick(1),layer.tick(2))

    def test_atomic_validation_idempotent_events(self):
        layer=BehaviorLayer();meta={'behavior_events':'[{"id":"x","type":"wink"}]'}
        layer.schedule(meta,0);layer.schedule(meta,1)
        self.assertEqual(len(layer.events),1)
        invalid=[{'behavior_events':'bad'}, {'style':'bad'}, {'style':'laugh','style_duration_ms':900},
                 {'idle_gain':-1},{'auto_blink':1},{'behavior_seed':2},
                 {'behavior_events':[{'id':'x','type':'sigh'}]},
                 {'behavior_events':[{'id':'z','type':'brow_flash','duration_ms':200}]}]
        before=layer.metadata()
        for meta in invalid:
            with self.assertRaises(ValueError):layer.schedule(meta,2)
            self.assertEqual(layer.metadata(),before)

    def test_invalid_output_inputs_and_timeline_metadata(self):
        layer=BehaviorLayer();layer.schedule({'idle_mode':'idle'},1)
        layer.schedule({'idle_mode':'listening'},3)
        self.assertEqual(len(layer.metadata()['behavior_config_timeline']),3)
        with self.assertRaises(ValueError):layer.tick(float('nan'))
        with self.assertRaises(ValueError):layer.apply(dict(BASE,jaw_y=float('nan')),0)
        with self.assertRaises(ValueError):layer.apply_array([0],[],[0])

    def test_future_event_and_repeat_style_no_restart(self):
        layer=BehaviorLayer();layer.schedule({'style':'laugh'},0)
        layer.schedule({'style':'laugh'},.2)
        self.assertEqual(len(layer.events),1)
        layer.schedule({'behavior_events':[{'id':'f','type':'wink','offset_ms':500}]},1)
        self.assertEqual(layer.tick(1.3)['left_upper_lid_y'],.5)
        self.assertLess(layer.tick(1.62)['left_upper_lid_y'],.05)

    def test_cancel_removes_previously_queued_future_events(self):
        layer=BehaviorLayer()
        layer.schedule({'behavior_events':[{'id':'future','type':'wink','offset_ms':2000}]},0)
        layer.schedule({'style':'none'},1)
        self.assertEqual(layer.tick(2.12),BASE)

    def test_audio_peak_cooldown(self):
        layer=BehaviorLayer();layer.schedule({'auto_brow':True},0)
        for t,rms in [(0,0),(.1,.08),(.2,0),(.3,.1),(.8,0),(.9,.2)]:layer.observe_audio(t,rms)
        self.assertEqual(len(layer.events),2)

    def test_asymmetric_bias_and_unused_channels(self):
        b=_emotion_bias('cheekiness',1,left_scale=0,right_scale=1.5)
        self.assertEqual(b['left_mouth_y'],0);self.assertGreater(b['right_mouth_y'],0)
        for channel in ('noseSneerLeft','cheekPuff','mouthShrugUpper','mouthShrugLower'):
            out=arkit52_to_current_head_dofs({channel:1},bias_scale=0)
            self.assertNotEqual(out,BASE)


class DenseExportTests(unittest.TestCase):
    def recorder(self):
        r=SessionRecorder();r.behavior_layer=BehaviorLayer()
        r.behavior_layer.schedule({'style':'laugh'},0)
        features={k:0. for k in ('jaw_open','mouth_round','mouth_wide','mouth_left_right','upper_face_activity','eye_activity','blink_like')}
        for t in np.arange(0,1.01,.1):
            combined=r.behavior_layer.apply(BASE,t)
            r.append_frame(t,np.zeros(169),features,combined,audio_rms=0,base_motors=BASE)
        r.append_audio_16k(np.zeros(16000));return r

    def test_export_90hz_evaluates_carrier_not_10hz_interpolation(self):
        r=self.recorder()
        p,n=build_payload_from_recorder(r,0,1000,session_id='dense',segment_index=0,emotion='neutral',intensity=1,is_final=True,export_fps=90)
        expected=np.array([r.behavior_layer.tick(float(t))['jaw_y'] for t in p['frame_times_30fps']])
        np.testing.assert_allclose(p['motor_values_30fps'][:,-1],expected,atol=1e-7)
        self.assertEqual(n,90)
        naive=np.interp(p['frame_times_30fps'],r.frame_times,np.array(r.motor_values)[:,-1])
        self.assertGreater(np.max(np.abs(naive-expected)),.02)

    def test_save_30hz_matches_overlay(self):
        r=self.recorder()
        with tempfile.TemporaryDirectory() as d:
            p=np.load(r.save(str(Path(d)/'test.npy')),allow_pickle=True).item()
        expected=[r.behavior_layer.tick(float(t))['jaw_y'] for t in p['frame_times_30fps']]
        np.testing.assert_allclose(p['motor_values_30fps'][:,-1],expected,atol=1e-7)
        self.assertIn('motor_base_values_native',p)


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(1);x=rng.uniform(0,.3,(500,52))
        w=rng.normal(0,.02,(13,52));w[-1]=np.abs(w[-1])
        y=x@w.T+NEUTRAL
        return x,y

    def test_ridge_recovers_pairs_and_preserves_neutral(self):
        x,y=self.fixture();fit=fit_calibration(x,y,'test_robot',.001)
        self.assertLess(max(fit['validation']['holdout_rmse']),.001)
        self.assertFalse(fit['approved'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'calib_test_robot.json';path.write_text(json.dumps(fit))
            with self.assertRaises(ValueError):RetargetCalibration(path)
            c=RetargetCalibration(path,allow_candidate=True)
            self.assertEqual(c.predict({}),BASE)
            with self.assertRaises(ValueError):RetargetCalibration(path,'wrong',allow_candidate=True)
            fit['approved']=True;path.write_text(json.dumps(fit))
            rt=OnlineRetargeter(calibration_path=path)
            self.assertIsNotNone(rt.calibration)

    def test_invalid_training_and_paths_rejected(self):
        x,y=self.fixture()
        for robot in ('../escape','','a/b'):
            with self.assertRaises(ValueError):fit_calibration(x,y,robot)
        with self.assertRaises(ValueError):fit_calibration(x[:10],y[:10],'r')
        with self.assertRaises(ValueError):fit_calibration(x,y,'r',alpha=0)
        with self.assertRaises(ValueError):fit_calibration(x,y,'r',groups=np.zeros(len(x)))


@unittest.skipUnless(os.environ.get('A2F_TEST_MODEL'),'Set A2F_TEST_MODEL')
class RealP26Tests(unittest.TestCase):
    def test_grpc_behavior_events_are_idempotent_and_exported(self):
        from a2f_tts_motion_bridge.entrypoints.tts_action_npy_server import VoiceAgentCompat,vapb2,PCM16,ERROR
        service=VoiceAgentCompat(os.environ['A2F_TEST_MODEL'])
        meta={'style':'laugh','behavior_events':'[{"id":"wink1","type":"wink","offset_ms":100}]','save_server_npy':'false'}
        requests=[vapb2.AudioRequest(session_id='p26-grpc',request_id='p26-grpc',codec=PCM16,sample_rate=16000,
                    channels=1,audio_chunk=b'\0'*32000,meta=meta,end_of_stream=i==1) for i in range(2)]
        context=mock.Mock();context.peer.return_value='test'
        replies=list(service.StreamSession(iter(requests),context))
        self.assertFalse([r.error_msg for r in replies if r.type==ERROR])
        payloads=[np.load(io.BytesIO(r.action_npz),allow_pickle=True).item() for r in replies if r.action_npz]
        self.assertTrue(payloads)
        self.assertEqual(len(payloads[-1]['meta']['behavior_events']),2)
        for p in payloads:
            self.assertIn('motor_base_values_native',p)
            self.assertTrue(np.isfinite(p['motor_values_30fps']).all())

    def test_stream_styles_recording_and_independent_idle_tick(self):
        model=os.environ['A2F_TEST_MODEL']
        s=MotorStreamSession(model,SessionConfig('p26-real',extra={'style':'laugh','idle_mode':'listening'}))
        frames=s.process_chunk(AudioChunk('p26-real',0,0,b'\0'*32000,16000))
        frames+=s.finalize()
        self.assertTrue(frames);self.assertTrue(all(np.isfinite(f.motor_values).all() for f in frames))
        self.assertEqual(len(s.recorder.base_motor_values),len(s.recorder.motor_values))
        count=s.engine.infer_frames;idle=s.tick_idle(3.1)
        self.assertEqual(s.engine.infer_frames,count);self.assertEqual(len(idle.motor_values),13)
        previous=s.emotion_control;nevents=len(s.behavior_layer.events)
        with self.assertRaises(ValueError):s.update_emotion_meta({'emotion':'fear','behavior_events':'bad'})
        self.assertIs(s.emotion_control,previous);self.assertEqual(len(s.behavior_layer.events),nevents)


if __name__=='__main__':unittest.main()
