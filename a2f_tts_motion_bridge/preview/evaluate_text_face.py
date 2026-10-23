"""Real A2F comparison using a previously generated KeyframeFace clip.

Keeps audio, random seed and emotion equal; checks lower-face invariance and
exports native/30/90Hz evidence plus optional videos. Not a human quality score.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from a2f_tts_motion_bridge.motion_core.a2f_stream_infer import A2FModelRuntime
from a2f_tts_motion_bridge.motion_core.motion_session import MotorStreamSession
from a2f_tts_motion_bridge.motion_core.types import SessionConfig, AudioChunk
from a2f_tts_motion_bridge.motion_core.text_face import TextFaceClip, REGIONS
from a2f_tts_motion_bridge.outputs.npy_segment_exporter import build_payload_from_recorder
from a2f_tts_motion_bridge.entrypoints.tts_action_npy_server import VoiceAgentCompat, vapb2, PCM16, ERROR


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--clip-dir',type=Path,required=True)
    p.add_argument('--clip',required=True)
    p.add_argument('--wav',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New directory; no overwrite')
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Output directory exists')
    os.environ['A2F_TEXT_FACE_DIR']=str(args.clip_dir.resolve())
    doc=json.loads((args.clip_dir/(args.clip+'.json')).read_text())
    clip=TextFaceClip(doc)
    wav,sr=sf.read(args.wav,dtype='float32')
    if wav.ndim==2:wav=wav.mean(axis=1)
    divisor=math.gcd(sr,16000)
    wav=resample_poly(wav,16000//divisor,sr//divisor)
    samples=int(round(clip.duration*16000))
    wav=np.pad(wav[:samples],(0,max(0,samples-len(wav))))
    pcm=(np.clip(wav,-1,1)*32767).astype('<i2')
    runtime=A2FModelRuntime(args.model,warmup=1)
    args.output.mkdir(parents=True)
    report={'clip':args.clip,'model':args.model,'audio':str(args.wav),
            'duration_s':clip.duration,'provenance':doc['provenance'],
            'keyframe_upper_dofs':[clip.sample(float(t)) for t in clip.times if t<clip.duration],
            'checks':{},'videos':[]}
    sessions={};payloads={}
    common={'auto_blink':'true','behavior_seed':'42'}
    for mode in ('baseline','baseline_repeat','text'):
        meta=dict(common)
        if mode=='text':meta.update(text_face_clip=args.clip,text_face_region='upper_face')
        session=MotorStreamSession(args.model,SessionConfig(session_id='textface-'+mode,save_npy=False,extra=meta),runtime=runtime)
        for i,start in enumerate(range(0,len(pcm),6400)):
            session.process_chunk(AudioChunk(session.config.session_id,i,start//16,
                pcm[start:start+6400].tobytes(),16000,emotion_meta=meta))
        session.finalize();sessions[mode]=session
        for fps in (30,90):
            payload,_=build_payload_from_recorder(session.recorder,0,samples//16,
                session_id=session.config.session_id,segment_index=0,emotion='neutral',
                intensity=1.,is_final=True,export_fps=fps)
            payloads[(mode,fps)]=payload
            np.save(args.output/(mode+'_'+str(fps)+'hz.npy'),payload,allow_pickle=True)
    names=list(sessions['baseline'].motor_names)
    lower=[i for i,k in enumerate(names) if k not in REGIONS['upper_face']]
    upper=[i for i,k in enumerate(names) if k in REGIONS['upper_face']]
    for fps in (30,90):
        a,b=(payloads[(mode,fps)]['motor_values_30fps'] for mode in ('baseline','text'))
        repeat=payloads[('baseline_repeat',fps)]['motor_values_30fps']
        repeat_error=float(np.max(np.abs(a[:,lower]-repeat[:,lower])))
        independent_error=float(np.max(np.abs(a[:,lower]-b[:,lower])))
        # Separate GPU/model repeatability from the mixing invariant. Reuse EXACTLY
        # the same pre-overlay recorder data for a paired export, not new inference.
        recorder=sessions['baseline'].recorder
        original=recorder.behavior_layer
        recorder.behavior_layer=sessions['text'].behavior_layer
        try:
            paired,_=build_payload_from_recorder(recorder,0,samples//16,
                session_id='paired',segment_index=0,emotion='neutral',intensity=1.,
                is_final=True,export_fps=fps)
        finally:recorder.behavior_layer=original
        np.testing.assert_array_equal(a[:,lower],paired['motor_values_30fps'][:,lower])
        # Fail substantial independent-run drift even when the paired check passes.
        if independent_error>max(2*repeat_error+1e-7,1e-6) or independent_error>1e-4:
            raise AssertionError(f'Independent drift {independent_error}, repeat control {repeat_error}')
        difference=float(np.max(np.abs(a[:,upper]-b[:,upper])))
        if difference<=1e-5:raise AssertionError('Text did not affect upper face')
        if not np.isfinite(b).all() or np.min(b)<0 or np.max(b)>1:raise AssertionError('Unsafe DOFs')
        report['checks'][str(fps)+'hz']={'paired_lower_face_max_error':0.,
            'independent_lower_face_max_error':independent_error,'baseline_repeat_max_error':repeat_error,
            'upper_face_max_change':difference,'frames':len(b)}
    # Exercise actual gRPC handler, including repeated clip packets and mid-stream cancellation.
    service=VoiceAgentCompat(args.model,runtime=runtime);context=mock.Mock();context.peer.return_value='textface-evaluation'
    requests=[]
    for i,start in enumerate(range(0,len(pcm),6400)):
        meta=dict(common,text_face_clip=args.clip if start<samples//2 else 'none')
        requests.append(vapb2.AudioRequest(session_id='textface-protocol',codec=PCM16,
            sample_rate=16000,channels=1,audio_chunk=pcm[start:start+6400].tobytes(),
            meta=meta,end_of_stream=start+6400>=len(pcm)))
    responses=list(service.StreamSession(iter(requests),context))
    errors=[r.error_msg for r in responses if r.type==ERROR]
    if errors or not responses[-1].end_of_response:raise AssertionError(errors or 'Missing EOS')
    exported=[np.load(io.BytesIO(r.action_npz),allow_pickle=True).item() for r in responses if r.action_npz]
    schedule=exported[-1]['meta']['text_face_schedule']
    if [e['id'] for e in schedule]!=[args.clip,'none']:raise AssertionError('Repeat/cancel schedule incorrect')
    report['checks']['grpc']={'segments':len(exported),'schedule_ids':[e['id'] for e in schedule],'errors':errors}
    if args.render:
        renderer=Path(__file__).with_name('render_current_head_13dof_npy.py')
        for mode in ('baseline','text'):
            video=args.output/(mode+'.mp4')
            subprocess.run([sys.executable,str(renderer),'--input_npy',str(args.output/(mode+'_90hz.npy')),
                '--output_video',str(video),'--transcript',mode+' / '+args.clip],stdin=subprocess.DEVNULL,check=True)
            report['videos'].append(str(video))
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
    print(json.dumps(report['checks'],indent=2),flush=True)


if __name__=='__main__':main()
