"""Remote reproducible P16 smoke/calibration artifacts; no human score claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from ..motion_core.a2f_stream_infer import A2FModelRuntime, _infer_input_array
from ..motion_core.audio_stream import resample_linear
from ..motion_core.motion_session import MotorStreamSession
from ..motion_core.motion_recorder import SessionRecorder
from ..motion_core.settings import EXPLICIT_EMOTION_LABELS
from ..motion_core.types import SessionConfig, AudioChunk


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--audio',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    out=Path(args.output)
    audio,sr=sf.read(args.audio,dtype='float32')
    if audio.ndim==2:audio=audio.mean(axis=1)
    audio=resample_linear(audio,sr,16000);sr=16000
    if len(audio)<160000 or not np.isfinite(audio).all():
        raise ValueError('Use finite audio at least 10 seconds long')
    out.mkdir(parents=True,exist_ok=False)
    audio=audio[:160000].copy()
    for a,b in ((0,1),(4,5),(8,10)):audio[a*sr:b*sr]=0
    sf.write(out/'evaluation_audio.wav',audio,sr,subtype='PCM_16')
    runtime=A2FModelRuntime(args.model,warmup=1)
    report=dict(model_sha256=sha(args.model),db_sha256=sha(runtime.emotion_latent.path),
                providers=runtime.providers,source_audio_sha256=sha(args.audio),
                silence_intervals_s=[[0,1],[4,5],[8,10]],
                latent_calibration=[],cases=[],human_recognition_rate=None,
                note='Engineering smoke/calibration only; no human or physical robot validation.')
    report['code_sha256']={str(p.relative_to(Path(__file__).resolve().parents[1])):sha(p)
                          for p in Path(__file__).resolve().parents[1].joinpath('motion_core').glob('*.py')}
    # Same audio input per comparison, no explicit block or retarget bias: isolate
    # the ten actor centroids' actual model effect over a gain sweep.
    for stimulus,x in [('silence',np.zeros((1,1,8320),np.float32)),
                       ('speech',audio[16000:24320].reshape(1,1,8320))]:
        base_inputs={i.name:_infer_input_array(i,x,emotion_vector=np.zeros(26)) for i in runtime.inputs_meta}
        baseline=runtime.session.run(None,base_inputs)[0]
        for index,label in enumerate(EXPLICIT_EMOTION_LABELS):
            for gain in (.25,.5,1.):
                v=np.r_[runtime.emotion_latent.anchors[index]*gain,np.zeros(10)].astype(np.float32)
                inputs={i.name:_infer_input_array(i,x,emotion_vector=v) for i in runtime.inputs_meta}
                y=runtime.session.run(None,inputs)[0]
                if not np.isfinite(y).all():raise AssertionError('Non-finite ONNX output')
                report['latent_calibration'].append(dict(stimulus=stimulus,emotion=label,gain=gain,
                    implicit_l2=float(np.linalg.norm(v[:16])),raw169_l2_vs_zero=float(np.linalg.norm(y-baseline))))
    for name,mode,speech_scale in [('centroid','centroid','.25'),('sequence','sequence','.25'),
                                  ('random','random','.25'),('sequence_ungated','sequence','1')]:
        initial={'emotion_mix':'{"joy":0.6,"amazement":0.4}', 'emotion_latent_mode':mode,
                 'emotion_latent_speech_scale':speech_scale,'emotion_latent_seed':'601'}
        config=SessionConfig('p16_'+name,npy_save_dir=str(out),extra=initial)
        session=MotorStreamSession(args.model,config,runtime=runtime)
        script={0:initial,10:{'emotion_mix':'{"anger":0.7,"fear":0.3}'},
                20:{'emotion_va':'[0.8,0.4]'},30:{'emotion':'neutral'}}
        for index,start in enumerate(range(0,len(audio),3200)):
            pcm=(np.clip(audio[start:start+3200],-1,1)*32767).astype('<i2').tobytes()
            session.process_chunk(AudioChunk(config.session_id,index,int(start/sr*1000),pcm,sr,
                                              emotion_meta=script.get(index,{})))
        session.finalize()
        path=Path(session.save_artifact())
        payload=np.load(path,allow_pickle=True).item()
        t=payload['frame_times_native'];v=payload['emotion_vectors_native'];m=payload['motor_values_native']
        if not (np.isfinite(v).all() and np.isfinite(m).all() and np.all((m>=0)&(m<=1))):
            raise AssertionError('Invalid recording')
        payload['frame_times_90hz'],payload['motor_values_90hz']=SessionRecorder._interp_to_fps(t,m,90)
        payload['meta']=dict(payload['meta'],export_fps=90)
        path90=out/(path.stem+'_90hz.npy');np.save(path90,payload,allow_pickle=True)
        windows={}
        for a,b in ((0,1),(1,2),(2,4),(4,5),(5,6),(6,8),(8,10)):
            mask=(t>=a)&(t<b)
            windows[f'{a}-{b}']=dict(dof_mean=m[mask].mean(0).tolist(),
                latent_l2_mean=float(np.linalg.norm(v[mask,:16],axis=1).mean()),
                jaw_mean=float(m[mask,-1].mean()))
        case=dict(name=name,meta=initial,events={str(k*.2):v for k,v in script.items()},
                  npy=path.name,npy90=path90.name,frames=len(t),dof_names=payload['meta']['motor_names'],windows=windows)
        if args.render:
            video=out/(path.stem+'.mp4')
            subprocess.run([sys.executable,'-m','a2f_tts_motion_bridge.preview.render_motor_npy',
                            '--input_npy',str(path),'--output_video',str(video),'--blind_id',name],
                           stdin=subprocess.DEVNULL,check=True)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
            streams=info['streams'];vs=next(s for s in streams if s['codec_type']=='video')
            assert int(vs['nb_frames'])==300
            assert any(s['codec_type']=='audio' for s in streams)
            case.update(video=video.name,video_frames=int(vs['nb_frames']),duration=float(info['format']['duration']))
        report['cases'].append(case)
        print('P16 sample complete:',name,flush=True)
    report['completed']=True
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Saved',out/'evaluation.json')


if __name__=='__main__':main()
