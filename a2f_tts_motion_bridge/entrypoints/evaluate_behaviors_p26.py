"""Reproducible P26 procedural previews + explicitly synthetic calibration demo."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from ..motion_core.behavior_layer import BehaviorLayer
from ..motion_core.motion_recorder import SessionRecorder
from ..motion_core.arkit52_to_motor import arkit52_to_current_head_dofs,debug_features_from_arkit52
from ..motion_core.retarget_calibration import fit_calibration,ARKIT_52_NAMES,DOFS
from ..motion_core.settings import MOTOR_CFG
from ..motion_core.motion_session import MotorStreamSession
from ..motion_core.types import SessionConfig,AudioChunk
from ..outputs.npy_segment_exporter import build_payload_from_recorder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--render',action='store_true');p.add_argument('--model')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    base={k:c['neutral'] for k,c in MOTOR_CFG.items()}
    report=dict(human_evaluation=None,cases=[],calibration_note='Synthetic teacher only; NOT a physical robot calibration')
    report['code_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in Path(__file__).resolve().parents[1].joinpath('motion_core').glob('*.py')}
    def event(kind,offset,i,**extra):return dict(id=f'{kind}-{i}',type=kind,offset_ms=offset,**extra)
    cases={
        'laugh':{'behavior_events':[event('laugh',t,i) for i,t in enumerate((500,2000,3500,5000))]},
        'sigh':{'behavior_events':[event('sigh',t,i) for i,t in enumerate((500,3500))]},
        'sob':{'behavior_events':[event('sob',t,i) for i,t in enumerate((500,2500,4500))]},
        'listening':{'idle_mode':'listening'},
        'micro':{'behavior_events':[event('brow_flash',1000,0),event('wink',2000,1,side='left'),
                                   event('smile_suppress',3000,2),event('wink',4500,3,side='right')]},
        'speech_yield':{'behavior_events':[event('laugh',t,i) for i,t in enumerate((500,2000,3500,5000))]},
    }
    def save_case(name,r,kind):
        path=out/('p26_'+name+'.npy');r.save(str(path))
        dense,n=build_payload_from_recorder(r,0,6000,session_id=name,segment_index=0,
                        emotion='neutral',intensity=1,is_final=True,export_fps=90)
        path90=out/('p26_'+name+'_90hz.npy');np.save(path90,dense,allow_pickle=True)
        m=dense['motor_values_30fps']
        assert m.shape==(540,13) and np.isfinite(m).all() and np.all((m>=0)&(m<=1))
        item=dict(name=name,source=kind,npy=path.name,npy90=path90.name,
                  dof_mean=m.mean(0).tolist(),dof_std=m.std(0).tolist(),dof_names=DOFS)
        if args.render:
            video=out/('p26_'+name+'.mp4')
            subprocess.run([sys.executable,'-m','a2f_tts_motion_bridge.preview.render_motor_npy',
                            '--input_npy',str(path),'--output_video',str(video),'--blind_id',name],
                           stdin=subprocess.DEVNULL,check=True)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))
            vs=next(s for s in info['streams'] if s['codec_type']=='video')
            assert int(vs['nb_frames'])==180
            item.update(video=video.name,video_frames=int(vs['nb_frames']))
        report['cases'].append(item);print('P26 sample complete:',name,flush=True)
    for name,meta in cases.items():
        r=SessionRecorder();layer=BehaviorLayer(42);layer.schedule(meta,0);r.behavior_layer=layer
        speech=name=='speech_yield'
        for t in np.arange(0,6,.1):
            dofs=layer.apply(base,float(t),1. if speech else 0.)
            features=debug_features_from_arkit52({},dofs,1. if speech else 0.)
            r.append_frame(float(t),np.zeros(169),features,dofs,base_motors=base,
                           audio_rms=.07 if speech else 0.,debug_extra={'speech_gate':float(speech)})
        audio=np.sin(np.arange(96000)*2*np.pi*180/16000)*.1 if speech else np.zeros(96000)
        r.append_audio_16k(audio.astype(np.float32));save_case(name,r,'procedural_neutral_base')
    if args.model:
        s=MotorStreamSession(args.model,SessionConfig('p26_model_mix',emotion='joy',extra={
            'emotion_latent_mode':'sequence','idle_mode':'listening','style':'laugh'}))
        for i in range(30):
            meta={'behavior_events':[event('wink',0,100)]} if i==12 else {}
            s.process_chunk(AudioChunk('p26_model_mix',i,i*200,b'\0'*6400,16000,emotion_meta=meta))
        s.finalize();save_case('model_mix',s.recorder,'real_onnx_silent_audio_plus_actor_emotion')
    rng=np.random.default_rng(26);x=rng.uniform(0,.5,(700,52)).astype(np.float32)
    y=np.array([list(arkit52_to_current_head_dofs(dict(zip(ARKIT_52_NAMES,row)),bias_scale=0).values()) for row in x])
    np.savez(out/'synthetic_pairs_example.npz',arkit_values=x,dof_values=y,
             arkit_names=np.array(ARKIT_52_NAMES),dof_names=np.array(DOFS),pose_groups=np.arange(len(x)))
    fit=fit_calibration(x,y,'example',alpha=.01,seed=26)
    fit['provenance']='SYNTHETIC_HANDWRITTEN_TEACHER_NOT_ROBOT_LABELS'
    fit['source_sha256']=hashlib.sha256((out/'synthetic_pairs_example.npz').read_bytes()).hexdigest()
    (out/'calib_example.json').write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n',encoding='utf8')
    report['example_calibration_validation']=fit['validation'];report['completed']=True
    (out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('Saved',out/'evaluation.json')


if __name__=='__main__':main()
