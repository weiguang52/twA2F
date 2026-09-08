"""Reproducible 10 x 5 model/retarget regression and anonymous video evaluation.

Run as a module from the project root. All artifacts stay under --output.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from ..motion_core.a2f_stream_infer import A2FModelRuntime
from ..motion_core.arkit52_to_motor import OnlineRetargeter
from ..motion_core.audio_stream import MockStreamConfig, iter_mock_tts_stream_from_wav
from ..motion_core.emotion_control import EmotionControl, canonical_emotion
from ..motion_core.motion_session import MotorStreamSession
from ..motion_core.settings import EXPLICIT_EMOTION_LABELS, EMOTION_BIAS_SCALE
from ..motion_core.types import SessionConfig


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def profile(values,names):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=len(names) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid DOF matrix')
    return {name:dict(mean=float(values[:,i].mean()),std=float(values[:,i].std()),
                     min=float(values[:,i].min()),max=float(values[:,i].max()),
                     saturation=float(np.mean((values[:,i]<=1e-6)|(values[:,i]>=1-1e-6))))
            for i,name in enumerate(names)}


def score(output, scores):
    key=json.loads((output/'evaluation/answer_key.json').read_text())
    rows=list(csv.DictReader(Path(scores).open(encoding='utf-8-sig')))
    seen=set(); predictions=[]
    for row in rows:
        trial=row['trial_id']; guessed=row.get('predicted_emotion','').strip()
        if not guessed: continue
        if trial not in key or trial in seen: raise ValueError(f'Unknown/duplicate trial: {trial}')
        seen.add(trial); guessed=canonical_emotion(guessed)
        predictions.append((key[trial]['emotion'],guessed))
    if not predictions: raise ValueError('No human predictions filled in; recognition rate is not yet available')
    labels=EXPLICIT_EMOTION_LABELS+['neutral']
    confusion={e:{p:0 for p in labels} for e in EXPLICIT_EMOTION_LABELS}
    for actual,predicted in predictions: confusion[actual][predicted]+=1
    result=dict(scored=len(predictions),expected=len(key),complete=len(predictions)==len(key),
                accuracy=sum(a==b for a,b in predictions)/len(predictions),confusion=confusion,
                per_emotion={e:(sum(a==b==e for a,b in predictions)/sum(a==e for a,b in predictions)
                                if any(a==e for a,b in predictions) else None) for e in EXPLICIT_EMOTION_LABELS})
    write_json(output/'evaluation/human_scores.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def render_one(item):
    output,case,trial=item
    dest=output/'videos'/f'{case}.mp4'
    with (output/'logs'/f'{case}.render.log').open('w') as log:
        subprocess.run([sys.executable,'-m','a2f_tts_motion_bridge.preview.render_motor_npy',
                        '--input_npy',str(output/'emotion_test_npy'/f'{case}.npy'),
                        '--output_video',str(dest),'--blind_id',trial],check=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)]))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    if int(video.get('nb_frames','0'))<=0 or not any(s['codec_type']=='audio' for s in probe['streams']):
        raise RuntimeError(f'Invalid video/audio: {dest}')
    blind=output/'blind'/f'{trial}.mp4'
    os.link(dest,blind)
    print(f'RENDERED {case} => {trial}',flush=True)
    return dict(case=case,trial_id=trial,frames=int(video['nb_frames']),duration=float(probe['format']['duration']),
                width=video['width'],height=video['height'],sha256=sha256(dest))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_wav')
    parser.add_argument('--model')
    parser.add_argument('--output',required=True)
    parser.add_argument('--intensities',nargs=5,type=float,default=[0.25,0.5,0.75,1.,1.25])
    parser.add_argument('--bias_scale',type=float,default=EMOTION_BIAS_SCALE)
    parser.add_argument('--force_cpu',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed',type=int,default=601)
    parser.add_argument('--baseline',help='Prior profiles.json for numeric regression comparison')
    parser.add_argument('--max_mean_delta',type=float,default=0.02)
    parser.add_argument('--scores',help='Score a completed human CSV; no generation')
    args=parser.parse_args(); output=Path(args.output).resolve()
    if args.scores: score(output,args.scores); return
    if not args.input_wav or not args.model: parser.error('--input_wav and --model are required for generation')
    if output.exists(): raise FileExistsError(f'Use a fresh output directory to protect existing baselines: {output}')
    if args.workers<1 or len(set(args.intensities))!=5 or any(not np.isfinite(x) or x<=0 for x in args.intensities):
        parser.error('Require positive workers and five distinct, finite, positive intensities')
    for v in args.intensities: EmotionControl('neutral',v,bias_scale=args.bias_scale)
    for sub in ('emotion_test_npy','emotion_test_90hz','videos','blind','evaluation','logs'):
        (output/sub).mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    model=Path(args.model).resolve(); wav=Path(args.input_wav).resolve()
    cases=[(f'p06_{e}_{v:.2f}',e,v) for e in EXPLICIT_EMOTION_LABELS for v in args.intensities]
    if len({x[0] for x in cases})!=50: raise ValueError('Intensities must remain distinct at two decimal places')
    shuffled=cases.copy(); random.Random(args.seed).shuffle(shuffled)
    answer={f'trial_{i+1:03d}':dict(case=c,emotion=e,intensity=v) for i,(c,e,v) in enumerate(shuffled)}
    trial_by_case={row['case']:trial for trial,row in answer.items()}
    write_json(output/'evaluation/answer_key.json',answer)
    manifest=dict(model_sha256=sha256(model),wav_sha256=sha256(wav),audio=str(wav),model=str(model),
                  labels=EXPLICIT_EMOTION_LABELS,intensities=args.intensities,bias_scale=args.bias_scale,
                  seed=args.seed,chunk_ms=200,style='p06',human_recognition_rate=None,
                  code_sha256={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts},
                  model_assets_sha256={p.name:sha256(p) for p in sorted(model.parent.glob('*.npz'))})
    write_json(output/'manifest.json',manifest)
    runtime=A2FModelRuntime(str(model),force_cpu=args.force_cpu,warmup=2)
    manifest['providers']=runtime.providers
    write_json(output/'manifest.json',manifest)
    all_profiles={}
    for case,emotion,intensity in cases:
        control=EmotionControl(emotion,intensity,bias_scale=args.bias_scale)
        config=SessionConfig(case,emotion=emotion,intensity=intensity,save_npy=False,
                             extra={'emotion_bias_scale':str(args.bias_scale)})
        session=MotorStreamSession(str(model),config,runtime=runtime)
        stream=MockStreamConfig(emotion=emotion,intensity=intensity,chunk_sec_min=.2,chunk_sec_max=.2)
        for chunk in iter_mock_tts_stream_from_wav(str(wav),session_id=case,config=stream):
            session.process_chunk(chunk)
        session.finalize()
        path=output/'emotion_test_npy'/f'{case}.npy'
        session.recorder.set_meta_extra(regression_style='p06',**control.metadata())
        session.recorder.save(str(path))
        payload=np.load(path,allow_pickle=True).item()
        model_only=OnlineRetargeter(model_dir=str(model.parent),calib_frames=20)
        pure=[]
        for w,rms in zip(session.recorder.weights,session.recorder.audio_rms):
            _,dofs=model_only.update(w,audio_rms=rms,control=replace(control,bias_scale=0))
            pure.append([dofs[name] for name in session.motor_names])
        native_times=payload['frame_times_native']; steady=native_times>=2.0
        combined=np.asarray(session.recorder.motor_values)
        if not steady.any(): raise ValueError('Regression audio must cover more than 2 seconds')
        all_profiles[case]=dict(emotion=emotion,intensity=intensity,dof_names=session.motor_names,
                               native_frames=len(native_times),emotion_vector=control.vector().tolist(),
                               combined=profile(combined,session.motor_names),
                               steady_after_2s=profile(combined[steady],session.motor_names),
                               model_only=profile(pure,session.motor_names),
                               model_only_steady_after_2s=profile(np.asarray(pure)[steady],session.motor_names))
        at90=dict(payload); at90['meta']=dict(payload['meta'],export_fps=90.0)
        for key in ('weights','features','arkit52','motor_values','audio_rms','speech_gate'):
            t,v=session.recorder._interp_to_fps(native_times,payload[key+'_native'],90.)
            at90[key+'_30fps']=v  # Existing payload key convention; meta declares actual fps.
        at90['frame_times_30fps']=t; at90['weights_30fps_3d']=at90['weights_30fps'][:,None,:]
        np.save(output/'emotion_test_90hz'/f'{case}_90hz.npy',at90,allow_pickle=True)
        write_json(output/'profiles.json',all_profiles)
        print(f'GENERATED {case} native={len(native_times)}',flush=True)
    with (output/'dof_means.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['case','emotion','intensity','path',*session.motor_names])
        for case,p in all_profiles.items():
            for kind in ('combined','model_only','steady_after_2s','model_only_steady_after_2s'):
                writer.writerow([case,p['emotion'],p['intensity'],kind,*[p[kind][n]['mean'] for n in session.motor_names]])
    if args.baseline:
        baseline=json.loads(Path(args.baseline).read_text()); comparisons={}
        if set(baseline)!=set(all_profiles): raise ValueError('Baseline case set differs')
        for case,p in all_profiles.items():
            if baseline[case]['dof_names']!=p['dof_names']: raise ValueError('Baseline DOF names differ')
            comparisons[case]={n:p['combined'][n]['mean']-baseline[case]['combined'][n]['mean'] for n in session.motor_names}
        maximum=max(abs(v) for row in comparisons.values() for v in row.values())
        write_json(output/'comparison.json',dict(max_abs_mean_delta=maximum,threshold=args.max_mean_delta,
                    passed=maximum<=args.max_mean_delta,deltas=comparisons))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        videos=list(pool.map(render_one,[(output,c,trial_by_case[c]) for c,e,v in cases]))
    write_json(output/'video_checks.json',videos)
    with (output/'blind/scores_template.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['trial_id','predicted_emotion','confidence'])
        writer.writerows((trial,'','') for trial in answer)
    style='<style>body{font:16px sans-serif;background:#f5f5f5;padding:24px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}video{width:240px}article{display:inline-block;padding:12px}</style>'
    page='<!doctype html><meta charset="utf-8">'+style+'<h1>P06: 10 emotions × 5 intensities</h1><p><a href="blind/index.html">Blind evaluation</a> · <a href="profiles.json">DOF profiles</a></p><table><tr><th>Emotion</th>'+''.join(f'<th>{v:.2f}</th>' for v in args.intensities)+'</tr>'
    for e in EXPLICIT_EMOTION_LABELS:
        page+=f'<tr><th>{html.escape(e)}</th>'
        for v in args.intensities:
            page+=f'<td><video controls preload="none" src="videos/p06_{e}_{v:.2f}.mp4"></video></td>'
        page+='</tr>'
    (output/'index.html').write_text(page+'</table>',encoding='utf-8')
    blind='<!doctype html><meta charset="utf-8">'+style+'<h1>Anonymous emotion evaluation</h1><p>Choose one label per trial: '+', '.join(EXPLICIT_EMOTION_LABELS)+'. Use the same audio level. Do not open the labeled matrix or answer key before scoring.</p><p><a href="scores_template.csv">Download scoring template</a></p>'
    for trial in answer:
        blind+=f'<article><h2>{trial}</h2><video controls preload="none" src="{trial}.mp4"></video></article>'
    (output/'blind/index.html').write_text(blind,encoding='utf-8')
    manifest.update(completed=True,cases=len(cases),videos=len(videos))
    write_json(output/'manifest.json',manifest)
    print(f'COMPLETE {output}: {len(cases)} cases, {len(videos)} verified videos; human scores pending',flush=True)


if __name__=='__main__': main()
