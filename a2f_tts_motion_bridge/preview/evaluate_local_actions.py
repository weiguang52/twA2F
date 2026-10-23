"""Archive the 50-case deterministic execution benchmark and representative videos.

These clips are programmatic poses, not neural facial animation or new A2F inference.
"""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from a2f_tts_motion_bridge.tests.test_local_actions import evaluate_case
from a2f_tts_motion_bridge.preview.local_action_cases import cases
from a2f_tts_motion_bridge.motion_core.settings import MOTOR_CFG


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--render',action='store_true');p.add_argument('--planner-results',type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Output already exists')
    args.output.mkdir(parents=True);results=[];rows=[]
    neural={}
    if args.planner_results:
        for line in args.planner_results.read_text(encoding='utf-8').splitlines():
            record=json.loads(line);neural[record['id']]=record
    videos={'case-02','case-11','case-20','case-29','case-38','case-46','case-50'}
    for case in cases():
        result=evaluate_case(case);ident=case['id'];video=''
        if 'times' in result:
            times=result.pop('times');values=result.pop('values');meta=result.pop('meta')
            duration=float(times[-1]+1/90)
            payload={'frame_times_30fps':times.astype(np.float32),'motor_values_30fps':values.astype(np.float32),
                'audio_16k_mono':np.zeros(round(duration*16000),np.float32),
                'meta':dict(meta,motor_names=list(MOTOR_CFG),target_sr=16000,source='structured_action_not_neural_animation')}
            np.save(args.output/(ident+'_90hz.npy'),payload,allow_pickle=True)
            if args.render and ident in videos:
                video=ident+'.mp4'
                subprocess.run([sys.executable,str(Path(__file__).with_name('render_current_head_13dof_npy.py')),
                    '--input_npy',str(args.output/(ident+'_90hz.npy')),'--output_video',str(args.output/video),
                    '--transcript','STRUCTURED ACTION / '+ident],stdin=subprocess.DEVNULL,check=True,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        result.update(text=case['text'],expected=case['expected']);results.append(result)
        model=neural.get(ident);status='未运行' if model is None else ('通过' if model['exact_match'] else '未通过')
        rows.append('<tr><td>'+ident+'</td><td>'+html.escape(case['text'])+'</td><td>通过</td><td>'+status+'</td><td>'+
            ('<a href="'+video+'">程序化视频</a>' if video else '—')+'</td></tr>')
    summary=dict(total=len(results),passed=len(results),results=results,
        model_parsing_total=len(neural),model_parsing_exact=sum(r['exact_match'] for r in neural.values()),
        scope='Fixed explicit instruction acceptance; not generalization or hardware validation')
    (args.output/'execution.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A2F 局部动作 · 50条验收</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 20px;background:#f3f6f3;color:#183b36}h1{font-size:32px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #d8e2db}a{color:#096655}small{color:#526b60}.wrap{overflow:auto}</style><h1>局部动作：模型理解与确定性执行分开验收</h1><p>50 条固定中文指令。执行层使用预先定义的期望计划校验；模型列独立统计真实 Qwen3 解析是否完全一致。视频为程序化动作，不是 KeyframeFace 生成，也不是本轮新生成的 A2F 口型。</p>'''
    page+=f'<p>执行验收：{len(results)}/50；真实模型计划完全匹配：{summary["model_parsing_exact"]}/{len(neural)}。</p>'
    page+='<p>覆盖左右侧、强度、起势/保持/释放、定时顺序、取消和不支持动作拒绝。固定模板全通过不代表自由中文表达可靠；仍需人工验收、更多改写/否定/歧义测试和真机标定。</p><div class="wrap"><table><thead><tr><th>编号</th><th>指令</th><th>执行</th><th>模型解析</th><th>预览</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p><small>NPY 与视频留在远端；失败不会被手工纠正为通过。执行数据见 execution.json，模型原始响应见 planner_50_v1/cases.jsonl。</small></p></html>'
    (args.output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))


if __name__=='__main__':main()
