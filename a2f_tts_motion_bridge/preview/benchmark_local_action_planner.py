"""Run 50 fixed Chinese instructions against a real local model (no mocked parsing)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from a2f_tts_motion_bridge.motion_core.local_action_planner import LocalActionPlanner,validate_plan
from a2f_tts_motion_bridge.preview.local_action_cases import cases


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    args.output.mkdir(parents=True)
    planner=LocalActionPlanner(args.base,args.device);results=[]
    with (args.output/'cases.jsonl').open('x',encoding='utf-8') as log:
        for case in cases():
            row=dict(case)
            raw,seconds=planner.parse(case['text']);row.update(raw=raw,seconds=seconds)
            try:
                actual=validate_plan(raw);row.update(actual=actual,schema_valid=True,exact_match=actual==case['expected'])
            except (ValueError,TypeError) as exc:
                row.update(schema_valid=False,exact_match=False,error=str(exc))
            results.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            print(case['id'],'PASS' if row['exact_match'] else 'FAIL',round(seconds,2),flush=True)
    summary=dict(total=len(results),schema_valid=sum(r['schema_valid'] for r in results),
        exact_match=sum(r['exact_match'] for r in results),failed_ids=[r['id'] for r in results if not r['exact_match']],
        mean_seconds=float(np.mean([r['seconds'] for r in results])),
        p95_seconds=float(np.percentile([r['seconds'] for r in results],95)),
        provenance=planner.provenance,scope='50 fixed explicit Chinese prompts; not a generalization or human perceptual score')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
