"""Compile one natural-language instruction to local-action metadata offline."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motion_core.local_action_planner import LocalActionPlanner,validate_plan,plan_to_meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--device',default='cuda:1')
    p.add_argument('--text',required=True);p.add_argument('--request-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output already exists')
    planner=LocalActionPlanner(args.base,args.device)
    raw,seconds=planner.parse(args.text);plan=validate_plan(raw)
    metadata=plan_to_meta(plan,args.request_id)
    record=dict(text=args.text,plan=plan,meta=metadata,raw=raw,seconds=seconds,provenance=planner.provenance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':main()
