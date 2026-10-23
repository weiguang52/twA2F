"""Compile prompt(s) offline to a trusted clip consumed by streaming meta."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motion_core.keyframeface_backend import KeyframeFaceBackend, validate_script


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,help='Local compatible Qwen3-14B directory')
    parser.add_argument('--adapter',required=True,help='Local official KeyframeFace LoRA directory')
    parser.add_argument('--device',default='cuda:0')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt',help='One semantic pose; use --script for timed changes')
    group.add_argument('--script',type=Path,help='JSON duration_s/keyframes[{time_s,description}]')
    parser.add_argument('--duration',type=float,default=5.)
    parser.add_argument('--output',type=Path,required=True,help='New operator-owned clip JSON (no overwrite)')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output already exists; use a new asset ID')
    script=json.loads(args.script.read_text(encoding='utf-8')) if args.script else {
        'duration_s':args.duration,'keyframes':[{'time_s':0,'description':args.prompt}]}
    validate_script(script)
    backend=KeyframeFaceBackend(args.base,args.adapter,args.device)
    doc=backend.compile(script)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(doc,f,ensure_ascii=False,allow_nan=False,indent=2)
    print('Compiled',args.output,flush=True)


if __name__=='__main__':main()
