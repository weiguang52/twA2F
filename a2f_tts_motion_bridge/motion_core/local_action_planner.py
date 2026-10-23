"""Optional offline natural-language -> validated action-plan compiler.

Uses an ordinary local Qwen3 model, not KeyframeFace LoRA or keyword matching.
Schema validation is NOT proof that the language instruction was understood.
"""
import json
from pathlib import Path
import time

from .local_actions import ACTIONS, validate_action, strict_json

SYSTEM='''你是机器人局部眉眼动作计划解析器。只返回JSON，不要思考过程或Markdown。
支持动作 action: raise_brow(抬眉), frown(皱眉下压), close_eye(闭眼), open_eye(睁大眼), squint(眯眼)。
side: left/right/both，左右均按机器人自身解剖方向，不按观看者方向。
每个动作必须含 action,side,strength,attack_ms,hold_ms,release_ms,offset_ms。
strength范围0到1；默认0.6；默认attack_ms=200,hold_ms=1000,release_ms=300,offset_ms=0。
毫秒为时间单位，1秒=1000毫秒。offset_ms是相对本计划开始时间；有多个动作时按请求给出的时间安排。
attack_ms 50..2000,hold_ms 0..10000,release_ms 50..3000,offset_ms 0..60000。
不要生成口型、嘴部、头部、视线或其他不支持的动作。超出范围或有不支持动作时整条拒绝，不要近似替代。
格式严格为 {"actions":[动作对象],"cancel_all":false,"unsupported":false}。
取消所有局部动作时 actions为空,cancel_all=true,unsupported=false。
不支持或不明确的请求返回 actions为空,cancel_all=false,unsupported=true。
不要添加id，不要添加其他字段。'''


def validate_plan(raw):
    plan=strict_json(raw)
    if not isinstance(plan,dict) or set(plan)!={'actions','cancel_all','unsupported'}:raise ValueError('Invalid plan schema')
    if type(plan['cancel_all']) is not bool or type(plan['unsupported']) is not bool:raise ValueError('Plan flags must be booleans')
    actions=plan['actions']
    if not isinstance(actions,list) or len(actions)>8:raise ValueError('Plan must contain at most 8 actions')
    if plan['unsupported'] or plan['cancel_all']:
        if actions or (plan['unsupported'] and plan['cancel_all']):raise ValueError('Rejected/cancel plan cannot contain actions')
    elif not actions:raise ValueError('Empty action plan')
    normalized=[]
    for i,a in enumerate(actions):
        if not isinstance(a,dict) or set(a)!={'action','side','strength','attack_ms','hold_ms','release_ms','offset_ms'}:
            raise ValueError('All action-plan fields must be explicit')
        action=validate_action(dict(a,id='a'+str(i)));action.pop('id');normalized.append(action)
    return dict(plan,actions=normalized)


def plan_to_meta(plan,request_id):
    from .local_actions import identifier
    identifier(request_id)
    if len(request_id)>48:raise ValueError('request_id maximum is 48 characters')
    plan=validate_plan(plan)
    if plan['unsupported']:raise ValueError('Planner rejected unsupported instruction')
    if plan['cancel_all']:
        return {'local_action_cancels':json.dumps([{'id':request_id,'targets':'all','release_ms':200}])}
    return {'local_actions':json.dumps([dict(a,id=request_id+'-'+str(i)) for i,a in enumerate(plan['actions'])])}


class LocalActionPlanner:
    def __init__(self,base,device='cuda:1'):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        root=Path(base).resolve();config=json.loads((root/'config.json').read_text())
        if config.get('model_type')!='qwen3':raise ValueError('Validated backend requires a local Qwen3 base')
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(str(root),local_files_only=True,
            trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,device_map={'':device}).eval()
        self.provenance={'model_type':'qwen3','hidden_size':config.get('hidden_size'),
            'quantization':config.get('quantization_config'),'keyframeface_adapter':False,
            'method':'neural_structured_plan','do_sample':False,'thinking':False}

    def parse(self,text):
        if not isinstance(text,str) or not text.strip() or len(text)>2000:raise ValueError('Instruction must be 1..2000 characters')
        prompt=self.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},
            {'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(prompt,return_tensors='pt').to(self.model.device);start=time.monotonic()
        with self.torch.inference_mode():
            outputs=self.model.generate(**inputs,do_sample=False,max_new_tokens=512,pad_token_id=self.tokenizer.eos_token_id)
        raw=self.tokenizer.decode(outputs[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
        seconds=time.monotonic()-start
        return raw,seconds
