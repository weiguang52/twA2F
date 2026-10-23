"""Offline KeyframeFace LoRA inference; never imported by the audio hot path.

Upstream: https://github.com/wjc12345123/KeyframeFace (MIT).
Requires the official adapter and a compatible Qwen3-14B base, not a base LLM alone.
"""
import hashlib
import json
from pathlib import Path
import re
import time

from .text_face import SCHEMA, POSE_NAMES, TextFaceClip, finite, normalize_arkit
from .a2f169_to_arkit52 import ARKIT_52_NAMES

SYSTEM = (
    'You are a professional facial animator. Generate all 61 ARKit facial parameters '
    'for exactly one requested semantic keyframe. The first 52 blendshapes are in '
    '[0,1], and the final 9 head/eye pose parameters are in [-1,1]. '
    'Return strict JSON only in the form {"Keyframe_x":{...}}. '
    'Include each of these exact channels once, with a numeric value; do not invent channels: '
    + ', '.join(n[0].upper()+n[1:] for n in ARKIT_52_NAMES) + ', ' + ', '.join(POSE_NAMES) + '.'
)
ORDINALS = ('One','Two','Three','Four','Five','Six','Seven','Eight')


def parse_response(text):
    """Reject incomplete JSON/channels; never silently replace inference with presets."""
    text = text.strip()
    if text.startswith('<think>'):
        if '</think>' not in text: raise ValueError('Truncated model reasoning')
        text = text.split('</think>',1)[1].strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.S)
        if match is None: raise ValueError('Invalid model JSON fence')
        text = match.group(1)
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('Duplicate model JSON key: '+key)
            result[key]=value
        return result
    result = json.loads(text,object_pairs_hook=pairs)
    if not isinstance(result,dict) or len(result)!=1:
        raise ValueError('Expected exactly one model keyframe')
    key = next(iter(result))
    if not key.lower().startswith('keyframe_'):
        raise ValueError('Missing Keyframe_x wrapper')
    return normalize_arkit(result[key])


def validate_script(script):
    if not isinstance(script,dict): raise ValueError('Script must be an object')
    duration=finite(script.get('duration_s'),.1,60,'duration_s')
    frames=script.get('keyframes')
    if not isinstance(frames,list) or not 1<=len(frames)<=8:
        raise ValueError('Expected 1..8 prompt keyframes')
    previous=-1
    for frame in frames:
        if not isinstance(frame,dict): raise ValueError('Invalid prompt keyframe')
        t=finite(frame.get('time_s'),0,duration,'time_s')
        if t<=previous: raise ValueError('Prompt times must increase strictly')
        previous=t
        prompt=frame.get('description')
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2000:
            raise ValueError('Each description must contain 1..2000 characters')
    if float(frames[0]['time_s'])!=0: raise ValueError('First prompt must start at zero')
    for key in ('background','emotion'):
        if key in script and (not isinstance(script[key],str) or len(script[key])>2000):
            raise ValueError(key+' must be a string of at most 2000 characters')
    return duration,frames


class KeyframeFaceBackend:
    def __init__(self, base, adapter, device='cuda:0'):
        base=Path(base).resolve();adapter=Path(adapter).resolve()
        config=json.loads((adapter/'adapter_config.json').read_text())
        if config.get('peft_type')!='LORA' or 'Qwen3-14B' not in config.get('base_model_name_or_path',''):
            raise ValueError('Expected official Qwen3-14B KeyframeFace LoRA')
        weights=adapter/'adapter_model.safetensors'
        if not weights.is_file(): raise ValueError('KeyframeFace safetensors adapter is required')
        base_config=json.loads((base/'config.json').read_text())
        if (base_config.get('model_type'),base_config.get('hidden_size'),base_config.get('num_hidden_layers'))!=('qwen3',5120,40):
            raise ValueError('Base must be Qwen3-14B, not another Qwen size')
        # Optional dependencies stay out of motion_core's default installation.
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        if not device.startswith('cuda:') or not torch.cuda.is_available():
            raise ValueError('This deployment requires a CUDA device, e.g. cuda:1')
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(
            str(base),local_files_only=True,trust_remote_code=False,use_safetensors=True,
            torch_dtype=torch.bfloat16,device_map={'':device})
        self.model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
        self.model.eval()
        self.provenance={
            'model':'KeyframeFace','upstream':'https://github.com/wjc12345123/KeyframeFace',
            'adapter_repository':'chaochaooooooo/KeyframeFace',
            'adapter_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'base_architecture':'Qwen3-14B',
            'quantization':base_config.get('quantization_config'),
            'thinking':False,'do_sample':False,'retargeting':'neutral_bias_free_upper_face_only',
        }
        manifest=base/'download_manifest.json'
        if manifest.is_file():
            self.provenance['base_download_manifest']=json.loads(manifest.read_text())

    def compile(self, script):
        duration,frames=validate_script(script)
        sequence='Keyframe Description Sequence:\n'+'\n'.join(
            '- Keyframe '+ORDINALS[i]+': '+f['description'] for i,f in enumerate(frames))
        sequence+='\nBackground: '+script.get('background','A person expressing the requested facial action.')
        sequence+='\nEmotion: '+script.get('emotion','As described in the keyframes.')
        output=[];raw=[]
        for i,frame in enumerate(frames):
            user=sequence+'\nTask Instruction: Generate the facial expression parameters for Keyframe '+ORDINALS[i]+'.'
            prompt=self.tokenizer.apply_chat_template(
                [{'role':'system','content':SYSTEM},{'role':'user','content':user}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=self.tokenizer(prompt,return_tensors='pt').to(self.model.device)
            if inputs.input_ids.shape[1]>8192: raise ValueError('Prompt exceeds 8192 tokens')
            started=time.monotonic()
            with self.torch.inference_mode():
                tokens=self.model.generate(**inputs,max_new_tokens=1800,do_sample=False,
                                           pad_token_id=self.tokenizer.eos_token_id)
            text=self.tokenizer.decode(tokens[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
            raw.append({'description':frame['description'],'response':text,'seconds':time.monotonic()-started})
            try:
                arkit=parse_response(text)
            except ValueError as exc:
                raise ValueError(f'Keyframe {i+1} rejected: {exc}; raw response: {text}') from exc
            output.append({'time_s':float(frame['time_s']),'arkit':arkit})
        doc={'schema':SCHEMA,'duration_s':duration,'keyframes':output,
             'provenance':dict(self.provenance,script=script,raw_responses=raw)}
        TextFaceClip(doc)  # Verify projection and complete schema before publishing an asset.
        return doc
