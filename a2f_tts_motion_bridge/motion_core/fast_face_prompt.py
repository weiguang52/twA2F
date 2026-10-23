"""Conservative full-command grammar, not a learned motion generator.

Only complete matches bypass the optional LLM. No keyword substring dispatch:
negation, sequences and extra context fall back unchanged or are rejected.
"""
import re
from .standalone_face import validate_face_plan

TONES={'温柔':'warm','伤感':'sad','愤怒':'angry','调皮':'playful'}
DEGREES={'轻微':.25,'稍微':.25,'轻轻':.25,'普通':.6,'明显':.85,'很':.85,'最大':1.}
ALIASES={}
for word,kind in [('微笑','smile'),('皱眉','frown'),('抬眉','raise_brow'),
                  ('抬起眉毛','raise_brow'),('闭眼','close_eye'),('睁大眼','open_eye'),
                  ('眯眼','squint'),('眨眼','blink'),('眨一下眼','blink')]:
    ALIASES[word]=(kind,'both')
for zh,side in [('左','left'),('右','right')]:
    for word,kind in [('抬起'+zh+'眉','raise_brow'),('抬'+zh+'眉','raise_brow'),
                      (zh+'眼眨一下','wink'),('眨一下'+zh+'眼','wink'),
                      ('闭上'+zh+'眼','close_eye'),('睁大'+zh+'眼','open_eye'),
                      ('眯起'+zh+'眼','squint')]:
        ALIASES[word]=(kind,side)
WORDS='|'.join(re.escape(x) for x in sorted(ALIASES,key=len,reverse=True))
PATTERN=re.compile(r'(?:请)?(?:(温柔|伤感|愤怒|调皮)地?)?'
                   r'(?:(轻微|稍微|轻轻|普通|明显|很|最大)地?)?'
                   r'('+WORDS+r')(?:，?保持([0-9]+(?:\.[0-9]+)?)秒)?[。！!]?')

def parse_fast_face(text):
    if not isinstance(text,str) or not text.strip() or len(text)>2000:
        raise ValueError('Instruction must be 1..2000 characters')
    match=PATTERN.fullmatch(text.strip())
    if match is None:return None
    tone_word,degree,word,hold=match.groups()
    kind,side=ALIASES[word];tone=TONES.get(tone_word,'neutral')
    attack,sustain,release=200,1000,300
    if kind in ('blink','wink'):attack,sustain,release=80,40,140
    elif tone in ('warm','sad'):attack,release=500,700
    elif tone=='angry':attack,release=120,400
    if hold is not None:
        # Duration must be exact in integer ms; do not silently round a command.
        from decimal import Decimal
        ms=Decimal(hold)*1000
        if ms!=ms.to_integral_value() or not 0<=ms<=10000:
            raise ValueError('Hold duration must be 0..10 seconds in whole milliseconds')
        sustain=int(ms)
    plan={'actions':[dict(action=kind,side=side,
          strength=DEGREES.get(degree,1. if kind in ('blink','wink') else .6),
          tone=tone,attack_ms=attack,hold_ms=sustain,release_ms=release,offset_ms=0)],
          'unsupported':False}
    validate_face_plan(plan)
    return plan
