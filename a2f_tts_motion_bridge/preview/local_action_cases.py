"""50 explicit Chinese instructions with independently specified expected plans.

Deliberately narrow acceptance corpus, not a natural-language generalization score.
"""
from itertools import product


def action(kind='raise_brow',side='left',strength=.6,**times):
    result=dict(action=kind,side=side,strength=strength,attack_ms=200,hold_ms=1000,release_ms=300,offset_ms=0)
    result.update(times)
    return result


def plan(actions=None,cancel=False,unsupported=False):
    return dict(actions=actions or [],cancel_all=cancel,unsupported=unsupported)


def cases():
    verbs={'raise_brow':'眉毛抬高','frown':'皱眉下压','close_eye':'眼睛闭合','open_eye':'眼睛睁大','squint':'眼睛眯起'}
    sides={'left':'机器人自身左侧','right':'机器人自身右侧','both':'机器人自身左右两侧'}
    result=[]
    for kind,side,strength in product(verbs,sides,(.2,.6,1.)):
        text=f'让{sides[side]}的{verbs[kind]}，强度{strength}，从现在开始，起势200毫秒，保持1秒，释放300毫秒。'
        result.append(dict(id=f'case-{len(result)+1:02d}',text=text,expected=plan([action(kind,side,strength)])))
    extras=[
        ('从现在开始，机器人左眉抬高，强度0.6，起势200毫秒，保持1秒，释放300毫秒；在本计划开始后1500毫秒，机器人右眼闭合，强度1.0，起势200毫秒，保持1秒，释放300毫秒。',plan([action(),action('close_eye','right',1.,offset_ms=1500)])),
        ('取消所有正在执行和等待执行的局部眉眼动作。',plan(cancel=True)),
        ('把头向左倾斜20度。',plan(unsupported=True)),
        ('张大嘴巴保持两秒。',plan(unsupported=True)),
        ('机器人左眉抬高，强度0.6，从现在开始，起势120毫秒，保持2秒，释放400毫秒。',plan([action(attack_ms=120,hold_ms=2000,release_ms=400)]))]
    for text,expected in extras:result.append(dict(id=f'case-{len(result)+1:02d}',text=text,expected=expected))
    return result
