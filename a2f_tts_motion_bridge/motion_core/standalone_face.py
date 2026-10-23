"""Audio-independent prompt-face planning and A2F-compatible motor segments.

Only normalized software poses are generated. Hardware calibration remains downstream.
"""
import math
import numpy as np

from .local_actions import LocalActionTrack, validate_action, targets, strict_json, identifier
from .local_action_planner import LocalActionPlanner
from .motion_recorder import SessionRecorder
from .settings import MOTOR_CFG, NUM_FEATURES
from .a2f169_to_arkit52 import ARKIT_52_NAMES
from ..outputs.npy_segment_exporter import ActionSegment, build_payload_from_recorder, payload_to_npy_bytes

ACTIONS = ('raise_brow', 'frown', 'close_eye', 'open_eye', 'squint', 'smile', 'blink', 'wink', 'sad_brow')
TONES = ('neutral', 'warm', 'sad', 'angry', 'playful')
SYSTEM = '''你是独立机器人表情指令解析器，只输出JSON。
格式严格为 {"actions":[{"action":"smile","side":"both","strength":0.6,"tone":"warm","attack_ms":300,"hold_ms":1000,"release_ms":400,"offset_ms":0}],"unsupported":false}。
动作仅支持 raise_brow抬眉,frown皱眉,close_eye闭眼,open_eye睁大眼,squint眯眼,smile微笑,blink眨眼,wink单眼眨眼,sad_brow内眉抬外眉压。
side为机器人自身left/right/both。wink必须指定单侧，不明确则拒绝。tone必须为neutral/warm/sad/angry/playful；无感情色彩用neutral。tone只修饰明确动作，不凭空增加动作。
程度：轻微/稍微0.25，普通默认0.6，很/明显0.85，最大1；明确数值优先。眨眼默认strength=1。
感情色彩与动作程度独立：愤怒、伤感、温柔、调皮本身不是程度副词，不得自行提高strength；未给程度的愤怒皱眉仍为0.6。
默认attack_ms=200,hold_ms=1000,release_ms=300,offset_ms=0。
blink/wink默认attack_ms=80,hold_ms=40,release_ms=140；慢眨眼200/80/400。
温柔或伤感且未指定时间时attack_ms=500,release_ms=700。愤怒时默认attack_ms=120,release_ms=400。
所有字段必须显式给出。strength 0..1,attack_ms 50..2000,hold_ms 0..10000,release_ms 50..3000,offset_ms 0..60000。最多8个动作，可按时间编排；微笑结束后会自动恢复中性。
不支持头部/视线/语音/硬件指令。不明确、越界、要求不支持的动作时整条返回 {"actions":[],"unsupported":true}，不要近似替换。'''


class PromptFacePlanner(LocalActionPlanner):
    system_prompt = SYSTEM


def validate_face_plan(raw):
    plan = strict_json(raw)
    if not isinstance(plan, dict) or set(plan) != {'actions', 'unsupported'}:
        raise ValueError('Invalid standalone face plan schema')
    if type(plan['unsupported']) is not bool:
        raise ValueError('unsupported must be boolean')
    if plan['unsupported']:
        if plan['actions'] != []:
            raise ValueError('Rejected plan must be empty')
        raise ValueError('Unsupported facial instruction')
    if not isinstance(plan['actions'], list) or not 1 <= len(plan['actions']) <= 8:
        raise ValueError('Expected 1..8 actions')
    actions = []
    fields = {'action', 'side', 'strength', 'tone', 'attack_ms', 'hold_ms', 'release_ms', 'offset_ms'}
    for i, raw_action in enumerate(plan['actions']):
        if not isinstance(raw_action, dict) or set(raw_action) != fields:
            raise ValueError('All face action fields must be explicit')
        kind, tone = raw_action['action'], raw_action['tone']
        if kind not in ACTIONS or tone not in TONES:
            raise ValueError('Unknown action or tone')
        if kind == 'wink' and raw_action['side'] == 'both':
            raise ValueError('wink requires one side')
        proxy = dict(raw_action, id='face-' + str(i), action='raise_brow')
        proxy.pop('tone')
        action = validate_action(proxy)
        action.update(action=kind, tone=tone)
        actions.append(action)
    return actions


def face_targets(action):
    kind, strength, tone = action['action'], action['strength'], action['tone']
    if kind in ('blink', 'wink'):
        return targets(dict(action, action='close_eye'))
    if kind not in ('smile', 'sad_brow'):
        result = targets(action)
        # Affective color changes the contour, not merely the global amplitude.
        if kind == 'frown' and tone == 'sad':
            return face_targets(dict(action, action='sad_brow'))
        return result
    result = {}
    for side in ('left', 'right') if action['side'] == 'both' else (action['side'],):
        if kind == 'sad_brow':
            values = {'inner_brow_y': .5 + .23*strength, 'outer_brow_y': .5 - .13*strength}
        else:
            scale = .65 if tone == 'playful' and side == 'right' else 1.
            values = {'mouth_x': .5 + .18*strength*scale, 'mouth_y': .5 + .22*strength*scale}
            if tone == 'warm':
                values.update(upper_lid_y=.5-.08*strength, lower_lid_y=.5+.08*strength)
            elif tone == 'sad':
                values.update(inner_brow_y=.5+.12*strength, outer_brow_y=.5-.08*strength)
        result.update({side+'_'+key: value for key, value in values.items()})
    return result


class PromptFaceSession:
    """One validated plan per session; no audio, ONNX or A2F inference needed.

    Independent source: callers must not send this and A2F simultaneously to motors.
    """
    def __init__(self, plan, session_id='prompt-face'):
        self.session_id = identifier(session_id)
        self.actions = validate_face_plan(plan)
        self.track = LocalActionTrack()
        events = [dict(a, start=a['offset_ms']/1000, order=i, target=face_targets(a))
                  for i, a in enumerate(self.actions)]
        self.track.commit((events, {a['id']: a for a in self.actions}, {}, {}, 0.))
        self.duration_ms = 100 + math.ceil(max(sum(a[k] for k in ('offset_ms', 'attack_ms', 'hold_ms', 'release_ms')) for a in self.actions))
        self.neutral = {name: cfg['neutral'] for name, cfg in MOTOR_CFG.items()}

    def sample(self, time_s):
        return self.track.apply(self.neutral, time_s)

    def segments(self, export_fps=30, segment_ms=640):
        if export_fps not in (30, 90) or isinstance(export_fps, bool):
            raise ValueError('export_fps must be 30 or 90')
        if type(segment_ms) is not int or not 1 <= segment_ms <= 10000:
            raise ValueError('segment_ms must be integer 1..10000')
        recorder = SessionRecorder()
        # Include every native sample; exporter keeps the established half-open timeline.
        for t in np.arange(math.ceil(self.duration_ms*90/1000)+1)/90:
            motors = self.sample(float(t))
            feats = dict(jaw_open=motors['jaw_y'], mouth_round=0., mouth_wide=0.,
                         mouth_left_right=0., upper_face_activity=0., eye_activity=0., blink_like=0.)
            recorder.append_frame(t, np.zeros(NUM_FEATURES, np.float32), feats, motors,
                                  arkit52={name: 0. for name in ARKIT_52_NAMES})
        # Re-evaluate envelopes exactly on each export clock, rather than interpolate them.
        owner = self
        class ExactSampler:
            def apply_array(self, times, base, speech, names):
                return np.asarray([[owner.sample(float(t))[n] for n in names] for t in times], dtype=np.float32).reshape(-1, len(names))
            def metadata(self):
                return {'motion_source': 'standalone_prompt_face', 'synthetic_auxiliary_channels': True,
                        'native_fps_est': 90., 'face_plan': owner.actions,
                        'calibration_applied': False, 'audio_required': False}
        recorder.behavior_layer = ExactSampler()
        for i, start in enumerate(range(0, self.duration_ms, segment_ms)):
            end = min(start+segment_ms, self.duration_ms)
            final = end == self.duration_ms
            payload, count = build_payload_from_recorder(recorder, start, end,
                session_id=self.session_id, segment_index=i, emotion='neutral', intensity=0.,
                is_final=final, export_fps=export_fps)
            yield ActionSegment(self.session_id, i, start, end, final, count, payload_to_npy_bytes(payload))
