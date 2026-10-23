"""Real local model smoke benchmark; retain raw plans without correction."""
import argparse
import json
from pathlib import Path
from ..motion_core.standalone_face import PromptFacePlanner, PromptFaceSession

CASES = [
    ('slight_smile', '轻微微笑', 'smile', 'both', .25, 'neutral'),
    ('strong_smile', '明显微笑', 'smile', 'both', .85, 'neutral'),
    ('warm_smile', '温柔地微笑', 'smile', 'both', .6, 'warm'),
    ('sad_frown', '伤感地皱眉', 'frown', 'both', .6, 'sad'),
    ('angry_frown', '愤怒地皱眉', 'frown', 'both', .6, 'angry'),
    ('blink', '眨一下眼', 'blink', 'both', 1., 'neutral'),
    ('left_wink', '左眼眨一下', 'wink', 'left', 1., 'neutral'),
    ('left_brow', '稍微抬起左眉', 'raise_brow', 'left', .25, 'neutral'),
    ('playful_smile', '调皮地微笑', 'smile', 'both', .6, 'playful'),
    ('unsupported', '把头向右倾斜', None, None, None, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    planner = PromptFacePlanner(args.base)
    rows = []
    for name, text, action, side, strength, tone in CASES:
        raw, seconds = planner.parse(text)
        row = dict(name=name, prompt=text, raw=raw, seconds=seconds, valid=False, semantic_match=False)
        try:
            data = json.loads(raw)
            if action is None:
                row['valid'] = row['semantic_match'] = data == {'actions': [], 'unsupported': True}
            else:
                session = PromptFaceSession(raw, name)
                row['valid'] = True
                a = session.actions[0]
                # Equivalent unilateral blink is accepted, not rewritten.
                row['semantic_match'] = len(session.actions)==1 and a['action'] in ((action, 'blink') if action=='wink' else (action,)) and a['side']==side and a['strength']==strength and a['tone']==tone
                for seg in session.segments():
                    (root/f'{name}-{seg.segment_index:03d}.npy').write_bytes(seg.npy_bytes)
        except (ValueError, TypeError) as exc:
            row['error'] = str(exc)
        rows.append(row)
        (root/'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, row['valid'], row['semantic_match'], round(seconds, 2), flush=True)
    print(json.dumps({'cases':len(rows), 'valid':sum(r['valid'] for r in rows), 'semantic_match':sum(r['semantic_match'] for r in rows)}))


if __name__ == '__main__': main()
