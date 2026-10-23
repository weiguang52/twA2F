# Explicit prompt fast path (experimental)

This is a strict command grammar plus the existing deterministic action
executor, **not** a trained expression-motion model. A2F's audio path is unchanged.

```bash
# CPU command mode, no Qwen weights needed:
python -m a2f_tts_motion_bridge.entrypoints.prompt_face_server --fast-prompts
# Hybrid: load Qwen once; unmatched full commands fall back unchanged:
python -m a2f_tts_motion_bridge.entrypoints.prompt_face_server --fast-prompts \
  --base /home/cxm2/data/a2f-textface-work/models/base --device cuda:1
```

Without the flag, behavior remains unchanged. Default listen address remains
127.0.0.1:50062. Do not expose this unauthenticated service publicly. This
change does not start or replace any running service.

## Supported grammar

Complete sentence only: optional 请, optional tone, optional degree, action,
optional 保持N秒, optional final 。/！/!. Leading/trailing whitespace is ignored.
No internal text is discarded. Tone precedes degree.

- Tones: 温柔/伤感/愤怒/调皮, optionally followed by 地.
- Degree: 轻微/稍微/轻轻=.25, 普通=.6, 明显/很=.85, 最大=1.
- Bilateral actions: 微笑, 皱眉, 抬眉/抬起眉毛, 闭眼, 睁大眼, 眯眼,
  眨眼/眨一下眼.
- Unilateral actions: 抬起左眉/抬左眉, 左眼眨一下/眨一下左眼,
  闭上左眼, 睁大左眼, 眯起左眼; same aliases for 右.
- Example: 请温柔地轻微微笑，保持2秒。 Hold means plateau duration,
  excluding attack/release. Allowed 0–10s in whole milliseconds.

Default timing and strength follow the standalone face prompt contract.
Out-of-range recognized durations are rejected, not silently clipped.
Negation, conditional instructions, sequences, repeated blinks, numeric strength,
novel adjectives and additional clauses do not match. With a model configured,
these go to Qwen, whose existing semantic limitations still apply; without a
model the request errors. No substring keyword triggering or silent approximation.

`action_description` includes `source=fast_grammar`, `source=qwen3`, or
`source=plan`. Existing NPY layout and model_version fields are unchanged.
Supported fast commands do not acquire the model inference lock, but RPC worker
availability still affects latency under concurrent load.

## Scope and validation

Current tests cover aliases, sides, intensity/tone independence, duration,
negative/compound near-misses, bypassing the model and unchanged fallback text.
On 2026-10-23 the remote suite ran 171 tests: 161 passed, 10 skipped.
Speed and raw interpreted plans are archived under the remote
`artifacts/fast_prompt_20261023_154513/results.json` (not committed).
Strict schema validity is not a claim of open-language semantic accuracy.

No new learned expression generator was trained. Future model work needs
licensed text/motion pairs, reviewed actor trajectories, held-out semantic and
temporal evaluation, and robot retarget calibration. Generated teacher output
must not be treated as ground truth without screening.
