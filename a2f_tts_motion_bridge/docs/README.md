# Retarget fix package

P26 style/idle/micro-events and candidate robot calibration: [BEHAVIOR_PROTOCOL_P26.md](BEHAVIOR_PROTOCOL_P26.md).

P16 continuous emotion library, envelopes and per-chunk mixtures: [EMOTION_PROTOCOL_P16.md](EMOTION_PROTOCOL_P16.md).

Current P06 emotion protocol and regression guide: [EMOTION_PROTOCOL_P06.md](EMOTION_PROTOCOL_P06.md). The notes below describe the earlier retarget fix.

## Why this fix
Your metrics strongly suggest the main problem is in retargeting, not in the raw 169-d output:

- raw_max_corr_audio_env_vs_jaw ≈ 0.709  -> raw layer tracks audio fairly well
- raw_silence_open_ratio = 0.0           -> raw layer closes during silence
- motor max_corr ≈ 0.145                 -> motor layer became much worse
- motor silence_open_ratio ≈ 0.71        -> mouth stays open during silence

So the primary fix is to change `retarget.py` and add a light audio-aware gate.

## Files
- retarget.py: full replacement for `motorstream_service/retarget.py`
- model_patch.diff: minimal change to `motorstream_service/model.py`
- pipeline_patch.diff: minimal change to `motorstream_service/pipeline.py`

## Apply
1) Replace:
   `motorstream_service/retarget.py`
   with `retarget.py`

2) Edit `motorstream_service/model.py`
   and apply the hunk in `model_patch.diff`

3) Edit `motorstream_service/pipeline.py`
   and apply the hunk in `pipeline_patch.diff`

## Then re-run
- regenerate `.npy`
- re-run `evaluate_npy_metrics.py`
- compare:
  - max_corr_audio_env_vs_jaw
  - silence_jaw_mean
  - silence_open_ratio
  - offset_mae_ms

Expected direction:
- max_corr: higher
- silence_jaw_mean: lower
- silence_open_ratio: much lower
- offset_mae_ms: much lower
