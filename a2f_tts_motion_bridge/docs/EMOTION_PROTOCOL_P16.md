# P16 continuous emotion controls

P06 `emotion`, `intensity`, raw explicit[10]/implicit[16] vectors remain supported.
Existing requests stay immediate and explicit-only. P16 options are opt-in; a
mixture, VA selection or enabled latent mode automatically enables EmotionTrack.
All gRPC meta values are strings; Python AudioChunk also accepts `emotion_mix` dict.

## Examples (gRPC meta)

```json
{
  "emotion_mix": "{\"joy\":0.6,\"amazement\":0.4}",
  "intensity": "0.8",
  "emotion_latent_mode": "centroid",
  "emotion_latent_gain": "0.5",
  "emotion_latent_speech_scale": "0.25"
}
```

Send `{"emotion_mix":"{\"fear\":0.7,\"sadness\":0.3}"}` on a later chunk
to crossfade. Send `{"emotion":"neutral"}` to release toward zero (library
mode remains configured, but neutral has no nonzero emotion weights).
To return to exact P06 handling, set `emotion_latent_mode=off` and
`emotion_track=false`. To clear a mixture back to the current label, send
`emotion_mix=null`; use `{}` for zero explicit weights instead.

`emotion_va="[0.7,0.4]"` selects valence/arousal in [-1,1], mapped by normalized
Gaussian radial weights (sigma .45) onto ten anchors. Coordinates are engineering
defaults in `emotion_latent.VA_COORDINATES`, NOT fitted actor annotations. To drive
the implicit block, also enable a latent mode. The central point is a mixture,
not a guaranteed neutral face; use `emotion=neutral` for neutral.

## Weight and source semantics

- Exactly one of `emotion_mix`, `emotion_va`, `emotion_explicit_weights` per update.
  Mixtures accept JSON objects, not colon/comma shorthand. Aliases resolve and sum.
  Weights must be finite and nonnegative. Explicit weights are NOT normalized and
  multiply intensity exactly once, matching P06.
- Library latent mixture weights normalize only if their sum exceeds 1; smaller
  sums retain attenuation. The resulting actor vector is multiplied by intensity
  and `emotion_latent_gain` (default .5, range 0..2). It is NOT normalized to 0..1.
- `off` (default), `centroid`, `sequence`, `random` are supported latent modes.
  A shared runtime reads `implicit_emo_db.npz` once before ONNX warmup and indexes
  all specs by name; the ten emotion anchors are the means of `mute_<label>`.
  Missing DB allows legacy operation but rejects enabled latent mode.
- `sequence` linearly interpolates and loops actor frames. `random` uses seeded,
  counter-based samples with interpolation, reproducible across chunk sizes and
  independent sessions. `emotion_latent_seed` is uint32, default 0.
  `emotion_latent_fps` defaults to 30 (0 < fps <=120): a configurable playback
  cadence, not a measured source capture FPS. 121 frames loop in about 4.03 s.
- Library mode and directly supplied `emotion_implicit_vector` are mutually
  exclusive: disable library mode explicitly before raw-vector passthrough.
  New mixture/VA selectors clear stale raw latent vectors. Other omitted fields
  persist. A label resets P06 vectors but does not reset P16 mode/timing options.
- SpeechGate uses the same centered 80 ms RMS as model output. Only library-generated
  implicit components are suppressed, down toward `emotion_latent_speech_scale`
  (default .25, range 0..1). Explicit conditioning remains active; raw P06 latent
  vectors are not speech-scaled. This mitigates, but cannot guarantee removal of,
  mouth interference: the latent dimensions are not independently mouth/eye axes.

## EmotionTrack: sample clock, not network arrival time

Events take effect at the start of the next audio chunk, using total resampled
16 kHz sample count. `pts_ms` retains its old bookkeeping role; it is not an
emotion scheduling offset. Split audio at phrase boundaries for precise scripting.
Delayed model windows evaluate control at the window center, so a newly arrived
chunk cannot overwrite an earlier frame's P16 emotion. Final padding continues
the same track. Gaps without audio do not advance emotion time.

Defaults (seconds):

| Dominant emotion | Attack | Sustain | Release | Residual after timed sustain |
|---|---:|---|---:|---:|
| amazement / surprise | .12 | .25 | .9 | .15 |
| sadness | .6 | hold | 1.5 | .25 |
| grief | .7 | hold | 1.8 | .2 |
| anger | .15 | hold | .6 | .35 |
| joy | .35 | hold | .8 | .2 |
| cheekiness | .25 | hold | .7 | .2 |
| disgust | .25 | hold | .6 | .15 |
| fear / pain | .15 | hold | .8 | .2 |
| outofbreath | .3 | hold | .8 | .2 |

For mixtures, the largest explicit weight chooses timing defaults; ties use model
label order. Initial attack starts from zero. A new mixture crossfades over .4 s;
same-selection intensity changes ramp over .25 s; neutral uses the outgoing
emotion's release, ending at zero. Timed sustain releases to its residual baseline
until an explicit neutral request. Repeated unchanged chunk metadata does not
restart attack, texture phase or sustain. Smoothstep envelopes supplement, not
replace, retarget EMA/RateLimiter. Native model sampling remains 10 Hz, so a 120 ms
attack is only coarsely sampled; interpolation does not increase model bandwidth.

Overrides: `emotion_track=true|false`, `emotion_attack_ms`, `emotion_sustain_ms`,
`emotion_release_ms`, `emotion_crossfade_ms`, `emotion_ramp_ms` (0..60000).
The first three accept JSON `null` to return to the emotion's default; `null`
sustain on amazement therefore restores .25 s, not infinite hold.

The model encoder accepts a validated ready 26-vector without reapplying intensity.
The same effective per-frame control reaches retarget and recorder. Existing
`emotion_vectors_native` records the actual post-envelope, post-gate model input;
`emotion_bias_scales_native` records effective bias scale. Session label/intensity
metadata still describes the latest requested selection, not the whole history:
use recorded per-frame vectors for regression and replay.
Latest timing/library settings are also archived as `emotion_track_options` and
`emotion_latent_options`, with `emotion_timeline_encoding=p16_sample_clock_v1`.

## Validation and calibration

Actual actor mute sequences have ranges approximately -0.177..0.182; centroids
have L2 norms .077.. .142. The .5 default gain is conservative engineering tuning,
not a completed human or hardware calibration. Compare gain 0/.25/.5/1 on the same
speech, silence, and bias setting. Record mouth DOFs as well as emotion similarity;
human expression identification and real robot mouth preservation remain required.

Run remotely from project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
A2F_TEST_MODEL="$PWD/audio2face-3d-model/network.onnx" \
/home/cxm2/data/conda_envs/a2f_env/bin/python -m unittest discover \
  -s a2f_tts_motion_bridge/tests -v
```

Reproduce the ten-anchor gain sweep (silence and speech) plus four scripted
centroid/sequence/random/ungated samples and 90Hz viewer exports in a NEW folder:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/cxm2/data/conda_envs/a2f_env/bin/python -m \
  a2f_tts_motion_bridge.entrypoints.evaluate_emotion_p16 \
  --model audio2face-3d-model/network.onnx \
  --audio a2f_offline_motion_pipeline/audio_prep/stream_10s.wav \
  --output artifacts/p16_new_run --render
```

`evaluation.json` archives code/model/DB hashes, gains, raw169 deltas, script event
times, per-window DOF/jaw means and video checks. Render overlays use case IDs,
not static latest-emotion labels (which would misrepresent a dynamic script).
