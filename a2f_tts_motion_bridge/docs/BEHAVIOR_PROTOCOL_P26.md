# P26 style, idle behaviors, micro-events and robot calibration

All new behaviors are opt-in. No hardware command is sent by these modules.
13 normalized DOFs are supported; there is no eye-gaze, head-tilt or shoulder axis.
Those movements cannot be represented faithfully and are not faked with mouth DOFs.
Existing emotion aliases (including `laugh -> joy`) remain compatible: use the new
`style` field to request a laugh rhythm rather than silently changing old clients.

## Meta examples (all gRPC meta values are strings)

```json
{
  "emotion": "joy",
  "style": "laugh",
  "style_weight": "0.7",
  "style_duration_ms": "600",
  "idle_mode": "listening",
  "behavior_seed": "42",
  "auto_brow": "true"
}
```

```json
{
  "behavior_events": "[{\"id\":\"word17\",\"type\":\"brow_flash\",\"duration_ms\":120,\"offset_ms\":80},{\"id\":\"wink1\",\"type\":\"wink\",\"side\":\"left\",\"offset_ms\":500}]"
}
```

Controls flow through the existing per-chunk `emotion_meta` dictionary (the name is
historical; it now also carries behavior controls). Explicit events require stable
IDs: repeated IDs with the same payload are ignored; changed payloads under the
same ID are errors. At most 64 explicit events per packet and 10000 IDs per session.
`offset_ms` is 0..60000 relative to the chunk's sample-clock start, not wall time.
Events have `id`, `type`, optional `duration_ms`, `weight` (0..1), `side` (left/right).

| Type | Default duration | Behavior |
|---|---:|---|
| smile | 1200 ms | soft mouth-corner lift and lid narrowing |
| laugh | 600 ms (allowed 300..800) | 4.5 Hz jaw carrier, squint and corner lift |
| sigh | 2000 ms | one slow jaw opening/closing envelope |
| sob | 1200 ms | seeded frequency-modulated 2.1..2.9 Hz jaw spasms |
| brow_flash | 120 ms (allowed 80..150) | brief bilateral brow raise |
| wink | 240 ms | one-sided closure |
| smile_suppress | 500 ms | quick smile followed by small downward suppression |

`style` accepts none/smile/laugh/sigh/sob. A changed style starts one burst;
repeated identical style metadata does not restart it. For repeated laughter, send
explicit `behavior_events` with different IDs. `style=none` cancels older queued or
active procedural events, but not automatic idle blinks. Cancellation is immediate.
`style_gain` and `idle_gain` are 0..1, default 1. `style_weight` affects a newly
triggered burst, not an already active one. `style_duration_ms` likewise applies on
trigger; for precise retriggering use the event API.

## Independent idle behavior

`idle_mode=off|idle|listening`, default off. Selecting idle/listening enables
`auto_blink` unless explicitly overridden in that packet. `auto_blink=true|false`
also works without idle mouth motion. Seed is a uint32 set at session start only.

Primary blink intervals are **1.5 s + exponential(mean 1.5 s)**: mean 3 s and
standard deviation 1.5 s, with a refractory period. This is a refractory-Poisson
renewal model, not an unmodified Poisson process (which would have std = mean).
There is a 15% chance of a second blink 320 ms later. Listening adds a slow blink
every 6 s, gentle brow lift, very slight smiling corners and a .23 Hz breath cycle.
Idle jaw displacement is only .008 normalized units at full gain.

`auto_brow=true` adds .12 s half-weight brow flashes on audio-energy upcrossings,
with a .6 s refractory interval. It observes the aligned model RMS at native rate;
it is a heuristic, NOT lexical stress detection, and can miss sub-100ms peaks.
Explicit word-aligned events are preferred for reliable accent timing.

The behavior clock is independent of the model. A host can call
`session.tick_idle(time_s)` at 30/90 Hz with no PCM or ONNX call. It returns a
MotorFrame and does not alter the inference audio timeline or recording. Use an
idle-session clock in the host scheduler; this does not create a background thread
or make the existing request-driven gRPC server spontaneously send packets while
it is waiting for input. For the existing audio stream path, silence PCM advances
its timeline normally. No physical actuation is performed by this API.

## Mixing, sampling and replay

The base path is model -> ARKit -> default/calibrated mapping -> emotion bias ->
EMA/rate limiter. Style and idle overlays are applied **after** base smoothing.
Do not run the combined output through the old 10 Hz EMA again: that would erase
the 4.5 Hz carrier and short micro-events. OnlineRetargeter.update exposes the
`time_code_s` hook; session streaming and NPY exports evaluate analytic events at
their output timestamps. 30/90 Hz export is true procedural resampling, not just
interpolation of the 10 Hz composite pose. 90 Hz does not increase model bandwidth.

Mouth/jaw overlay amplitude is scaled by `1 - speech*(1-behavior_speech_scale)`;
speech uses the larger of the retarget gate and RMS/.035, clipped to 0..1. Default
`behavior_speech_scale=.2` leaves most speech articulation to the model. Eye/brow
events remain expressive. All combined DOFs clamp to [0,1], summed additive deltas
cap at .3. Blinks close toward the existing lid limits without reopening an already
closed model eye. These are normalized preview limits, NOT certified mechanical
velocity, acceleration, torque or collision limits. Apply robot-specific safety
constraints before real hardware use.

`motor_values_native` records the composite native poses;
`motor_base_values_native` preserves pre-overlay poses. Dense `motor_values_30fps`
contains the actual output-rate overlays (the legacy key can hold 90Hz in segmented
exports, with `meta.export_fps` declaring the real cadence). The event list, seed,
config and version are stored in metadata. Only the dense jaw debug feature is
recomputed after overlay; other abstract debug features remain native-interpolated.

`bias_left_scale` and `bias_right_scale` (0..2, default 1) scale the corresponding
side of the hand-authored emotion bias, not the actor model or calibration matrix.
Cheekiness already has asymmetric components; these knobs allow side tuning.

## Per-robot ridge calibration

Collect real paired data in an NPZ with `allow_pickle=False`:

- `arkit_values`: [N,52], finite in [0,1]; `dof_values`: [N,13] ideal normalized DOFs.
- `arkit_names`: canonical ARKit_52_NAMES; `dof_names`: MOTOR_CFG order (Unicode arrays).
- Optional `pose_groups`: one identifier per row. Put correlated frames from the
  same recorded pose/take in the same group to avoid validation leakage.

```bash
python -m a2f_tts_motion_bridge.entrypoints.fit_robot_calibration \
  --samples robotA_pairs.npz --robot robotA \
  --output calibration/calib_robotA.json --alpha 0.1
```

At least 66 paired rows and 5 independent groups are required. The fitter splits
20% of pose groups for validation, reports per-DOF RMSE against a constant baseline,
then refits all rows. Ridge uses all 52 channels with a fixed neutral intercept, so
zero ARKit preserves mechanism neutral. No universal error threshold certifies a
robot; examine coverage, held-out poses and actual safe motions.

Files are generated with `approved:false`. After real-data and hardware review,
the operator may approve the file, then set `A2F_CALIBRATION_DIR` and supply
`robot_id=robotA` on session initialization. IDs are restricted to ASCII letters,
digits, '_' and '-', and resolve only to `calib_<robot>.json` in that trusted
directory. Robot identity cannot switch mid-session. Remote meta cannot supply
arbitrary file paths. No selected calibration means the handwritten mapping;
an explicitly selected missing/invalid/unapproved file is an error, not a silent
fallback. Direct Python retarget callers can supply a trusted `calibration_path`.

Without calibration, small fallback combinations use noseSneer -> eyelids/corners,
cheekPuff -> mouth spread/lower lids, mouthShrug -> corner height. These are proxies,
not a true nose or cheek mechanism. A calibrated matrix replaces the base mapping,
then retains emotion bias, smoothing and behavioral layers.

There are currently no supplied real robot paired labels. Any `calib_example.json`
produced by the regression/demo uses a synthetic teacher and is explicitly marked
as such, unapproved and refused by live loading. It demonstrates the pipeline;
it is **not** an individual robot's calibration.

## Remote regression and previews

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
A2F_TEST_MODEL="$PWD/audio2face-3d-model/network.onnx" \
/home/cxm2/data/conda_envs/a2f_env/bin/python -m unittest discover \
  -s a2f_tts_motion_bridge/tests -v

/home/cxm2/data/conda_envs/a2f_env/bin/python -m \
  a2f_tts_motion_bridge.entrypoints.evaluate_behaviors_p26 \
  --output artifacts/p26_new_run --model audio2face-3d-model/network.onnx --render
```

The demo makes six procedural neutral-base scenes and one real-model silent-audio
scene. These are engineering previews, not recordings of a laughing/sobbing actor
or real hardware. Each scene has 30/90Hz NPY, optional MP4, and per-DOF statistics.
The speech-yield scene uses a synthetic tone to isolate amplitude reduction.
Config history and cancellation timestamps are also archived for behavioral replay.
