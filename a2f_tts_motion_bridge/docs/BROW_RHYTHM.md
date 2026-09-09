# Sustained brow rhythm

The behavior layer now defaults to `brow_rhythm=true` and
`brow_rhythm_gain=0.35`. These keys are accepted in session/chunk meta;
boolean values use `true`/`false`, gain is finite in [0,1]. Set
`brow_rhythm=false` (or gain 0) for the previously committed static behavior.
The existing `auto_blink` default is unchanged.

This is a procedural tension envelope, not reconstructed actor dynamics or
word-stress detection. Every four-second window contains a seeded event with
jittered start and duration (1.8–3.1 s): broad relaxation and a smaller rebound.
Each brow's signed displacement from 0.5 is scaled, preserving pose direction
and left/right asymmetry. Per-channel residual is capped at 0.06. Neutral brows
remain neutral. Jaw, mouth and eyelids are not modified by this layer.

Evaluation uses absolute session time after DOF smoothing, including 30/90 Hz
exports. Stateless seed/time sampling gives repeatable results independent of
chunk boundaries or replay order. Existing config metadata archives the flags,
gain and seed. Mid-session switch/gain changes are immediate; retain a fixed
configuration through a phrase if a continuous envelope is desired.

The separate `auto_brow` energy-peak flash is not enabled by this change.
There is no added brow horizontal motion or hardware calibration. Real robot
travel and perceived expressiveness still require hardware evaluation.
