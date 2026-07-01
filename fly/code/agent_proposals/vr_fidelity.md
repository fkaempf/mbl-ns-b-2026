# Proposal: validate closed-loop VR fidelity (latency + gain + no-drop) from the logs

## Question
Every spatial claim (penetration depth, wall-behind, approach geometry, fixation)
assumes the VR faithfully follows the ball in real time. `load_combined` interpolates
`vrpos` onto fictrac on a shared clock but never checks the loop. Three measurable
fidelity questions: (1) **latency** — how long from a fictrac sample to its rendered VR
update? (2) **gain/conversion** — does the rendered VR displacement equal the ball
motion (and is `BALL_RADIUS_MM=3.065` self-consistent)? (3) **integrity** — are fictrac
samples dropped or duplicated in the consumed stream, biasing position?

## Method
`vr_fidelity.py` over all 15 barrier runs. The key channels are verified present:
`vrpos` carries both `unity_time` (render clock) and `last_timestamp` (the fictrac
sample it consumed) on the **same unix clock as fictrac** (nearest-neighbor gap
<5 us, confirmed on run 30), plus `last_delta_x/y` (consumed ball step) and cumulative
`proper_position_x/y`.
1. **Latency.** Per frame, `(unity_time - last_timestamp)`; report per-fly
   median/p5/p95/jitter and fraction negative (clock-sanity). (Run 30: median 12.4 ms,
   p5-p95 7.8-16.9, jitter 3.0, 0% negative -> ~1.5 fictrac frames.)
2. **Integrity.** Each unique `last_timestamp` -> nearest fictrac `timestamp`; count
   skipped fictrac samples (drops) and stale frames (Unity re-rendering an un-advanced
   sample, expected since vrpos 250 Hz > fictrac 120 Hz). Quantify drop rate and the
   max consecutive-stale gap (the worst-case position lag).
3. **Gain/conversion.** Regress cumulative `proper_position_x/y` increments on the
   consumed fictrac `integrated_position_lab` increments between matched samples:
   slope = effective gain, intercept ~0, R^2. Confirm `proper_position` step == summed
   `last_delta` (already exact on run 30) and back out the mm-per-radian factor implied
   by VR units to check it against `BALL_RADIUS_MM`.
4. **Geometry closure.** For each `CREATE RectMaze` in `vrcmd`, confirm the barrier
   `position_x/y/rotation_z` is fixed in the same VR frame the fly moves through
   (constant during the trial; `vrcollisions` `collider_position` matches the commanded
   wall), so penetration is measured against a stationary, correctly-placed wall.

## Why rigorous
Uses only directly logged, cross-validated clock channels; no behavioural modeling, no
derived quantity. Latency and drop rate are arithmetic on `unity_time`/`last_timestamp`;
gain is a regression with a ground-truth slope of 1. Each number is a falsifiable bound
on a fidelity assumption the whole pipeline currently takes on faith.

## Output
One panel per fly-aggregate: latency histogram (+per-fly p95 bar), drop/stale-rate bar,
gain scatter (VR vs ball increment, slope annotated), and a verdict line
(e.g. "loop latency 12 +/- 3 ms, 0 drops, gain 1.00, walls static -> VR is faithful").

## Caveats
This bounds *internal* fidelity: ball-in -> render-out. It cannot verify the **optical**
latency (monitor photon emission) or that the ball-to-fly coupling is slip-free — no
photodiode or camera channel exists in these logs. So the claim is "the software loop
adds ~12 ms with no sample loss and unit gain"; absolute display/optomotor latency would
need a photodiode the logs do not contain.
