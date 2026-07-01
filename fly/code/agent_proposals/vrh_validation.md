# Proposal: what is vrh? Validate proper_rotation_z against the raw FicTrac yaw channel

## Question
Many analyses (menotaxis, heading persistence, fixation, the mean-vector roses) rest on
`vrh` (`proper_rotation_z` / `last_heading`), which is decoupled from the walked path
(R~0.04) yet tracks each barrier's `rotation_z`. Is vrh (a) the fly's **body/ball yaw**,
(b) a VR-rendering artifact, (c) an arbitrary offset, or (d) a genuine navigational
heading? This determines whether every heading result is about the fly or about the rig.

## Method
A script (`vrh_validation.py`) over all 15 barrier runs (`bo.barrier_experiments`),
interpolating channels onto one clock as `load_combined` does.
1. **Identity chain.** Confirm `proper_rotation_z == degrees(last_heading)` and that vrh
   equals **`-integrated_heading_lab`** (FicTrac integrated ball yaw): per-fly circular
   correlation and the constant sign/offset.
2. **What it is NOT.** Show vrh vs FicTrac `animal_movement_direction_lab` (translation
   direction) is ~0 — the decoupling is intrinsic to FicTrac (ball yaw independent of walk
   direction), inherited by vrh, not introduced by Unity.
3. **Conditioning.** Recompute vrh-vs-walking circular correlation in speed and |turn-rate|
   bins: if vrh is true yaw, coupling to walk direction should rise with speed and fall
   during turns; flat-at-zero across all speeds means yaw genuinely uncoupled from
   translation here.
4. **Geometry closure.** Integrate `delta_rotation_lab_2` and confirm it reproduces vrh;
   check vrh's modal direction at each `CREATE RectMaze` against that wall's `rotation_z`
   (the ~8 deg match) to test whether barriers are placed relative to the live heading.

## Why rigorous
Grounded in verified raw channels, not derived quantities. On run 72 I already confirmed:
vrh = `degrees(last_heading)` to 0.002 deg; vrh vs `integrated_heading_lab` circular
corr **-0.998** (vrh IS sign-flipped FicTrac ball yaw, not artifact/offset); FicTrac's own
`integrated_heading_lab` vs `animal_movement_direction_lab` corr **-0.005** (decoupling is
FicTrac-intrinsic); `last_position` all zeros; `delta_rotation_lab_2` present for the
integration check. No new behavioural claim, no fishing — it characterises a signal six
other proposals depend on.

## Output
One panel: per-fly vrh-vs-yaw and vrh-vs-walk correlations (bar), the speed/turn-binned
coupling curve, and the integrated-yaw overlay; plus a one-line verdict
(`vrh = body yaw, sign-flipped FicTrac integrated_heading_lab`).

## Caveats
This proves vrh = FicTrac integrated ball yaw, i.e. the **closed-loop yaw the fly drove**.
It cannot prove that yaw equals true tethered body angle: FicTrac yaw can drift, and there
is no independent head/body-angle ground truth in these logs (no camera angle channel).
So the achievable claim is "vrh is the genuine integrated ball-yaw heading, not a rendering
artifact or offset"; an absolute-accuracy claim would need an external body-angle measure
that the logs do not contain.
