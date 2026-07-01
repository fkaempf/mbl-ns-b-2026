# Meta-completeness: the single most important remaining gap

## The gap

**The laser-on timestamp (`light_sugar_commands.csv`) is the t=0 of nearly every peri-event
finding, yet its clock alignment to the position stream has never been validated.** Two
fidelity audits were done and both are good as far as they go: `vr_fidelity.md` validated the
*internal* VR loop (fictrac sample -> rendered vrpos, ~12 ms, gain 1, no drops) and
`vrh_validation.md` validated the *heading signal* (vrh = integrated ball yaw). Neither touches
the **third, independent stream** that defines the bounce: the aversive-laser command log. Every
escape/heading result aligns the trajectory to `tb = laser_on_times(folder)` pulled from
`light_sugar_commands.csv` and divides `timestamp` by `1e9` on the assumption it is the same unix
clock, frame-synchronous with vrpos/fictrac. That assumption is asserted in `ANALYSIS_DECISIONS.md`
("They share one wall clock") but never *measured* for the light_sugar stream the way it was for
vrpos.

The pipeline already contains the evidence that something is off. `bounces.py` matches each
laser-on to the nearest `vrcollisions` event and accepts a slop of **`MATCH_S = 1.0 s`**.
`ANALYSIS_DECISIONS.md` (bounce_clusters section) records that with the best config-barrier
reconstruction the fly is inside the hurt zone for only **~55% of bounces (23/36, 19/36, 9/28)**;
for the rest "the laser fires while the fly is 4-39 mm from the nearest active wall." This is
treated purely as a *spatial* (barrier-to-bounce matching) nuisance. But laser-on and a wall
collision are two physical proxies for the **same** event - the fly entering the hurt zone - so if
the clocks were truly frame-aligned they would agree to within one or two FicTrac frames (<20 ms),
not up to 1 s. A 4-39 mm offset at the ~7-14 mm/s approach/escape speeds in `shock_response.py`
corresponds to **roughly 0.3-3 s** of travel: exactly the magnitude of an unvalidated clock skew
or a laser-command logging latency, and squarely inside the windows every peri-event analysis uses.

## Why it matters most

If `tb` is systematically offset from the true hurt-zone entry by even a few hundred milliseconds
to a second, it does not knock out one finding - it **shifts the t=0 of the entire peri-event
pipeline at once**, and that pipeline is where nearly all the headline claims live:

- **Escape disrupts heading, +31 deg (`escape_lmm.py`/`escape_correction.py`)** - the keystone
  result and the template the LMM was built to make defensible. It compares heading in
  `[-3,-1] s` ("pre") vs `[+1,+3] s` ("post") *around `tb`*. If `tb` lands during or after the
  escape turn rather than at zone entry, the "pre" window is already contaminated by the turn and
  the "post" window measures the tail - the measured pre->post jump is then partly an artifact of
  where t=0 sits, not of the escape. The +31 deg LMM is rigorous about *pseudoreplication* but
  inherits this alignment wholesale (it just calls `ec.gather()`).
- **Shock response (turn 91->471 deg/s, speed 2x at +0.2-0.4 s)** - the peak *latencies* are read
  straight off `tb`. A clock skew rescales every reported latency and can manufacture or erase the
  "stereotyped 200 ms" timing.
- **No heading recovery / post-escape settling (`heading_recovery.py`, `post_escape.py`)** - both
  define the pre-collision reference window relative to `tb`; a late `tb` poisons the reference
  with the turn itself, biasing "no recovery."
- **The ~62% wall-behind and ~48% walk-through** - depth-at-laser-on and the in/out geometry in
  `bounces.py` are evaluated at `tb`; `through_predictors.md` deliberately uses depth-at-laser-on
  to avoid outcome leakage, but that very quantity is only meaningful if `tb` marks the true entry.

Crucially, the controls already in the proposal set **do not catch this**.
`spontaneous_baseline.md` (matched ITI null) re-references magnitudes but aligns its pseudo-events
to random times, so a constant skew cancels in the *difference* yet still corrupts the *absolute
latencies and the pre/post split* that the headline narrative reports.
`avoidance_selection_redteam.md` catches the distance-trigger censoring confound, a different
issue. `vr_fidelity.md`/`vrh_validation.md` validate the two streams that are *not* the laser log.
So this is a confound that none of the 14 proposals catch, sitting under the measurement that
defines t=0 for the largest cluster of findings - which is the definition of the highest-leverage
gap.

## The fix

A short, falsifiable clock/latency audit of the laser stream against position - arithmetic on
already-logged channels, no new behavioural modeling:

1. **Laser-vs-collision lag distribution.** For every laser-on `tb`, take the signed gap to the
   nearest `vrcollisions` event, `dt = ct_nearest - tb`, across all 15 runs. If the clocks are
   aligned this is a tight zero-centred distribution (sub-frame). A non-zero **median** is a fixed
   skew (correctable); a wide or multi-modal spread is jitter/mismatch. Report median, IQR, and
   the fraction within +/-1 FicTrac frame. (The current 1 s `MATCH_S` is hiding exactly this
   number.)
2. **Laser-on vs. independent zone entry from geometry.** Using the config wall geometry that the
   gallery already trusts (`rotation_z` + `laser_margin`), compute the first time the vrpos
   trajectory crosses into each bounce's reconstructed hurt zone, and compare *that* time to `tb`.
   This is independent of the collision log. A consistent offset is the true command/log latency.
3. **Skew-sensitivity sweep.** Re-run `escape_lmm` and `shock_response` with `tb` shifted by the
   measured median offset (and +/-250, +/-500 ms brackets) and report how the +31 deg coefficient,
   its CI, and the response peak latencies move. If they are flat across the bracket, the findings
   are robust and this gap is closed honestly; if they move materially, t=0 must be redefined as
   the geometric zone-entry, not the laser command.
4. **Verdict line**, mirroring the other two fidelity audits: e.g. "laser log skew = X ms (IQR Y);
   +31 deg escape effect = Z over the +/-500 ms bracket -> alignment robust / t0 redefined."

If the data cannot resolve it (collisions too sparse, geometry match too loose), the honest
fallback is to anchor every peri-event analysis to the **geometric hurt-zone entry** instead of the
laser command, and report findings against that anchor.

## What it would change

- **Best case (skew is sub-frame, sweep is flat):** the largest finding cluster gains the same
  grounding the VR loop and the heading signal already have. The "+31 deg, holds" claim, the
  stereotyped-escape latencies, and the no-recovery result become defensible at the *measurement*
  level, not just the statistical level - closing the last untested link in the chain from raw log
  to headline.
- **Worst case (a real skew of a few hundred ms to ~1 s):** the t=0 of the escape-disruption,
  shock-response, and recovery analyses is mis-set, the "pre" baselines are partly post-escape, and
  the headline effect sizes and latencies need recomputation against geometric zone-entry. The
  qualitative story (a bounce evokes a large turn) would likely survive; the *numbers that are
  being reported as the result* - +31 deg, +0.2 s peak, "no recovery" - might not.

Either way this is the one check that, run once, either certifies or rewrites more of the findings
than any other single analysis - and it is currently the only major fidelity assumption in the
pipeline that has been asserted but never measured.
