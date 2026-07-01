# Exploratory findings

Running log of creative/exploratory analyses on the fly VR data (the `/loop` work).
Each entry: the question, the script, the result, and whether it holds up.

## 1. Bounces trigger a stereotyped escape: sharp turn + speed surge (`shock_response.py`)

**Question.** We had the *geometry* of bounces (incidence bins, exit direction) but not
their *dynamics*. Does the fly startle in the instant after the laser fires?

**Method.** Pooled all 100 shocks (laser-on, runs 72/55/46); aligned fulltrack walking
speed and VR-heading turn rate to the shock at 0.1 s resolution, mean +/- SEM.

**Result - a clean, fast escape response:**
- **Turn rate spikes ~5x**: 91 -> **471 deg/s**, peaking **+0.2 s** after the shock and
  back to baseline within ~1 s. A sharp, stereotyped reorientation.
- **Speed nearly doubles**: 6.9 -> **13.7 mm/s** (+99%), peaking ~+0.4 s and staying
  elevated for ~2 s before decaying - a sustained run-away.

**Why it matters / reconciliation.** This resolves the earlier negative ("exit direction
is ~random, ~50% reverse"): the *direction* of the escape turn is random (left/right),
but its *magnitude and timing are stereotyped* - a ~470 deg/s turn + speed doubling
within 200-400 ms. So a bounce is a genuine escape turn of random handedness, not a
non-response. The aversive laser clearly drives behaviour even though the wall is only a
soft (passable) trigger zone. **Holds up:** tight SEM, consistent across the 3 runs.

Plot: `plots/exploratory/shock_response.png`.

## 2. The escape is a fixed, large, committed turn - no adaptation, no handedness (`shock_adaptation.py`)

**Question.** Does the escape (finding #1) adapt over the session, and is its turn biased
left/right? Per shock: speed surge, peak turn rate, and signed net turn over the first 1 s.

**Results:**
- **No adaptation.** Escape magnitude is flat across the run: speed surge vs session
  position Spearman **rho=-0.11 (p=0.26)**, peak turn **rho=-0.07 (p=0.47)**. Early vs
  late medians ~unchanged (surge 18->16, turn 721->633 deg/s). A fixed reflex - matches
  the flat inter-bounce interval (no learning at the response level either).
- **The turn is large and committed, not graded.** Net-turn distribution is **bimodal at
  ~+/-100-150 deg with a dip near 0** - the fly throws a big reorientation either way,
  rarely a small correction.
- **No robust handedness.** Pooled 61/100 left looks significant (binomial p=0.035) but
  that is **pseudoreplication** (100 shocks, 3 flies). Per fly: 0.50 / 0.64 / 0.64 left,
  p = 0.85 / 0.13 / 0.13 - not significant. At the honest n=3 flies there is **no L/R
  bias**, only a weak (2-of-3) lean.

**Picture so far (1+2):** a bounce = a fast (~200 ms), large (~120 deg), committed escape
turn of essentially random handedness, plus a ~2x speed surge, and it does not adapt.

Plots: `plots/exploratory/shock_adaptation.png`, `shock_handedness.png`.

## 3. Weak idiosyncratic menotaxis with ~10 s heading memory (`heading_persistence.py`)

**Question.** Is the fly holding a preferred heading during the barrier task, how
persistent is it, and does it drift? (`proper_rotation_z`, the integrated VR heading.)

**Results:**
- **Each fly holds its own broad preferred heading**: directions 122 / -174 / -132 deg,
  resultant **R = 0.23 / 0.38 / 0.47** (clearly anisotropic roses, but broad). The
  per-fly idiosyncrasy is the signature of menotaxis to an internal reference.
- **Stable over the session**: early-half vs late-half mean heading barely move
  (drift ~+32 / +9 / +4 deg), so it is a held preference, not a slow rotation.
- **~10 s heading memory**: the circular autocorrelation `<cos Δh>` decays to 1/e in
  **7-14 s** and plateaus at exactly **R^2** (the weak fixed direction it returns to).

**Link to bounces.** Persistence (~10 s) << inter-bounce interval (~100 s): between
zaps the fly reorients several times *within* its preferred envelope rather than holding
one rigid heading. The bounce escape (#1-2, a sharp ~120 deg turn) is a discrete
reorientation riding on this slow drift. Open question for next: does the escape turn
bring the heading *back toward* the preferred direction (error-correcting) or away?

Plots: `plots/exploratory/heading_rose.png`, `heading_autocorr.png`.

## 4. The escape DISRUPTS the goal heading, it doesn't correct it (`escape_correction.py`)

**Question (capstone).** Does the escape turn (#1-2) bring the heading back toward the
fly's preferred direction (#3, error-correction) or scatter it? Per bounce: |heading -
preferred| just before (-3..-1 s) vs just after the escape (+1..+3 s).

**Result - the opposite of correction (counter-intuitive):**
- Deviation from preferred **increases**: median **53 -> 103 deg** across the bounce
  (most points above the diagonal). Pooled Wilcoxon p<1e-3; per fly all three go the
  same way, significant in 2 (p=0.002, 0.000), trending in the third (p=0.195).
- Before the bounce the fly is biased **toward** preferred (53 deg < the 90 deg of a
  uniform heading) - it walks into walls placed ahead along its menotaxis direction.
  The escape then **randomises** the heading (post ~90-100 deg ~= uniform).

**The whole story (1-4).** The barrier task is a cycle: *menotaxis toward a goal heading
(#3) -> walk into a wall placed ahead -> fast/large/random escape turn that disrupts the
heading (#1,2,4) -> slow ~10 s drift back toward the goal (#3) -> repeat.* The escape is
a reflexive disruption, not a goal-directed correction; the "navigation" back to the
goal is the slow recovery, not the turn itself. The aversive laser drives a robust,
non-adapting startle that competes with - rather than serves - the menotaxis drive.

Plot: `plots/exploratory/escape_correction.png`.
