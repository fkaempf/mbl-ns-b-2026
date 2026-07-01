# Statistical review: escape_lmm.py and avoidance_learning.py

Reviewer stance: adversarial. Files reviewed in full: `escape_lmm.py`,
`avoidance_learning.py`, `escape_correction.py`, `bounces.py` (`wall_trials`,
`laser_on_times`, `wall_trials` geometry), `utils.load_combined`; proposals
`rigor_stats.md`, `avoidance_learning.md`, plus the corroborating
`avoidance_selection_redteam.md`. The models could not be re-run here (no
`statsmodels` in this env, data dir outside repo), so points below are derived
from the code and statistics, not from a re-fit. Where a number depends on the
fit (e.g. whether the RE covariance is singular) I say what to print to settle it.

---

## escape_lmm issues

The core upgrade (per-fly LMM to replace a pooled, pseudoreplicated Wilcoxon over
418 within-fly-correlated bounces) is the right move and is correctly motivated.
The structure is mostly sound. Real problems, in priority order:

### 1. `m.converged` does NOT detect the failure mode that actually matters (boundary/singular RE)
The fallback gate is:
```python
m = smf.mixedlm("dev ~ post", long, groups=long.eid, re_formula="~post").fit(...)
if not m.converged: raise RuntimeError   # -> random-intercept fallback
```
`converged` reports whether lbfgs's gradient norm hit tolerance. It does **not**
report whether the random-slope variance was pinned at the boundary (slope
variance to 0, or intercept/slope correlation to +/-1). statsmodels routinely
returns `converged=True` on a degenerate `cov_re` for a 3-parameter RE
covariance estimated from only 15 groups. In that case the script proudly prints
"random slope" while the slope RE is effectively absent, and the fixed-effect SE
(hence p=2.5e-8) is whatever the near-singular fit produced. **The convergence
check is checking the wrong thing.**

Fix: after the slope fit, inspect `m.cov_re` and fall back if it is degenerate,
e.g.
```python
C = m.cov_re.to_numpy()
slope_var = C[1, 1]
corr = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
singular = (slope_var < 1e-6) or (abs(corr) > 0.999) or not np.all(np.linalg.eigvalsh(C) > 1e-8)
if not m.converged or singular: raise RuntimeError
```
And **print which branch ran plus `re_sd` for both intercept and slope** so a
reader can see the RE actually has variance. Right now `re_sd` only pulls
`cov_re.iloc[0,0]` (the intercept), so even the printed diagnostic can't reveal a
collapsed slope.

### 2. The Gaussian-residual caveat is written but never enforced
`dev = |heading - preferred|` is bounded [0, 180] and piles up near 0 (proposal
caveat #1 says exactly this). A +31 deg mean shift sitting against a hard floor at
0 produces right-skewed, heteroscedastic residuals; a Gaussian LMM's model-based
SE is then anticonservative, which is precisely the regime that manufactures a
p=2.5e-8. The proposal promised "check residuals and, if needed, refit on per-fly
medians." The code computes the per-fly median Wilcoxon (`wp`) and prints it, but
**nothing gates the headline on it** and **no residual diagnostic is produced.**
The 2.5e-8 is reported as the result irrespective of fit quality.

Fix: (a) actually emit a residual/QQ check (or at least print residual skew and a
scale-location slope); (b) make the **per-fly Wilcoxon (n=15) the headline-grade
number**, not a parenthetical, since it is the assumption-light test over the 15
independent units the whole exercise exists to respect. If the LMM p (2.5e-8) and
the Wilcoxon p disagree by orders of magnitude, trust the Wilcoxon and report the
LMM as corroborative only. Given a clean per-fly slope plot this is likely still
significant, but the *defensible* p is the Wilcoxon, not 2.5e-8.

### 3. `re_formula="~post"` on a single binary predictor is barely identifiable with 15 groups
A random slope of a 0/1 phase indicator means each fly gets its own pre and post
level — effectively a per-fly random intercept for pre and for post, i.e. a 2x2
unstructured RE covariance (3 params) from 15 clusters. This is the direct cause
of issue 1. It is not wrong in principle, but it is the first thing to collapse.
The honest default here is random-intercept (which the fallback already is); the
slope should be reported only if issue-1's singularity guard passes.

### 4. (minor) residual within-fly autocorrelation is unmodeled
Even with a per-fly random intercept, consecutive bounces within a fly are
temporally autocorrelated (same bout, drifting state). The RE absorbs the mean
fly offset, not serial correlation, so the effective N per fly is < its bounce
count and the LMM SE is still mildly optimistic. The per-fly Wilcoxon (one number
per fly) sidesteps this entirely -- another reason to lead with it.

**escape_lmm net:** the direction and magnitude (+31 deg, disruption) are almost
certainly real and the per-fly slope plot is the honest artifact. The specific
number p=2.5e-8 is not trustworthy as stated: it rides on an unverified RE
covariance and an unchecked Gaussian assumption against a bounded skewed outcome.
Lead with the n=15 per-fly Wilcoxon, add the singularity guard, print both RE SDs.

---

## avoidance_learning issues

The proposal is unusually self-aware (it pre-states that locomotor slowing is the
likely truth and that `ttf` is undefined for zero-bounce trials). The
implementation nonetheless lands two of the exact traps the prompt flags.

### 1. "ttf decreases -25 s" is a SURVIVORSHIP/COLLIDER artifact, and the result table reports it anyway
`db = d[d.bounce == 1].dropna(subset=["ttf"])` — the ttf model is fit only on
trials that bounced. `ttf` is *undefined* otherwise. Conditioning the regression
on `bounce == 1` conditions on a downstream consequence of the very process being
modeled, which opens a collider path between `sess` and `ttf`. Two concrete
consequences:
- If avoidance increased late, the trials that *still* bounce late would be the
  hardest-to-avoid (fast, head-on approaches) -> ttf would drop for selection
  reasons, not learning. That is the textbook survivorship reading.
- The sign here is the *opposite* of the proposal's own learning prediction
  ("time-to-first-bounce **up**"). ttf went **down** 25 s. Under the stated
  avoidance hypothesis a decrease is anti-learning. The only way "ttf down" is
  even discussed as a learning-adjacent finding is by quietly reinterpreting it,
  which is not allowed once the sign flipped.

Decisive cross-check the code already contains: **bounce-rate is flat (p=0.66).**
If the population of bouncing trials is not shrinking over the session, there is
no avoidance gradient to "select" the survivors -- so the ttf decrease is not
avoidance at all; it is a within-bouncer dynamic (e.g. flies that bounce do so
sooner late in the session, plausibly because they are slower/closer at onset, or
pure regression noise on a selected subset). Either way **"ttf decreases = a real
avoidance effect" is unsupported.** With bounce-rate flat, the honest headline is
"no avoidance learning"; the ttf line should be demoted to "a within-bouncer
timing change of unclear cause, on a selected subset, opposite in sign to the
learning prediction" -- not reported as a speed-adjusted p<0.05 result on equal
footing.

Fix: (a) report ttf strictly as secondary and label it selection-prone in the
panel, not just the docstring; (b) if any causal reading of ttf is wanted, model
the joint outcome without conditioning on the collider -- e.g. a Tobit/censored
model treating non-bounce trials as right-censored at `toff`, or a time-to-event
(discrete-time hazard `bounce_in_dt ~ sess + ...`) so non-bouncers contribute as
censored, not as silently-dropped rows; (c) make the primary verdict rest on the
binary bounce model (p=0.66), which it correctly does in the title but not in the
emphasis of the ttf panel.

### 2. The off-wall ~2% baseline is a DISTANCE/exposure confound, not a no-stimulus control
The off-wall metric asks whether the fly enters the *removed* wall's footprint
during the ITI (`toff..next_on`). The on-wall 80% vs off-wall 2% gap is being read
as "stimulus drives the 80x increase in entering the zone." It cannot be:
- **Start distance.** The sibling redteam (`avoidance_selection_redteam.md`)
  verified that DESTROY fires at a hard ~200 mm fly-wall distance (median 166,
  max 201 mm). So the ITI *begins* with the fly ~200 mm from that footprint and
  typically moving away. The 2% is dominated by "the fly is nowhere near the
  strip," not by "the fly chooses not to enter an aversive strip." Distance, not
  avoidance -- exactly the prompt's suspicion.
- **Asymmetric geometry of "being there."** An on-wall trial *starts when a wall
  is placed in the fly's path*; the footprint strip is `|dp| < LASER_MARGIN`
  (2.5 mm) x `|da| < w/2` -- a thin band the fly was steered into on-wall and is
  ~200 mm away from off-wall. The two rates are computed over non-comparable
  exposure to the strip, so their ratio is uninterpretable.
- **ITI length / coverage.** `next` is `toff+60` for the last trial but the true
  next onset otherwise; the ITI window length (hence opportunity to wander back)
  varies trial to trial and is not matched to the on-wall trial duration.

The 2% is therefore not "the hit-rate you'd see with no stimulus." A valid
no-stimulus baseline must caliper-match the ITI pseudo-onset to the same start
distance and bearing as wall onset (precisely what the redteam doc prescribes:
"start-distance caliper matching is essential"), then compare time-to-reach-strip.
As implemented, the off-wall control should not be presented as evidence of
avoidance; at best it shows the fly is usually far from the old footprint, which
is mechanically guaranteed by the distance trigger.

### 3. GEE logistic: the covariate is outcome-dependent (endogenous), and 15 clusters is too few for the sandwich SE
Two distinct problems in `gee("bounce ~ sess + appsp", "eid", ...)`:

a) **`appsp` is defined differently for the two outcome classes.** Line 42:
`end = b_in[0] if nb else toff`. For *bounce* trials the approach-speed average
runs onset->first bounce; for *non-bounce* trials it runs onset->wall-off
(`toff`), a much longer, distance-censored window. So the "fatigue covariate"
is computed over systematically different windows depending on the outcome it is
supposed to predict. `appsp` thus partially encodes the outcome (an endogenous
regressor), which biases the `sess` coefficient it is meant to purge. This is a
concrete bug, not a nuance: the speed control is not measuring the same quantity
across the two groups it conditions on. Fix: compute `appsp` over a fixed,
outcome-independent early window (e.g. onset..onset+W, same W for all trials, as
the redteam's uncensored window), or onset->min(first_bounce, onset+W).

b) **Anticonservative SE with 15 clusters.** GEE's robust sandwich variance is
asymptotic in the number of clusters; with ~15 flies it is known to under-cover
(empirically needs ~40+ clusters, or a small-sample correction -- Mancl-DeRouen /
Kauermann-Carroll / Fay-Graubard). statsmodels `GEE.fit()` applies none of these
by default. The bounce p=0.66 is null so this does not change the conclusion
here, but any *significant* GEE p in this file (it is the inferential backbone)
would need a bias-corrected covariance or a cluster-robust df adjustment before
being believed. Also `bounce ~ sess` ignores within-fly serial dependence of
consecutive trials beyond the working-independence assumption; that is what the
cluster-robust SE is for, but see the cluster-count caveat.

c) (minor) GEE with `family=Binomial()` and default `cov_struct` is
working-independence -- fine for a population-average slope, but then the random
intercept the proposal promised ("weight flies equally via the random intercept")
is **not** what GEE delivers; GEE gives a marginal, not a conditional, estimate.
The proposal text and the method disagree on what "per fly" means. Pick one and
say so.

**avoidance_learning net:** the primary conclusion -- **no avoidance learning**
(bounce-rate flat) -- is correct and well-supported, and the docstring honesty is
commendable. But two of the three reported "findings" are not admissible as
stated: the "ttf decreases" line is a selected-subset/collider result with the
wrong sign for the hypothesis, and the "2% off-wall vs 80% on-wall" contrast is a
distance-trigger artifact, not a control. The GEE has a real endogenous-covariate
bug (`appsp` window) plus a too-few-clusters SE caveat.

---

## Verdict

Neither analysis is fraudulent and both have a defensible core, but **neither is
publishable as the numbers are currently reported.**

- **escape_lmm:** effect is real (disruption, ~+31 deg, consistent per-fly slopes);
  the headline **p=2.5e-8 is not trustworthy** -- it rides on an unverified
  random-slope covariance (`m.converged` cannot catch a singular RE) and an
  unchecked Gaussian assumption on a bounded, floor-piled, skewed outcome. Demote
  to the n=15 per-fly Wilcoxon as the headline; add a real RE-singularity guard;
  print both RE SDs and a residual diagnostic. Cheap fixes, conclusion survives.

- **avoidance_learning:** the *negative* headline (no avoidance learning;
  bounce-rate flat) is sound and honestly framed. The two supporting "positives"
  are not: **"ttf decreases" is a survivorship/collider artifact on the
  bounce==1 subset, with a sign opposite to the learning prediction**, and the
  **off-wall 2% baseline is a distance-trigger confound, not a no-stimulus
  control** (the fly starts the ITI ~200 mm from the footprint by construction --
  verified in the sibling redteam). Plus a concrete endogenous-covariate bug:
  `appsp`'s averaging window depends on the outcome (`end = b_in[0] if nb else
  toff`), and GEE's sandwich SE is anticonservative at 15 clusters.

**Most serious single issue:** in avoidance_learning, the "ttf decreases -25 s,
p<0.05" learning claim is a survivorship/collider artifact -- ttf is computed only
on trials that bounced, the bounce-rate is flat (p=0.66) so no avoidance gradient
exists to interpret, and the decrease is the *opposite* sign to the stated
learning hypothesis -- so it must not be reported as evidence of avoidance.
