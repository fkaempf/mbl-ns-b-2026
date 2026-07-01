# Proposal: linear mixed-effects model for the escape-correction claim

## 1. Which claim/plot it improves

Finding #4 (`escape_correction.py`, `plots/exploratory/escape_correction.png`): "the
escape DISRUPTS the goal heading, deviation rises 53 -> 103 deg." The headline test is
`wilcoxon(dev_pre, dev_post)` over **all pooled bounces**, which the script itself prints
as `(pooled, pseudoreplicated)`. The same pooled-Wilcoxon / pooled-Spearman pattern is
the inferential backbone of `wall_response.py` (speed surge), `shock_adaptation.py`
(adaptation, handedness) and `heading_recovery.py`. Fixing it here fixes the template.

## 2. The method

A **linear mixed-effects model** (`statsmodels.formula.api.mixedlm`) on the per-bounce
deviation, long form (one row per bounce, columns `dev`, `phase` in {pre, post}, `eid`):

`dev ~ phase + (1 | eid)` with a random **slope** of phase where it converges
(`re_formula="~phase"`), else random intercept. The fixed-effect `phase` coefficient and
its Wald/LRT p-value are the honest test; `eid` (fly) is the grouping factor. Report the
coefficient, 95% CI, p, and the per-fly random-effect spread.

The data now supports this: `barrier_experiments("good")` is **15 flies / 418 bounces,
median 28 per fly (range 8-49)** — no longer the 3 flies / 100 bounces the FINDINGS text
still assumes. 15 groups with ~28 replicates each is comfortably enough for a two-level
LMM with one fixed effect.

## 3. Why it's rigorous / the pitfall it fixed

Pooling 418 bounces treats correlated within-fly observations as independent, so the
Wilcoxon n is inflated ~14x and its p-value is uninterpretable. The current honest
fallback ("trust the per-fly tests, n=3") throws away 12 flies and has no power. The LMM
partitions fly-level from bounce-level variance, so the p-value reflects the **15
independent units**, and partial pooling stabilises the flies with only 8-11 bounces
instead of dropping them. This is the textbook fix for exactly the pseudoreplication the
code already flags in three files.

## 4. Concrete output

Replace the pooled Wilcoxon line with a printed LMM summary (phase coef, CI, p, random-
effect SD) and add a panel to `escape_correction.png`: a **per-fly paired slope plot**
(median dev_pre -> dev_post, one line per fly, 15 lines) with the model's fixed-effect
estimate +/- CI overlaid. One number a reviewer can trust, plus the per-fly consistency
shown honestly.

## 5. Caveats

- `dev` (abs angular deviation, 0-180) is bounded and right-skew; check residuals and, if
  needed, refit on the per-fly pre/post **medians** (paired Wilcoxon, n=15) as a
  distribution-free cross-check.
- Random slopes may not converge with the most uneven flies; fall back to random
  intercept and say so.
- It tightens inference, not effect size: if the LMM agrees with the pooled result the
  claim simply becomes defensible; if it shrinks to non-significant, that is the real,
  honest answer and #4 must be softened.
- Same recipe transfers to the speed-surge and adaptation tests, but each needs its own
  convergence check, not a blanket reuse.
