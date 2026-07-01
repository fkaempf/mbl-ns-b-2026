# through_predictors

## Question
`wall_through.py` shows a near coin-flip: ~48% of shocks the fly powers THROUGH the soft
laser wall, ~51% it bounces back. Is that split *predictable* from the kinematics of each
approach, or is it noise at this n? Specifically, does the binary outcome through (1) vs
bounce (0) depend on: approach speed, incidence angle, penetration depth at the moment the
laser fires, laser power (125 vs 255), or prior experience (bounce ordinal within the
wall)? None of the 8 existing proposals model this binary outcome predictively:
trajectory_geometry *describes* the penetration axis, avoidance_learning models
cross-probability vs wall-onset *time* (not per-event kinematics), reproducibility_batch
only flags the power/date confound.

## Method
One row per shock (drop "around-end" events: `|along| >= w/2` confounds geometry with
escape). Predictors, all from existing extractors: `spre` (approach speed −3..−1 s,
`wall_response.py`), `incidence` (deg from wall, `bo.bounces`), `laser_power` (125/255),
`bounce_ordinal` (1st/2nd/… within that wall presentation = within-wall experience).
Penetration depth is **outcome-contaminated** (it is *defined by* crossing), so it is NOT a
predictor; instead use the **depth at laser-on** = signed perpendicular distance at t=0,
which is causally prior. Fit a per-fly mixed-effects logistic regression:
`through ~ z(spre) + z(incidence) + z(depth_at_on) + laser_power + z(bounce_ordinal) + (1|eid)`
(statsmodels BinomialBayesMixedGLM or a Firth-penalized GLMM via R `lme4`/`brms` if
separation appears). Predictors z-scored within fly to separate within- from between-fly
effects. Null = intercept-only `through ~ 1 + (1|eid)`; compare by LRT / WAIC and report
each coefficient's odds ratio + 95% CI. **Honesty gate:** if the full model does not beat
the null (ΔWAIC < 4 / LRT p > 0.05) and every CI crosses OR=1, report **"unpredictable at
this n"** — do not cherry-pick a surviving term.

## Why rigorous
- `(1|eid)` removes the pseudoreplication the docs flag 3×; effects are within-fly.
- Uses `depth_at_on` not max penetration, breaking the outcome-leak that would
  trivially "predict" through.
- Soft-trigger geometry handled by excluding around-end events and z-scoring incidence.
- Pre-registered null + WAIC/LRT gate + explicit OR CIs prevent fishing; laser power kept
  as a fixed effect precisely because reproducibility_batch shows it is date-confounded.

## Output / plot
Forest plot of odds ratios (95% CI) per predictor; ROC/AUC of full vs null on
leave-one-fly-out CV (AUC≈0.5 = the honest "coin-flip" verdict); partial-effect curve for
the strongest predictor.

## Caveats
n≈200 non-around shocks across ~15 flies → power for one or two effects only; expect to
detect a strong predictor (likely incidence or approach speed) or nothing. Laser power has
only 2 levels and is date-confounded, so its coefficient is descriptive, not causal.
`depth_at_on` is small (hurt zone is ±2.5 mm) and may be near-constant by trigger design —
if its variance is trivial, drop it and say so.
