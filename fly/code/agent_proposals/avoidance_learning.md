# Proposal: does the fly learn to avoid the wall over the session, controlling for fatigue

## 1. The question

`shock_adaptation.py` showed the *escape* (the response after the zap) does not change
over the session. The untested question is the *approach* side: does the fly learn to
**not reach the hurt zone** as the session goes on? Operationalised per wall presentation
(`bo.wall_trials`, the CREATE->DESTROY trial already used in `wall_trials.py`), the
avoidance metrics are: time-to-first-bounce (median ~5 s now), bounces-per-trial
(`nb`), and the zero-bounce fraction (already reported as "% trials with 0 bounces").
Learning predicts time-to-first-bounce **up**, bounces/trial **down**, zero-bounce
fraction **up**, regressed on the wall's onset time in the session, **per fly**.

## 2. The method

For every wall trial collect, alongside `nb`/`ttf`: (a) the wall onset time `ton`
(session clock), and (b) a **within-trial approach-speed control** = the fly's mean
fulltrack walking speed over the trial's pre-bounce window (`load_combined` speed, the
same stream `wall_trials.py` already loads). Fit a **per-fly mixed/within-fly model**
(15 flies): `ttf ~ session_time + approach_speed + (1|eid)` and, for the binary outcome,
`bounce_in_trial ~ session_time + approach_speed + (1|eid)` (logistic). The learning
claim survives **only if the `session_time` coefficient is significant after
`approach_speed` is partialled out**. Report the per-fly Spearman of `ttf` vs `ton`
beside the pooled model (the house pattern from `06_offspeed_over_time` and
`shock_adaptation`).

## 3. Why rigorous + the confound control

Three confounds, each addressed:
- **Overall slowing / fatigue.** A tired fly reaches any wall later regardless of
  learning. Controlled by `approach_speed` as a covariate: if `ttf` rises *only*
  because the fly walks slower, `session_time` goes non-significant. The honest
  positive is a `ttf` increase **at matched approach speed**, i.e. the fly takes a more
  evasive *path*, not just a slower one.
- **Spontaneous-crossing control.** Reaching a wall is partly just locomotion. Compute
  the same `ttf`/cross-rate against the **just-removed wall's geometry during the
  wall-OFF gap** (no laser present, ~60 s ITI from `interwall_analysis`): a no-stimulus
  baseline. Real avoidance = the on-wall trend exceeds the off-wall trend.
- **Regression-to-mean / fewer barriers late.** Use the continuous `ton` regression
  (not early-vs-late medians, which RTM distorts and which unequal late-trial counts
  bias); weight flies equally via the random intercept so a long fly can't dominate.

## 4. Output / plot

`plots/exploratory/avoidance_learning.png`: (a) `ttf` vs `ton` per fly with the
partial-regression slope; (b) zero-bounce fraction in session thirds, on-wall vs
off-wall control; (c) coefficient forest of `session_time` (raw vs speed-adjusted).
Title carries the adjusted `session_time` p-value.

## 5. Caveats

If `session_time` dies once `approach_speed` is in the model, the honest conclusion is
**no separable avoidance learning, only locomotor slowing** — and given the flat escape
and flat off-speed already on record, that is the likely outcome; this test is built to
*report* that cleanly, not to manufacture a signal. Wall-off is distance-triggered, so
trial duration is selection, not behaviour (per `wall_trials.py`) — hence metrics use
approach/`ttf`, never duration. `ttf` is undefined for zero-bounce trials, which is why
the binary cross model carries the avoidance signal and `ttf` is the secondary readout.
