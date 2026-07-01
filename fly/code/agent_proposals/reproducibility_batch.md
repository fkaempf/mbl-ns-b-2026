# Proposal: are the headline effects reproducible across recording dates, or a batch confound?

## 1. The question

The four headline effects (escape disrupts heading 53->103 deg, #4; wall-behind /
reverse-walking ~62%; inter-wall heading persistence; no recovery / no adaptation) are all
reported on the **pooled "good set"**. But that set is silently a multi-day mixture:
`good_set()` spans **4 recording dates, not 3 — 20260625, 20260626, 20260627, 20260629 — at
4 / 1 / 5 / 5 flies** (one lone fly from 20260626 the brief misses entirely). Different
days = different fly cohorts, fresh tether/ball mounts, and (documented in
ANALYSIS_DECISIONS) **different laser power** (255 on 20260625 vs 125 later). So a "robust"
pooled effect could be one strong day dragging two flat ones, and nobody would see it. The
question: does each headline effect hold **within each date**, or does date moderate it?

## 2. The method

Per-date effect sizes with CIs, plus a date-moderation test, on the per-fly LMM that
`rigor_stats.md` already establishes (no new pipeline):

- **Per-date estimates.** Refit the escape-correction LMM `dev ~ phase + (1|eid)`
  *separately within each date* and report the phase effect + 95% CI per date. Repeat for
  the reverse-walking fraction and the inter-wall R. Overlay the four (or three usable)
  date estimates on one axis: **if the CIs overlap and all point the same way, the effect
  reproduces; if one date carries it, flag the confound.**
- **Moderation test.** On the pooled model add date as a fixed effect and test the
  **date x phase interaction** by likelihood-ratio test against the no-interaction model. A
  non-significant interaction (with the per-date CIs as the real read) is the evidence of
  consistency; the main effect of date alone just shifts baselines, harmlessly.
- Treat date as fixed (only 3-4 levels — too few for a random effect), fly nested in date.

## 3. Why it is rigorous

It targets the one threat pooling cannot rule out: that the "good set" averages over a
day/rig/laser batch structure. Per-date CIs with an interaction LRT is the standard
fixed-batch check, reuses the validated LMM rather than inventing a test, and the laser-
power difference is a *named, documented* covariate, not a fishing variable. It can only
confirm or break an existing claim, never manufacture a new one.

## 4. Concrete output

`plots/exploratory/reproducibility_batch.png`: a forest plot per headline effect — one row
per date (effect +/- 95% CI) plus the pooled estimate, with the LRT interaction p in the
title. A small table: per-date n flies, n bounces, laser power, effect, CI.

## 5. Caveats (the honest core)

**3-4 dates is too few to cleanly separate a batch effect from noise, and one date
(20260626) has a single fly — its "date effect" and "fly effect" are mathematically the
same thing and cannot be told apart.** With 1/4/5/5 flies the per-date CIs will be wide,
especially 20260625 and 20260626; a non-significant interaction here means
"underpowered to detect a batch effect," **not** "no batch effect." So the result is
diagnostic, not confirmatory: it can *catch* a gross confound (one date flips sign or
carries the whole effect) but cannot *certify* its absence. The honest framing is a
consistency check with explicit power limits — and the genuinely needed fix is more
recording days with balanced flies per day, ideally with laser power held constant or
counterbalanced. State plainly that the single-fly date should likely be dropped from any
moderation test, and that laser power is perfectly confounded with date (255 only on
20260625), so the two can never be separated in this dataset.
