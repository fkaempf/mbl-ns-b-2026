# menotaxis_avoidance_link

## Question
Are the two systems coupled? When the fly holds its menotaxis goal heading *more tightly*,
is wall-avoidance **stronger** (more often walks away, bigger escape turn) and/or **less
perturbing** (smaller pre->post heading scatter)? This is the INTERACTION the existing 11
proposals leave open: heading_persistence/recovery describe the heading system alone;
escape_correction shows the bounce disrupts heading on average but never asks whether that
disruption *scales with* how stable the heading was going in. One directed hypothesis (more
stable heading -> stronger, less-disrupted avoidance), tested once. No fishing.

## Method
**Within-fly, epoch-level** primary design (cross-fly n=15 is underpowered; see caveats).
One row per clean bounce:
- **Predictor (heading state, causally prior):** `R_local` = resultant length of `vrh`
  (`proper_rotation_z`, speed>=5) in **[-30, -3] s** before laser-on - how tightly the fly
  held a heading approaching *this* wall. Drop if <15 gated samples.
- **Outcomes (3, pre-registered):** (a) `away` = `out_dir > 90` (0/1, the ~62%,
  `bounces.py`); (b) `peakturn` escape vigor (`shock_adaptation.py`); (c) `disruption` =
  `dev_post - dev_pre` (`escape_correction.py`, >0 = bounce scattered heading).
- **Models:** `(1|eid)` mixed-effects, `R_local` **z-scored within fly** so the slope is the
  within-fly epoch effect not the cross-fly trait: logistic `away ~ z(R_local) + (1|eid)`;
  LMM `disruption ~ z(R_local) + (1|eid)`; LMM `peakturn ~ z(R_local) + (1|eid)`. Covariates:
  incidence, approach speed `spre`.
- **Null:** per-fly **circular block-shuffle** of `R_local`-to-bounce assignment (1000x),
  preserving each fly's marginals and count, so the slope reads against drift, not 0.
  Cross-fly trait correlation (per-fly mean R vs %away / median disruption, n=15) reported
  **secondarily** with CI.

## Why rigorous
`R_local` precedes the bounce, so direction is causal. Within-fly z-scoring + `(1|eid)`
removes the pseudoreplication the docs flag 3x and splits the within-fly link from the
between-fly trait. The block-shuffle null calibrates against heading autocorrelation (~10 s
< the 27 s window) a naive test would mistake for coupling. Three outcomes pre-named, not
mined.

## Output
`plots/exploratory/menotaxis_avoidance_link.png`: 3 panels (away, disruption, peakturn vs
binned `R_local`), within-fly slope +/- CI on the shuffle-null band; inset = per-fly slopes
(15 lines) + cross-fly trait scatter with R, CI.

## Caveats
- **Cross-fly correlation is underpowered at n=15** (needs |r|>~0.5); report it as
  suggestive only. The defensible test is the within-fly epoch link (~400 bounces).
- `R_local` over 27 s is noisy per bounce and partly reflects pre-bounce slowing; `spre`
  guards this. If most windows fail the gate, report n usable and stop.
- If the within-fly slope sits inside the shuffle band for all 3 outcomes, conclude **the
  systems are uncoupled at this n** - state it, do not salvage one.
