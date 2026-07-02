# FC2 + behaviour analysis plan

Relate the FC2 fan-shaped-body bump (phase, magnitude) to the VR barrier task and
behaviour, built on the `<fly>_unified.pkl` files from `imaging_unify.py` (4 flies:
Fly1, Fly2_001, Fly5_002, Fly6_002; all yellow sun). Run under the `nsb_fly` conda env;
figures go to `plots/imaging/analysis/` (PNG) with SVGs in the `svg/` subfolder, house
style (pink / yellow / white from `cxstyle`).

## Bump fit (foundation)

At each timepoint the 16 column ROIs (`c1_roi_1..16`) are fit to `A*cos(theta-phi)+C`
across the 16 ROIs (the per-frame sinusoid, analogous to PVA). For 16 evenly spaced
columns this is the first Fourier harmonic in closed form: `A` = magnitude, `phi` =
phase, `C` = baseline (the "shelf"). The fit `phi` equals the existing PVA phase exactly;
`A` is the true cosine modulation depth and is the canonical magnitude everywhere.

## Two questions the plots answer

1. **What does the bump magnitude mean?** Make magnitude the organising variable and show
   what co-varies with it.
2. **How do FC2 phase and magnitude correlate with heading and walking speed?**

## Shared module: `fc2_analysis.py`

`fit_bump` (A, phi, C, r2), `add_fit` (attaches fit + heading + heading-minus-phase
offset), event edges (shock / wall on / wall off), event-triggered averaging, smoothed
bump angular velocity `|dphi/dt|`, within-fly windowed heading-vs-phase circular
correlation, and shared helpers (`binned`, `FLY_COLORS`). Correlations are computed within
fly (the FC2 offset differs across flies); the honest unit is the fly.

## Plot scripts

- `fc2_magnitude_meaning.py` (Q1): magnitude on the x-axis vs windowed heading-phase
  coupling, |heading-bump offset|, walking speed, and bump `|dphi/dt|`. A strong bump marks
  engaged, stable, well-tracked goal-directed walking.
- `fc2_fc2_vs_behaviour.py` (Q2): per-fly 2D density of heading vs bump phase (the goal
  tracking, with circular r), plus magnitude-vs-speed, bump-motion-vs-speed, and a per-fly
  correlation summary.
- `fc2_window_sweep.py`: for Fly5+Fly6 and Fly5 alone, heading-phase coupling in windows
  of {0.5..5} s before vs after the first shock of each wall trial, plus magnitude-vs-
  coupling and magnitude-vs-speed. Coupling is positive before the first shock and
  collapses after it (clearest in Fly5), recovering only at long windows.
- `fc2_offset_hist.py`: per-fly histogram of the heading-minus-phase offset (moving
  frames), with a strong-bump overlay.
- `fc2_first_shock.py`: bump profile and fit metrics before vs after the first shock.
- `fc2_events.py`: bump magnitude and motion event-aligned to shock, wall on, wall off
  (the neural analogue of `shock_response.py`).

## Cross-cutting decisions

- "bump" / "shock" / "wall hit" all mean the aversive-laser event (the repo's "bounce" =
  laser-on), not the neural bump.
- Honest n = flies (4), not samples. Fly2_001's bump is too weak (median A ~0.09) to
  contribute; Fly5_002 is the cleanest example throughout.
- Speed-gate at >= 5 mm/s for anything directional (heading, phase). Keep all speeds for
  the magnitude-vs-speed relationship itself.
- A "trial" (in `fc2_window_sweep`) is one wall presentation; its "first shock" is the
  first laser onset during that wall; before/after windows are relative to that shock.
