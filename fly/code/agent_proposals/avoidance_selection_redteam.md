# avoidance_selection_redteam

## The confound
The headline "the fly walks away from the aversive wall" (wall behind walking ~62%;
`fixation_polar`, `wall_egocentric`) pools samples over each presentation's whole life
(wall-on to wall-off). But **wall-off is distance-triggered**: I verified fly-to-wall
distance at every DESTROY across the 242 good-set trials is tightly capped (median 166 mm,
max 201, CV 0.22) - a hard ~200 mm cutoff, not a behavioural endpoint. Every presentation
is censored exactly when the fly is far from the wall, so "ends up far / wall behind" is
partly tautological: late samples are oversampled at large distance and rear-bearing by
construction. No existing proposal tests this - `spontaneous_baseline` recalibrates
peri-bounce magnitudes, `avoidance_learning` models time-to-bounce.

## The test
Two parts.

1. **Uncensored window.** Recompute the wall-behind / distance statistics using only
   `t_on .. t_on+W` for fixed `W` (~20 s, below the 1st-quartile wall-on duration of 39 s),
   dropping shorter trials. Selection is gone: every trial gives the same early window
   regardless of when DESTROY later fires.
2. **Hazard vs matched null.** Model time-from-onset to first reaching the off-distance as
   survival. Build the null from inter-wall ITIs (laser-off, median 64 s): from each ITI
   take a pseudo-onset at a matched start distance and measure time to drift `D_off` away
   with no wall. Per-fly discrete-time hazard `reach_off ~ wall_present + (1|eid)` tests
   whether the wall *accelerates* departure over baseline locomotion.

Avoidance is genuine only if the wall-behind fraction stays >50% in the fixed window AND
the wall raises the departure hazard above the laser-off null.

## Why rigorous
The fixed window is immune to the trigger by construction; the ITI null is the same flies'
own laser-free walking from matched start distances, netting out baseline speed and
menotaxis (finding #3) - the drift the cutoff converts into spurious "avoidance". The
per-fly random effect removes the pseudoreplication the docs flag 3x.

## Output / plot
(a) wall-behind fraction and median bearing vs window `W` (does 62% survive as `W->0`?);
(b) Kaplan-Meier departure curves, wall-present vs ITI null, with per-fly hazard ratio
(95% CI).

## Caveats
If most trials are too short for a clean pre-cutoff window, the uncensored estimator loses
power - report usable n and say so. The ITI null starts wherever the previous wall left the
fly, so start-distance caliper matching is essential. If the wall-behind fraction collapses
to ~50% at small `W` AND the hazard-ratio CI covers 1, the honest verdict is: avoidance
**cannot be separated** from the distance trigger in these logs, and disentangling it would
require presentations terminated on a *fixed timer*, not on distance.
