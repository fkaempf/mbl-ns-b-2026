# Proposal: the perpendicular penetration profile through the soft hurt zone

## 1. The question
When the aversive laser fires, how far does the fly push *into* (and through) the
wall plane before it escapes, and how long does it dwell in the hurt zone getting
re-zapped? Concretely: is a bounce an instantaneous turn at the wall surface, or a
graded "push-in then retreat", and does the penetration depth / dwell predict the
escape vigour (turn size, speed surge)?

## 2. The method
Per laser-on, build the config wall frame already used in `bounce_clusters.py`
(origin = wall centre, normal = `rotation_z`, oriented to the fly side). Take the
single signed coordinate `v(t)` = perpendicular distance to the wall plane (this is
already computed; just kept, not collapsed to a direction). For each event measure:
(a) **max penetration** `min v(t)` (how deep past the plane), (b) **dwell time** =
seconds with `|v| < thickness/2 + laser_margin_y` (time inside the band where the
laser actually fires), and (c) **re-fire count** = raw laser-on events inside that
dwell *before* the 1 s refractory collapses them. Then plot the depth-aligned mean
`v(t)` and regress dwell / depth against the existing escape metrics (peak turn rate,
speed surge from `shock_response.py`).

## 3. Why it is rigorous / what it adds
It is built directly on the two established quirks. The soft trigger means `v` is a
real, continuous, mechanically-unclamped signal (verified: 57% of bounces cross the
plane, median 3.2 mm past it, up to 38 mm), so penetration is a measurable dose-like
geometry, not an artefact. The decoupling means this must use the *path* coordinate
`v(t)`, never heading — which it does. It uses the re-fire cadence the refractory
currently discards (verified: ~53% of fires <2 s apart in newer runs; median dwell
2.75 s, up to 16 s), turning a deduplication nuisance into the readout. No current
analysis quantifies the one axis the soft trigger makes informative: depth and dwell
along the normal. Incidence/exit measure *direction*; reflect/around/through is a
3-way label; clusters mix both axes. None give a graded penetration profile.

## 4. The concrete output
One panel: depth-aligned mean +/- SEM `v(t)` (entry -> deepest -> exit), with the
hurt-zone band shaded. Plus two scatters: dwell-in-zone vs peak turn rate, and max
penetration vs speed surge, pooled over runs with per-fly colour and a per-fly (n~3)
significance note. Histograms of dwell time and penetration depth alongside.

## 5. Honest caveats
- Bounce-to-barrier matching is imperfect (~55% inside the reconstructed zone), so the
  zone-edge threshold for dwell is approximate; report depth from the wall *plane*
  (robust) as primary, dwell (threshold-dependent) as secondary.
- Laser amplitude differs by run (255 vs 125), so pool depth/dwell *within* run or
  z-score before comparing escape coupling.
- n~3 flies: any depth->vigour correlation is pseudoreplicated; show it per fly.
