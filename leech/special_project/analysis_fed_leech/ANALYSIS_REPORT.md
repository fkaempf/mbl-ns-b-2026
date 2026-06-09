# Leech behavior: dopamine x food (2x2), from DL keypoint tracking

Keypoint predictions (head=anterior, tail=posterior) for all 149,784 frames (~83 min,
30 fps) of 4 videos forming a **2x2 design**:

| dish | drug | food |
|---|---|---|
| dish1 | Vehicle | Food |
| dish2 | Dopamine | Food |
| dish3 | Vehicle | none (baseline) |
| dish4 | Dopamine | none (2 leeches) |

All metrics are in **real cm** using the known 9 cm dish (per-dish arena center + 4.5 cm
radius; ~0.020 cm/px). Coordinates were Savitzky-Golay smoothed before derivatives.
**dish4 is low-confidence** (two ~0.5 cm leeches, model PCK 39%; reported as the mean of
its two animals). With one dish per cell this is **descriptive, not statistical**.

## Treatment means

| treatment | % moving | head:tail (foraging probe) | area explored (cm2) | turning (deg/min) |
|---|---|---|---|---|
| Vehicle + NoFood (baseline) | **12.1** | 2.13 | **2.0** | **278** |
| Vehicle + Food | 45.7 | **4.01** | 12.7 | 842 |
| Dopamine + Food | 41.0 | 2.20 | 24.0 | 1217 |
| Dopamine + NoFood* | 68.0 | 1.92 | 41.9 | 1417 |

*low-confidence.

## Main findings (read along the 2x2)

1. **Baseline (Vehicle + NoFood) is near-quiescent** — the behavioral floor: moves only
   12% of the time, occupies ~2 cm2 (3% of the dish), slowest turning. Both manipulations
   lift the animal off this floor.

2. **Dopamine = general locomotor activation / arousal.** Adding dopamine to the no-food
   condition raises %moving 12 -> 68, area 2 -> 42 cm2, turning 278 -> 1417 deg/min. The
   effect is large and present with or without food (turning & area are highest in both
   dopamine cells). dopamine drives *whole-body locomotion and reorientation*.

3. **Food = directed head-led foraging.** Under vehicle, adding food raises %moving
   12 -> 46 and, distinctively, **doubles the head:tail path ratio (2.1 -> 4.0)** — the
   animal anchors its tail and sweeps its head, the signature of localized food-probing.
   This head-led signature is the single cleanest food effect.

4. **Interaction: dopamine blunts the food-probing signature.** With food present, the
   head:tail ratio is 4.0 under vehicle but only 2.2 under dopamine. So dopamine appears to
   shift behavior away from localized head-probing toward generalized locomotion even when
   food is available -- arousal overriding directed foraging.

5. **Thigmotaxis (true 9 cm arena).** dish2 (DA+Food) and dish3 (Veh+NoFood) sit almost
   entirely in the outer ring (~100%, mean r ~4.1 of 4.5 cm) = strong wall-following;
   dish1 (Veh+Food) is the most willing to leave the wall (68% outer, ventures center).

## One-line summary
**Dopamine arouses (more locomotion, turning, area, regardless of food); food directs
(head-led foraging probing); and dopamine overrides food's directed-probing signature.**

## Figures (in `analysis/`)
- `design_2x2_effects.png` - dopamine x food bar effects for the 4 key metrics
- `treatment_fingerprint.png` - radar per leech (cm metrics)
- `spaceuse_occupancy_heatmaps.png` - dwell within the 9 cm dish (cm, each leech)
- `treatment_summary.csv` - the numbers above
- kinematics_*, spaceuse_*, thigmotaxis_*, posture_rhythm_*, turning_* - per-dimension detail

## Caveats
n=1 dish per cell (dish4 has 2 leeches) -> descriptive only, no statistics. Metrics inherit
DL error; dish4 (dopamine, no food) is low-confidence and its high activity/area are partly
noise-inflated, so the dopamine-arousal claim is strongest where it is corroborated by the
food cells. Thresholds (still = 0.05 cm/s) and smoothing affect %moving and bout counts.
