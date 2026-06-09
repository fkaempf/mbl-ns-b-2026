# Food-orientation foraging assay: analysis report

A population foraging assay: 5 dishes, each with many leeches (5 to 27), digitized at
~25 to 31 time snapshots. For each leech in each frame we have head (anterior), mid and
tail (posterior) positions, the dish center and radius, and the food position. From these
we derive how well each leech is **aimed at food** (food_align_deg, 0 = straight at food),
how **close** it is, and its **body posture** (straight vs bent).

All distances are normalized by the per-dish arena radius (0 = at food/center, 1 = wall);
there is no cm scale in this dataset. Short labels: 2855, 2857, PXL1-d0, PXL1-d1, PXL2-d1.

## Headline result

**Leeches genuinely orient toward food in 4 of 5 dishes, and individuals approach it,
but orientation is driven by location, not by an at-range homing signal or by posture.**

- 4 of 5 dishes show statistically significant orientation toward food (Rayleigh p < 0.001),
  with circular means within +/-9 deg of the food direction. **2857 is the lone null**
  (R = 0.06, p = 0.58, indistinguishable from random).
- Across the stable-id dishes, **80 to 100% of individuals end up closer to food** than they
  started: leeches are net approachers, not random wanderers.
- Posture does **not** predict aiming: straight vs bent leeches aim equally well
  (Mann-Whitney p = 0.86; Spearman rho = -0.006). A straight body is not a "directed crawl
  toward food" marker in these snapshots.
- Closeness and aiming are only weakly and inconsistently coupled (pooled Spearman rho = +0.11):
  some dishes show closer = better aimed (the homing direction), others the opposite. No
  consistent active-homing-at-range signature.
- **PXL1-d0 is the standout forager**: tightest orientation (R = 0.70, 70% aimed within 45 deg),
  closest to food (median 0.18 r, 71% of frames at food), most center-occupying, and highest
  active-forager fraction (38%).

## Per-dish summary

| Dish | Aim: mean align (deg) | Rayleigh R (p) | Closest: median dist (r) | % at food (<0.25r) | Active forager % |
|------|----------------------:|---------------:|-------------------------:|-------------------:|-----------------:|
| PXL1-d0 | 37 | 0.70 (~0) | 0.18 | 71 | 38 |
| PXL2-d1 | 47 | 0.61 (~0) | ~0.45 | 47 | 26 |
| 2855 | 54 | 0.48 (~0) | ~0.45 | int. | 16 |
| PXL1-d1 | 67 | 0.31 (1.5e-5) | 0.88 | 12 | 11 |
| 2857 | 85 | 0.06 (0.58) | ~0.48 | 15 | 12 |

PXL1-d1 is the instructive case: food sits dead-center (food rho 0.03) while the leeches
hug the wall (mean rho 0.77). They are weakly aimed at food and far from it, yet 100% of
them net-approach over time, so directed movement is happening slowly against a strong
wall-following (thigmotaxis) baseline.

## Figures (in `plots/`)

**Orientation dynamics** (`orientation_dynamics.py`)
- `orient_food_align_over_time.png` : alignment vs time per dish, with population mean line and 90 deg chance line.
- `orient_distribution_and_wellaimed.png` : alignment histograms + fraction well-aimed (<30/<45/<90 deg) per dish.
- `orient_rose_circular.png` : per-dish polar rose of signed alignment with circular mean arrow, R and Rayleigh p.

**Approach / distance** (`approach_distance.py`)
- `approach_distance_distributions.png` : per-dish violins of normalized head-to-food distance.
- `approach_individual_trajectories.png` : per-leech distance vs time (stable-id dishes) with population median.
- `approach_population_summary.png` : fraction near food over time + mean distance / fraction near food bars.

**Posture and directed crawling** (`posture_directed.py`)
- `posture_straight_fraction.png` : straight fraction + body-bend distribution + straight-fraction over time.
- `posture_vs_orientation.png` : the key test, food_align for straight vs bent (Mann-Whitney) + bend-vs-aim scatter (Spearman).
- `posture_straight_aimed_2x2.png` : straight x aimed contingency (chi-square / Fisher).

**Orientation x distance coupling** (`coupling_foraging.py`)
- `couple_align_vs_distance.png` : alignment vs normalized distance, binned mean + Spearman, plus hexbin.
- `couple_polar_food_centric.png` : food-centric polar map (theta = aim-vs-food, r = distance) per dish.
- `couple_active_forager.png` : active-forager fraction (aimed AND straight AND near) per dish and over time.

**Spatial ecology** (`spatial_maps.py`)
- `spatial_arena_maps.png` : the money map, every leech drawn in-dish as a body segment colored by alignment, food marked.
- `spatial_radial_thigmotaxis.png` : leech radial position vs food radial position per dish.
- `spatial_foodcentric_density.png` : leech density recentered on food (food at origin).

Metrics tables for each theme are in `metrics/*.csv`.

## Caveats

- These are **population snapshots, not tracked trajectories**. "Over time" mixes different
  leeches across frames, so time trends and per-frame correlations are pseudoreplicated and
  are not within-animal dynamics.
- **food_align measures where a leech points, not where it moves.** It is a plausibility proxy
  for foraging, not confirmed approach (the distance/approach analysis is the movement evidence).
- Food annotation is incomplete in two dishes (2855 47%, 2857 65%), and food position moves
  between frames in three dishes, so food-relative metrics for 2855/2857 use a biased subset.
- No cm scale; cross-dish absolute comparisons assume comparable arena and food geometry.
