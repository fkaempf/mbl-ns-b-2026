# Analysis decisions

Rationale behind the non-obvious choices in the `fly/code` analyses. Keep this in
sync when a decision changes.

## Data & coordinate frames

- **Two trajectory signals.** `fulltrack` (FicTrac ball integration) is the fly's
  walked path in the **lab/ball frame**; `vrpos` `proper_position` is the fly's
  position in the **VR world frame**. They are different frames.
  - Use **vrpos** whenever the trajectory must line up with VR objects (the
    confinement enclosure, the barriers) — those objects live in the VR frame.
  - Use **fulltrack** for raw walking speed and for lab-frame trajectory overviews.
- **Clocks.** fulltrack `timestamp` is unix **nanoseconds**; vrpos `last_timestamp`,
  vrcmd `unity_time`, and light_sugar `timestamp` are unix **seconds** (vrcmd/laser
  also nanoseconds in `timestamp`). They share one wall clock (vrpos starts ~3 s
  after fulltrack), which is what lets `load_combined` align them.
- **`load_combined`** (utils.py) interpolates vrpos onto the fulltrack timestamps
  (heading interpolated **circularly** via cos/sin, not linearly), derives walking
  speed from fulltrack, and returns `t, t_unix, fx/fy/fh, vrx/vry/vrh, speed`.
  - `min_speed_mm_s` drops standing samples; `max_speed_mm_s` drops FicTrac
    **glitch frames** (bad frames hit >100,000 mm/s vs realistic <40 mm/s — they
    also corrupt the VR position, so cutting on real walking speed cleans both).
    Barrier/bounce plots use `min_speed=0, max_speed=50`.
- **Heading.** Use the **integrated/`proper_rotation_z`** heading, never the
  per-sample movement direction (the latter is pure noise at low speed — thousands
  of deg/s). The two VR headings `last_heading` (rad) and `proper_rotation_z` (deg)
  are the same signal up to sign.
- **Units.** VR units ≈ mm (the 100 mm-diameter enclosure spans ~50 units radius),
  so the 100-unit scalebar is labelled "10 cm".

## Confinement scalloping (`confinement_scalloping.py`)

- exp_06 is **menotaxis (0–900 s) → confinement (900–1800 s) → menotaxis (1800 s–
  end)**. Enclosure is **100 mm diameter** (fitted radius 47.6 mm ✓).
- A *contact* is a prominent radial maximum near the wall (peak-based, height
  0.85·R, prominence 3 mm) — a parameter sweep showed the count is stable ~45 there;
  a hysteresis detector is kept for the deeper-excursion scale.
- The scalloping rhythm is **broadband (~0.2 Hz), not periodic** — an honest
  negative result, shown via detrended Welch PSD.

## Menotaxis heading before/after + shuffle test

- Heading sampled from `vrpos` `proper_rotation_z`, **gated at speed ≥ 5 mm/s**
  (direction is meaningless while standing), in windows `before (10–15 min)` and
  `after (30–35 min)` (editable in `vrpos_heading_hist.py`).
- Summary is a **before-vs-after scatter** (each fly one point) — circular data on
  a linear axis, so points near ±180° can look far off-diagonal for a small true
  shift (note the wrap).
- **Shuffle test (`heading_shift_shuffle.py`)** works on the **5-min average per
  window** (one before + one after value per fly, n = 5). Null = each fly's *after*
  paired with a **different fly's before** (between-fly mismatched pairs; the 5
  diagonal/correct pairs are excluded → 20 null points vs 5 actual). Earlier
  per-sample shuffling was **pseudoreplication** — ~30 k autocorrelated 120 Hz
  samples collapsed the null onto 0; the integrated autocorrelation time is ~5 s,
  so only ~18–90 independent samples exist per window. At the honest n = 5 the
  before→after change is **not** significant.

## Barriers (`barrier_traces.py`)

- Barrier paradigm `eternarig_experiment_logic_barrier`. Each barrier is a
  **`RectMaze` wall** logged in `vrcmd.csv` as `CREATE RectMaze` with centre
  (`position_x/y`), orientation (`rotation_z`), size `scale` = (width 100 ×
  thickness 2 × height 5), in the **vrpos frame**.
- `vrcmd.csv` **cannot be read with pandas** (commas inside the `color`/`serial`
  fields) — it is parsed by splitting off the leading comma-free fields.
- The **heat zone** is the wall inflated by the `laser_margin_x/y` (0.05 / 2.5) from
  `config.yaml` — the band where the aversive laser fires.
- Analysis is restricted to the long, clean maze runs **72 / 55 / 46** (>10 min;
  shorter runs are aborted tests). Trace is colour-by-time, with a black start dot
  and scalebar.

## Single-bounce analysis (`bounces.py`, `bounce_analysis.py`)

- A **bounce** = a laser-on event: `light_sugar_commands.csv`
  `laser_exponential_set_end_level` ramps to a **positive level** (`> 0`), deduped with
  a **1 s refractory**. Detect on `> 0`, not `== 255`: the **laser power differs by
  run** — older runs (20260625) ramp to 255, newer ones (20260626/27) to 125, so a
  hardcoded 255 silently misses every newer bounce.
- Each bounce is matched to the **nearest collision** (`vrcollisions.csv`, within
  1 s) to get the wall surface point (origin) and its normal. (The `vrcollisions`
  header is shifted — read positionally.)
- The trajectory is rotated so the **wall is horizontal**, the normal points to the
  fly's side (approach from above), then **mirror-flipped so the approach comes
  from the left** (handedness fold).
- **Incidence** = direction of travel over the **5 s before the hit** (moving
  samples ≥ 5 mm/s only), as the **angle from the wall surface: 0° grazing → 90°
  head-on**. Bounces below **15°** are dropped; bins **15–40 / 40–70 / 70–90°**.
- Only **collision glitches** are dropped: a bounce whose trace dips past the wall
  (`< -CROSS_TOL = 2 mm`) **within ±GLITCH_WIN_S = 2 s** of the hit — the fly jumps
  *through* the wall. **Walk-arounds are kept** (rounding the wall end is real
  behaviour) — those cross later / at the wall end, not at the hit. 83/100 kept.
- Pooled over 72/55/46 (100 bounces); plotted per window **10/30/60/120 s** in
  separate folders. Longer windows smear (far-from-hit parts are the fly elsewhere),
  so **10–30 s are the informative ones**.
- **Mean trajectory drawing.** The mean ± SEM is taken from a **fixed window**
  (`MEAN_WINDOW_S = 10 s`, same in every plot, so it doesn't change with the display
  window) and is drawn **only while the trajectories stay directionally coherent**
  (mean step / mean individual step ≥ `COH_MIN = 0.4`); it stops where it would
  collapse instead of smearing toward the centre. Individual traces use the *display*
  window and a higher alpha (`FAINT_ALPHA = 0.2`).

### Exploratory: reflection — continue vs reverse (`reflection.png`)

Added because the abs-angle in/out scatter throws away the **sign** of the
departure, so it can't tell a **specular reflection** (fly continues along the wall,
billiard-style) from a **reversal** (fly turns back the way it came — an avoidance).
`out_dir` is the **signed** exit direction in the wall frame: `<90°` = continue,
`90°` = straight off, `>90°` = reverse. **Finding:** the exit direction is roughly
**uniform, ~51 % reverse** — the bounce is **not** specular; it is a roughly
**random reorientation**. That is why the departure mean collapses (approaches are
clean and incidence-structured; departures are not). Caveat: ~20 bounces per
incidence bin, so the histogram is noisy — suggestive, not conclusive.

## Per-bounce gallery + clustering (`bounce_clusters.py`)

One figure per bounce (all 100 in `plots/bounce_gallery/`), 30 s before / 60 s after
the laser, in the wall frame with the true finite wall, coloured by time, titled with
the bounce order (`bounce 13/36`), its time in the session, and its k-means type.

- **Bounce = laser activation**, the same definition as everywhere else
  (`bo.laser_on_times`: `laser_exponential_set_end_level -> 255`, 1 s refractory).
  `tau = 0` is the laser-on time; the black dot is the fly's VR position at that
  instant. The collision log is **not** used here at all.
- **The barrier is a soft laser TRIGGER zone, not a physical wall.** This was the key
  realisation while debugging "the fly phases through the wall":
  - *Config* (`eternarig_experiment_logic_barrier`) defines a barrier only by
    geometry (`barrier_width 100`, `barrier_thickness 2`, `barrier_distance_to_fly 5`)
    and the laser hurt zone (`laser_margin_x 0.05`, `laser_margin_y 2.5`). There is
    **no** collider / clamp / rigidbody / pushback parameter.
  - *Data*: `proper_position` is the only real VR position (`last_position` is all
    zeros, unused) and it crosses the wall **smoothly at walking speed** (8 ms
    sampling, no teleport, speed under the 50 mm/s cap). It is **not** clamped.
  - So the fly's VR position is never mechanically stopped: it walks straight through
    the wall and the laser is the only consequence. The "correction" in this paradigm
    is the aversive **laser**, not a physical barrier. Wall crossings are **real
    behaviour**, not glitches or a geometry bug.
- **"Phasing" breaks down as:** of the bounces that pass the wall plane, ~42/100 walk
  **around the finite wall ends** (`|along| > 50 mm`; the wall is only 100 mm wide),
  and ~17/100 push **through the mid-wall**, gradually, at normal walking speed. Both
  are real. (Distinct from the true VR *teleport* glitches in `barrier_traces`, which
  are single ~800-unit jumps; none of those occur here.)
- **Wall placement is from the config barrier geometry, not the collision log.** Each
  bounce is mapped to a `RectMaze` (its `position` = wall centre, `rotation_z`,
  `width`), and the frame is built from that. The collision normal (`vrcollisions`)
  was tried and **rejected**: it aligns with the true wall orientation only ~half the
  time (parallel/crossing walls and noisy normals), whereas `rotation_z` is exact and
  `barrier_traces` confirms the config geometry overlays the trajectory correctly.
- **Caveat - bounce-to-barrier matching is imperfect.** Barriers persist 360 s and
  the fly revisits them, so neither "most-recent-created" nor "nearest active" wins:
  both put the fly inside the reconstructed hurt zone (wall `+/- laser_margin`) for
  only ~55 % of bounces (23/36, 19/36, 9/28); for the rest the laser fires while the
  fly is 4-39 mm from the nearest active wall. The residual gap is consistent with the
  `barrier_distance_to_fly = 5` spawn offset plus a laser-trigger condition the logs
  do not fully expose. So the drawn wall is the **best config-based estimate** of the
  barrier, not a guaranteed exact hurt-zone fit. This is why some bounces show the fly
  zapped slightly in front of, or behind, the drawn wall.
- **Clustering** is plain k-means (sklearn; `tslearn` is not installed) on the
  resampled `(U, V)` trajectory over `CLUST_WIN = -10..+30 s`, standardised, `k = 5`,
  relabelled 0..4 by size for stable colours. `cluster_overview.png` shows each type's
  members + mean; the single plots carry the type in their title and filename. The
  types separate roughly into along-wall, straight-off-the-wall, crossing, and
  lingering responses (type 0 is the large low-displacement "lingers near wall" group).
- **Scale bar dodges the trace:** `utils.emptiest_corner` places it in whichever axis
  corner holds the fewest trajectory points.

### Hand-curation of bounces (`bounce_select.py` -> `*_good/`)

Because the auto bounce-to-barrier matching is imperfect (above), bounces are curated
by eye. `bounce_select.py` is a small **napari** tool: it stacks every bounce thumbnail,
lets you tag good ones (`g`), and **autosaves** the selection to
`plots/bounce_gallery/good_bounces.csv` (keyed by `eid` + laser-on index). Each bounce
in `bo.bounces`/`extract` now carries that `(eid, i)` key so the CSV maps back exactly.
When the CSV exists, `bounce_analysis` and `bounce_clusters` auto-restrict to the
curated set (`bo.good_set()`, toggled by `USE_GOOD`) and write to parallel `*_good/`
folders, leaving the all-bounces outputs intact. `interwall_analysis` can also restrict
(a trial is kept only if the just-removed wall had a curated-good bounce), but defaults
`USE_GOOD = False`: the curation is keyed by laser bounce, so it would drop exactly the
zero-bounce walls that the interwall analysis is built to include. Note the collision-
frame analysis (`bounce_analysis`) keeps fewer than the gallery curated (e.g. 44 of 52):
bounces without a collision logged within 1 s have no collision-frame trace, so they
exist in the config-barrier gallery but not in the pooled wall-aligned analysis.

## Interwall analysis (`interwall_analysis.py`)

Replaced the earlier bounce-to-bounce trial. The old trial was zap-to-zap
(fly-driven), which **dropped the walls the fly never hit** (5 of 33 here) and needed a
`MAX_TRIAL_S` cap purely to throw away inter-bounce gaps that straddled a wall
vanishing. The wall on/off cycle is the experiment's real unit, so the trial is now
defined by it.

- **A trial is one interwall gap:** from a wall's `DESTROY` (wall off, the origin,
  `t = 0`) to the next wall's `CREATE` (wall on). `wall_cycles()` parses
  `CREATE`/`DESTROY` from `vrcmd.csv`, matched by `RectMaze#id` (walls are presented
  strictly **one at a time**), giving each wall's `t_on`, `t_off` and geometry. N walls
  give **N-1** trials and **every** transition is included. Runs **72/55/46** give
  **32/20/27 = 79 trials**.
- **The off-gap is a near-constant ITI:** median **60 s** (`05_inter_wall_interval`).
  This is the protocol's fixed cooldown, not fly behaviour, so unlike the old bimodal
  IBI there is nothing to read from its distribution. **`MAX_TRIAL_S = 180 s`** now only
  drops pathological gaps (a session pause or a missing next wall), not the signal.
- **Frames:** trajectory and tortuosity use **vrpos** (the wall lives there); speed
  uses the **fulltrack** walking speed from `load_combined`. Same split as the rest of
  the barrier work.
- **Trajectory alignment is translate-only:** each trial's wall-off position is moved to
  the origin, VR-world orientation **kept**, so trials fan out in every direction (the
  raw data). Both walls are drawn in the trial frame, true to config (100 x 2) plus the
  laser margins: the **next wall** (wall on, solid, where the trial is heading) and the
  **just-removed wall** (faint, where the fly came from).
- **Speed is shown twice:** **unnormalised** (vs seconds since wall off, mean drawn only
  where `>= MIN_TRIALS` trials still run) and **time-normalised** (vs gap phase
  0 = wall off, 1 = wall on), since gaps vary slightly in length.
- **Tortuosity is aligned to wall off** (`04_tortuosity`): path length / net displacement
  in **10 s bins**, four before (`-40..0`, wall still up) and four after (`0..+40`, no
  wall), pooled over **all 81 walls** with a logged `DESTROY`. Bins below
  `MIN_DISP_MM = 1 mm` of net displacement are dropped (NaN). It **dips while the wall is
  up and rises once the wall is gone**: the path is straighter with the wall present.
- **Off-period adaptation test** (`06_offspeed_over_time`): each trial's mean off-period
  speed vs its wall-off time in the session, pooled Spearman. Replaces the old
  IBI-over-time learning test, which is meaningless now the ITI is fixed; the result is
  in the panel title.

`USE_GOOD` is **off by default** here (see the curation note above): the bounce-keyed
curation would exclude exactly the zero-bounce walls this analysis exists to keep.
