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
  `laser_exponential_set_end_level → 255`, deduped with a **1 s refractory**.
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

## Bounce-to-bounce analysis (`bounce2bounce.py`)

- Reuses the **same bounce definition** as the single-bounce analysis
  (`bo.laser_on_times`: laser-on, 1 s refractory) and the same runs **72/55/46**.
  It imports `bounces.py` **read-only** and adds no compute there.
- A **bounce-to-bounce trial** is the interval between two **consecutive** bounces
  (the fly is zapped, walks, is zapped again). The trial structure needs only the
  bounce **times**; the trajectory is aligned by **translation only** (see below).
- **`MAX_TRIAL_S = 120 s`** drops over-long inter-bounce gaps. Consecutive laser-ons
  can straddle a barrier vanishing (`barrier_duration 360 s`) and the next one
  appearing (`cooldown_duration 60 s`), which would make a "trial" that is mostly
  walking with no wall. 100 bounces give 60 trials after the cap.
- **Frames:** trajectory and tortuosity use **vrpos** (the wall lives there); speed
  uses the **fulltrack** walking speed from `load_combined`. Same split as the rest
  of the barrier work.
- **Trajectory alignment is translate-only:** each trial's start bounce is moved to
  the origin, the VR-world orientation is **kept** (no rotation). This is the raw
  data, so the trials fan out in every direction rather than being folded onto a
  common wall frame.
- **Each trial's own wall is drawn** on the trajectory plot, true to config
  (width 100, thickness 2) plus the laser margins. The bounce is matched to the
  nearest collision (within `bo.MATCH_S = 1 s`) and the logged `RectMaze` nearest
  that contact gives the wall geometry, translated into the trial frame. Because the
  alignment is translate-only the walls keep their own orientations, so they overlay
  as a small rosette around the bounce (the 100 mm wall is much shorter than the
  inter-bounce excursions). All 60 trials matched a wall.
- **Speed is shown twice** because inter-bounce intervals differ: an **unnormalised**
  version (speed vs seconds since the bounce, trials ending at their own next bounce,
  mean drawn only where `>= MIN_TRIALS` trials still run) and a **time-normalised**
  version (speed vs phase 0->1 of the interval), so the shape is comparable across
  trials of different length.
- **Tortuosity** = path length / straight-line net displacement (always `>= 1`),
  computed per bounce in **10 s bins**, four before and four after (`-40..0` and
  `0..+40 s`), bounce-aligned and pooled over **all** bounces (not just trial-internal
  ones). Bins with net displacement below `MIN_DISP_MM = 1 mm` give an undefined
  ratio and are dropped (NaN), so a near-stationary bin cannot blow the ratio up.

### Exploratory: the trial structure itself (`05_inter_bounce_interval.png`, `06_ibi_over_time.png`)

Added because the analysis is named for a "trial structure" but never characterised
it: how long does the fly stay away from the wall between zaps, and does that change
over the session (avoidance learning)? The inter-bounce interval (IBI) is the
trial duration; both plots use **all** bounces (uncapped), pooled over 72/55/46
(97 IBIs).

- **The IBI distribution is bimodal** (median 103 s). A **short mode** (~18 % of
  IBIs under 20 s) is the fly **lingering in the hurt zone and being re-zapped**, not
  a left-and-returned trip; a **trough** sits around 30-70 s; the main **return-trip
  mode** peaks at ~90-200 s, with a long tail (the barrier on/off cycle and
  wandering). This is why **`MIN_TRIAL_S` exists** (default `0` = keep all): setting
  it to ~20-30 s isolates genuine return trips from re-zaps. It is left **off by
  default** so the other four plots are unchanged unless deliberately filtered.
- **No learning trend** (honest negative, like the scalloping rhythm). Across the
  session the capped IBI is flat: pooled **Spearman rho = -0.01, p = 0.91** (n = 60
  trials `<= cap`), and per fly none is significant (p = 0.19 / 0.99 / 0.77).
  First-half vs second-half median IBI is unchanged (~85 vs ~87 s). The fly does
  **not** learn to avoid the wall, nor sensitise, over a single session.
