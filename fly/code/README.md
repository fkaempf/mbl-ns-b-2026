# fly/code

Analysis and plotting of tethered-fly walking data: FicTrac `fulltrack` CSV exports
(ball-on-air trajectory + heading) and the Unity VR `vrpos` logs (virtual position +
heading).

## Layout

```
code/
  utils.py                      shared loaders (fulltrack, vrpos, combined) + plotting helpers
  scalloping.py                 compute for the confinement analysis + circular stats

  trajectory_time_windows.py    one experiment, split into time windows
  trajectories_3panel.py        side-by-side trajectories of 3 experiments
  overview_all.py               4-panel raw-trace overview per experiment (triage)
  traces_overlay.py             all flies' traces in one figure (overlaid + juxtaposed)

  heading_analysis.py           one experiment: trajectory + heading over time
  menotaxis_heading.py          preferred heading before vs after confinement (fulltrack)
  vrpos_heading_hist.py         VR heading before/after, speed-gated: histograms + scatter
  heading_sampling_traces.py    where on the trace the VR heading is sampled (red on black)

  confinement_scalloping.py     scalloping in a virtual enclosure + within-fly controls

  barrier_traces.py             all barrier runs as vector PDF: trace + walls, red while
                                shocked, heading@create; good runs marked; angles/arrows/plain
  barrier_approaches.py         fly's reaction to one barrier over time, barrier-aligned

  bounces.py                    bounce compute: laser-on detection, wall-aligned frames,
                                incidence/exit angles, the good_set() curation helper
  bounce_analysis.py            pooled bounce analysis: mean +/- SEM, incidence bins,
                                in/out scatter, reflection (continue vs reverse)
  interwall_analysis.py         interwall trials (wall off -> next wall on, from the
                                CREATE/DESTROY cycle): trajectories, speed (raw +
                                time-normalised), tortuosity around wall off, inter-wall
                                interval (trial structure + off-period adaptation test)
  bounce_clusters.py            per-bounce gallery (one figure each, true finite wall,
                                30s before/60s after) + k-means clustering of bounce types
  bounce_select.py              napari tool to hand-curate good bounces -> good_bounces.csv
                                (double-click start_bounce_select.command to launch)
  wall_trials.py                wall-presentation trials (wall-ON to wall-OFF, paired by
                                serial): wall-frame trajectories, approach/retreat, summary
```

Barriers (paradigm `eternarig_experiment_logic_barrier`) are `RectMaze` walls
logged in `vrcmd.csv` as `CREATE RectMaze` (centre = position_x/y, orientation =
rotation_z, size = scale = width x thickness), in the vrpos world frame. The
aversive-laser "hurt zone" is the wall inflated by `laser_margin_x/y` from
`config.yaml`. `vrcollisions.csv` logs every wall contact.

**Curating bounces.** `bounce_select.py` (napari) tags good bounces into
`plots/bounce_gallery/good_bounces.csv` (autosaves on every tag). When that file
exists, the bounce analyses auto-restrict to the curated set and write to parallel
`*_good/` folders (`bounces_good/`, `bounce_gallery_good/`, `interwall_good/`); the
`USE_GOOD` flag at the top of each script toggles this off.

Figures are written to **`fly/plots/`** (one level up from `code/`; git-ignored),
sorted into one folder **per plot type**:

```
plots/
  traces/         raw trajectory traces (per-experiment overviews + all-fly overlays)
  barriers/       barrier runs (PDF): angles/ arrows/ plain/ variants
  bounces/        pooled bounce analysis (10s/30s/60s/120s windows)
  bounce_gallery/ one figure per bounce (true finite wall) + k-means cluster overview
  interwall/      interwall (wall off -> on) trial trajectories, speed, tortuosity
  wall_trials/    wall-presentation (wall-on to wall-off) trial trajectories + dynamics
  *_good/         the same analyses restricted to the hand-curated bounces
  scalloping/     confinement scalloping panels
  controls/       confinement vs menotaxis control panels
  menotaxis/      menotaxis heading panels
  heading/        VR heading histograms, before/after scatter, sampling traces
```

## The experiment

`rig1_experiment_06` (and the other "good" recordings) run **menotaxis (0-900 s) ->
confinement in a 100 mm virtual enclosure (900-1800 s) -> menotaxis (1800 s-end)**.
During confinement the walked path fills a disc whose rim is *scalloped*: repeated
out-and-back excursions to the wall.

- `confinement_scalloping.py` characterises those scallops (geometry, timing,
  handedness, speed coupling, spectral rhythm), the bounce-aligned mean ± SEM, and
  the within-fly controls. Reusable compute (circle fit, contact detection, scallop
  geometry, bounce alignment, circular stats) lives in `scalloping.py`.
- `vrpos_heading_hist.py` compares the VR heading just before vs just after the
  confinement, using only samples where the fly walks faster than 5 mm/s
  (`utils.load_combined` merges fulltrack speed with the vrpos heading on one
  timeline). `heading_sampling_traces.py` shows where those samples sit on the trace.

## Running

Run from the `code/` directory, e.g.:

```sh
python overview_all.py          # triage which recordings to use
python confinement_scalloping.py
python vrpos_heading_hist.py
```

Each script is self-contained: set the experiment(s) and parameters in the config
block at the top, then it loads, computes, plots, and saves — read top to bottom.

## Choosing the data

Edit the `EXPERIMENT` / `EXPERIMENTS` constant at the top of each script. It is an
experiment *folder* (relative to `fly/data`, or an absolute path); the fulltrack /
vrpos CSV inside it is found automatically. Every figure is labelled with its source
experiment so it is always clear which data a figure shows.
