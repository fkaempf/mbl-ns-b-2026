# fly/code

Analysis and plotting of FicTrac `fulltrack` CSV exports (ball-on-air walking trajectories).

## Layout

```
code/
  utils.py                      shared helpers (loading + plotting)
  scalloping.py                 compute helpers for the confinement analysis
  trajectories_3panel.py        side-by-side trajectories of 3 experiments
  trajectory_time_windows.py    one experiment, split into time windows
  heading_analysis.py           one experiment: trajectory + heading over time
  confinement_scalloping.py     scalloping in a virtual enclosure + controls
  menotaxis_heading.py          preferred heading before vs after confinement
  plots/                        saved figures (output; git-ignored)
```

`rig1_experiment_06` runs **menotaxis (0-900 s) -> confinement in a 100 mm virtual
enclosure (900-1800 s) -> menotaxis (1800 s-end)**. During confinement the walked
path fills a disc whose rim is *scalloped*: repeated out-and-back excursions to the
wall. `confinement_scalloping.py` characterises those scallops (geometry, timing,
handedness, speed coupling, spectral rhythm) and compares the confinement window to
the two menotaxis windows as within-fly controls. `menotaxis_heading.py` compares
the preferred walking direction before vs after the confinement. The reusable
computation (circle fit, contact detection, scallop geometry, bounce alignment,
circular stats) lives in `scalloping.py`.

These two scripts write **one standalone image per panel** into a per-analysis
subfolder of `plots/` (e.g. `plots/<experiment>_scalloping/`,
`_controls/`, `_menotaxis/`) rather than a single composite figure. Among them,
`09_bounce_average_trajectory.png` and `10_bounce_average_radial.png` show the
**bounce-aligned mean ± SEM**: every wall contact is re-centred on the contact
point with the wall normal as a common axis and the turn handedness folded, so the
stereotyped approach → contact → retreat excursion can be averaged across bounces.

Each script is self-contained: set the experiment(s) at the top, then it loads,
computes, plots, and saves — read top to bottom. Shared logic lives in `utils.py`.

## Running

Run from the `code/` directory:

```sh
python trajectory_time_windows.py     # or trajectories_3panel.py / heading_analysis.py
```

## Choosing the data

Edit the `EXPERIMENT` / `EXPERIMENTS` constant at the top of each script. It is an
experiment *folder* (relative to `fly/data`, or an absolute path); the fulltrack CSV
inside it is found automatically.

Every figure is labelled with its source experiment — as a title on the image and as
a prefix on the saved filename (e.g. `rig1_experiment_06_trajectory_0_900s.png`), so
it is always clear which data a figure shows. Figures are written to `code/plots/`.
