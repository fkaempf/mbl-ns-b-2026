"""Interwall analysis: trial structure inferred from the wall on/off cycle.

The barrier paradigm (``eternarig_experiment_logic_barrier``) presents RectMaze walls
one at a time: a wall is created (``CREATE RectMaze`` in ``vrcmd.csv`` = *wall on*),
the fly walks and is sometimes zapped, the wall is removed (``DESTROY`` = *wall off*),
then after a roughly fixed ~60 s gap with no wall the next wall appears somewhere else.

A *wall-cycle trial* is the inter-wall gap: it starts at one wall's ``DESTROY`` (wall
off, the origin) and ends at the next wall's ``CREATE`` (wall on). N walls give N-1
trials, and every transition is included (unlike the old zap-driven trials, which
dropped the walls the fly never hit). Trials are pooled over the barrier experiments.
The script draws:
  * 01_trajectories - every trial's vrpos path from the wall-off origin to where the
    next wall appears, all overlaid; the removed wall (faint) and the next wall (solid)
    are drawn in each trial's frame,
  * 02_speed_vs_time - speed vs time since wall off, one faint line per trial plus
    mean +/- SEM (unnormalised: trials end at their own next wall-on),
  * 03_speed_vs_phase - the time-normalised version, speed vs fraction of the gap
    (0 = wall off, 1 = wall on),
  * 04_tortuosity - path tortuosity (path length / net displacement) in 10 s bins
    around the wall-off transition, pooled over all walls (before = wall still up,
    after = no wall),
  * 05_inter_wall_interval - the off-gap durations (the trial structure / fixed ITI),
  * 06_offspeed_over_time - off-period mean speed across the session (adaptation test).

Trajectory and tortuosity use the vrpos (VR-world) path - the frame the wall lives
in; speed uses the fulltrack-derived walking speed from ``load_combined``. Reuses
``bounces.py`` read-only (``wall_trials`` for the wall cycles, ``laser_on_times`` for
optional curation).

Run from this directory:  python interwall_analysis.py
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

import bounces as bo
from utils import add_scalebar, experiment_id, load_combined, save_fig

# --- config (edit) ---------------------------------------------------------
MODE = "good"      # "good" = the curated runs, "all" = every barrier run
EXPERIMENTS = bo.barrier_experiments(MODE)
MAX_TRIAL_S = 180.0      # drop pathological gaps (session pauses / missing next wall); None = keep all
MIN_TRIAL_S = 0.0        # drop gaps shorter than this; 0 = keep all
DT_S = 0.5               # resample step for the unnormalised speed mean (s)
N_PHASE = 50             # resample points for the time-normalised speed mean
MIN_SAMPLES = 5          # a trial needs at least this many trajectory samples
MIN_TRIALS = 5           # only draw the unnormalised mean where >= this many trials still run
TORT_BIN_S = 10.0        # tortuosity bin width (s)
TORT_NBINS = 4           # bins each side of wall off (4 -> -40..0 and 0..+40 s)
MIN_DISP_MM = 1.0        # bins with less net displacement give an undefined tortuosity
LASER_MARGIN_X = 0.05    # config: aversive-laser x margin (hurt-zone inflation)
LASER_MARGIN_Y = 2.5     # config: aversive-laser y margin

# good_bounces.csv is keyed by laser bounce, so it excludes exactly the zero-bounce
# walls this analysis wants. Off by default; when on, a trial is kept only if the wall
# that just turned off had >= 1 curated-good bounce during its on-epoch.
USE_GOOD = False
SUBDIR = "interwall" + ("" if MODE == "good" else "_all")
DPI = 300
TRACE, MEAN = "#4477aa", "#cc3311"
START, END = "#0FFF50", "#d62728"


def minimal(ax):
    """Drop the top/right spines for a cleaner look."""
    ax.spines[["top", "right"]].set_visible(False)


def good_bounce_times(folder, good):
    """Unix times of the curated-good bounces for this experiment (empty if none)."""
    if good is None:
        return None
    bt = bo.laser_on_times(folder)
    eid = experiment_id(folder)
    return np.array([bt[i - 1] for (e, i) in good
                     if e == eid and 1 <= i <= len(bt)])


def _wall_geom(wall, x0=0.0, y0=0.0):
    """Wall geometry (cx, cy, rot, w, th) from a ``bo.wall_trials`` tuple
    (t_on, t_off, cx, cy, rot, w, th), translated so (x0, y0) is the origin."""
    _, _, cx, cy, rot, w, th = wall
    return (cx - x0, cy - y0, rot, w, th)


def trials(folder, good=None):
    """Wall-cycle (off -> on) trials for one experiment.

    Each trial dict has ``tau`` (s since wall off), ``dur`` (inter-wall gap, s),
    ``x``/``y`` (vrpos path, mm, translated so the wall-off position is the origin),
    ``speed`` (mm/s), ``on_wall`` (the next wall, wall on, in the trial frame),
    ``off_wall`` (the just-removed wall), ``eid`` and ``t0_sess`` (wall-off time from
    the start of the recording, s). With ``good`` set, only trials whose just-removed
    wall had a curated-good bounce are kept."""
    walls = bo.wall_trials(folder)
    if len(walls) < 2:
        return []
    eid = experiment_id(folder)
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                   df.vry.to_numpy(), df.speed.to_numpy())
    t_start = t[0]
    gt = good_bounce_times(folder, good)
    out = []
    for k in range(len(walls) - 1):
        w_off, w_on = walls[k], walls[k + 1]
        t0, t1 = w_off[1], w_on[0]                              # this wall off -> next wall on
        if gt is not None and not np.any((gt >= w_off[0]) & (gt <= t0)):
            continue                                            # removed wall not curated-good
        dur = t1 - t0
        if dur < MIN_TRIAL_S or (MAX_TRIAL_S is not None and dur > MAX_TRIAL_S):
            continue
        m = (t >= t0) & (t <= t1)
        if m.sum() < MIN_SAMPLES:
            continue
        xi, yi = x[m], y[m]
        x0, y0 = xi[0], yi[0]
        out.append(dict(tau=t[m] - t0, dur=dur, x=xi - x0, y=yi - y0, speed=sp[m],
                        on_wall=_wall_geom(w_on, x0, y0),
                        off_wall=_wall_geom(w_off, x0, y0),
                        eid=eid, t0_sess=t0 - t_start))
    return out


def tortuosity(x, y):
    """Path length / straight-line net displacement (>= 1; NaN if barely moving)."""
    if len(x) < 2:
        return np.nan
    path = np.hypot(np.diff(x), np.diff(y)).sum()
    disp = np.hypot(x[-1] - x[0], y[-1] - y[0])
    return path / disp if disp >= MIN_DISP_MM else np.nan


def wall_off_tortuosity(folder, edges, good=None):
    """Per-wall tortuosity in each time bin around wall off (rows = walls, cols = bins).

    Bins are relative to each wall's ``DESTROY``: negative = wall still up, positive =
    no wall. With ``good`` set, only curated-good walls are included."""
    walls = bo.wall_trials(folder)
    if not walls:
        return np.empty((0, len(edges) - 1))
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y = df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy()
    gt = good_bounce_times(folder, good)
    rows = []
    for w in walls:
        t_on, tb = w[0], w[1]
        if gt is not None and not np.any((gt >= t_on) & (gt <= tb)):
            continue
        row = [tortuosity(x[(t >= tb + lo) & (t < tb + hi)],
                          y[(t >= tb + lo) & (t < tb + hi)])
               for lo, hi in zip(edges[:-1], edges[1:])]
        rows.append(row)
    return np.array(rows) if rows else np.empty((0, len(edges) - 1))


# --- plots -----------------------------------------------------------------
def draw_wall(ax, wall, faint=False):
    """One wall (true width x thickness from config) + its laser hurt zone. ``faint``
    dims it for the wall that has just been removed (gone during the trial)."""
    cx, cy, rot, w, th = wall
    hx, hy = w / 2 + LASER_MARGIN_X, th / 2 + LASER_MARGIN_Y
    zone_a, wall_a = (0.02, 0.05) if faint else (0.04, 0.1)
    ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, angle=rot,
                           rotation_point="center", facecolor="red", alpha=zone_a,
                           edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((cx - w / 2, cy - th / 2), w, th, angle=rot,
                           rotation_point="center", facecolor="0.4", alpha=wall_a,
                           edgecolor="none", zorder=1))


def plot_trajectories(group):
    fig, ax = plt.subplots(figsize=(8, 8))
    for tr in group:
        draw_wall(ax, tr["off_wall"], faint=True)
        draw_wall(ax, tr["on_wall"])
    for tr in group:
        ax.plot(tr["x"], tr["y"], color=TRACE, alpha=0.25, lw=0.6, zorder=2)
    ax.scatter([tr["x"][-1] for tr in group], [tr["y"][-1] for tr in group],
               s=10, color=END, alpha=0.5, zorder=3)
    ax.scatter(0, 0, s=50, color=START, edgecolor="k", lw=0.5, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    add_scalebar(ax)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=START,
               markeredgecolor="k", ms=8, label="wall off (origin)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=END, ms=6,
               label="at next wall on"),
        Patch(facecolor="0.4", alpha=0.5, label="next wall (100 x 2)"),
        Patch(facecolor="0.4", alpha=0.2, label="removed wall"),
        Patch(facecolor="red", alpha=0.3, label="hurt zone"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"wall off -> on trajectories (n={len(group)} trials)", fontsize=10)
    save_fig(fig, "01_trajectories.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_speed_time(group):
    gmax = MAX_TRIAL_S if MAX_TRIAL_S else max(tr["dur"] for tr in group)
    grid = np.arange(0, gmax + DT_S, DT_S)
    S = np.full((len(group), len(grid)), np.nan)
    for i, tr in enumerate(group):
        S[i] = np.interp(grid, tr["tau"], tr["speed"], left=np.nan, right=np.nan)
    m, sem = bo.mean_sem(S)
    valid = np.sum(~np.isnan(S), axis=0) >= MIN_TRIALS

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tr in group:
        ax.plot(tr["tau"], tr["speed"], color=TRACE, alpha=0.1, lw=0.5, zorder=1)
    ax.fill_between(grid[valid], (m - sem)[valid], (m + sem)[valid],
                    color=MEAN, alpha=0.25, zorder=2)
    ax.plot(grid[valid], m[valid], color=MEAN, lw=2.5, zorder=3, label="mean ± SEM")
    ax.axvline(0, color="0.5", ls="--", lw=1)
    ax.set_xlim(0, gmax)
    ax.set_xlabel("time since wall off (s)")
    ax.set_ylabel("speed (mm/s)")
    ax.set_title(f"wall off -> on speed (n={len(group)})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    minimal(ax)
    save_fig(fig, "02_speed_vs_time.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_speed_phase(group):
    phase = np.linspace(0, 1, N_PHASE)
    P = np.full((len(group), N_PHASE), np.nan)
    for i, tr in enumerate(group):
        if tr["dur"] > 0 and len(tr["tau"]) >= 2:
            P[i] = np.interp(phase, tr["tau"] / tr["dur"], tr["speed"])
    m, sem = bo.mean_sem(P)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tr in group:
        if tr["dur"] > 0:
            ax.plot(tr["tau"] / tr["dur"], tr["speed"],
                    color=TRACE, alpha=0.1, lw=0.5, zorder=1)
    ax.fill_between(phase, m - sem, m + sem, color=MEAN, alpha=0.25, zorder=2)
    ax.plot(phase, m, color=MEAN, lw=2.5, zorder=3, label="mean ± SEM")
    ax.set_xlim(0, 1)
    ax.set_xlabel("gap phase (0 = wall off, 1 = wall on)")
    ax.set_ylabel("speed (mm/s)")
    ax.set_title(f"wall off -> on speed, time-normalised (n={len(group)})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    minimal(ax)
    save_fig(fig, "03_speed_vs_phase.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_tortuosity(T, edges):
    centers = (edges[:-1] + edges[1:]) / 2
    m, sem = bo.mean_sem(T)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, c in enumerate(centers):
        yj = T[:, j][~np.isnan(T[:, j])]
        ax.scatter(np.full(yj.size, c), yj, color=TRACE, alpha=0.15, s=8, zorder=1)
    ax.errorbar(centers, m, yerr=sem, fmt="o-", color=MEAN, capsize=3, lw=2,
                zorder=3, label="mean ± SEM")
    ax.axvline(0, color="red", ls="--", lw=1, label="wall off")
    hi = np.nanpercentile(T, 95)
    ax.set_ylim(0.95, max(hi, np.nanmax(m + sem) * 1.1))
    ax.set_xlabel("time relative to wall off (s)\n<0 = wall still up    >0 = no wall")
    ax.set_ylabel("tortuosity (path / displacement)")
    ax.set_title(f"tortuosity before vs after wall off (n={T.shape[0]} walls)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    minimal(ax)
    save_fig(fig, "04_tortuosity.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_iwi_distribution(iwi):
    """Histogram of all inter-wall (off) intervals - the trial structure itself."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0, np.ceil(iwi.max() / 20) * 20 + 20, 20)
    ax.hist(iwi, bins=bins, color=TRACE, alpha=0.85, edgecolor="white")
    if MAX_TRIAL_S is not None:
        ax.axvline(MAX_TRIAL_S, color="0.4", ls="--", lw=1, label=f"cap {MAX_TRIAL_S:g} s")
    ax.axvline(np.median(iwi), color=MEAN, lw=1.5, label=f"median {np.median(iwi):.0f} s")
    ax.set_xlabel("inter-wall (off) interval (s)")
    ax.set_ylabel("count")
    ax.set_title(f"trial structure: inter-wall intervals (n={len(iwi)})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    minimal(ax)
    save_fig(fig, "05_inter_wall_interval.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_offspeed_over_time(records):
    """Off-period mean speed vs time in the session, per fly - the adaptation test."""
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(records):
        ax.scatter(r["t0"] / 60, r["v"], s=18, color=cyc[i % len(cyc)],
                   alpha=0.7, label=r["eid"])
    t0 = np.concatenate([r["t0"] for r in records])
    v = np.concatenate([r["v"] for r in records])
    keep = np.isfinite(v)
    rho, p = spearmanr(t0[keep], v[keep]) if keep.sum() >= 3 else (np.nan, np.nan)
    ax.set_xlabel("wall-off time in session (min)")
    ax.set_ylabel("off-period mean speed (mm/s)")
    ax.set_title(f"off-period speed over the session "
                 f"(Spearman rho={rho:.2f}, p={p:.2f}, n={keep.sum()})", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    minimal(ax)
    save_fig(fig, "06_offspeed_over_time.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def main():
    global SUBDIR
    good = bo.good_set() if USE_GOOD else None
    if good is not None:
        SUBDIR = "interwall_good"
        print(f"restricting to walls with a curated-good bounce -> plots/{SUBDIR}/")
    print("=== interwall (off -> on) trials ===")
    group = []
    for folder in EXPERIMENTS:
        tr = trials(folder, good)
        group += tr
        print(f"  {experiment_id(folder)}: {len(tr)} trials")
    print(f"total trials: {len(group)}")
    if not group:
        return

    plot_trajectories(group)
    plot_speed_phase(group)

    edges = np.arange(-TORT_NBINS, TORT_NBINS + 1) * TORT_BIN_S
    T = np.vstack([wall_off_tortuosity(folder, edges, good) for folder in EXPERIMENTS])
    print(f"tortuosity over {T.shape[0]} walls, {T.shape[1]} bins of {TORT_BIN_S:g}s")
    plot_tortuosity(T, edges)

    # trial structure: inter-wall intervals + per-trial off-period speed over the session
    iwi = np.array([tr["dur"] for tr in group])
    print(f"inter-wall intervals: n={len(iwi)} median={np.median(iwi):.0f}s")
    plot_iwi_distribution(iwi)

    by_eid = defaultdict(lambda: dict(t0=[], v=[]))
    for tr in group:
        by_eid[tr["eid"]]["t0"].append(tr["t0_sess"])
        by_eid[tr["eid"]]["v"].append(np.nanmean(tr["speed"]))
    recs = [dict(eid=e, t0=np.array(d["t0"]), v=np.array(d["v"]))
            for e, d in by_eid.items()]
    plot_offspeed_over_time(recs)
    print(f"done -> plots/{SUBDIR}/")


if __name__ == "__main__":
    main()
