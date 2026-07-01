"""Bounce-to-bounce analysis: infer a trial structure from consecutive wall bounces.

A *bounce* is a laser-on event - the same definition as ``bounces.py``
(``bo.laser_on_times``: the aversive laser ramps to 255, deduped with a refractory
period). A *bounce-to-bounce trial* is the interval between two consecutive bounces:
the fly is zapped, walks, and is zapped again. Trials are pooled over the barrier
experiments. The script draws:
  * 01_trajectories - every trial's vrpos path from its start bounce (translated to
    the origin, no rotation) to the next bounce, all overlaid (the raw data),
  * 02_speed_vs_time - speed vs time since the bounce, one faint line per trial plus
    mean +/- SEM (unnormalised: trials end at their own next bounce),
  * 03_speed_vs_phase - the time-normalised version, speed vs fraction of the
    inter-bounce interval (0 = this bounce, 1 = next bounce),
  * 04_tortuosity - path tortuosity (path length / net displacement) in 10 s bins
    before and after the bounce, bounce-aligned and pooled over all bounces.

Trajectory and tortuosity use the vrpos (VR-world) path - the frame the wall lives
in; speed uses the fulltrack-derived walking speed from ``load_combined``. Reuses
``bounces.py`` read-only.

Run from this directory:  python bounce2bounce.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

import bounces as bo
from utils import DATA_DIR, add_scalebar, experiment_id, load_combined, save_fig

# --- config (edit) ---------------------------------------------------------
EXPERIMENTS = [
    "20260625/rig1_experiment_72",
    "20260625/rig1_experiment_55",
    "20260625/rig1_experiment_46",
]
MAX_TRIAL_S = 120.0      # drop inter-bounce gaps longer than this (barrier gone/cooldown); None = keep all
MIN_TRIAL_S = 0.0        # drop trials shorter than this (re-zaps while lingering at the wall); 0 = keep all
DT_S = 0.5               # resample step for the unnormalised speed mean (s)
N_PHASE = 50             # resample points for the time-normalised speed mean
MIN_SAMPLES = 5          # a trial needs at least this many trajectory samples
MIN_TRIALS = 5           # only draw the unnormalised mean where >= this many trials still run
TORT_BIN_S = 10.0        # tortuosity bin width (s)
TORT_NBINS = 4           # bins each side of the bounce (4 -> -40..0 and 0..+40 s)
MIN_DISP_MM = 1.0        # bins with less net displacement give an undefined tortuosity
LASER_MARGIN_X = 0.05    # config: aversive-laser x margin (hurt-zone inflation)
LASER_MARGIN_Y = 2.5     # config: aversive-laser y margin

SUBDIR = "bounce2bounce"
DPI = 300
TRACE, MEAN = "#4477aa", "#cc3311"
START, END = "#0FFF50", "#d62728"


def minimal(ax):
    """Drop the top/right spines for a cleaner look."""
    ax.spines[["top", "right"]].set_visible(False)


def barrier_walls(folder):
    """RectMaze walls (cx, cy, rot_deg, width, thickness) from ``vrcmd.csv``. Parsed
    by splitting (pandas chokes on the commas inside the color/serial fields)."""
    cmd = glob.glob(os.path.join(DATA_DIR, folder, "*vrcmd.csv"))[0]
    out = []
    with open(cmd) as fh:
        next(fh, None)
        for line in fh:
            p = line.split(",")
            if len(p) > 13 and p[1] == "CREATE" and p[2] == "RectMaze":
                out.append((float(p[5]), float(p[6]), float(p[10]),
                            float(p[11]), float(p[12])))
    return out


def match_wall(tb, x0, y0, walls, ct, bx, by):
    """The wall the fly hit at this bounce, in the trial frame (origin = bounce).

    The bounce is matched to the nearest collision in time (within ``bo.MATCH_S``),
    then the logged RectMaze nearest that contact point gives the true wall geometry
    (width 100, thickness 2 from config). Returns (cx, cy, rot, w, th) translated so
    the bounce sits at the origin, or None if no collision or wall matches."""
    if not walls or len(ct) == 0:
        return None
    j = int(np.argmin(np.abs(ct - tb)))
    if abs(ct[j] - tb) > bo.MATCH_S:
        return None
    k = int(np.argmin([np.hypot(w[0] - bx[j], w[1] - by[j]) for w in walls]))
    cx, cy, rot, w, th = walls[k]
    return (cx - x0, cy - y0, rot, w, th)


def trials(folder):
    """Bounce-to-bounce trials for one experiment.

    Each trial dict has ``tau`` (s since the start bounce), ``dur`` (inter-bounce
    interval, s), ``x``/``y`` (vrpos path, mm, translated so the start bounce is at
    the origin), ``speed`` (mm/s) and ``wall`` (the hit wall in the trial frame, or
    None)."""
    bt = bo.laser_on_times(folder)
    if len(bt) < 2:
        return []
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                   df.vry.to_numpy(), df.speed.to_numpy())
    walls = barrier_walls(folder)
    ct, bx, by, _, _ = bo.collisions(folder)
    out = []
    for t0, t1 in zip(bt[:-1], bt[1:]):
        dur = t1 - t0
        if dur < MIN_TRIAL_S or (MAX_TRIAL_S is not None and dur > MAX_TRIAL_S):
            continue
        m = (t >= t0) & (t <= t1)
        if m.sum() < MIN_SAMPLES:
            continue
        xi, yi = x[m], y[m]
        x0, y0 = xi[0], yi[0]
        out.append(dict(tau=t[m] - t0, dur=dur, x=xi - x0, y=yi - y0, speed=sp[m],
                        wall=match_wall(t0, x0, y0, walls, ct, bx, by)))
    return out


def tortuosity(x, y):
    """Path length / straight-line net displacement (>= 1; NaN if barely moving)."""
    if len(x) < 2:
        return np.nan
    path = np.hypot(np.diff(x), np.diff(y)).sum()
    disp = np.hypot(x[-1] - x[0], y[-1] - y[0])
    return path / disp if disp >= MIN_DISP_MM else np.nan


def bounce_tortuosity(folder, edges):
    """Per-bounce tortuosity in each time bin (rows = bounces, cols = bins)."""
    bt = bo.laser_on_times(folder)
    if len(bt) == 0:
        return np.empty((0, len(edges) - 1))
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y = df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy()
    rows = []
    for tb in bt:
        row = [tortuosity(x[(t >= tb + lo) & (t < tb + hi)],
                          y[(t >= tb + lo) & (t < tb + hi)])
               for lo, hi in zip(edges[:-1], edges[1:])]
        rows.append(row)
    return np.array(rows)


# --- plots -----------------------------------------------------------------
def draw_wall(ax, wall):
    """One trial's wall (true width x thickness from config) + its laser hurt zone."""
    cx, cy, rot, w, th = wall
    hx, hy = w / 2 + LASER_MARGIN_X, th / 2 + LASER_MARGIN_Y
    ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, angle=rot,
                           rotation_point="center", facecolor="red", alpha=0.04,
                           edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((cx - w / 2, cy - th / 2), w, th, angle=rot,
                           rotation_point="center", facecolor="0.4", alpha=0.1,
                           edgecolor="none", zorder=1))


def plot_trajectories(group):
    fig, ax = plt.subplots(figsize=(8, 8))
    nwall = sum(tr["wall"] is not None for tr in group)
    for tr in group:
        if tr["wall"] is not None:
            draw_wall(ax, tr["wall"])
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
               markeredgecolor="k", ms=8, label="bounce"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=END, ms=6,
               label="next bounce"),
        Patch(facecolor="0.4", alpha=0.5, label="wall (100 x 2)"),
        Patch(facecolor="red", alpha=0.3, label="hurt zone"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"bounce-to-bounce trajectories (n={len(group)}, {nwall} with wall)",
                 fontsize=10)
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
    ax.set_xlabel("time since bounce (s)")
    ax.set_ylabel("speed (mm/s)")
    ax.set_title(f"bounce-to-bounce speed (n={len(group)})", fontsize=10)
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
    ax.set_xlabel("inter-bounce phase (0 = this bounce, 1 = next bounce)")
    ax.set_ylabel("speed (mm/s)")
    ax.set_title(f"bounce-to-bounce speed, time-normalised (n={len(group)})", fontsize=10)
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
    ax.axvline(0, color="red", ls="--", lw=1, label="bounce")
    hi = np.nanpercentile(T, 95)
    ax.set_ylim(0.95, max(hi, np.nanmax(m + sem) * 1.1))
    ax.set_xlabel("time relative to bounce (s)")
    ax.set_ylabel("tortuosity (path / displacement)")
    ax.set_title(f"tortuosity before vs after bounce (n={T.shape[0]} bounces)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    minimal(ax)
    save_fig(fig, "04_tortuosity.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_ibi_distribution(ibi):
    """Histogram of all inter-bounce intervals - the trial structure itself."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0, np.ceil(ibi.max() / 20) * 20 + 20, 20)
    ax.hist(ibi, bins=bins, color=TRACE, alpha=0.85, edgecolor="white")
    if MAX_TRIAL_S is not None:
        ax.axvline(MAX_TRIAL_S, color="0.4", ls="--", lw=1, label=f"cap {MAX_TRIAL_S:g} s")
    ax.axvline(np.median(ibi), color=MEAN, lw=1.5, label=f"median {np.median(ibi):.0f} s")
    ax.set_xlabel("inter-bounce interval (s)")
    ax.set_ylabel("count")
    ax.set_title(f"trial structure: inter-bounce intervals (n={len(ibi)})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    minimal(ax)
    save_fig(fig, "05_inter_bounce_interval.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def plot_ibi_over_time(records):
    """Inter-bounce interval vs time in the session, per fly - the learning test."""
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(records):
        ax.scatter(r["t0"] / 60, r["ibi"], s=18, color=cyc[i % len(cyc)],
                   alpha=0.7, label=r["eid"])
    t0 = np.concatenate([r["t0"] for r in records])
    ibi = np.concatenate([r["ibi"] for r in records])
    keep = ibi <= (MAX_TRIAL_S if MAX_TRIAL_S is not None else np.inf)
    rho, p = spearmanr(t0[keep], ibi[keep])
    if MAX_TRIAL_S is not None:
        ax.axhline(MAX_TRIAL_S, color="0.6", ls="--", lw=1)
    ax.set_xlabel("time of bounce in session (min)")
    ax.set_ylabel("inter-bounce interval (s)")
    ax.set_title(f"inter-bounce interval over the session "
                 f"(Spearman rho={rho:.2f}, p={p:.2f}, n={keep.sum()} <= cap)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    minimal(ax)
    save_fig(fig, "06_ibi_over_time.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def main():
    print("=== bounce-to-bounce trials ===")
    group = []
    for folder in EXPERIMENTS:
        tr = trials(folder)
        group += tr
        print(f"  {experiment_id(folder)}: {len(tr)} trials")
    print(f"total trials: {len(group)}")
    if not group:
        return

    plot_trajectories(group)
    plot_speed_time(group)
    plot_speed_phase(group)

    edges = np.arange(-TORT_NBINS, TORT_NBINS + 1) * TORT_BIN_S
    T = np.vstack([bounce_tortuosity(folder, edges) for folder in EXPERIMENTS])
    print(f"tortuosity over {T.shape[0]} bounces, {T.shape[1]} bins of {TORT_BIN_S:g}s")
    plot_tortuosity(T, edges)

    # trial structure: the inter-bounce intervals themselves (all bounces, uncapped)
    recs = []
    for folder in EXPERIMENTS:
        bt = bo.laser_on_times(folder)
        if len(bt) >= 2:
            recs.append(dict(eid=experiment_id(folder), t0=bt[:-1] - bt[0], ibi=np.diff(bt)))
    ibi = np.concatenate([r["ibi"] for r in recs])
    print(f"inter-bounce intervals: n={len(ibi)} median={np.median(ibi):.0f}s")
    plot_ibi_distribution(ibi)
    plot_ibi_over_time(recs)
    print(f"done -> plots/{SUBDIR}/")


if __name__ == "__main__":
    main()
