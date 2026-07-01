"""Wall-presentation trials: one trial = a barrier from wall-ON to wall-OFF.

A more natural trial unit than bounce-to-bounce: a wall appears in front of the fly
(CREATE), the fly walks to it and is usually zapped once, then it vanishes (DESTROY),
followed by a ~60 s cooldown before the next wall. This reframes the barrier analysis
around that cycle:
  * 01_trajectories - every trial's vrpos path during the wall's lifetime, in the wall
    frame (wall horizontal, fly side up), coloured by time since wall-on,
  * 02_dynamics     - distance-to-wall and speed vs time since wall-on, plus the fly's
    distance to the wall at wall-off. (Wall-off is DISTANCE-triggered: the wall is
    removed once the fly has walked far enough away, so the trial ends when the fly
    disengages - do not read the speed at wall-off as a response, it is selection.)
  * 03_summary      - bounces per trial, time-to-first-bounce, trial duration.

Trial = `bo.wall_trials` (CREATE->DESTROY paired by serial). Trajectory/speed from
`load_combined` (vrpos + fulltrack speed). Run:  python wall_trials.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

import bounces as bo
from utils import experiment_id, load_combined, save_fig

MODE = "good"      # "good" = the curated runs, "all" = every barrier run
EXPERIMENTS = bo.barrier_experiments(MODE)
SUBDIR = "wall_trials" + ("" if MODE == "good" else "_all")
ON_WIN, DT = 40.0, 0.5                      # wall-on alignment window (s) and resample step
TRIM_PRE, TRIM_POST = 5.0, 12.0             # clip 01_trajectories to this window around the bounce
TRACE = "#4477aa"


def trials(folder):
    """Per-trial wall-frame trajectory + metrics for one experiment."""
    tr = bo.wall_trials(folder)
    bt = bo.laser_on_times(folder)
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                   df.vry.to_numpy(), df.speed.to_numpy())
    eid, out = experiment_id(folder), []
    for ton, toff, cx, cy, rot, w, th in tr:
        m = (t >= ton) & (t <= toff)
        if m.sum() < 5:
            continue
        a = np.radians(rot)
        c, s = np.cos(a), np.sin(a)
        dx, dy = x[m] - cx, y[m] - cy
        u = dx * c + dy * s                      # along the wall's long axis
        v = -dx * s + dy * c                     # perpendicular to the wall
        if np.nanmean(v) < 0:                    # put the fly's side up
            v = -v
        inwin = (bt >= ton) & (bt <= toff)
        ttf = bt[inwin][0] - ton if inwin.any() else np.nan
        out.append(dict(eid=eid, tau=t[m] - ton, u=u, v=v, sp=sp[m],
                        dur=toff - ton, nb=int(inwin.sum()), ttf=ttf, w=w, toff=toff))
    return out


def grid_mean(items, key_t, key_y, grid):
    A = np.full((len(items), len(grid)), np.nan)
    for i, it in enumerate(items):
        A[i] = np.interp(grid, it[key_t], it[key_y], left=np.nan, right=np.nan)
    n = np.sum(~np.isnan(A), 0)
    m = np.nanmean(A, 0)
    sem = np.nanstd(A, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, sem, n


def plot_trajectories(G):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Rectangle((-G[0]["w"] / 2, -bo.LASER_MARGIN - bo.WALL_TH / 2), G[0]["w"],
                 2 * bo.LASER_MARGIN + bo.WALL_TH, facecolor="red", alpha=0.15, zorder=1))
    ax.add_patch(Rectangle((-G[0]["w"] / 2, -bo.WALL_TH / 2), G[0]["w"], bo.WALL_TH,
                 facecolor="0.4", edgecolor="k", lw=0.8, zorder=2))
    for it in G:
        pts = np.column_stack([it["u"], it["v"]]).reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(seg, cmap="viridis", array=it["tau"][:-1],
                                         lw=0.7, alpha=0.7, zorder=3))
    ax.set_xlabel("along wall (mm)"); ax.set_ylabel("perpendicular to wall (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"wall-presentation trajectories (n={len(G)} trials, coloured by time since wall-on)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "01_trajectories.png", subdir=SUBDIR)
    plt.close(fig)


def plot_trajectories_clipped(G):
    """Alternative to 01: each trace clipped to a fixed window around its bounce, so
    all traces are the same length and the plot zooms in on the wall interaction
    instead of sprawling out to the ~200 mm wall-off cutoff."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-G[0]["w"] / 2, -bo.LASER_MARGIN - bo.WALL_TH / 2), G[0]["w"],
                 2 * bo.LASER_MARGIN + bo.WALL_TH, facecolor="red", alpha=0.15, zorder=1))
    ax.add_patch(Rectangle((-G[0]["w"] / 2, -bo.WALL_TH / 2), G[0]["w"], bo.WALL_TH,
                 facecolor="0.4", edgecolor="k", lw=0.8, zorder=2))
    lc, nb = None, 0
    for it in G:
        if np.isnan(it["ttf"]):                            # only trials with a bounce
            continue
        tau = it["tau"]
        m = (tau >= it["ttf"] - TRIM_PRE) & (tau <= it["ttf"] + TRIM_POST)
        if m.sum() < 2:
            continue
        u, v, rel = it["u"][m], it["v"][m], tau[m] - it["ttf"]
        pts = np.column_stack([u, v]).reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap="viridis", array=rel[:-1], lw=0.9, alpha=0.7, zorder=3)
        lc.set_clim(-TRIM_PRE, TRIM_POST)
        ax.add_collection(lc)
        i0 = int(np.argmin(np.abs(rel)))
        ax.scatter(u[i0], v[i0], color="black", s=8, zorder=5)   # the bounce
        nb += 1
    ax.set_xlabel("along wall (mm)"); ax.set_ylabel("perpendicular to wall (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"wall trajectories clipped to -{TRIM_PRE:.0f}..+{TRIM_POST:.0f}s around the bounce "
                 f"(n={nb}; black = bounce)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if lc is not None:
        fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("time from bounce (s)")
    save_fig(fig, "01b_trajectories_clipped.png", subdir=SUBDIR)
    plt.close(fig)


def plot_dynamics(G):
    on_grid = np.arange(0, ON_WIN + DT, DT)
    dist = [dict(t=it["tau"], y=np.abs(it["v"])) for it in G]
    spon = [dict(t=it["tau"], y=it["sp"]) for it in G]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, items, yl, ttl, col in [
        (ax[0], dist, "distance to wall (mm)", "approach/retreat after wall-on", "#aa3377"),
        (ax[1], spon, "speed (mm/s)", "speed after wall-on", TRACE)]:
        m, sem, _ = grid_mean(items, "t", "y", on_grid)
        a.fill_between(on_grid, m - sem, m + sem, color=col, alpha=0.25)
        a.plot(on_grid, m, color=col, lw=2)
        a.set_xlabel("time since wall-on (s)"); a.set_ylabel(yl)
        a.set_title(ttl, fontsize=10); a.spines[["top", "right"]].set_visible(False)
    # wall-off is DISTANCE-triggered: the wall is removed once the fly is far enough.
    doff = np.array([np.hypot(it["u"][-1], it["v"][-1]) for it in G])
    ax[2].hist(doff, bins=15, color="#228833", edgecolor="white")
    ax[2].axvline(np.median(doff), color="k", lw=1.5, label=f"median {np.median(doff):.0f} mm")
    ax[2].set_xlabel("fly distance to wall at wall-off (mm)"); ax[2].set_ylabel("trials")
    ax[2].set_title("wall-off trigger: fly has walked this far away", fontsize=10)
    ax[2].legend(frameon=False, fontsize=8); ax[2].spines[["top", "right"]].set_visible(False)
    save_fig(fig, "02_dynamics.png", subdir=SUBDIR)
    plt.close(fig)


def plot_summary(G):
    nb = np.array([it["nb"] for it in G])
    ttf = np.array([it["ttf"] for it in G])
    dur = np.array([it["dur"] for it in G])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hist(nb, bins=np.arange(-0.5, nb.max() + 1.5), color=TRACE, edgecolor="white")
    ax[0].set_xlabel("bounces per trial"); ax[0].set_ylabel("trials")
    ax[0].set_title(f"{(nb == 0).mean()*100:.0f}% trials with 0 bounces", fontsize=10)
    ax[1].hist(ttf[~np.isnan(ttf)], bins=15, color="#aa3377", edgecolor="white")
    ax[1].axvline(np.nanmedian(ttf), color="k", lw=1.5,
                  label=f"median {np.nanmedian(ttf):.0f}s")
    ax[1].set_xlabel("time to first bounce after wall-on (s)"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("how fast the fly reaches a new wall", fontsize=10)
    ax[2].hist(dur, bins=15, color="0.5", edgecolor="white")
    ax[2].axvline(np.median(dur), color="k", lw=1.5, label=f"median {np.median(dur):.0f}s")
    ax[2].set_xlabel("wall-on duration (s)"); ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_title("how long each wall stays up", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "03_summary.png", subdir=SUBDIR)
    plt.close(fig)


def main():
    G = []
    for f in EXPERIMENTS:
        tr = trials(f)
        G += tr
        print(f"  {experiment_id(f)}: {len(tr)} wall trials")
    print(f"total: {len(G)} wall-presentation trials")
    plot_trajectories(G)
    plot_trajectories_clipped(G)
    plot_dynamics(G)
    plot_summary(G)
    nb = np.array([it["nb"] for it in G])
    print(f"  bounces/trial: {np.bincount(nb)}  | median wall-on "
          f"{np.median([it['dur'] for it in G]):.0f}s  | median time-to-first-bounce "
          f"{np.nanmedian([it['ttf'] for it in G]):.0f}s")
    print(f"done -> plots/{SUBDIR}/")


if __name__ == "__main__":
    main()
