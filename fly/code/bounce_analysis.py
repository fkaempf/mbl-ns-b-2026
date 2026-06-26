"""Single-bounce analysis: how the fly approaches and leaves a barrier when the
laser fires. Wall-aligned, approach-from-the-left bounce trajectories pooled over
the barrier experiments.

For each plot window (10 s, 30 s, 1 min, 2 min, each in its own folder) it draws:
  * all bounces - faint individual traces + mean +/- SEM, approach vs departure,
  * incidence-binned (drop <15 deg; 15-40, 40-70, 70-90 deg) - one plot per bin and
    a united plot,
  * a scatter of incidence-in vs outgoing angle, coloured by bin.
The laser hurt zone is drawn on every trajectory plot.

Run from this directory:  python bounce_analysis.py
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import save_fig

EXPERIMENTS = [
    "20260625/rig1_experiment_72",
    "20260625/rig1_experiment_55",
    "20260625/rig1_experiment_46",
]
WINDOWS_S = [10, 30, 60, 120]
BINS = [(15, 40), (40, 70), (70, 90)]
MAXWIN = 120.0
MIN_INCIDENCE = 15.0
GLITCH_WIN_S = 2.0       # s around the hit to check for a collision glitch through the wall
CROSS_TOL = 2.0          # mm; a dip past the wall within GLITCH_WIN_S = a glitch (dropped)
MEAN_WINDOW_S = 10.0     # outer cap: the mean +/- SEM is never drawn beyond this window
FAINT_ALPHA = 0.2        # individual-trace opacity
COH_MIN = 0.4            # stop drawing the mean once directional coherence drops below this

APPROACH, DEPART = "#1f77b4", "#d62728"
BIN_COLORS = ["#4477aa", "#228833", "#aa3377"]


def draw_wall(ax):
    ax.axhspan(-bo.LASER_MARGIN - bo.WALL_TH / 2, bo.LASER_MARGIN + bo.WALL_TH / 2,
               color="red", alpha=0.15, zorder=0, label="hurt zone")
    ax.axhspan(-bo.WALL_TH / 2, bo.WALL_TH / 2, color="0.4", zorder=1, label="wall")


def mean_arrays(group):
    U = np.array([b["U"] for b in group])
    V = np.array([b["V"] for b in group])
    mU, sU = bo.mean_sem(U)
    mV, sV = bo.mean_sem(V)
    n = np.sum(~np.isnan(U), axis=0)
    return mU, sU, mV, sV, n


def mean_mask(group, tau, k=7):
    """Where the mean is drawn: within the fixed window and while the trajectories
    stay directionally **coherent** (mean step / mean individual step >= COH_MIN).
    The mean collapses where the traces fan out, so it is not drawn there."""
    U = np.array([b["U"] for b in group])
    V = np.array([b["V"] for b in group])
    dU, dV = np.gradient(U, axis=1), np.gradient(V, axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coh = np.hypot(np.nanmean(dU, 0), np.nanmean(dV, 0)) / (
            np.nanmean(np.hypot(dU, dV), 0) + 1e-9)
    coh = np.convolve(np.nan_to_num(coh), np.ones(k) / k, mode="same")
    return (coh >= COH_MIN) & (np.abs(tau) <= MEAN_WINDOW_S)


def plot_mean(ax, tau, m, wmask, color, label=None):
    mU, sU, mV, sV, n = m
    valid = wmask & ~np.isnan(mU) & ~np.isnan(mV) & (n >= 5)
    for seg, c in [(valid & (tau <= 0), color), (valid & (tau >= 0), color)]:
        idx = np.where(seg)[0]
        if len(idx) < 2:
            continue
        P = np.column_stack([mU[idx], mV[idx]])
        T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        N = np.column_stack([-T[:, 1], T[:, 0]])
        sig = np.sqrt((N[:, 0] * sU[idx]) ** 2 + (N[:, 1] * sV[idx]) ** 2)
        up, lo = P + sig[:, None] * N, P - sig[:, None] * N
        ax.fill(np.r_[up[:, 0], lo[::-1, 0]], np.r_[up[:, 1], lo[::-1, 1]],
                color=c, alpha=0.22, lw=0, zorder=3)
    ax.plot(mU[valid], mV[valid], color=color, lw=2.5, zorder=4, label=label)


def plot_faint(ax, group, tau, wmask):
    pre, post = wmask & (tau <= 0), wmask & (tau >= 0)
    for b in group:
        ax.plot(b["U"][pre], b["V"][pre], color=APPROACH, alpha=FAINT_ALPHA, lw=0.5, zorder=2)
        ax.plot(b["U"][post], b["V"][post], color=DEPART, alpha=FAINT_ALPHA, lw=0.5, zorder=2)


def finish(ax, title):
    draw_wall(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("along wall (mm)"); ax.set_ylabel("distance from wall (mm)")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def plot_all(group, tau, W, sub):
    wmask = np.abs(tau) <= W
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_faint(ax, group, tau, wmask)
    m = mean_arrays(group)
    cmask = mean_mask(group, tau)
    plot_mean(ax, tau, m, cmask & (tau <= 0), APPROACH, "approach (mean)")
    plot_mean(ax, tau, m, cmask & (tau >= 0), DEPART, "departure (mean)")
    finish(ax, f"all bounces (n={len(group)}) +/- SEM, +/-{W}s")
    ax.legend(fontsize=7, loc="upper right")
    save_fig(fig, "all_bounces.png", subdir=sub); plt.close(fig)


def plot_binned(groups, tau, W, sub):
    wmask = np.abs(tau) <= W
    # one plot per bin
    for (lo, hi), col in zip(BINS, BIN_COLORS):
        g = groups[(lo, hi)]
        fig, ax = plt.subplots(figsize=(7, 7))
        if g:
            plot_faint(ax, g, tau, wmask)
            plot_mean(ax, tau, mean_arrays(g), mean_mask(g, tau), col)
        finish(ax, f"incidence {lo}-{hi} deg (n={len(g)}), +/-{W}s")
        save_fig(fig, f"bin_{lo}_{hi}.png", subdir=sub); plt.close(fig)
    # united: bin means together
    fig, ax = plt.subplots(figsize=(7, 7))
    for (lo, hi), col in zip(BINS, BIN_COLORS):
        g = groups[(lo, hi)]
        if g:
            plot_mean(ax, tau, mean_arrays(g), mean_mask(g, tau), col, f"{lo}-{hi} deg (n={len(g)})")
    finish(ax, f"incidence bins, mean +/- SEM, +/-{W}s")
    ax.legend(fontsize=7, loc="upper right")
    save_fig(fig, "binned_all.png", subdir=sub); plt.close(fig)


def plot_scatter(group, sub):
    inc = np.array([b["incidence"] for b in group])
    out = np.array([b["outgoing"] for b in group])
    ok = ~np.isnan(out)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 90], [0, 90], color="0.8", ls="--", lw=1)
    for (lo, hi), col in zip(BINS, BIN_COLORS):
        m = ok & (inc >= lo) & (inc < hi)
        ax.scatter(inc[m], out[m], color=col, s=30, label=f"{lo}-{hi} deg")
    ax.scatter(inc[ok & (inc < MIN_INCIDENCE)], out[ok & (inc < MIN_INCIDENCE)],
               facecolor="none", edgecolor="0.6", s=25, label="<15 deg (dropped)")
    ax.set(xlim=(0, 90), ylim=(0, 90), xlabel="incidence in (deg)",
           ylabel="outgoing angle (deg)", title="bounce: in vs out angle")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "in_out_scatter.png", subdir=sub); plt.close(fig)


def plot_reflection(group, sub):
    """Does the fly continue along the wall (reflect) or turn back (reverse)?"""
    inc = np.array([b["incidence"] for b in group])
    od = np.array([b["out_dir"] for b in group])
    ok = ~np.isnan(od)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 180, 19)
    for (lo, hi), col in zip(BINS, BIN_COLORS):
        m = ok & (inc >= lo) & (inc < hi)
        if m.sum():
            ax.hist(od[m], bins=bins, histtype="step", lw=2, color=col,
                    label=f"{lo}-{hi} deg (n={m.sum()})")
    ax.axvline(90, color="0.5", ls="--", lw=1)
    binned = ok & (inc >= MIN_INCIDENCE)
    rev = np.mean(od[binned] > 90) if binned.sum() else np.nan
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("exit direction, wall frame (deg)\n<90 = continue along wall    >90 = reverse / turn back")
    ax.set_ylabel("count")
    ax.set_title(f"bounce: continue vs reverse  (reverse {100 * rev:.0f}%)")
    ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "reflection.png", subdir=sub); plt.close(fig)


def glitched(b):
    """A collision glitch: the trace dips past the wall within GLITCH_WIN_S of the
    hit (the fly jumps *through* the wall). Walk-arounds (crossing later / at the
    wall end) are not flagged, so they are kept."""
    v = b["V"][np.abs(b["tau"]) <= GLITCH_WIN_S]
    v = v[~np.isnan(v)]
    return len(v) > 0 and v.min() < -CROSS_TOL


def main():
    group = []
    for f in EXPERIMENTS:
        b = bo.bounces(f, half_win_s=MAXWIN)
        group += b
        print(f"  {f.split('/')[-1]}: {len(b)} bounces")
    group = [b for b in group if not glitched(b)]          # drop collision glitches only
    print(f"total bounces: {len(group)} (after glitch drop; walk-arounds kept)")
    tau = group[0]["tau"]

    kept = [b for b in group if b["incidence"] >= MIN_INCIDENCE]
    groups = {(lo, hi): [b for b in kept if lo <= b["incidence"] < hi] for lo, hi in BINS}
    for W in WINDOWS_S:
        sub = f"bounces/{W}s"
        plot_all(group, tau, W, sub)
        plot_binned(groups, tau, W, sub)
        plot_scatter(group, sub)
        plot_reflection(group, sub)
    print("done")


if __name__ == "__main__":
    main()
