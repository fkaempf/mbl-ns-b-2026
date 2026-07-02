"""Windowed heading-phase coupling before vs after the first shock of each trial.

For Fly5+Fly6 (top row) and Fly5 alone (bottom row):

A *trial* is one wall presentation (wall on -> wall off); its *first shock* is the first
aversive-laser onset during that wall. For every trial and every window size
W in {0.5, 1, ..., 5} s, we take the window just before the first shock ([ts-W, ts)) and
just after it ([ts, ts+W)) and, over moving frames (speed >= MOVE_CUTOFF), compute:
  * the circular correlation of fly heading vs FC2 bump phase,
  * the window-mean bump magnitude A,
  * the window-mean walking speed.
Windows are relative to the shock (not clamped to the wall on/off), and a window is kept
only if it holds at least MIN_SAMP moving frames.

Panels (per row):
  1. heading-phase r vs window size, before vs after the first shock (mean +/- SEM over
     trials x windows-of-that-size);
  2. window-mean magnitude A vs the heading-phase r in that window (does a stronger bump
     mean tighter tracking?), pooled over trials/windows, Spearman rho;
  3. window-mean magnitude A vs window-mean speed, Spearman rho.

One figure in plots/imaging/analysis: fc2_window_sweep.

Run:  python fc2_window_sweep.py
"""

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt

import fc2_analysis as fa
from cxstyle import PINK, WHITE
from utils import save_fig

WINDOWS = np.arange(0.5, 5.01, 0.5)
MIN_SAMP = 10                          # min moving frames in a window to compute a correlation
BEFORE_C, AFTER_C = "#37e0d0", PINK    # teal = before the shock, pink = after


def trial_first_shocks(df):
    """First laser onset within each wall presentation (one event per shocked trial)."""
    ons, offs = fa.wall_on_times(df), fa.wall_off_times(df)
    sh = fa.shock_onset_times(df)
    events = []
    for ton, toff in zip(ons, offs):
        s = sh[(sh >= ton) & (sh <= toff)]
        if len(s):
            events.append(float(s[0]))
    return events


def records(flies):
    """One row per (trial, window, side) with the heading-phase r, mean A, mean speed."""
    rows = []
    for fly in flies:
        df = fa.add_fit(fa.load(fly))
        t = df["time"].to_numpy()
        head, phase = df["head"].to_numpy(), df["fit_phase"].to_numpy()
        A, spd = df["fit_A"].to_numpy(), df["speed"].to_numpy()
        mov = spd >= fa.MOVE_CUTOFF
        for ts in trial_first_shocks(df):
            for W in WINDOWS:
                for side, lo, hi in (("before", ts - W, ts), ("after", ts, ts + W)):
                    m = (t >= lo) & (t < hi) & mov
                    if m.sum() < MIN_SAMP:
                        continue
                    rows.append(dict(fly=fly, W=W, side=side,
                                     r=fa.circ_corr(head[m], phase[m]),
                                     A=float(np.mean(A[m])), speed=float(np.mean(spd[m]))))
    return pd.DataFrame(rows)


def plot_row(axes, rec, title):
    ax0, ax1, ax2 = axes
    # panel 1: r vs window size, before vs after
    for side, color in (("before", BEFORE_C), ("after", AFTER_C)):
        g = rec[rec.side == side]
        mu = g.groupby("W").r.mean()
        se = g.groupby("W").r.sem()
        ax0.plot(mu.index, mu.values, "-o", color=color, ms=4, label=side)
        ax0.fill_between(mu.index, mu.values - se.values, mu.values + se.values,
                         color=color, alpha=0.2)
    ax0.axhline(0, color=WHITE, lw=0.6, alpha=0.4)
    ax0.set_xlabel("window size (s)"); ax0.set_ylabel("heading-phase r")
    ax0.set_title(f"{title}: coupling vs window", fontsize=9); ax0.legend(fontsize=7)

    # panel 2: mean A vs r (does a stronger bump track better)
    for side, color in (("before", BEFORE_C), ("after", AFTER_C)):
        g = rec[rec.side == side]
        ax1.scatter(g.A, g.r, s=10, color=color, alpha=0.4)
    rho, p = st.spearmanr(rec.A, rec.r)
    ax1.set_xlabel("window mean bump magnitude A"); ax1.set_ylabel("heading-phase r")
    ax1.set_title(f"magnitude vs coupling (rho={rho:+.2f}, p={p:.1e})", fontsize=9)

    # panel 3: mean A vs mean speed
    for side, color in (("before", BEFORE_C), ("after", AFTER_C)):
        g = rec[rec.side == side]
        ax2.scatter(g.A, g.speed, s=10, color=color, alpha=0.4)
    rho2, p2 = st.spearmanr(rec.A, rec.speed)
    ax2.set_xlabel("window mean bump magnitude A"); ax2.set_ylabel("window mean speed (mm/s)")
    ax2.set_title(f"magnitude vs speed (rho={rho2:+.2f}, p={p2:.1e})", fontsize=9)


def main():
    variants = [("Fly5+Fly6", ["Fly5_002", "Fly6_002"]), ("Fly5 alone", ["Fly5_002"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
    for row, (title, flies) in zip(axes, variants):
        rec = records(flies)
        plot_row(row, rec, title)
        n_ev = rec[rec.W == WINDOWS[-1]].groupby("side").size().to_dict()
        print(f"{title:10s} rows={len(rec)}  trials/windows kept; "
              f"n(before/after) at W={WINDOWS[-1]}s = {n_ev}")
    fig.suptitle("Heading-phase coupling before (teal) vs after (pink) the first shock of "
                 "each trial", fontsize=12)
    save_fig(fig, "fc2_window_sweep.png", subdir="imaging/analysis")
    plt.close(fig)
    print("saved plots/imaging/analysis/fc2_window_sweep.png")


if __name__ == "__main__":
    main()
