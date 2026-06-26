"""Overview of every experiment's raw trajectory, for triage.

For each ``*_fictrac_node_fulltrack.csv`` under ``fly/data`` this writes a 2x2
panel — the three 15-minute protocol windows plus the full track — so you can scan
which recordings show the confinement/scalloping pattern (a bounded, scalloped
disc in the 15-30 min window) and decide which to analyse.

Saves plots/traces/<experiment>.png. Run from this directory:
    python overview_all.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from utils import DATA_DIR, draw_trajectory, load_trajectory, save_fig

# (label, t_start_s, t_end_s); the fourth panel is the full track.
WINDOWS = [("0-15 min", 0, 900), ("15-30 min (confinement?)", 900, 1800),
           ("30+ min", 1800, np.inf)]
STRIDE = 5  # plot every Nth sample (120 Hz -> 24 Hz) to keep figures light


def overview(folder):
    data, x, y, time_s = load_trajectory(folder)
    rel = os.path.relpath(folder, DATA_DIR)
    eid = rel.replace(os.sep, "__")

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = WINDOWS + [("full", 0, np.inf)]
    for ax, (name, lo, hi) in zip(axes.ravel(), panels):
        m = (time_s >= lo) & (time_s < hi)
        if m.sum() > 2:
            draw_trajectory(ax, x[m][::STRIDE], y[m][::STRIDE],
                            mark_start=(name == "full"))
        else:
            ax.axis("off")
        ax.set_title(name, fontsize=9)
    save_fig(fig, f"{eid}.png", title=rel, subdir="traces")
    plt.close(fig)
    return rel, time_s[-1] / 60


if __name__ == "__main__":
    csvs = sorted(glob.glob(
        os.path.join(DATA_DIR, "**", "*fictrac_node_fulltrack.csv"), recursive=True))
    print(f"{len(csvs)} experiments found")
    for c in csvs:
        try:
            rel, dur = overview(os.path.dirname(c))
            print(f"  ok   {rel}  ({dur:.0f} min)")
        except Exception as e:
            print(f"  FAIL {c}: {e}")
