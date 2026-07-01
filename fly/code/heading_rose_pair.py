"""Clean two-panel preferred-heading rose for two example runs (26 and 25).

Reuses heading_persistence.session: the walking-heading distribution (rose) with the
mean direction in the early half (green) and late half (red) of the session, to show how
steady / drifting each run's preferred heading is.

Run from this directory:  python heading_rose_pair.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import heading_persistence as hp
from utils import experiment_id, save_fig

PAIR = ["20260627/rig1_experiment_26", "20260627/rig1_experiment_25"]
EARLY, LATE = "#228833", "#cc3311"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), subplot_kw={"projection": "polar"})
    for ax, f in zip(axes, PAIR):
        s = hp.session(f)
        ax.hist(np.radians(s["h"]), bins=48, color="0.72", edgecolor="white", linewidth=0.2)
        top = ax.get_ylim()[1]
        for ang, col in [(s["drift"][0], EARLY), (s["drift"][1], LATE)]:
            ax.annotate("", xy=(np.radians(ang), top), xytext=(0, 0),
                        arrowprops=dict(color=col, lw=4, arrowstyle="-|>"))
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_yticklabels([]); ax.tick_params(labelsize=8)
        ax.set_title(f"run {experiment_id(f).split('_')[-1]}     R = {s['R']:.2f}",
                     fontsize=13, pad=18)
    fig.legend([Line2D([], [], lw=0), Line2D([], [], lw=0)],
               ["early-half mean", "late-half mean"], loc="lower center", ncol=2,
               frameon=False, fontsize=12, labelcolor=[EARLY, LATE],
               handlelength=0, handletextpad=0, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("preferred walking heading with early vs late mean direction", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_fig(fig, "heading_rose_pair.png", subdir="exploratory")
    plt.close(fig)


if __name__ == "__main__":
    main()
