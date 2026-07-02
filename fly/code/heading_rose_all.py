"""Preferred-heading mean-vector rose for every good run (a grid, one rose per run).

For each good run: the walking-heading distribution (vrh, moving samples) as a rose, with
the early-half (green) and late-half (red) mean heading VECTORS drawn as arrows whose
length is each half's resultant R. This is the fly's own goal heading and its drift -
distinct from wall_egocentric (where the wall sits). Run:  python heading_rose_all.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bounces as bo
import heading_persistence as hp
from utils import experiment_id, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
EARLY, LATE = "#228833", "#cc3311"
SUBDIR = "exploratory"


def main():
    n = len(EXPERIMENTS)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, EXPERIMENTS):
        s = hp.session(f)
        ax.hist(np.radians(s["h"]), bins=40, color="0.74", edgecolor="white", linewidth=0.15)
        top = ax.get_ylim()[1]
        for half, col in [(s["drift"][0], EARLY), (s["drift"][1], LATE)]:
            ax.annotate("", xy=(np.radians(half), top), xytext=(0, 0),
                        arrowprops=dict(color=col, lw=3, arrowstyle="-|>"))
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_yticklabels([]); ax.tick_params(labelsize=6)
        ax.set_title(f"{experiment_id(f).split('_')[-1]}   R={s['R']:.2f}", fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend([Line2D([], [], lw=0), Line2D([], [], lw=0)],
               ["early-half mean", "late-half mean"], loc="lower center", ncol=2,
               frameon=False, fontsize=11, labelcolor=[EARLY, LATE],
               handlelength=0, handletextpad=0)
    fig.suptitle("preferred walking-heading mean-vector rose, all good runs "
                 "(arrow = early/late mean heading, length = R)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_fig(fig, "heading_rose_all.png", subdir=SUBDIR)
    plt.close(fig)
    print(f"done: {n} runs")


if __name__ == "__main__":
    main()
