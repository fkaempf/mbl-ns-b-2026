"""Inter-trial heading evolution.

For each good run, draw the walking-heading rose (proper_rotation_z, only where
speed >= MIN_SPEED) before the first barrier and in each inter-wall interval
(wall-off_i -> wall-on_{i+1}, the free walking between two walls), for the first N
barriers. One grid per experiment is saved under plots/heading/intertrial/.

It also makes one combined figure (plots/heading/intertrial_meanvectors.png): each
interval reduced to just its mean heading **vector**, plotted as a per-experiment path
on a single polar axis (radius = interval number, marker size ~ concentration R), so
all runs' heading evolution can be compared at a glance.

Run from this directory:  python intertrial_heading.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import experiment_id, load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
N = 15                                       # first N barriers / inter-wall intervals
MIN_SPEED = 5.0                              # mm/s; only count heading while walking
SUBDIR = "heading"


def circ(deg):
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    return np.degrees(np.arctan2(s, c)), np.hypot(c, s)


def segments(folder, n=N):
    walls = bo.wall_trials(folder)                       # sorted (t_on, t_off, ...)
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, h, sp = df.t_unix.to_numpy(), df.vrh.to_numpy(), df.speed.to_numpy()
    spans = [("before barriers", t[0], walls[0][0])]
    for i in range(min(n, len(walls) - 1)):
        spans.append((f"after barrier {i + 1}", walls[i][1], walls[i + 1][0]))
    out = []
    for lab, a, b in spans:
        m = (t >= a) & (t <= b) & (sp >= MIN_SPEED)
        out.append((lab, h[m], b - a))
    return out


def plot_grid(folder):
    segs = segments(folder)
    n = len(segs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.1 * nrow),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()
    for ax, (lab, hh, dur) in zip(axes, segs):
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_yticklabels([]); ax.tick_params(labelsize=6)
        if len(hh) >= 5:
            ax.hist(np.radians(hh), bins=24, color="#88a", alpha=0.85)
            mean, R = circ(hh)
            ax.annotate("", xy=(np.radians(mean), ax.get_ylim()[1] * R), xytext=(0, 0),
                        arrowprops=dict(color="red", lw=2.5, arrowstyle="-|>"))
            ax.set_title(f"{lab}\nR={R:.2f}  n={len(hh)}  ({dur:.0f}s)", fontsize=8)
        else:
            ax.set_title(f"{lab}\n(too few moving samples)", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{experiment_id(folder)}: walking heading (>= {MIN_SPEED:.0f} mm/s) "
                 f"before barriers and in each inter-wall interval", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, f"{experiment_id(folder)}.png", subdir=f"{SUBDIR}/intertrial")
    plt.close(fig)


def plot_meanvectors(experiments):
    """Mean heading per interval as an x-y plot: x = interval, y = mean heading (deg),
    one line per run, marker size ~ concentration R."""
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("turbo")
    for j, f in enumerate(experiments):
        segs = segments(f)
        xs, ys, sz = [], [], []
        for k, (lab, hh, dur) in enumerate(segs):
            if len(hh) < 5:
                continue
            mean, R = circ(hh)
            xs.append(k); ys.append(mean); sz.append(10 + 50 * R)
        if not ys:
            continue
        ref, _ = circ(np.array(ys))                          # this run's average heading
        dy = [(y - ref + 180) % 360 - 180 for y in ys]       # deviation, centred on 0
        color = cmap(j / max(len(experiments) - 1, 1))
        ax.plot(xs, dy, "-", color=color, lw=1, alpha=0.6,
                label=experiment_id(f).split("_")[-1])
        ax.scatter(xs, dy, s=sz, color=color, zorder=3, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.axvline(0.5, color="0.7", ls=":", lw=1)
    ax.set_xlabel("interval (0 = before barriers, then each inter-wall interval)")
    ax.set_ylabel("mean heading - run average (deg)")
    ax.set_ylim(-190, 195); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title("inter-trial heading deviation from each run's average (aligned to 0; "
                 "marker ~ concentration R)", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="run", title_fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "intertrial_meanvectors.png", subdir=SUBDIR)
    plt.close(fig)


def main():
    for f in EXPERIMENTS:
        plot_grid(f)
        print(f"  grid {experiment_id(f)}")
    plot_meanvectors(EXPERIMENTS)
    print(f"done -> plots/{SUBDIR}/intertrial/ + intertrial_meanvectors.png")


if __name__ == "__main__":
    main()
