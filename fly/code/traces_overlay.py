"""All experiments' walking trajectories overlaid in one figure.

Each fly's fulltrack path is centred on its start and drawn in its own colour,
labelled fly 1..N. Set WINDOW to a (start_s, end_s) tuple to overlay just the
confinement window instead of the whole recording.

Run from this directory:  python traces_overlay.py
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import add_scalebar, load_trajectory, save_fig

EXPERIMENTS = [
    "20260624/rig1_experiment_05",
    "20260624/rig1_experiment_06",
    "20260624/rig1_experiment_16",
    "20260624/rig1_experiment_18",
    "20260624/rig1_experiment_20",
]
# Each entry is a window to overlay: None = full recording, or (start_s, end_s).
WINDOWS = [None, (900, 1800)]    # full track, and the confinement window (discs)
STRIDE = 5                       # plot every Nth sample to keep the figure light
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def overlay(window):
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, folder in enumerate(EXPERIMENTS):
        _, x, y, t = load_trajectory(folder)
        m = np.ones_like(t, bool) if window is None else (t >= window[0]) & (t < window[1])
        x, y = x[m], y[m]
        x, y = x - np.median(x), y - np.median(y)      # centre each trace on its centroid
        ax.plot(x[::STRIDE], y[::STRIDE], lw=0.6, color=COLORS[i % 10],
                label=f"fly {i + 1}")
        ax.scatter(x[0], y[0], color="black", s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)   # start of each trace
    ax.set_aspect("equal"); ax.axis("off")
    add_scalebar(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              borderaxespad=0)
    suffix = "" if window is None else f"_{window[0]}_{window[1]}s"
    title = "all flies" + ("" if window is None else f" · {window[0]}-{window[1]}s")
    save_fig(fig, f"_all_flies_overlay{suffix}.png", subdir="traces", title=title)


def juxtapose(window):
    """Same traces, but each shifted into its own slot so they sit side by side
    without overlapping (tight)."""
    traces = []
    for folder in EXPERIMENTS:
        _, x, y, t = load_trajectory(folder)
        m = np.ones_like(t, bool) if window is None else (t >= window[0]) & (t < window[1])
        x, y = x[m], y[m]
        traces.append((x - x.min(), y - y.min()))      # shift each bbox to the origin
    gap = 0.12 * max(x.max() for x, _ in traces)
    max_h = max(y.max() for _, y in traces)
    total_w = sum(x.max() for x, _ in traces) + gap * (len(traces) - 1)

    fig, ax = plt.subplots(figsize=(14, max(3.0, 14 * max_h / total_w)))
    cur = 0.0
    for i, (x, y) in enumerate(traces):
        ax.plot(x[::STRIDE] + cur, y[::STRIDE], lw=0.6, color=COLORS[i % 10])
        ax.scatter(x[0] + cur, y[0], color="black", s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)
        ax.text(cur + x.max() / 2, -0.04 * max_h, f"fly {i + 1}", ha="center",
                va="top", color=COLORS[i % 10])
        cur += x.max() + gap
    ax.set_aspect("equal"); ax.axis("off")
    add_scalebar(ax)
    suffix = "" if window is None else f"_{window[0]}_{window[1]}s"
    title = "all flies (juxtaposed)" + ("" if window is None else f" · {window[0]}-{window[1]}s")
    save_fig(fig, f"_all_flies_juxtaposed{suffix}.png", subdir="traces", title=title)


def main():
    for window in WINDOWS:
        overlay(window)
        juxtapose(window)


if __name__ == "__main__":
    main()
