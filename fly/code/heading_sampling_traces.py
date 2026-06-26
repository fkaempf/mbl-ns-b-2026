"""Show where on the walking trace the heading samples come from.

The full fulltrack trace is drawn in black, and the points actually used for the
heading histograms - in the before/after window AND faster than MIN_SPEED - in red.
All flies are drawn at the **same spatial scale** (common span, each centred on its
own bounding box) so trace sizes are comparable; a combined figure and one
standalone figure per fly are saved. Windows/speed gate come from vrpos_heading_hist.

Run from this directory:  python heading_sampling_traces.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from utils import add_scalebar, experiment_id, load_combined, save_fig
from vrpos_heading_hist import AFTER_MIN, BEFORE_MIN, MIN_SPEED, VR_HEADING

EXPERIMENTS = ["20260624/rig1_experiment_05",
               "20260624/rig1_experiment_16",
               "20260624/rig1_experiment_20"]
SUBDIR = "heading"
STRIDE = 3   # thin the trace for a lighter LineCollection


def in_window(t, w):
    return (t >= w[0] * 60) & (t < w[1] * 60)


def load(folder):
    df = load_combined(folder, vr_heading=VR_HEADING, min_speed_mm_s=0)
    t, fx, fy = df.t.to_numpy()[::STRIDE], df.fx.to_numpy()[::STRIDE], df.fy.to_numpy()[::STRIDE]
    speed = df.speed.to_numpy()[::STRIDE]
    sampled = ((in_window(t, BEFORE_MIN) | in_window(t, AFTER_MIN)) & (speed >= MIN_SPEED))
    return experiment_id(folder), fx, fy, sampled


def draw_trace(ax, fx, fy, sampled, span):
    pts = np.column_stack([fx, fy]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    red = sampled[:-1] & sampled[1:]
    ax.add_collection(LineCollection(segs, colors=np.where(red, "red", "black"),
                                     linewidths=np.where(red, 1.1, 0.5)))
    cx, cy = (fx.min() + fx.max()) / 2, (fy.min() + fy.max()) / 2   # centre, common span
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal"); ax.axis("off")
    add_scalebar(ax)


def main():
    data = [load(f) for f in EXPERIMENTS]
    span = 1.05 * max(max(fx.max() - fx.min(), fy.max() - fy.min()) for _, fx, fy, _ in data)
    ttl = (f"heading samples in red · before {BEFORE_MIN} + after {AFTER_MIN} min, "
           f"speed > {MIN_SPEED:g} mm/s")

    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 6.5))
    for ax, (eid, fx, fy, s) in zip(np.atleast_1d(axes), data):
        draw_trace(ax, fx, fy, s, span); ax.set_title(eid)
    save_fig(fig, "_heading_sampling_traces.png", subdir=SUBDIR, title=ttl)
    plt.close(fig)

    for eid, fx, fy, s in data:
        fig, ax = plt.subplots(figsize=(5, 6.5))
        draw_trace(ax, fx, fy, s, span); ax.set_title(eid)
        save_fig(fig, f"{eid}_sampling_trace.png", subdir=SUBDIR, title=ttl)
        plt.close(fig)
        print(f"  saved {eid}")


if __name__ == "__main__":
    main()
