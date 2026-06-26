"""The fly's reaction to a single barrier over time, in a barrier-aligned frame.

The vrpos trajectory (united with fulltrack so tracking-glitch jumps are dropped)
is restricted to the barrier's active window - the barrier exists for
BARRIER_DURATION_S from its CREATE command - then rotated/translated into the
barrier's frame (barrier centred at the origin, long axis horizontal). The trace is
coloured by time since the barrier appeared; the aversive-laser "hurt zone" (barrier
inflated by the laser margins) is drawn in red.

Run from this directory:  python barrier_approaches.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from utils import DATA_DIR, add_scalebar, experiment_id, load_combined, save_fig

MAX_SPEED_MM_S = 50           # drop FicTrac tracking-glitch jumps
BARRIER_DURATION_S = 360      # config.yaml barrier_duration

EXPERIMENTS = [
    "20260625/rig1_experiment_53",   # the long single-barrier run (12 min)
]
SUBDIR = "barriers"
DPI = 600
LASER_MARGIN_X, LASER_MARGIN_Y = 0.05, 2.5   # config.yaml aversive-laser margins
VIEW = 80                     # half-window (units) shown around the barrier


def barriers(folder):
    """CREATE RectMaze commands -> (cx, cy, rot_deg, width, thickness, t_create)."""
    cmd = glob.glob(os.path.join(DATA_DIR, folder, "*vrcmd.csv"))[0]
    out = []
    with open(cmd) as fh:
        next(fh, None)
        for line in fh:
            p = line.split(",")
            if len(p) > 13 and p[1] == "CREATE" and p[2] == "RectMaze":
                out.append((float(p[5]), float(p[6]), float(p[10]),
                            float(p[11]), float(p[12]), float(p[0])))
    return out


def to_barrier_frame(x, y, cx, cy, rot_deg):
    """Translate to the barrier centre and rotate so the barrier is axis-aligned."""
    a = np.radians(rot_deg)
    rx, ry = x - cx, y - cy
    return rx * np.cos(a) + ry * np.sin(a), -rx * np.sin(a) + ry * np.cos(a)


def main():
    for folder in EXPERIMENTS:
        eid = experiment_id(folder)
        walls = barriers(folder)
        if not walls:
            print(f"  {eid}: no barrier"); continue
        cx, cy, rot, w, th, t_create = walls[0]                 # the single barrier
        df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=MAX_SPEED_MM_S)
        active = (df.t_unix >= t_create) & (df.t_unix <= t_create + BARRIER_DURATION_S)
        x, y, t = df.vrx[active].to_numpy(), df.vry[active].to_numpy(), df.t_unix[active].to_numpy()
        if len(x) < 5:
            print(f"  {eid}: no trajectory while barrier active"); continue
        t = (t - t[0]) / 60                                     # min since barrier appeared
        u, v = to_barrier_frame(x, y, cx, cy, rot)

        fig, ax = plt.subplots(figsize=(10, 10))
        hx, hy = w / 2 + LASER_MARGIN_X, th / 2 + LASER_MARGIN_Y
        ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, facecolor="red", alpha=0.25,
                               edgecolor="red", lw=0.8, zorder=1, label="hurt zone"))
        ax.add_patch(Rectangle((-w / 2, -th / 2), w, th, facecolor="0.35",
                               edgecolor="black", lw=0.6, zorder=2, label="barrier"))
        pts = np.column_stack([u, v]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=t[:-1], lw=0.6, zorder=3)
        ax.add_collection(lc)
        ax.scatter(u[0], v[0], color="black", s=30, zorder=5)  # start (barrier onset)

        ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW)
        ax.set_aspect("equal"); ax.axis("off")
        add_scalebar(ax)
        fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("time since barrier on (min)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.set_title(f"{eid} · reaction to barrier (barrier-aligned, while present)")
        save_fig(fig, f"{eid}_approach.png", subdir=SUBDIR, dpi=DPI)
        plt.close(fig)
        print(f"  {eid}: {len(x)} samples while barrier active ({t[-1]:.1f} min)")


if __name__ == "__main__":
    main()
