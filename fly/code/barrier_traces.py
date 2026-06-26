"""Raw VR trajectory with the barriers (RectMaze walls) overlaid, one figure per
experiment, the trace coloured by time in the experiment.

Barriers are logged in ``vrcmd.csv`` as ``CREATE RectMaze`` commands, each with a
centre (position_x, position_y), orientation (rotation_z, deg) and size
(scale = width x thickness x height). They live in the same world frame as the
``vrpos`` trajectory, so this plots the fly's path (shaded light->dark blue over
time) with every barrier drawn as a rotated grey rectangle.

``vrcmd.csv`` can't be read with pandas (the colour/serial fields contain commas),
so the command log is parsed by splitting off the leading, comma-free fields.

Run from this directory:  python barrier_traces.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from utils import DATA_DIR, add_scalebar, experiment_id, load_combined, save_fig

# The barrier analysis uses these three long, clean maze runs. Set to None to
# auto-discover every experiment that has a barrier + a vrpos trajectory instead.
EXPERIMENTS = [
    "20260625/rig1_experiment_72",
    "20260625/rig1_experiment_55",
    "20260625/rig1_experiment_46",
]
BARRIER_DATASET = "20260625"
MIN_DURATION_S = 600        # only keep runs longer than 10 min (skip aborted/short runs)
MAX_SPEED_MM_S = 50         # drop FicTrac tracking-glitch jumps (real walking is < ~40 mm/s)
LASER_MARGIN_X, LASER_MARGIN_Y = 0.05, 2.5   # config.yaml aversive-laser ("heat") margins
SUBDIR = "barriers"
STRIDE = 3          # thin the trace for a lighter LineCollection
DPI = 600           # high-resolution output

# blue colourmap truncated so even early time is a visible (not near-white) blue
BLUES = LinearSegmentedColormap.from_list("blues_t", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))


def barriers(folder):
    """Parse ``CREATE RectMaze`` commands into (cx, cy, rot_deg, width, thickness)."""
    cmd = glob.glob(os.path.join(DATA_DIR, folder, "*vrcmd.csv"))[0]
    out = []
    with open(cmd) as fh:
        next(fh, None)
        for line in fh:
            p = line.split(",")
            if len(p) > 13 and p[1] == "CREATE" and p[2] == "RectMaze":
                out.append((float(p[5]), float(p[6]), float(p[10]), float(p[11]), float(p[12])))
    return out


def vrpos_xyt(folder):
    """VR trajectory (vrpos), united with fulltrack so glitch-speed jumps are dropped."""
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=MAX_SPEED_MM_S)
    return df.vrx.to_numpy(), df.vry.to_numpy(), df.t.to_numpy()


def all_wall_experiments():
    """Every experiment in the barrier dataset that has a barrier and a vrpos."""
    out = []
    for cmd in sorted(glob.glob(os.path.join(DATA_DIR, BARRIER_DATASET, "*", "*vrcmd.csv"))):
        folder = os.path.dirname(cmd)
        if glob.glob(os.path.join(folder, "*vrpos.csv")) and barriers(folder):
            out.append(os.path.relpath(folder, DATA_DIR))
    return out


def add_barriers(ax, walls):
    """Draw each barrier as a grey wall surrounded by its red laser ("heat") zone."""
    for cx, cy, rot, w, th in walls:
        hw, hth = w + 2 * LASER_MARGIN_X, th + 2 * LASER_MARGIN_Y       # inflated heat zone
        ax.add_patch(Rectangle((cx - hw / 2, cy - hth / 2), hw, hth, angle=rot,
                               rotation_point="center", facecolor="red", alpha=0.3,
                               edgecolor="red", lw=0.6, zorder=2))
        ax.add_patch(Rectangle((cx - w / 2, cy - th / 2), w, th, angle=rot,
                               rotation_point="center", facecolor="0.35",
                               edgecolor="black", lw=0.8, zorder=3))


def draw_trace(eid, x, y, t, walls):
    fig, ax = plt.subplots(figsize=(11, 11))
    pts = np.column_stack([x[::STRIDE], y[::STRIDE]]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=BLUES, array=t[::STRIDE][:-1] / 60, lw=0.7, zorder=1)
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], color="black", s=30, zorder=4)        # start of trace
    add_barriers(ax, walls)
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    add_scalebar(ax)
    fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("time in experiment (min)")
    ax.set_title(f"{eid} · trace + {len(walls)} barriers (red = laser heat zone)")
    save_fig(fig, f"{eid}_barriers.png", subdir=SUBDIR, dpi=DPI)
    plt.close(fig)


def main():
    experiments = EXPERIMENTS or all_wall_experiments()
    for folder in experiments:
        eid = experiment_id(folder)
        x, y, t = vrpos_xyt(folder)
        if len(x) < 5 or t[-1] < MIN_DURATION_S:
            print(f"  {eid}: skip ({t[-1]/60:.1f} min, too short)"); continue
        walls = barriers(folder)
        draw_trace(eid, x, y, t, walls)
        print(f"  {eid}: {len(walls)} barriers, {t[-1]/60:.1f} min")


if __name__ == "__main__":
    main()
