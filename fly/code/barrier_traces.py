"""Raw VR trajectory with the barriers (RectMaze walls) overlaid, one figure per
experiment, saved as a **vector PDF** (infinite zoom).

Barriers are logged in ``vrcmd.csv`` as ``CREATE RectMaze`` commands, each with a
centre (position_x, position_y), orientation (rotation_z, deg), size (scale = width x
thickness x height) and a creation time. They live in the same world frame as the
``vrpos`` trajectory, so this plots the fly's path (shaded light->dark blue over time)
with every barrier drawn as a rotated grey rectangle inside its red laser zone.

On top of that:
  * the trace turns **red while the fly is being shocked** (laser on, from the
    ``light_sugar`` command log),
  * each barrier is annotated with the fly's **heading at the moment the wall was
    created** (a green arrow in that direction + the angle), since the barrier is
    spawned a fixed distance in front of the fly along its heading.

``vrcmd.csv`` can't be read with pandas (the colour/serial fields contain commas), so
the command log is parsed by splitting off the leading, comma-free fields.

Run from this directory:  python barrier_traces.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from utils import DATA_DIR, PLOTS_DIR, add_scalebar, experiment_label, load_combined

# None = plot every experiment in BARRIER_DATASET that has a barrier + a vrpos.
EXPERIMENTS = None
# The clean runs used for the quantitative barrier/bounce analysis (marked in green).
# Keyed by date_experiment so it never collides across datasets.
GOOD = {"20260625_rig1_experiment_72", "20260625_rig1_experiment_55",
        "20260625_rig1_experiment_46",
        "20260625_rig1_experiment_60",
        "20260627_rig1_experiment_29", "20260627_rig1_experiment_26",
        "20260627_rig1_experiment_25", "20260627_rig1_experiment_23",
        "20260627_rig1_experiment_21", "20260626_rig1_experiment_30",
        "20260629_rig1_experiment_01", "20260629_rig1_experiment_02",
        "20260629_rig1_experiment_08", "20260629_rig1_experiment_15",
        "20260629_rig1_experiment_16",
        "20260630_rig1_experiment_05", "20260630_rig1_experiment_06"}
BARRIER_DATASETS = ["20260625", "20260626", "20260627", "20260629", "20260630"]
MAX_SPEED_MM_S = 50         # drop FicTrac tracking-glitch jumps (real walking is < ~40 mm/s)
LASER_MARGIN_X, LASER_MARGIN_Y = 0.05, 2.5   # config.yaml aversive-laser ("heat") margins
HEADING_OFFSET = 20         # mm to nudge the heading-angle label off the barrier
HEADING_ARROW = 250         # mm length of the heading-at-creation arrow
# (sub-folder, show angle labels, show heading arrows) - one folder per variant
VARIANTS = [("angles", True, False), ("arrows", False, True), ("plain", False, False)]
SUBDIR = "barriers"
STRIDE = 3          # thin the trace for a lighter (still vector) LineCollection

# blue colourmap truncated so even early time is a visible (not near-white) blue
BLUES = LinearSegmentedColormap.from_list("blues_t", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))


def barriers(folder):
    """``CREATE RectMaze`` -> (cx, cy, rot_deg, width, thickness, t_create)."""
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


def laser_intervals(folder):
    """Shock intervals (unix s): laser ramps to 255 (on) until it ramps to 0 (off)."""
    ls = glob.glob(os.path.join(DATA_DIR, folder, "*light_sugar*commands.csv"))
    if not ls:
        return []
    d = pd.read_csv(ls[0])
    c = "laser_exponential_set_end_level"
    ev = d.loc[d[c].notna(), ["timestamp", c]]                   # any level-set row
    ev = ev.assign(t=ev.timestamp.astype(float) / 1e9).sort_values("t")
    out, on = [], None
    for t, lv in zip(ev.t.to_numpy(), ev[c].to_numpy()):
        if lv > 0 and on is None:                               # ramp to any positive level = on
            on = t
        elif lv == 0 and on is not None:
            out.append((on, t)); on = None
    if on is not None:
        out.append((on, on + 0.5))
    return out


def all_wall_experiments():
    """Every experiment across BARRIER_DATASETS that has a barrier and a vrpos."""
    out = []
    cmds = sorted(c for ds in BARRIER_DATASETS
                  for c in glob.glob(os.path.join(DATA_DIR, ds, "*", "*vrcmd.csv")))
    for cmd in cmds:
        folder = os.path.dirname(cmd)
        if glob.glob(os.path.join(folder, "*vrpos.csv")) and barriers(folder):
            out.append(os.path.relpath(folder, DATA_DIR))
    return out


def shocked_mask(t_unix, intervals):
    m = np.zeros(len(t_unix), bool)
    for a, b in intervals:
        m |= (t_unix >= a) & (t_unix <= b)
    return m


def fly_state_at(df, times):
    """Fly VR position (interp) and heading (nearest sample) at each ``time``."""
    t = df.t_unix.to_numpy()
    fx = np.interp(times, t, df.vrx.to_numpy())
    fy = np.interp(times, t, df.vry.to_numpy())
    idx = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    return fx, fy, df.vrh.to_numpy()[idx]


def add_barriers(ax, walls):
    """Each barrier as a grey wall inside its red laser ("heat") zone."""
    for cx, cy, rot, w, th, _ in walls:
        hw, hth = w + 2 * LASER_MARGIN_X, th + 2 * LASER_MARGIN_Y
        ax.add_patch(Rectangle((cx - hw / 2, cy - hth / 2), hw, hth, angle=rot,
                               rotation_point="center", facecolor="red", alpha=0.3,
                               edgecolor="red", lw=0.6, zorder=2))
        ax.add_patch(Rectangle((cx - w / 2, cy - th / 2), w, th, angle=rot,
                               rotation_point="center", facecolor="0.35",
                               edgecolor="black", lw=0.8, zorder=3))


def add_angle_labels(ax, walls, h):
    """The fly's heading angle (deg) at each barrier's creation, next to the barrier."""
    for (cx, cy, rot, w, th, _), hh in zip(walls, h):
        ax.text(cx, cy + 0.6 * th + HEADING_OFFSET, f"{hh:.0f}°", color="green",
                fontsize=6, ha="center", va="bottom", zorder=7)


def add_heading_arrows(ax, walls, h):
    """Arrow along the fly's recorded heading at each barrier's creation. Because the
    wall is spawned with rotation_z = that heading, the arrow runs *along* the wall."""
    for (cx, cy, rot, w, th, _), hh in zip(walls, h):
        d = np.array([np.cos(np.radians(hh)), np.sin(np.radians(hh))])
        tip = np.array([cx, cy]) + d * HEADING_ARROW
        ax.annotate("", xy=tip, xytext=(cx, cy), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="green", lw=1.0, alpha=0.9))


def draw_trace(name, df, walls, intervals, h, good, variant, angles, arrows):
    x, y = df.vrx.to_numpy(), df.vry.to_numpy()
    t, tu = df.t.to_numpy(), df.t_unix.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 11))
    pts = np.column_stack([x[::STRIDE], y[::STRIDE]]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=BLUES, array=t[::STRIDE][:-1] / 60, lw=0.7, zorder=1)
    ax.add_collection(lc)
    sm = shocked_mask(tu, intervals)[::STRIDE]
    seg_shock = sm[:-1] | sm[1:]
    if seg_shock.any():                                          # red while shocked
        ax.add_collection(LineCollection(segs[seg_shock], colors="red", lw=1.8, zorder=4))
    if intervals:                                                # red dot per shock onset
        on = [a for a, _ in intervals]
        ax.scatter(np.interp(on, tu, x), np.interp(on, tu, y), color="red", s=10,
                   zorder=6, linewidths=0)
    ax.scatter(x[0], y[0], color="black", s=30, zorder=5)        # start of trace
    add_barriers(ax, walls)
    if angles:
        add_angle_labels(ax, walls, h)
    if arrows:
        add_heading_arrows(ax, walls, h)
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    add_scalebar(ax)
    fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("time in experiment (min)")
    tag = "★ GOOD · " if good else ""
    extra = " · green = heading@create" if (angles or arrows) else ""
    ax.set_title(f"{tag}{name} · {len(walls)} barriers · red = shock{extra}",
                 color="green" if good else "black")
    if good:
        fig.add_artist(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=fig.transFigure,
                                     fill=False, edgecolor="green", lw=5, zorder=10))
    out = os.path.join(PLOTS_DIR, SUBDIR, variant)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{'GOOD_' if good else ''}{name}_barriers.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    experiments = EXPERIMENTS or all_wall_experiments()
    for folder in experiments:
        name = experiment_label(folder).replace("/", "_")        # e.g. 20260625_rig1_experiment_72
        try:
            df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=MAX_SPEED_MM_S)
        except Exception as e:                                   # missing/short logs
            print(f"  {name}: skip ({type(e).__name__})"); continue
        if len(df) < 5:
            print(f"  {name}: skip (empty)"); continue
        walls = barriers(folder)
        intervals = laser_intervals(folder)
        _, _, h = fly_state_at(df, [w[5] for w in walls])
        good = name in GOOD
        for variant, angles, arrows in VARIANTS:
            draw_trace(name, df, walls, intervals, h, good, variant, angles, arrows)
        print(f"  {name}: {len(walls)} barriers, {len(intervals)} shocks, "
              f"{df.t.to_numpy()[-1] / 60:.1f} min{' [GOOD]' if good else ''}")


if __name__ == "__main__":
    main()
