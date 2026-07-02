"""Where is the wall, relative to the fly, over each presentation (wall-on to wall-off)?

The fly sits at the centre facing up (0 deg = straight ahead). For every walking sample
of every presentation we take the nearest point of the wall and plot its egocentric
position - bearing (angle) and distance (radius) - as a polar density. Two frames:
  * left  - relative to the fly's facing (proper_rotation_z / vrh, the rig heading),
  * right - relative to the fly's walking direction (smoothed velocity),
because vrh is decoupled from the path here (R~0), so the two tell different stories.

Run from this directory:  python fixation_polar.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
MIN_SPEED = 5.0        # only count samples while the fly walks (so a direction exists)
SMOOTH_N = 25          # samples (~0.2 s) to smooth the velocity for the walking direction
SUBDIR = "exploratory"


def wrap(d):
    return (d + 180) % 360 - 180


def smooth(a, n=SMOOTH_N):
    return np.convolve(a, np.ones(n) / n, mode="same")


def egocentric(folder):
    walls = bo.wall_trials(folder)
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, h, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy(),
                      df.vrh.to_numpy(), df.speed.to_numpy())
    mvdir = np.degrees(np.arctan2(np.gradient(smooth(y)), np.gradient(smooth(x))))
    Bh, Bm, D = [], [], []
    for ton, toff, cx, cy, rot, w, th in walls:
        m = (t >= ton) & (t <= toff) & (sp >= MIN_SPEED)
        if m.sum() < 5:
            continue
        fx, fy = x[m], y[m]
        # wall long-axis in the vrx/vry frame is -rot (vrcmd rotation_z is y-flipped)
        dx, dy = np.cos(np.radians(-rot)), np.sin(np.radians(-rot))   # nearest point on the wall
        proj = np.clip((fx - cx) * dx + (fy - cy) * dy, -w / 2, w / 2)
        nx, ny = cx + proj * dx, cy + proj * dy
        ang = np.degrees(np.arctan2(ny - fy, nx - fx))
        # egocentric bearing relative to facing: atan2 + h - 90 (vrh is y-flipped)
        Bh += list(wrap(ang + h[m] - 90))
        Bm += list(wrap(ang - mvdir[m]))
        D += list(np.hypot(nx - fx, ny - fy))
    return np.array(Bh), np.array(Bm), np.array(D)


def panel(ax, B, D, rmax, title):
    tb = np.linspace(-180, 180, 49)
    rb = np.linspace(0, rmax, 25)
    H, _, _ = np.histogram2d(B, D, bins=[tb, rb])
    H = H / H.sum()
    Th, Rr = np.meshgrid(np.radians(tb), rb)
    ax.pcolormesh(Th, Rr, H.T, cmap="inferno", shading="auto")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.scatter([0], [0], marker="^", s=120, color="cyan", zorder=6, edgecolor="k")
    ax.set_rlabel_position(135); ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, pad=22)


def main():
    Bh, Bm, D = [], [], []
    for f in EXPERIMENTS:
        bh, bm, d = egocentric(f)
        Bh += list(bh); Bm += list(bm); D += list(d)
    Bh, Bm, D = np.array(Bh), np.array(Bm), np.array(D)
    rmax = np.nanpercentile(D, 95)

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    panel(ax1, Bh, D, rmax, "relative to FACING (vrh)")
    panel(ax2, Bm, D, rmax, "relative to WALKING direction")
    fig.suptitle("where the wall is relative to the fly, over each presentation (wall-on to off)\n"
                 f"fly (triangle) at centre facing up (0 deg);  radius = distance to wall (mm);  "
                 f"n={len(D)} walking samples", fontsize=10, y=0.99)
    fig.subplots_adjust(top=0.80, wspace=0.25)
    save_fig(fig, "wall_egocentric.png", subdir=SUBDIR)
    plt.close(fig)
    front_face = np.mean(np.abs(Bh) < 90)
    front_walk = np.mean(np.abs(Bm) < 90)
    print(f"n={len(D)} samples | wall ahead-of-FACING {100*front_face:.0f}% | "
          f"ahead-of-WALKING {100*front_walk:.0f}% | median dist {np.median(D):.0f} mm")


if __name__ == "__main__":
    main()
