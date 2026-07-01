"""Maimon-style polar density of the fly's heading from a sliding mean-vector window,
split by whether the fly was standing or moving in that window.

The heading (proper_rotation_z) is summarised in a sliding WIN-sample window with the
paper's mean-vector method: each window's heading samples are averaged as unit vectors ->
angle = mean heading, r = resultant (r~1 a steady heading, r~0 a turning window). Each
window is labelled STANDING or MOVING by its median speed vs MOVE_CUTOFF, and the two
polar densities are shown side by side (finely binned). Pooled over the good runs.

Run from this directory:  python heading_mv_rose.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # samples in the sliding mean-vector window
SUF = "" if WIN == 200 else f"_{WIN}"
STEP = 20          # slide step (samples) - finer
MOVE_CUTOFF = 5.0  # mm/s; a window is MOVING if its median speed >= this, else STANDING
NTHETA, NR = 72, 20    # fine polar bins (5 deg x 0.05 r)
SUBDIR = "exploratory"


def windows():
    A, R, moving = [], [], []
    for f in EXPERIMENTS:
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        h, sp = df.vrh.to_numpy(), df.speed.to_numpy()
        cos, sin = np.cos(np.radians(h)), np.sin(np.radians(h))
        for i in range(0, len(h) - WIN, STEP):
            c, s = cos[i:i + WIN].mean(), sin[i:i + WIN].mean()
            A.append(np.degrees(np.arctan2(s, c)))
            R.append(np.hypot(c, s))
            moving.append(np.median(sp[i:i + WIN]) >= MOVE_CUTOFF)
    return np.array(A), np.array(R), np.array(moving)


def panel(fig, pos, A, R, title):
    ax = fig.add_subplot(1, 2, pos, projection="polar")
    tb = np.linspace(-180, 180, NTHETA + 1)
    rb = np.linspace(0, 1, NR + 1)
    H, _, _ = np.histogram2d(A, R, bins=[tb, rb])
    H = H / H.sum() if H.sum() else H
    Th, Rr = np.meshgrid(np.radians(tb), rb)
    pcm = ax.pcolormesh(Th, Rr, H.T, cmap="inferno", shading="auto")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rticks([0.5, 1.0]); ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10, pad=16)
    fig.colorbar(pcm, ax=ax, fraction=0.045, pad=0.09, label="prob. density")


def main():
    A, R, mov = windows()
    fig = plt.figure(figsize=(13, 6.2))
    panel(fig, 1, A[~mov], R[~mov],
          f"STANDING  (median speed < {MOVE_CUTOFF:g} mm/s)\n"
          f"n={int((~mov).sum())},  median r={np.median(R[~mov]):.2f}")
    panel(fig, 2, A[mov], R[mov],
          f"MOVING  (median speed >= {MOVE_CUTOFF:g} mm/s)\n"
          f"n={int(mov.sum())},  median r={np.median(R[mov]):.2f}")
    fig.suptitle(f"heading mean-vector polar density  (sliding {WIN}-sample window, "
                 f"step {STEP}; moving cutoff = {MOVE_CUTOFF:g} mm/s)", fontsize=11, y=0.98)
    save_fig(fig, f"heading_meanvector_rose{SUF}.png", subdir=SUBDIR)
    plt.close(fig)
    print(f"standing n={int((~mov).sum())} (median R={np.median(R[~mov]):.2f}) | "
          f"moving n={int(mov.sum())} (median R={np.median(R[mov]):.2f}) | cutoff={MOVE_CUTOFF} mm/s")


if __name__ == "__main__":
    main()
