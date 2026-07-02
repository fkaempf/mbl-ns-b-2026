"""Replay every good run's VR trajectory as a ~15 s video - a grid, one panel per run.

All runs play on a single common time-compression (real time / C, where C makes the
longest run fill the video), so every fly walks at its own correct relative speed and
shorter runs finish earlier (their panel freezes). Each panel is in that run's own VR
coordinates, with its barrier walls drawn (and their laser zone), appearing and
vanishing as they do in the session.

Run from this directory:  python replay_video.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

import bounces as bo
from utils import PLOTS_DIR, experiment_id, load_combined

EXPERIMENTS = bo.barrier_experiments("good")
DURATION, FPS = 15, 25
NF = DURATION * FPS
NSAMP = 2500                 # resample each run to this many points (uniform in real time)
LASER_MARGIN = 2.5


def load():
    out = []
    for f in EXPERIMENTS:
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y = df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy()
        tr = t - t[0]
        grid = np.linspace(0, tr[-1], NSAMP)
        xs = np.interp(grid, tr, x)
        ys = np.interp(grid, tr, y)
        walls = [(won - t[0], woff - t[0], cx, cy, rot, w, th)
                 for won, woff, cx, cy, rot, w, th in bo.wall_trials(f)]
        out.append(dict(eid=experiment_id(f).split("_")[-1], gx=xs, gy=ys, gt=grid,
                        T=tr[-1], walls=walls))
    return out


def main():
    data = load()
    Tmax = max(d["T"] for d in data)
    n = len(data)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("turbo")
    arts = []
    for j, (ax, d) in enumerate(zip(axes, data)):
        ax.set_aspect("equal"); ax.axis("off")
        pad = 0.05 * (d["gx"].max() - d["gx"].min() + 1)
        ax.set_xlim(d["gx"].min() - pad, d["gx"].max() + pad)
        ax.set_ylim(d["gy"].min() - pad, d["gy"].max() + pad)
        col = cmap(j / max(n - 1, 1))
        line, = ax.plot([], [], color=col, lw=0.7, alpha=0.85)
        dot, = ax.plot([], [], "o", color="k", ms=4)
        wps = []
        for won, woff, cx, cy, rot, w, th in d["walls"]:
            ang = -rot                   # vrcmd rotation_z is y-flipped vs the vrx/vry frame
            heat = Rectangle((cx - w / 2, cy - th / 2 - LASER_MARGIN), w, th + 2 * LASER_MARGIN,
                             angle=ang, rotation_point="center", facecolor="red", alpha=0.18,
                             edgecolor="none", visible=False, zorder=1)
            wall = Rectangle((cx - w / 2, cy - th / 2), w, th, angle=ang, rotation_point="center",
                             facecolor="0.35", edgecolor="k", lw=0.4, visible=False, zorder=2)
            ax.add_patch(heat); ax.add_patch(wall)
            wps.append((won, woff, heat, wall))
        ax.set_title(d["eid"], fontsize=8, color=col)
        arts.append((d, line, dot, wps))
    for ax in axes[n:]:
        ax.axis("off")
    sup = fig.suptitle("", fontsize=12)

    def update(fr):
        tau = fr / NF * Tmax
        changed = [sup]
        for d, line, dot, wps in arts:
            done = tau >= d["T"]
            idx = NSAMP - 1 if done else int(tau / d["T"] * (NSAMP - 1))
            line.set_data(d["gx"][:idx + 1], d["gy"][:idx + 1])
            dot.set_data([d["gx"][idx]], [d["gy"][idx]])
            dot.set_alpha(0.25 if done else 1.0)
            for won, woff, heat, wall in wps:
                vis = (won <= tau <= woff) and not done
                heat.set_visible(vis); wall.set_visible(vis)
            changed += [line, dot]
        sup.set_text(f"all {n} runs replayed at true relative speed  -  "
                     f"{tau / 60:.0f} / {Tmax / 60:.0f} min")
        return changed

    ani = animation.FuncAnimation(fig, update, frames=NF, interval=1000 / FPS, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(PLOTS_DIR, "replay.mp4")
    try:
        ani.save(out, fps=FPS, dpi=110)
    except Exception as e:
        out = out[:-4] + ".gif"
        ani.save(out, fps=FPS, writer="pillow")
        print(f"(ffmpeg failed: {type(e).__name__}) -> GIF")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
