"""Separate the analysis by stimulus: GREEN sun vs WHITE sun (the menotaxis landmark).

The 'good' set mixes runs whose Sun landmark is white (default) or green (sun_color in the
config). Since the sun is the orienting cue, this compares the two groups on:
  (A,B) the fly-mean-aligned heading mean-vector rose (sliding 200-sample window),
  (C) per-fly menotaxis strength (aligned-heading R) and wall-avoidance (% of moving time
      the wall is behind the walking direction), green vs white.
Run from this directory:  python stimulus_split.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

WIN, STEP, MIN_SPEED, SMOOTH_N = 200, 40, 5.0, 25
NTHETA, NR = 60, 18
SUBDIR = "exploratory"
COLORS = {"green": "#228833", "white": "#888888"}


def wrap(d):
    return (d + 180) % 360 - 180


def cmean(deg):
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    return np.degrees(np.arctan2(s, c)), np.hypot(c, s)


def smooth(a, n=SMOOTH_N):
    return np.convolve(a, np.ones(n) / n, mode="same")


def windows_mv(ang, mask):
    mask = mask.astype(float)
    cc = np.concatenate([[0.0], np.cumsum(np.cos(np.radians(ang)) * mask)])
    sc = np.concatenate([[0.0], np.cumsum(np.sin(np.radians(ang)) * mask)])
    mc = np.concatenate([[0.0], np.cumsum(mask)])
    n = len(ang)
    if n <= WIN:
        return np.array([]), np.array([])
    idx = np.arange(0, n - WIN, STEP)
    cnt = mc[idx + WIN] - mc[idx]
    ok = cnt >= WIN / 2
    c = (cc[idx + WIN] - cc[idx])[ok] / cnt[ok]
    s = (sc[idx + WIN] - sc[idx])[ok] / cnt[ok]
    return np.degrees(np.arctan2(s, c)), np.hypot(c, s)


def group_data(exps):
    A, R, perfly_R, behind = [], [], [], []
    for f in exps:
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y, h, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy(),
                          df.vrh.to_numpy(), df.speed.to_numpy())
        mov = sp >= MIN_SPEED
        if mov.sum() < WIN:
            continue
        fm, _ = cmean(h[mov])
        a, r = windows_mv(wrap(h - fm), mov)
        A.extend(a); R.extend(r)
        perfly_R.append(cmean(wrap(h[mov] - fm))[1])
        mvdir = np.degrees(np.arctan2(np.gradient(smooth(y)), np.gradient(smooth(x))))
        bk, tot = 0, 0
        for ton, toff, cx, cy, rot, w, th in bo.wall_trials(f):
            m = (t >= ton) & (t <= toff) & mov
            if m.sum() < 5:
                continue
            brg = wrap(np.degrees(np.arctan2(cy - y[m], cx - x[m])) - mvdir[m])
            bk += int(np.sum(np.abs(brg) > 90)); tot += int(m.sum())
        if tot:
            behind.append(100 * bk / tot)
    return np.array(A), np.array(R), np.array(perfly_R), np.array(behind)


def rose(fig, pos, A, R, title, col):
    ax = fig.add_subplot(2, 2, pos, projection="polar")
    if len(A):
        tb = np.linspace(-180, 180, NTHETA + 1); rb = np.linspace(0, 1, NR + 1)
        H, _, _ = np.histogram2d(A, R, bins=[tb, rb]); H = H / H.sum()
        Th, Rr = np.meshgrid(np.radians(tb), rb)
        ax.pcolormesh(Th, Rr, H.T, cmap="inferno", shading="auto")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rticks([0.5, 1.0]); ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, color=col, pad=12)


def main():
    good = bo.barrier_experiments("good")
    G = {k: [f for f in good if bo.sun_group(f) == k] for k in ("green", "white")}
    D = {k: group_data(v) for k, v in G.items()}

    fig = plt.figure(figsize=(12, 10))
    rose(fig, 1, *D["green"][:2], f"aligned heading - GREEN sun (n={len(G['green'])} runs)", COLORS["green"])
    rose(fig, 2, *D["white"][:2], f"aligned heading - WHITE sun (n={len(G['white'])} runs)", COLORS["white"])

    ax = fig.add_subplot(2, 2, 3)
    for i, k in enumerate(("green", "white")):
        v = D[k][2]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.08, .08, len(v)),
                   v, color=COLORS[k], s=40, zorder=3)
        ax.bar(i, v.mean(), color=COLORS[k], alpha=0.3, width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["green", "white"])
    ax.set_ylabel("per-fly aligned-heading R"); ax.set_ylim(0, 1)
    ax.set_title("menotaxis strength (heading concentration)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(2, 2, 4)
    for i, k in enumerate(("green", "white")):
        v = D[k][3]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-.08, .08, len(v)),
                   v, color=COLORS[k], s=40, zorder=3)
        ax.bar(i, v.mean(), color=COLORS[k], alpha=0.3, width=0.6)
    ax.axhline(50, color="0.6", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["green", "white"])
    ax.set_ylabel("% of moving time wall is behind"); ax.set_ylim(0, 100)
    ax.set_title("wall avoidance (wall behind walking dir)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("analysis split by stimulus: GREEN sun vs WHITE sun", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "stimulus_split.png", subdir=SUBDIR)
    plt.close(fig)
    for k in ("green", "white"):
        print(f"  {k}: {len(G[k])} runs, aligned-R {np.mean(D[k][2]):.2f}, "
              f"wall-behind {np.mean(D[k][3]):.0f}%")


if __name__ == "__main__":
    main()
