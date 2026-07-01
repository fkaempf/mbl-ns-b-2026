"""Sliding-window (WIN-sample) mean-vector polar-density heatmaps, generalized across flies.

Same method as heading_meanvector_rose (a sliding window, each window's angles averaged as
unit vectors -> angle = window mean, r = window concentration). Three panels:
  direction to the wall RELATIVE TO THE FRONT of the fly (0 = ahead), split into
    - STANDING windows (< MOVE_CUTOFF mm/s) and
    - RUNNING windows (>= MOVE_CUTOFF mm/s);
  the heading ALIGNED to each fly's own circular-mean heading (0 = the fly's preferred
    direction, so the peak sits at 0 and the concentration generalizes across flies).

Window size is the 1st CLI arg (default 200).  Run:  python meanvector_roses.py [WIN]
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 200
STEP = 40
MOVE_CUTOFF = 5.0
NTHETA, NR = 60, 18
SUF = "" if WIN == 200 else f"_{WIN}"
SUBDIR = "exploratory"


def wrap(d):
    return (d + 180) % 360 - 180


def cmean(deg):
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    return np.degrees(np.arctan2(s, c)), np.hypot(c, s)


def windows_mv(ang_deg, mask):
    """Sliding WIN-sample mean vectors of an angle series, over the masked samples only."""
    mask = mask.astype(float)
    cos = np.cos(np.radians(ang_deg)) * mask
    sin = np.sin(np.radians(ang_deg)) * mask
    cc = np.concatenate([[0.0], np.cumsum(cos)])
    sc = np.concatenate([[0.0], np.cumsum(sin)])
    mc = np.concatenate([[0.0], np.cumsum(mask)])
    n = len(ang_deg)
    if n <= WIN:
        return np.array([]), np.array([])
    idx = np.arange(0, n - WIN, STEP)
    cnt = mc[idx + WIN] - mc[idx]
    ok = cnt >= WIN / 2
    c = (cc[idx + WIN] - cc[idx])[ok] / cnt[ok]
    s = (sc[idx + WIN] - sc[idx])[ok] / cnt[ok]
    return np.degrees(np.arctan2(s, c)), np.hypot(c, s)


def gather():
    out = {"wall_stand": ([], []), "wall_run": ([], []), "head": ([], [])}
    for f in EXPERIMENTS:
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y, h, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy(),
                          df.vrh.to_numpy(), df.speed.to_numpy())
        run, stand = sp >= MOVE_CUTOFF, sp < MOVE_CUTOFF
        if run.sum() < WIN:
            continue
        fm, _ = cmean(h[run])                                    # fly's own mean heading
        a, r = windows_mv(wrap(h - fm), run)                     # heading aligned (running)
        out["head"][0].extend(a); out["head"][1].extend(r)
        for ton, toff, cx, cy, rot, w, th in bo.wall_trials(f):
            m = (t >= ton) & (t <= toff)
            if m.sum() <= WIN:
                continue
            brg = wrap(np.degrees(np.arctan2(cy - y[m], cx - x[m])) - h[m])
            for key, msk in (("wall_run", run[m]), ("wall_stand", stand[m])):
                a, r = windows_mv(brg, msk)
                out[key][0].extend(a); out[key][1].extend(r)
    return {k: (np.array(A), np.array(R)) for k, (A, R) in out.items()}


def panel(fig, pos, A, R, title):
    ax = fig.add_subplot(1, 3, pos, projection="polar")
    if len(A):
        tb = np.linspace(-180, 180, NTHETA + 1)
        rb = np.linspace(0, 1, NR + 1)
        H, _, _ = np.histogram2d(A, R, bins=[tb, rb])
        H = H / H.sum() if H.sum() else H
        Th, Rr = np.meshgrid(np.radians(tb), rb)
        ax.pcolormesh(Th, Rr, H.T, cmap="inferno", shading="auto")
        md = np.median(R)
    else:
        md = float("nan")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rticks([0.5, 1.0]); ax.tick_params(labelsize=7)
    ax.set_title(f"{title}\nn={len(A)}, median r={md:.2f}", fontsize=9, pad=12)


def main():
    d = gather()
    fig = plt.figure(figsize=(17, 5.8))
    panel(fig, 1, *d["wall_stand"], "wall vs front of fly - STANDING (<5 mm/s)")
    panel(fig, 2, *d["wall_run"], "wall vs front of fly - RUNNING (>=5 mm/s)")
    panel(fig, 3, *d["head"], "heading ALIGNED to each fly's mean\n(0 = preferred heading)")
    fig.suptitle(f"sliding {WIN}-sample mean-vector heatmaps generalized across flies "
                 f"(0 = straight ahead / preferred heading)", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, f"meanvector_roses{SUF}.png", subdir=SUBDIR)
    plt.close(fig)
    ha = d["head"][0]
    pk = np.degrees(np.arctan2(np.sin(np.radians(ha)).mean(), np.cos(np.radians(ha)).mean()))
    print(f"WIN={WIN}: wall-stand n={len(d['wall_stand'][0])} r={np.median(d['wall_stand'][1]):.2f}; "
          f"wall-run n={len(d['wall_run'][0])} r={np.median(d['wall_run'][1]):.2f}; "
          f"heading-aligned n={len(ha)} r={np.median(d['head'][1]):.2f}, peak {pk:+.1f} deg (should be ~0)")


if __name__ == "__main__":
    main()
