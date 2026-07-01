"""Heading mean-vector polar density across sliding-window sizes in TIME (500..5000 ms),
produced SEPARATELY for standing (<5 mm/s) and moving (>=5 mm/s) samples. Shows how the
heading-stability (r) distribution depends on window length and on whether the fly walks.

Run from this directory:  python heading_mv_rose_windows.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
WINDOWS_MS = list(range(500, 5001, 500))   # 500, 1000, ..., 5000 ms
CUTOFF = 5.0                               # mm/s split: standing < CUTOFF <= moving
NTHETA, NR = 60, 18
SUBDIR = "exploratory"


def gather(experiments):
    data, dts = [], []
    for f in experiments:
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        data.append((np.radians(df.vrh.to_numpy()), df.speed.to_numpy()))
        dts.append(np.median(np.diff(df.t_unix.to_numpy())))
    return data, 1.0 / np.median(dts)


def cumsums(data, mask_fn):
    out = []
    for h, sp in data:
        m = mask_fn(sp).astype(float)
        cc = np.concatenate([[0.0], np.cumsum(np.cos(h) * m)])
        sc = np.concatenate([[0.0], np.cumsum(np.sin(h) * m)])
        mc = np.concatenate([[0.0], np.cumsum(m)])
        out.append((cc, sc, mc))
    return out


def window_points(cs, W, step):
    A, R = [], []
    for cc, sc, mc in cs:
        ccd, scd = cc[W:] - cc[:-W], sc[W:] - sc[:-W]
        cnt = mc[W:] - mc[:-W]
        ok = cnt >= W / 2
        if not ok.any():
            continue
        mcv, msv = ccd[ok] / cnt[ok], scd[ok] / cnt[ok]
        A.append(np.degrees(np.arctan2(msv, mcv))[::step])
        R.append(np.hypot(mcv, msv)[::step])
    return (np.concatenate(A), np.concatenate(R)) if A else (np.array([]), np.array([]))


def make_grid(cs, wins, label, title):
    tb = np.linspace(-180, 180, NTHETA + 1)
    rb = np.linspace(0, 1, NR + 1)
    Th, Rr = np.meshgrid(np.radians(tb), rb)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.6), subplot_kw={"projection": "polar"})
    axes = axes.ravel()
    meds = {}
    for ax, (ms, W) in zip(axes, wins):
        A, R = window_points(cs, W, step=max(10, W // 4))
        if len(R):
            H, _, _ = np.histogram2d(A, R, bins=[tb, rb])
            H = H / H.sum() if H.sum() else H
            ax.pcolormesh(Th, Rr, H.T, cmap="inferno", shading="auto")
            meds[ms] = round(float(np.median(R)), 2)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_rticks([0.5, 1.0]); ax.tick_params(labelsize=6)
        ax.set_title(f"{ms} ms  (~{W} samp)\nmedian r = {meds.get(ms, float('nan')):.2f}",
                     fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, f"heading_meanvector_rose_windows_{label}.png", subdir=SUBDIR)
    plt.close(fig)
    return meds


def main():
    data, fps = gather(EXPERIMENTS)
    wins = [(ms, max(2, round(ms / 1000 * fps))) for ms in WINDOWS_MS]
    for label, title, mask in [
        ("moving", f"heading mean-vector density vs window (ms) - MOVING (>= {CUTOFF:g} mm/s, ~{fps:.0f} Hz)",
         lambda sp: sp >= CUTOFF),
        ("standing", f"heading mean-vector density vs window (ms) - STANDING (< {CUTOFF:g} mm/s, ~{fps:.0f} Hz)",
         lambda sp: sp < CUTOFF)]:
        meds = make_grid(cumsums(data, mask), wins, label, title)
        print(f"{label}: median r per window (ms) = {meds}")


if __name__ == "__main__":
    main()
