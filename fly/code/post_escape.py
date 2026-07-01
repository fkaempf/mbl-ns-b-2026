"""'After the primary shock response', defined PER BOUNCE by when speed returns to baseline.

For each shock we find t_recover: the first time after the escape peak that walking speed
comes back to the pre-shock baseline (median over -5..-3 s) and holds there ~1 s. That is
this bounce's end-of-primary-response. We then (A) show the distribution of t_recover (how
long the speed response lasts), and (B,C) RE-ALIGN every bounce to its own t_recover and
look at speed and heading deviation afterwards - i.e. once the speed is back to baseline,
is the fly's heading settled, or has the escape left it pointing somewhere new?

Run from this directory:  python post_escape.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
PRE, POST, DT = 8.0, 25.0, 0.1
WALKWIN = 0.75            # +/- s for the windowed walking direction
SUBDIR = "exploratory"


def wrap(d):
    return (d + 180) % 360 - 180


def mean_sem(A):
    n = np.sum(~np.isnan(A), 0)
    return np.nanmean(A, 0), np.nanstd(A, 0, ddof=1) / np.sqrt(np.maximum(n, 1))


def t_recover(tau, sp, base):
    pk = (tau >= 0) & (tau <= 3)
    if not np.any(np.isfinite(sp[pk])):
        return np.nan
    k0 = int(np.nanargmax(np.where(pk, sp, -np.inf)))
    hold = int(round(1.0 / DT))
    thr = base * 1.15
    for k in range(k0, len(sp) - hold):
        seg = sp[k:k + hold]
        if np.all(np.isfinite(seg)) and np.all(seg <= thr):
            return tau[k]
    return np.nan


def gather():
    tau = np.arange(-PRE, POST + DT, DT)
    w = int(round(WALKWIN / DT))
    i8, i2 = int(np.argmin(np.abs(tau + 8))), int(np.argmin(np.abs(tau + 2)))
    rows = []
    for f in EXPERIMENTS:
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                       df.vry.to_numpy(), df.speed.to_numpy())
        for i, tb in enumerate(bt):
            nxt = (bt[i + 1] - tb) if i + 1 < len(bt) else np.inf
            sprow = np.interp(tau, t - tb, sp, left=np.nan, right=np.nan)
            xg = np.interp(tau, t - tb, x, left=np.nan, right=np.nan)
            yg = np.interp(tau, t - tb, y, left=np.nan, right=np.nan)
            sprow[tau >= nxt] = np.nan
            base = np.nanmedian(sprow[(tau >= -5) & (tau <= -3)])
            if not (np.isfinite(base) and np.isfinite(xg[i8]) and np.isfinite(xg[i2])):
                continue
            pre = np.degrees(np.arctan2(yg[i2] - yg[i8], xg[i2] - xg[i8]))
            dev = np.full(len(tau), np.nan)
            for k in range(w, len(tau) - w):
                dx, dy = xg[k + w] - xg[k - w], yg[k + w] - yg[k - w]
                if np.isfinite(dx) and sprow[k] >= 5 and tau[k] < nxt:   # only while moving (>=5 mm/s)
                    dev[k] = abs(wrap(np.degrees(np.arctan2(dy, dx)) - pre))
            rows.append(dict(sp=sprow, dev=dev, tr=t_recover(tau, sprow, base), base=base))
    return tau, rows


def main():
    tau, rows = gather()
    trs = np.array([r["tr"] for r in rows])
    rec = np.isfinite(trs)
    taup = np.arange(-3, 12 + DT, DT)
    SPp = np.array([np.interp(taup, tau - r["tr"], r["sp"], left=np.nan, right=np.nan)
                    for r in rows if np.isfinite(r["tr"])])
    DVp = np.array([np.interp(taup, tau - r["tr"], r["dev"], left=np.nan, right=np.nan)
                    for r in rows if np.isfinite(r["tr"])])
    msp, ssp = mean_sem(SPp)
    mdv, sdv = mean_sem(DVp)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.hist(trs[rec], bins=np.linspace(0, POST, 26), color="#1f77b4", edgecolor="white")
    ax.axvline(np.median(trs[rec]), color="k", lw=2, label=f"median {np.median(trs[rec]):.1f}s")
    ax.set_xlabel("t_recover: speed back to baseline (s from shock)")
    ax.set_ylabel("bounces")
    ax.set_title(f"primary-response duration (per bounce)\n{100*rec.mean():.0f}% recover within {POST:.0f}s",
                 fontsize=10)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axvspan(-3, 0, color="0.85", zorder=0)
    ax.fill_between(taup, msp - ssp, msp + ssp, color="#1f77b4", alpha=0.25)
    ax.plot(taup, msp, color="#1f77b4", lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("time from speed-recovery (s)"); ax.set_ylabel("speed (mm/s)")
    ax.set_title("speed, re-aligned to its own recovery", fontsize=10)
    ax.text(-1.5, ax.get_ylim()[1] * 0.9, "primary\nresponse", ha="center", fontsize=7, color="0.4")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ax.axhline(0, color="0.5", ls=":", lw=1, label="0 = pre-collision heading")
    ax.axhline(90, color="0.6", ls="--", lw=1, label="90 = random")
    ax.fill_between(taup, mdv - sdv, mdv + sdv, color="#cc3311", alpha=0.25)
    ax.plot(taup, mdv, color="#cc3311", lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_ylim(0, 180); ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("time from speed-recovery (s)")
    ax.set_ylabel("|walking dir - pre-collision dir| (deg)")
    ax.set_title("heading once the speed has recovered", fontsize=10)
    ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("after the primary shock response (anchored on speed return-to-baseline)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "post_escape.png", subdir=SUBDIR)
    plt.close(fig)
    print(f"recover {100*rec.mean():.0f}%, median t_recover {np.median(trs[rec]):.1f}s; "
          f"heading dev at recovery {np.nanmean(DVp[:, np.argmin(np.abs(taup))]):.0f} deg")


if __name__ == "__main__":
    main()
