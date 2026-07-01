"""Does the fly resume its pre-collision walking direction after clearing the wall?

For each bounce the reference is the fly's walking direction (net displacement) just
before the hit (-8..-2 s). We then measure the angular deviation of the walking direction
from that reference in successive windows after the bounce, to ask whether - once it has
escaped and cleared the wall - the fly drifts back to the heading it had before the
collision (deviation -> 0, it "carries on") or keeps a new randomised heading (deviation
stays ~90 deg, the collision wiped the memory).

Walking direction = net displacement (not vrh, which is decoupled from the path here),
gated at speed >= MIN_SPEED. Run from this directory:  python heading_recovery.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
PRE = (-8.0, -2.0)                                   # pre-collision reference window (s)
POST_BINS = [(2, 5), (5, 10), (10, 20), (20, 35), (35, 55)]   # windows after the bounce
CLEARED = (20, 35)                                   # "cleared the wall" window for the histogram
MIN_SPEED = 5.0
SUBDIR = "exploratory"


def wrap(d):
    return (d + 180) % 360 - 180


def walk_dir(t, x, y, sp, tb, lo, hi):
    """Net-displacement walking direction (deg) in [tb+lo, tb+hi], or NaN if barely moving."""
    m = (t >= tb + lo) & (t < tb + hi) & (sp >= MIN_SPEED)
    if m.sum() < 5:
        return np.nan
    xs, ys = x[m], y[m]
    return np.degrees(np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0]))


def gather():
    rows = []
    for f in EXPERIMENTS:
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                       df.vry.to_numpy(), df.speed.to_numpy())
        for tb in bt:
            pre = walk_dir(t, x, y, sp, tb, *PRE)
            if np.isnan(pre):
                continue
            rows.append([abs(wrap(walk_dir(t, x, y, sp, tb, lo, hi) - pre)) for lo, hi in POST_BINS])
    return np.array(rows)


def main():
    D = gather()
    centers = np.array([(lo + hi) / 2 for lo, hi in POST_BINS])
    n = np.sum(~np.isnan(D), 0)
    m = np.nanmean(D, 0)
    sem = np.nanstd(D, 0, ddof=1) / np.sqrt(np.maximum(n, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.axhline(90, color="0.6", ls="--", lw=1, label="90 deg = random (no memory)")
    ax1.axhline(0, color="0.6", ls=":", lw=1, label="0 = same as before collision")
    ax1.errorbar(centers, m, yerr=sem, fmt="o-", color="#cc3311", lw=2, capsize=3, zorder=3)
    ax1.set_xlabel("time since collision (s)")
    ax1.set_ylabel("|walking dir - pre-collision dir| (deg)")
    ax1.set_ylim(0, 180); ax1.set_yticks([0, 45, 90, 135, 180])
    ax1.set_title(f"does the fly resume its pre-collision direction? (n={D.shape[0]})", fontsize=10)
    ax1.legend(fontsize=7); ax1.spines[["top", "right"]].set_visible(False)

    ci = POST_BINS.index(CLEARED)
    cleared = D[:, ci][~np.isnan(D[:, ci])]
    ax2.hist(cleared, bins=np.linspace(0, 180, 19), color="0.5", edgecolor="white")
    ax2.axvline(np.median(cleared), color="tab:blue", lw=2.5, label=f"median {np.median(cleared):.0f} deg")
    ax2.axvline(90, color="0.6", ls="--", lw=1)
    ax2.set_xlabel("|walking dir - pre-collision dir| once cleared (deg)")
    ax2.set_ylabel("count"); ax2.set_xlim(0, 180); ax2.set_xticks([0, 45, 90, 135, 180])
    ax2.set_title(f"heading deviation after clearing the wall ({CLEARED[0]}-{CLEARED[1]} s)", fontsize=10)
    ax2.legend(fontsize=8); ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "heading_recovery.png", subdir=SUBDIR)
    plt.close(fig)
    print("median |dev| per post-window:",
          {f"{int(c)}s": round(float(np.nanmedian(D[:, i]))) for i, c in enumerate(centers)})


if __name__ == "__main__":
    main()
