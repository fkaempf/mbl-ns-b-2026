"""Exploratory: the shock-triggered escape response.

We have characterised the *geometry* of bounces but not their *dynamics*. This aligns
walking speed and turning rate tightly to the laser onset (the shock), pooled over
every shock in the good barrier runs, to ask: does the fly startle - accelerate and/or
turn sharply - in the second after it is zapped?

Speed comes from the fulltrack walking speed; turning from the VR heading
(proper_rotation_z), resampled on a fine grid so the per-sample heading noise averages
out. Run from this directory:  python shock_response.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
WIN = 5.0        # +/- seconds around the shock
DT = 0.1         # resample step (s)


def wrap(d):
    return (d + 180) % 360 - 180


def gather():
    tau = np.arange(-WIN, WIN + DT, DT)
    sp_rows, turn_rows = [], []
    for f in EXPERIMENTS:
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, sp, h = df.t_unix.to_numpy(), df.speed.to_numpy(), df.vrh.to_numpy()
        for tb in bt:
            sp_rows.append(np.interp(tau, t - tb, sp, left=np.nan, right=np.nan))
            # heading on the fine grid via unit vectors, then angular speed
            ch = np.interp(tau, t - tb, np.cos(np.radians(h)), left=np.nan, right=np.nan)
            sh = np.interp(tau, t - tb, np.sin(np.radians(h)), left=np.nan, right=np.nan)
            hg = np.degrees(np.arctan2(sh, ch))
            turn_rows.append(np.abs(wrap(np.diff(hg))) / DT)
    return tau, np.array(sp_rows), np.array(turn_rows)


def mean_sem(A):
    n = np.sum(~np.isnan(A), 0)
    m = np.nanmean(A, 0)
    s = np.nanstd(A, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, s


def main():
    tau, SP, TURN = gather()
    print(f"{SP.shape[0]} shocks pooled")
    msp, ssp = mean_sem(SP)
    mtu, stu = mean_sem(TURN)
    i0 = int(np.argmin(np.abs(tau)))
    far = np.nanmean(msp[(tau >= -5) & (tau <= -3)])     # baseline far from the wall (rest)
    at0 = msp[i0]                                        # speed at the shock = end of approach
    post = (tau >= 0) & (tau <= 2)
    peak, tp = np.nanmax(msp[post]), tau[post][np.nanargmax(msp[post])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True)
    # SPEED: the pre-0 rise is the approach (a shock can only fire while the fly walks
    # into the wall), so shade it; the escape is the surge AFTER the shock.
    ax1.axvspan(-1, 0, color="0.85", zorder=0)
    ax1.axvspan(0, tp + 1, color="#1f77b4", alpha=0.07, zorder=0)
    ax1.axhline(far, color="0.5", ls=":", lw=1)
    ax1.fill_between(tau, msp - ssp, msp + ssp, color="#1f77b4", alpha=0.25)
    ax1.plot(tau, msp, color="#1f77b4", lw=2)
    ax1.axvline(0, color="red", ls="--", lw=1)
    ax1.scatter([0, tp], [at0, peak], color="k", zorder=5, s=25)
    yt = ax1.get_ylim()[1]
    ax1.text(-0.5, yt, "approach\n(shock needs motion)", ha="center", va="top",
             fontsize=6.5, color="0.35")
    ax1.text(tp / 2 + 0.15, yt, "escape", ha="center", va="top", fontsize=7.5, color="#1f77b4")
    ax1.text(-3, far, " far baseline", va="bottom", fontsize=6.5, color="0.4")
    ax1.set_ylabel("speed (mm/s)")
    ax1.set_title(f"shock-triggered response (n={SP.shape[0]} shocks)\nfar baseline "
                  f"{far:.1f} -> at shock {at0:.1f} (approach) -> escape peak {peak:.1f} mm/s "
                  f"at +{tp:.1f}s", fontsize=8.5)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.fill_between(tau[:-1], mtu - stu, mtu + stu, color="#d62728", alpha=0.25)
    ax2.plot(tau[:-1], mtu, color="#d62728", lw=2)
    ax2.axvline(0, color="red", ls="--", lw=1)
    ax2.set_xlabel("time from shock (s)"); ax2.set_ylabel("turn rate (deg/s)")
    ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "shock_response.png", subdir="exploratory")
    plt.close(fig)
    print(f"SPEED far {far:.1f} -> at-shock {at0:.1f} (approach) -> escape peak {peak:.1f} "
          f"at +{tp:.1f}s; escape surge above approach = {peak - at0:.1f} mm/s")


if __name__ == "__main__":
    main()
