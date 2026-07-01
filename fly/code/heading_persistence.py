"""Exploratory: is the fly doing menotaxis during the barrier task?

`proper_rotation_z` is the integrated VR heading (valid even when standing). For each
good barrier run this asks:
  * preferred heading - is the walking heading concentrated (high resultant R, a
    menotaxis direction) or uniform? (speed-gated rose, R, mean),
  * persistence - the circular autocorrelation of heading vs lag, C(lag) =
    <cos(h(t+lag)-h(t))>; the 1/e decay time is how long a heading is held, and the
    long-lag plateau equals R^2 (a fixed preferred direction),
  * drift - does the preferred heading rotate from the first to the second half?

Run from this directory:  python heading_persistence.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import experiment_id, load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
GRID_DT = 0.2          # uniform resample step for the autocorrelation (s)
MAX_LAG = 60.0         # autocorrelation lag range (s)


def circ(deg):
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    return np.degrees(np.arctan2(s, c)), np.hypot(c, s)   # mean, resultant R


def session(f):
    df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, h, sp = df.t_unix.to_numpy(), df.vrh.to_numpy(), df.speed.to_numpy()
    grid = np.arange(t[0], t[-1], GRID_DT)
    hg = np.degrees(np.arctan2(np.interp(grid, t, np.sin(np.radians(h))),
                               np.interp(grid, t, np.cos(np.radians(h)))))
    lags = np.arange(0, MAX_LAG, GRID_DT * 2)
    C = np.array([np.mean(np.cos(np.radians(hg[int(L / GRID_DT):] -
                  hg[:len(hg) - int(L / GRID_DT)]))) for L in lags])
    g = sp >= 5
    mean, R = circ(h[g])
    half = t[0] + (t[-1] - t[0]) / 2
    m_early, _ = circ(h[g & (t < half)])
    m_late, _ = circ(h[g & (t >= half)])
    return dict(eid=experiment_id(f), h=h[g], R=R, mean=mean, lags=lags, C=C,
                drift=(m_early, m_late))


def main():
    S = [session(f) for f in EXPERIMENTS]
    # roses + drift
    fig = plt.figure(figsize=(2.4 * len(S), 4))
    for i, s in enumerate(S):
        ax = fig.add_subplot(1, len(S), i + 1, projection="polar")
        ax.hist(np.radians(s["h"]), bins=36, color="0.6")
        for ang, col, lab in [(s["drift"][0], "tab:green", "early"),
                              (s["drift"][1], "tab:red", "late")]:
            ax.annotate("", xy=(np.radians(ang), ax.get_ylim()[1]), xytext=(0, 0),
                        arrowprops=dict(color=col, lw=2, arrowstyle="-|>"))
        ax.set_title(f"{s['eid'].split('_')[-1]}: R={s['R']:.2f}\n"
                     f"drift {s['drift'][0]:.0f}->{s['drift'][1]:.0f} deg", fontsize=9)
        ax.set_yticklabels([])
    fig.suptitle("preferred walking heading (rose) + early(green)/late(red) mean")
    save_fig(fig, "heading_rose.png", subdir="exploratory")
    plt.close(fig)

    # autocorrelation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, s in enumerate(S):
        ax.plot(s["lags"], s["C"], color=cyc[i % len(cyc)], lw=2, label=f"{s['eid'].split('_')[-1]} (R^2={s['R']**2:.2f})")
        ax.axhline(s["R"] ** 2, color=cyc[i % len(cyc)], ls=":", lw=1)
    ax.axhline(1 / np.e, color="0.5", ls="--", lw=1, label="1/e")
    ax.set_xlabel("lag (s)"); ax.set_ylabel("heading autocorrelation  <cos Δh>")
    ax.set_title("how long the fly holds a heading (dotted = R^2 plateau)", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "heading_autocorr.png", subdir="exploratory")
    plt.close(fig)

    for s in S:
        tau_e = s["lags"][np.argmax(s["C"] < 1 / np.e)] if np.any(s["C"] < 1 / np.e) else np.nan
        print(f"  {s['eid']}: R={s['R']:.2f} mean={s['mean']:.0f}deg  "
              f"1/e persistence={tau_e:.0f}s  plateau R^2={s['R']**2:.2f}  "
              f"drift {s['drift'][0]:.0f}->{s['drift'][1]:.0f}")


if __name__ == "__main__":
    main()
