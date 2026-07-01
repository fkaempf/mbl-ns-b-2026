"""Exploratory: does the shock-triggered escape adapt, and is its turn biased?

Follow-up to shock_response.py (finding #1: bounces trigger a ~470 deg/s turn + speed
doubling). Per shock we measure the escape magnitude (speed surge, peak turn rate) and
the signed net turn over the first second, then ask:
  * adaptation - does the escape shrink (habituation) or grow (sensitization) across the
    session, regressed on the bounce's position in the run?
  * handedness - is the escape turn biased left/right, per fly (binomial test)?

Run from this directory:  python shock_adaptation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binomtest

import bounces as bo
from utils import experiment_id, load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
WIN, DT = 5.0, 0.1


def wrap(d):
    return (d + 180) % 360 - 180


def per_shock():
    tau = np.arange(-WIN, WIN + DT, DT)
    tt = tau[:-1]
    k0, k1 = int(np.argmin(np.abs(tau - 0))), int(np.argmin(np.abs(tau - 1)))
    rows = []
    for f in EXPERIMENTS:
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, sp, h = df.t_unix.to_numpy(), df.speed.to_numpy(), df.vrh.to_numpy()
        eid, n = experiment_id(f), len(bt)
        for i, tb in enumerate(bt):
            spg = np.interp(tau, t - tb, sp, left=np.nan, right=np.nan)
            ch = np.interp(tau, t - tb, np.cos(np.radians(h)), left=np.nan, right=np.nan)
            sh = np.interp(tau, t - tb, np.sin(np.radians(h)), left=np.nan, right=np.nan)
            hg = np.degrees(np.arctan2(sh, ch))
            turn = np.abs(wrap(np.diff(hg))) / DT
            base = np.nanmean(spg[(tau >= -3) & (tau < -1)])
            post = spg[(tau >= 0) & (tau <= 2)]
            tpost = turn[(tt >= 0) & (tt <= 1)]
            rows.append(dict(
                eid=eid, frac=i / (n - 1) if n > 1 else 0.0,
                surge=(np.nanmax(post) - base) if np.any(~np.isnan(post)) else np.nan,
                peakturn=np.nanmax(tpost) if np.any(~np.isnan(tpost)) else np.nan,
                net=wrap(hg[k1] - hg[k0])))
    return pd.DataFrame(rows)


def plot_adaptation(d, cols):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, lab in [(axes[0], "surge", "speed surge (mm/s)"),
                         (axes[1], "peakturn", "peak turn rate (deg/s)")]:
        sub = d.dropna(subset=[key])
        for eid, g in sub.groupby("eid"):
            ax.scatter(g.frac, g[key], s=20, alpha=0.7, color=cols[eid], label=eid)
        rho, p = spearmanr(sub.frac, sub[key])
        z = np.polyfit(sub.frac, sub[key], 1)
        xs = np.array([0, 1])
        ax.plot(xs, np.polyval(z, xs), color="k", lw=1.5, ls="--")
        ax.set_xlabel("position in session (0 = first bounce, 1 = last)")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab}  (Spearman rho={rho:.2f}, p={p:.2f})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    save_fig(fig, "shock_adaptation.png", subdir="exploratory")
    plt.close(fig)


def plot_handedness(d, cols):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    net = d.net.dropna().to_numpy()
    ax1.hist(net, bins=np.linspace(-180, 180, 25), color="0.6", edgecolor="white")
    ax1.axvline(0, color="red", ls="--", lw=1)
    ax1.set_xlabel("net turn over first 1 s (deg; <0 = right, >0 = left)")
    ax1.set_ylabel("count")
    ax1.set_title(f"escape turn direction (n={len(net)})", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    flies, lefts, ns, ps = [], [], [], []
    for eid, g in d.dropna(subset=["net"]).groupby("eid"):
        L = int((g.net > 0).sum()); N = len(g)
        flies.append(eid); lefts.append(L / N); ns.append(N)
        ps.append(binomtest(L, N, 0.5).pvalue)
    x = np.arange(len(flies))
    ax2.bar(x, lefts, color=[cols[e] for e in flies])
    ax2.axhline(0.5, color="red", ls="--", lw=1)
    for xi, (l, n, p) in enumerate(zip(lefts, ns, ps)):
        ax2.text(xi, l + 0.02, f"n={n}\np={p:.2f}", ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([e.split("_")[-1] for e in flies])
    ax2.set_ylim(0, 1); ax2.set_ylabel("fraction turning left")
    ax2.set_title("left/right bias per fly (binomial vs 0.5)", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "shock_handedness.png", subdir="exploratory")
    plt.close(fig)


def main():
    d = per_shock()
    print(f"{len(d)} shocks")
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cols = {e: cyc[i % len(cyc)] for i, e in enumerate(d.eid.unique())}
    plot_adaptation(d, cols)
    plot_handedness(d, cols)
    for key in ["surge", "peakturn"]:
        sub = d.dropna(subset=[key])
        rho, p = spearmanr(sub.frac, sub[key])
        print(f"  {key:9s} vs session position: rho={rho:+.2f} p={p:.3f} "
              f"(early {sub[sub.frac<0.5][key].median():.0f} vs late {sub[sub.frac>=0.5][key].median():.0f})")
    net = d.net.dropna()
    L = int((net > 0).sum()); N = len(net)
    print(f"  handedness: {L}/{N} left, pooled binomial p={binomtest(L, N, 0.5).pvalue:.3f}")


if __name__ == "__main__":
    main()
