"""Implements dynamics_temporal.md: fit the bounce->escape->recovery loop as a latent-state
sequence (Gaussian HMM on [speed, |turn rate|]) and measure recovery as a state process.

The HMM is fit WITHOUT laser times (10 Hz, pooled over runs, k chosen by BIC over {2,3}).
The escape state = the state with the highest turn-rate centroid. We then overlay bounces
and report: P(escape state | time-since-bounce) (the recovery curve, with its return time),
the escape-state dwell-time distribution, and the transition matrix. Run:
    python state_dynamics.py
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
DT = 0.1                       # resample to 10 Hz (behavioural states are slow)
SUBDIR = "exploratory"


def wrap(d):
    return (d + 180) % 360 - 180


def features(f):
    df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, sp, h = df.t_unix.to_numpy(), df.speed.to_numpy(), df.vrh.to_numpy()
    grid = np.arange(t[0], t[-1], DT)
    spg = np.interp(grid, t, sp)
    ch = np.interp(grid, t, np.cos(np.radians(h)))
    sh = np.interp(grid, t, np.sin(np.radians(h)))
    turn = np.abs(wrap(np.diff(np.degrees(np.arctan2(sh, ch))))) / DT
    turn = np.concatenate([[turn[0]], turn])
    return grid, np.column_stack([spg, turn])


def dwell_times(states, target):
    out, run = [], 0
    for s in states:
        if s == target:
            run += 1
        elif run:
            out.append(run * DT); run = 0
    if run:
        out.append(run * DT)
    return out


def main():
    grids, Xs, bts = [], [], []
    for f in EXPERIMENTS:
        grid, X = features(f)
        grids.append(grid); Xs.append(X); bts.append(bo.laser_on_times(f))
    Xall = np.vstack(Xs)
    lengths = [len(X) for X in Xs]
    mu, sd = Xall.mean(0), Xall.std(0)
    Z = (Xall - mu) / sd

    bics, models = {}, {}
    for k in (2, 3):
        m = GaussianHMM(n_components=k, covariance_type="full", n_iter=40, random_state=0)
        m.fit(Z, lengths)
        ll = m.score(Z, lengths)
        npar = k * k - 1 + k * 2 + k * 2 * 3      # transmat + means + full covars (rough)
        bics[k] = -2 * ll + npar * np.log(len(Z))
        models[k] = m
    k = min(bics, key=bics.get)
    m = models[k]
    means = m.means_ * sd + mu                    # back to mm/s, deg/s
    escape = int(np.argmax(means[:, 1]))          # highest turn-rate state

    tau = np.arange(-5, 20 + DT, DT)
    Prows, dwell = [], []
    raster = None
    for grid, X, bt, f in zip(grids, Xs, bts, EXPERIMENTS):
        st = m.predict((X - mu) / sd, [len(X)])
        dwell += dwell_times(st, escape)
        esc = (st == escape).astype(float)
        if raster is None:                        # keep one run for the example raster
            raster = (grid, st, bt)
        for tb in bt:
            Prows.append(np.interp(tau, grid - tb, esc, left=np.nan, right=np.nan))
    P = np.nanmean(Prows, 0)
    base = np.nanmean(P[(tau >= -5) & (tau <= -2)])
    pk = np.nanmax(P[(tau >= 0) & (tau <= 2)])
    ret = next((tt for tt, pp in zip(tau[tau > 0], P[tau > 0]) if pp <= base + 0.1 * (pk - base)), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # (a) state raster for one run (first 6 min)
    ax = axes[0, 0]
    grid, st, bt = raster
    w = (grid - grid[0]) <= 360
    ax.imshow(st[w][None, :], aspect="auto", cmap="Set1", interpolation="nearest",
              extent=[0, (grid[w][-1] - grid[0]) / 60, 0, 1])
    for tb in bt:
        rt = (tb - grid[0]) / 60
        if rt <= 6:
            ax.axvline(rt, color="k", lw=0.8)
    ax.set_yticks([]); ax.set_xlabel("time in run (min)")
    ax.set_title(f"inferred state (k={k}), escape=state {escape}; black = bounces", fontsize=9)
    # (b) P(escape) vs time since bounce
    ax = axes[0, 1]
    ax.axvspan(0, ret if np.isfinite(ret) else 2, color="#cc3311", alpha=0.08)
    ax.plot(tau, P, color="#cc3311", lw=2)
    ax.axhline(base, color="0.5", ls=":", lw=1, label="baseline")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("time from bounce (s)"); ax.set_ylabel("P(escape state)")
    ax.set_title(f"escape-state occupancy: peak {pk:.2f}, returns to baseline ~{ret:.1f}s",
                 fontsize=9); ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    # (c) dwell-time histogram
    ax = axes[1, 0]
    dwell = np.array(dwell)
    ax.hist(dwell[dwell <= 10], bins=30, color="#cc3311", edgecolor="white")
    ax.axvline(np.median(dwell), color="k", lw=2, label=f"median {np.median(dwell):.1f}s")
    ax.set_xlabel("escape-state dwell time (s)"); ax.set_ylabel("count")
    ax.set_title("how long an escape bout lasts", fontsize=9)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    # (d) transition matrix
    ax = axes[1, 1]
    im = ax.imshow(m.transmat_, cmap="magma", vmin=0, vmax=1)
    for a in range(k):
        for b in range(k):
            ax.text(b, a, f"{m.transmat_[a, b]:.2f}", ha="center", va="center",
                    color="white" if m.transmat_[a, b] < 0.6 else "black", fontsize=9)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title(f"transition matrix (escape = {escape})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045)

    cents = ", ".join(f"s{i}:({means[i,0]:.0f}mm/s,{means[i,1]:.0f}deg/s)" for i in range(k))
    fig.suptitle(f"behavioural-state dynamics (Gaussian HMM, k={k} by BIC; {cents})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "state_dynamics.png", subdir=SUBDIR)
    plt.close(fig)
    print(f"k={k} (BIC {bics}), escape state {escape}; P(escape) peak {pk:.2f} base {base:.2f} "
          f"return ~{ret:.1f}s; median dwell {np.median(dwell):.1f}s")


if __name__ == "__main__":
    main()
