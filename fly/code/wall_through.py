"""Does the fly walk THROUGH the (soft-trigger) wall, or bounce back?

The barrier is a soft laser zone: the fly is only zapped, never mechanically stopped, so
it can keep walking to the far side. Using the reliable config wall geometry (centre,
orientation, width from wall_trials - NOT the noisy collision log), for each shock we
track the SIGNED PERPENDICULAR distance to the wall plane (+ = approach side, - = far
side) and the along-wall coordinate, then classify the event:
  * THROUGH  - reaches the far side (perp < 0) while within the wall's extent (|along| < w/2)
  * AROUND   - reaches the far side, but only past a wall end (|along| >= w/2)
  * BOUNCE   - never reaches the far side (turns back)

Panels: (A) perpendicular distance vs time-from-shock, coloured through/bounce, so you can
see WHEN the crossing happens; (B) the through/around/bounce split; (C) how far past the
plane the through-walkers penetrate. Run:  python wall_through.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
WIN, DT = 8.0, 0.1               # +/- s around the shock, resample step
SUBDIR = "exploratory"


def gather():
    tau = np.arange(-WIN, WIN + DT, DT)
    rows = []
    for f in EXPERIMENTS:
        bt = bo.laser_on_times(f)
        walls = bo.wall_trials(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y = df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy()
        for tb in bt:
            w = next((ww for ww in walls if ww[0] <= tb <= ww[1]), None)
            if w is None:
                continue
            _, _, cx, cy, rot, width, _ = w
            rr = np.radians(rot)
            nrm = np.array([-np.sin(rr), np.cos(rr)])           # wall normal
            alg = np.array([np.cos(rr), np.sin(rr)])            # along wall
            m = (t >= tb - WIN) & (t <= tb + WIN)
            if m.sum() < 10:
                continue
            tw = t[m] - tb
            dp = (x[m] - cx) * nrm[0] + (y[m] - cy) * nrm[1]
            da = (x[m] - cx) * alg[0] + (y[m] - cy) * alg[1]
            ref = (tw >= -5) & (tw < -0.5)                      # orient: approach side = +
            if ref.sum() < 2:
                continue
            if np.median(dp[ref]) < 0:
                dp = -dp
            dpg = np.interp(tau, tw, dp, left=np.nan, right=np.nan)
            dag = np.interp(tau, tw, da, left=np.nan, right=np.nan)
            far = dpg < 0
            within = np.abs(dag) < width / 2
            if np.any(far & within):
                cls, depth = "through", float(-np.nanmin(dpg))
                tcross = tau[np.argmax(far & within)]
            elif np.any(far):
                cls, depth, tcross = "around", np.nan, np.nan
            else:
                cls, depth, tcross = "bounce", np.nan, np.nan
            rows.append(dict(tau=tau, dp=dpg, cls=cls, depth=depth, tcross=tcross))
    return rows, tau


def main():
    rows, tau = gather()
    cls = np.array([r["cls"] for r in rows])
    n = len(rows)
    C = {"through": "#cc3311", "around": "#ee9933", "bounce": "#4477aa"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             gridspec_kw={"width_ratios": [2.1, 1, 1.3]})

    ax = axes[0]                                                # perpendicular trace
    ax.axhspan(-bo.LASER_MARGIN, bo.LASER_MARGIN, color="red", alpha=0.08, zorder=0)
    ax.axhline(0, color="k", lw=1.2, zorder=1)
    ax.axvline(0, color="0.5", ls="--", lw=1)
    for r in rows:
        if r["cls"] != "around":
            ax.plot(r["tau"], r["dp"], color=C[r["cls"]], alpha=0.18, lw=0.6, zorder=2)
    for key in ("through", "bounce"):
        M = np.nanmean(np.array([r["dp"] for r in rows if r["cls"] == key]), 0)
        ax.plot(tau, M, color=C[key], lw=3, zorder=4, label=f"{key} (mean)")
    ax.set_ylim(-25, 40); ax.set_xlim(-WIN, WIN)
    ax.set_xlabel("time from shock (s)")
    ax.set_ylabel("perpendicular distance to wall (mm)\n(+ approach side  /  - far side)")
    ax.text(-WIN + 0.3, -22, "far side (walked through)", fontsize=8, color="#cc3311")
    ax.set_title(f"crossing the wall plane (n={n} shocks)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]                                               # classification
    order = ["through", "around", "bounce"]
    pct = [100 * np.mean(cls == k) for k in order]
    ax.bar(range(3), pct, color=[C[k] for k in order])
    for i, p in enumerate(pct):
        ax.text(i, p + 1, f"{p:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(["through\nwall", "around\nend", "bounce\nback"])
    ax.set_ylabel("% of shocks"); ax.set_ylim(0, 100)
    ax.set_title("does it walk through?", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]                                               # penetration depth
    depth = np.array([r["depth"] for r in rows if r["cls"] == "through"])
    depth = depth[np.isfinite(depth)]
    ax.hist(depth, bins=np.linspace(0, 40, 21), color=C["through"], edgecolor="white")
    ax.axvline(bo.WALL_TH, color="k", ls=":", lw=1, label=f"wall ({bo.WALL_TH:g} mm)")
    ax.axvline(np.median(depth), color="0.2", lw=2, label=f"median {np.median(depth):.0f} mm")
    ax.set_xlabel("max penetration past the plane (mm)")
    ax.set_ylabel("through-events")
    ax.set_title("how far through it goes", fontsize=10)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("walking through the soft wall", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "wall_through.png", subdir=SUBDIR)
    plt.close(fig)
    print(f"n={n}: through {pct[0]:.0f}%  around {pct[1]:.0f}%  bounce {pct[2]:.0f}%; "
          f"median through-depth {np.median(depth):.1f} mm")


if __name__ == "__main__":
    main()
