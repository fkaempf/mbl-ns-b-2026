"""Q2: how do FC2 phase and magnitude correlate with heading and walking speed?

Top row - phase vs heading. Per fly, the 2D density of fly heading (x) against bump phase
(y) over moving frames: a diagonal band is the goal-tracking correlation (offset by the
fly-specific angle; it wraps at the edges). The Jammalamadaka circular correlation r is
printed per fly.

Bottom row - FC2 vs walking speed and a correlation summary:
  * bump magnitude vs speed (per-fly binned, Spearman rho pooled in the title);
  * bump angular velocity |dphi/dt| vs speed (the bump moves less when the fly walks
    faster - opposite of a heading compass);
  * per-fly summary of the three correlations: r(phase, heading), rho(magnitude, speed),
    rho(|dphi/dt|, speed).

One figure in plots/imaging/analysis: fc2_fc2_vs_behaviour.

Run:  python fc2_fc2_vs_behaviour.py
"""

import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

import fc2_analysis as fa
from fc2_analysis import FLY_COLORS
from cxstyle import PINK_CMAP, WHITE
from utils import save_fig

SPEED_BINS = np.linspace(0, 30, 16)


def main():
    data = [(f, fa.add_fit(fa.load(f))) for f in fa.ANALYSIS_FLIES]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)

    summ = []                                                  # per-fly correlation triples
    for i, (fly, df) in enumerate(data):
        mov = df["speed"].to_numpy() >= fa.MOVE_CUTOFF
        head = df["head"].to_numpy(); phase = df["fit_phase"].to_numpy()
        A = df["fit_A"].to_numpy(); spd = df["speed"].to_numpy()
        av = fa.bump_angular_velocity(phase, df["time"].to_numpy())

        ax = axes[0][i]
        ax.hist2d(head[mov], phase[mov], bins=36, range=[[0, 360], [0, 360]], cmap=PINK_CMAP)
        r = fa.circ_corr(head[mov], phase[mov])
        ax.set_title(f"{fly}\nr(phase,heading)={r:+.2f}", fontsize=9)
        ax.set_xlabel("fly heading (deg)")
        if i == 0:
            ax.set_ylabel("FC2 bump phase (deg)")
        ax.set_xticks([0, 180, 360]); ax.set_yticks([0, 180, 360])

        # magnitude vs speed and bump-motion vs speed (bottom row, first two panels)
        fin = np.isfinite(spd) & np.isfinite(A)
        cx, cy, _ = fa.binned(spd[fin], A[fin], SPEED_BINS)
        axes[1][0].plot(cx, cy, "-o", color=FLY_COLORS[i], ms=4, label=fly)
        fv = fin & np.isfinite(av)
        bx, by, _ = fa.binned(spd[fv], av[fv], SPEED_BINS)
        axes[1][1].plot(bx, by, "-o", color=FLY_COLORS[i], ms=4)

        summ.append((fly, r,
                     st.spearmanr(spd[fin], A[fin]).correlation,
                     st.spearmanr(spd[fv], av[fv]).correlation))

    axes[1][0].set_xlabel("walking speed (mm/s)"); axes[1][0].set_ylabel("bump magnitude A")
    axes[1][0].set_title("magnitude vs speed", fontsize=9); axes[1][0].legend(fontsize=7)
    axes[1][1].set_xlabel("walking speed (mm/s)"); axes[1][1].set_ylabel("bump |dphi/dt| (deg/s)")
    axes[1][1].set_title("bump motion vs speed", fontsize=9)

    # summary of the three correlations, one dot per fly
    axs = axes[1][2]
    labels = ["r(phase,\nheading)", "rho(A,\nspeed)", "rho(|dphi/dt|,\nspeed)"]
    for i, (fly, r, ra, rv) in enumerate(summ):
        axs.scatter([0, 1, 2], [r, ra, rv], color=FLY_COLORS[i], s=45, zorder=3, label=fly)
    for j, vals in enumerate(zip(*[(r, ra, rv) for _, r, ra, rv in summ])):
        axs.plot([j - 0.25, j + 0.25], [np.mean(vals)] * 2, color=WHITE, lw=2)
    axs.axhline(0, color=WHITE, lw=0.6, alpha=0.4)
    axs.set_xticks([0, 1, 2]); axs.set_xticklabels(labels, fontsize=7)
    axs.set_ylabel("correlation"); axs.set_title("per-fly correlations (bar = mean)", fontsize=9)

    axes[1][3].axis("off")
    h, lab = axes[1][0].get_legend_handles_labels()
    axes[1][3].legend(h, lab, loc="center", fontsize=9, title="fly")
    fig.suptitle("FC2 phase and magnitude vs heading and walking speed", fontsize=13)
    save_fig(fig, "fc2_fc2_vs_behaviour.png", subdir="imaging/analysis")
    plt.close(fig)
    for fly, r, ra, rv in summ:
        print(f"{fly:10s} r(phase,heading)={r:+.2f}  rho(A,speed)={ra:+.2f}  rho(|dphi/dt|,speed)={rv:+.2f}")
    print("saved plots/imaging/analysis/fc2_fc2_vs_behaviour.png")


if __name__ == "__main__":
    main()
