"""Q1: what does the FC2 bump *magnitude* mean?

Make bump magnitude A the organising variable and show what co-varies with it, so the
meaning can be read straight off. Per fly (the honest unit) plus a pooled trend (white).
All panels share x = bump magnitude A:

  a) heading-phase coupling (windowed circular r) rises with magnitude
  b) |heading - bump offset| falls with magnitude (heading locks to the bump)
  c) walking speed rises with magnitude
  d) bump angular velocity |dphi/dt| falls with magnitude (the bump is steadier)

Read together: a strong bump marks moments of engaged, stable, well-tracked goal-directed
walking. One figure in plots/imaging/analysis: fc2_magnitude_meaning.

Run:  python fc2_magnitude_meaning.py
"""

import numpy as np
import matplotlib.pyplot as plt

import fc2_analysis as fa
from fc2_analysis import FLY_COLORS
from cxstyle import WHITE
from utils import save_fig

ABINS = np.linspace(0, 0.5, 11)


def main():
    data = [(f, fa.add_fit(fa.load(f))) for f in fa.ANALYSIS_FLIES]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    (a_cpl, a_off), (a_spd, a_stab) = axes
    pool = {k: ([], []) for k in ("cpl", "off", "spd", "stab")}

    for i, (fly, df) in enumerate(data):
        mov = df["speed"].to_numpy() >= fa.MOVE_CUTOFF
        A = df["fit_A"].to_numpy()
        off = np.abs(df["offset"].to_numpy())
        spd = df["speed"].to_numpy()
        av = fa.bump_angular_velocity(df["fit_phase"].to_numpy(), df["time"].to_numpy())
        w = fa.windowed(df)

        for ax, x, y, key, mask in (
                (a_cpl, w["mean_A"].to_numpy(), w["circ_r"].to_numpy(), "cpl", None),
                (a_off, A, off, "off", mov),
                (a_spd, A, spd, "spd", None),
                (a_stab, A, av, "stab", mov)):
            if mask is not None:
                x, y = x[mask], y[mask]
            cx, cy, _ = fa.binned(x, y, ABINS, min_n=(6 if key == "cpl" else 20))
            ax.plot(cx, cy, "-o", color=FLY_COLORS[i], ms=4, alpha=0.8, label=fly)
            pool[key][0].append(np.asarray(x)); pool[key][1].append(np.asarray(y))

    for ax, key in ((a_cpl, "cpl"), (a_off, "off"), (a_spd, "spd"), (a_stab, "stab")):
        X, Y = np.concatenate(pool[key][0]), np.concatenate(pool[key][1])
        px, py, pe = fa.binned(X, Y, ABINS, min_n=(12 if key == "cpl" else 40))
        ax.plot(px, py, "-", color=WHITE, lw=2.6, zorder=5)
        ax.fill_between(px, py - pe, py + pe, color=WHITE, alpha=0.15, zorder=4)

    a_cpl.axhline(0, color=WHITE, lw=0.6, alpha=0.4)
    a_cpl.set_ylabel("heading-phase coupling r (windowed)")
    a_cpl.set_title("a) coupling rises with magnitude", fontsize=9)
    a_cpl.legend(fontsize=7)
    a_off.set_ylabel("|heading - bump offset| (deg)")
    a_off.set_title("b) heading locks to the bump as magnitude rises", fontsize=9)
    a_spd.set_ylabel("walking speed (mm/s)")
    a_spd.set_title("c) faster walking when the bump is strong", fontsize=9)
    a_stab.set_ylabel("bump |dphi/dt| (deg/s)")
    a_stab.set_title("d) the bump is steadier when strong", fontsize=9)
    for ax in (a_spd, a_stab):
        ax.set_xlabel("bump magnitude A")
    fig.suptitle("What the FC2 bump magnitude means: a strong bump = engaged, stable, "
                 "well-tracked goal (white = pooled +/- SEM)", fontsize=12)
    save_fig(fig, "fc2_magnitude_meaning.png", subdir="imaging/analysis")
    plt.close(fig)
    print("saved plots/imaging/analysis/fc2_magnitude_meaning.png")


if __name__ == "__main__":
    main()
