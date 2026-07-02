"""Per-fly histogram of the heading-minus-FC2-bump-phase offset.

For each fly, over moving frames (speed >= MOVE_CUTOFF), the offset
``delta = wrap180(fly heading - bump phase)`` in degrees. A peaked distribution at a
fly-specific angle is the signature of a stable heading<->bump relationship (FC2 encoding
a goal at a fixed offset from heading); a flat one means no relationship. The circular
mean and resultant length R (concentration, 0 = uniform, 1 = perfectly locked) are marked.
A lighter overlay restricts to strong-bump frames (fit magnitude above the fly's median),
previewing whether the offset tightens when the bump is strong (item 5).

Run:  python fc2_offset_hist.py
"""

import numpy as np
import matplotlib.pyplot as plt

import fc2_analysis as fa
from cxstyle import PINK, WHITE
from utils import save_fig

BINS = np.linspace(-180, 180, 49)


def circ_mean_R(deg):
    z = np.mean(np.exp(1j * np.radians(deg)))
    return np.degrees(np.angle(z)), np.abs(z)


def main():
    flies = fa.ANALYSIS_FLIES
    fig, axes = plt.subplots(1, len(flies), figsize=(3.3 * len(flies), 3.3),
                             squeeze=False, constrained_layout=True)
    for ax, fly in zip(axes[0], flies):
        df = fa.add_fit(fa.load(fly))
        mov = df["speed"].to_numpy() >= fa.MOVE_CUTOFF
        off = fa.wrap180(df["head"].to_numpy() - df["fit_phase"].to_numpy())
        A = df["fit_A"].to_numpy()
        good = mov & np.isfinite(off)
        strong = good & (A >= np.nanmedian(A[good]))

        ax.hist(off[good], bins=BINS, density=True, color=PINK, alpha=0.65, label="all moving")
        ax.hist(off[strong], bins=BINS, density=True, histtype="step", color=WHITE, lw=1.4,
                label="strong bump")
        m_all, R_all = circ_mean_R(off[good])
        m_str, R_str = circ_mean_R(off[strong])
        ax.axvline(m_all, color=PINK, lw=1.4, ls="--")
        ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("heading - bump phase (deg)")
        ax.set_title(f"{fly}\nR={R_all:.2f} (all), {R_str:.2f} (strong)\nmean={m_all:.0f} deg",
                     fontsize=8)
    axes[0, 0].set_ylabel("density")
    axes[0, -1].legend(fontsize=7, loc="upper right")
    save_fig(fig, "fc2_offset_hist.png",
             title="Heading - FC2 bump offset per fly (moving frames; R = concentration)",
             subdir="imaging/analysis")
    plt.close(fig)
    print("saved plots/imaging/analysis/fc2_offset_hist.png")


if __name__ == "__main__":
    main()
