"""Is each fly's before->after heading change real? Permutation test.

For every fly the speed-gated VR heading samples from the before and after windows
are pooled and randomly re-split (same group sizes) many times; the circular
after-before shift of each shuffle builds a null distribution. The real shift is
overlaid with a permutation p-value (fraction of |null shift| >= |real shift|).
Windows / speed gate / experiments are imported from vrpos_heading_hist so the two
stay in sync.

Run from this directory:  python heading_shift_shuffle.py
"""

import numpy as np

import scalloping as sc
from utils import experiment_id, save_panel
from vrpos_heading_hist import EXPERIMENTS, before_after_heading

RNG = np.random.default_rng(0)


def cmean(deg):
    mu, _ = sc.circ_mean_R(np.radians(deg))
    return mu                              # radians


def wrap_deg(rad):
    return np.degrees((rad + np.pi) % (2 * np.pi) - np.pi)


def main():
    # one 5-min average heading per window per fly
    labels, before_avg, after_avg = [], [], []
    for folder in EXPERIMENTS:
        b, a = before_after_heading(folder)
        if len(b) == 0 or len(a) == 0:
            continue
        labels.append(experiment_id(folder).replace("rig1_experiment_", "fly "))
        before_avg.append(cmean(b)); after_avg.append(cmean(a))
    before_avg, after_avg = np.array(before_avg), np.array(after_avg)

    actual = wrap_deg(after_avg - before_avg)                      # within-fly shift
    # null: shuffle which before pairs with which after (between-fly differences)
    null = np.array([wrap_deg(after_avg[j] - before_avg[i])
                     for i in range(len(before_avg)) for j in range(len(after_avg)) if i != j])
    am, nm = np.mean(np.abs(actual)), np.mean(np.abs(null))
    p = float(np.mean(np.abs(null) <= am))                         # is within-fly smaller than chance?
    print(f"mean |shift|: within-fly {am:.0f} deg  vs  shuffled-pairing {nm:.0f} deg   p={p:.3f}")
    for lab, s in zip(labels, actual):
        print(f"  {lab}: {s:+.0f} deg")

    def draw(ax):
        bp = ax.boxplot([actual, null], positions=[0, 1], widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], ["#cc3311", "0.8"]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.scatter((RNG.random(len(actual)) - 0.5) * 0.18, actual, color="#cc3311", s=55, zorder=3)
        ax.scatter((RNG.random(len(null)) - 0.5) * 0.4 + 1, null, color="0.45", s=18, zorder=3)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["actual\n(within fly)", "shuffled\n(between flies)"])
        ax.set_xlim(-0.5, 1.7)
        ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_ylabel("heading shift after - before (deg)")
        ax.set_title(f"within-fly heading shift vs shuffled pairing\n"
                     f"mean |shift| {am:.0f}° vs {nm:.0f}°  (p={p:.3f})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    save_panel("stats", "heading_shift_shuffle.png", draw, figsize=(6.5, 5))


if __name__ == "__main__":
    main()
