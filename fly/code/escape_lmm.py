"""Implements rigor_stats.md: a per-fly linear mixed-effects model for the claim
"the escape disrupts the goal heading" (replacing the pooled, pseudoreplicated Wilcoxon
in escape_correction.py).

Model: dev ~ post + (post | eid)  [random slope; falls back to random intercept].
The fixed `post` coefficient (with CI, p) is the honest test over the 15 independent flies;
a per-fly median paired Wilcoxon (n=15) is the distribution-free cross-check. The plot shows
the 15 per-fly pre->post slopes with the model's fixed effect +/- CI overlaid.

Run from this directory:  python escape_lmm.py
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

import escape_correction as ec
from utils import save_fig


def main():
    d = ec.gather()                                  # per-bounce: eid, dev_pre, dev_post
    long = pd.concat([
        pd.DataFrame({"dev": d.dev_pre.to_numpy(), "post": 0.0, "eid": d.eid.to_numpy()}),
        pd.DataFrame({"dev": d.dev_post.to_numpy(), "post": 1.0, "eid": d.eid.to_numpy()}),
    ], ignore_index=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm("dev ~ post", long, groups=long.eid, re_formula="~post").fit(method="lbfgs")
            rs = "random slope"
            cr = np.atleast_2d(m.cov_re.to_numpy())     # reject a degenerate/singular RE covariance
            if (not m.converged) or np.linalg.det(cr) < 1e-8 or np.any(np.diag(cr) < 1e-6):
                raise RuntimeError
        except Exception:
            m = smf.mixedlm("dev ~ post", long, groups=long.eid).fit(method="lbfgs")
            rs = "random intercept"
    coef = m.params["post"]
    lo, hi = m.conf_int().loc["post"]
    p = m.pvalues["post"]
    b0 = m.params["Intercept"]
    re_sd = float(np.sqrt(m.cov_re.iloc[0, 0])) if m.cov_re.size else float("nan")

    g = d.groupby("eid").agg(pre=("dev_pre", "median"), post=("dev_post", "median")).reset_index()
    wp = wilcoxon(g.pre, g.post).pvalue

    fig, ax = plt.subplots(figsize=(7, 5.6))
    for _, r in g.iterrows():
        ax.plot([0, 1], [r.pre, r.post], color="0.65", alpha=0.6, marker="o", ms=4, lw=1, zorder=2)
    ax.plot([0, 1], [b0, b0 + coef], color="#cc3311", lw=3.5, marker="o", ms=8, zorder=5,
            label="LMM fixed effect")
    ax.errorbar([1], [b0 + coef], yerr=[[coef - lo], [hi - coef]], fmt="none",
                color="#cc3311", capsize=6, lw=2.5, zorder=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pre\n(-3..-1 s)", "post\n(+1..+3 s)"])
    ax.set_xlim(-0.3, 1.3); ax.set_ylabel("|heading - preferred| (deg)")
    ax.set_title(f"escape disrupts heading ({len(g)} flies, {len(d)} bounces)\n"
                 f"robust per-fly Wilcoxon p={wp:.1e}  |  LMM effect +{coef:.0f} deg [{lo:.0f}, {hi:.0f}] "
                 f"({rs}; Gaussian p={p:.0e} anticonservative on bounded dev)", fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "escape_lmm.png", subdir="exploratory")
    plt.close(fig)
    print(f"LMM ({rs}): post +{coef:.1f} [{lo:.1f},{hi:.1f}] p={p:.2e}, RE_sd={re_sd:.1f}; "
          f"per-fly Wilcoxon (n={len(g)}) p={wp:.3f}")


if __name__ == "__main__":
    main()
