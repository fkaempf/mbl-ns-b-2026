"""Implements avoidance_learning.md: does the fly learn to AVOID the wall over the
session, after controlling for locomotor slowing?

Per wall presentation (CREATE->DESTROY) we collect time-to-first-bounce (ttf), whether it
was hit at all (bounce), the wall onset time in the session, and the pre-bounce approach
speed (the fatigue covariate). Models (15 flies):
  ttf    ~ session_time + approach_speed + (1|eid)     (LMM, bounce trials)
  bounce ~ session_time + approach_speed               (GEE logistic, fly-clustered)
Learning survives only if session_time is significant AFTER approach_speed is partialled
out. An off-wall control (does the fly spontaneously enter the just-removed wall's footprint
during the laser-free ITI?) gives a no-stimulus baseline. Run:  python avoidance_learning.py
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import bounces as bo
from utils import load_combined, experiment_id, save_fig

EXPERIMENTS = bo.barrier_experiments("good")


def gather():
    rows = []
    for f in EXPERIMENTS:
        eid = experiment_id(f).split("_")[-1]
        walls = bo.wall_trials(f)
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                       df.vry.to_numpy(), df.speed.to_numpy())
        t0 = t[0]
        for i, (ton, toff, cx, cy, rot, w, th) in enumerate(walls):
            b_in = bt[(bt >= ton) & (bt <= toff)]
            nb = len(b_in)
            ttf = (b_in[0] - ton) if nb else np.nan
            m = (t >= ton) & (t <= ton + 3.0)          # FIXED 3 s pre-window, exogenous to outcome
            appsp = np.nanmean(sp[m]) if m.sum() > 3 else np.nan
            # off-wall control: does the fly enter the removed wall's footprint in the ITI?
            nxt = walls[i + 1][0] if i + 1 < len(walls) else toff + 60
            mi = (t >= toff) & (t <= nxt)
            off_cross = np.nan
            if mi.sum() > 3:
                rr = np.radians(rot)
                nrm, alg = np.array([-np.sin(rr), np.cos(rr)]), np.array([np.cos(rr), np.sin(rr)])
                dp = (x[mi] - cx) * nrm[0] + (y[mi] - cy) * nrm[1]
                da = (x[mi] - cx) * alg[0] + (y[mi] - cy) * alg[1]
                off_cross = int(np.any((np.abs(dp) < bo.LASER_MARGIN) & (np.abs(da) < w / 2)))
            rows.append(dict(eid=eid, ton_min=(ton - t0) / 60.0, nb=nb, ttf=ttf,
                             bounce=int(nb > 0), appsp=appsp, off_cross=off_cross))
    return pd.DataFrame(rows)


def main():
    d = gather().dropna(subset=["appsp"])
    d["sess"] = d.groupby("eid").ton_min.transform(lambda s: (s - s.min()) / max(s.max() - s.min(), 1e-9))
    db = d[d.bounce == 1].dropna(subset=["ttf"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_raw = smf.mixedlm("ttf ~ sess", db, groups=db.eid).fit(method="lbfgs")
        m_adj = smf.mixedlm("ttf ~ sess + appsp", db, groups=db.eid).fit(method="lbfgs")
        m_bin = smf.gee("bounce ~ sess + appsp", "eid", d, family=sm.families.Binomial()).fit()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    # (a) ttf vs session onset per fly
    ax = axes[0]
    for eid, gdf in db.groupby("eid"):
        ax.scatter(gdf.ton_min, gdf.ttf, s=10, alpha=0.4)
    ax.set_xlabel("wall onset in session (min)"); ax.set_ylabel("time-to-first-bounce (s)")
    pa = m_adj.pvalues["sess"]
    ax.set_title("time-to-first-bounce vs session (bounced trials only = SURVIVORSHIP)\n"
                 f"sess coef adj {m_adj.params['sess']:+.1f}s p={pa:.2f}  - not learning evidence",
                 fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    # (b) session thirds: on-wall bounce rate vs off-wall spontaneous-cross rate
    ax = axes[1]
    d["third"] = pd.cut(d.sess, [-.01, .33, .66, 1.01], labels=["early", "mid", "late"])
    on = d.groupby("third", observed=True).bounce.mean() * 100
    off = d.dropna(subset=["off_cross"]).groupby("third", observed=True).off_cross.mean() * 100
    xx = np.arange(3)
    ax.plot(xx, on.values, "o-", color="#cc3311", lw=2, label="on-wall (got zapped)")
    ax.plot(xx, off.values, "o--", color="0.5", lw=2, label="off-wall (spont. cross, ITI)")
    ax.set_xticks(xx); ax.set_xticklabels(["early", "mid", "late"])
    ax.set_ylabel("% of trials"); ax.set_ylim(0, 100)
    ax.set_title("hit-rate over the session\n(learning = on-wall drops below off-wall)", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    # (c) coefficient forest of session_time
    ax = axes[2]
    rows = [("ttf ~ sess (raw)", m_raw.params["sess"], *m_raw.conf_int().loc["sess"]),
            ("ttf ~ sess (speed-adj)", m_adj.params["sess"], *m_adj.conf_int().loc["sess"]),
            ("bounce ~ sess (adj, logit)", m_bin.params["sess"], *m_bin.conf_int().loc["sess"])]
    for j, (lab, c, lo, hi) in enumerate(rows):
        ax.errorbar(c, j, xerr=[[c - lo], [hi - c]], fmt="o", color="k", capsize=4)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("session_time coefficient"); ax.set_ylim(-0.5, 2.5)
    ax.set_title("does session_time survive the speed control?", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"no avoidance learning: hit-rate flat over the session (bounce~session p={m_bin.pvalues['sess']:.2f}, "
                 f"n={d.eid.nunique()} flies, {len(d)} trials).  "
                 f"ttf-decrease is survivorship (bounced trials only), NOT learning.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "avoidance_learning.png", subdir="exploratory")
    plt.close(fig)
    print(f"ttf~sess raw {m_raw.params['sess']:+.1f} (p={m_raw.pvalues['sess']:.2f}); "
          f"speed-adj {m_adj.params['sess']:+.1f} (p={pa:.2f}); "
          f"bounce~sess adj coef {m_bin.params['sess']:+.2f} (p={m_bin.pvalues['sess']:.2f})")


if __name__ == "__main__":
    main()
