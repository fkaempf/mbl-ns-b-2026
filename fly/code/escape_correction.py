"""Exploratory (capstone): is the bounce escape turn error-correcting?

Connects findings #1-3. Each fly has a preferred heading (#3); each bounce triggers a
big turn (#1-2). Question: does that turn bring the heading *back toward* the preferred
direction (a course-correction to the menotaxis goal), or is it directionally random?

Per bounce we take the circular-mean heading just before (-3..-1 s, moving) and just
after the escape (+1..+3 s, moving), and their absolute deviation from the fly's own
preferred heading. If the turn restores heading, deviation drops (post < pre).

Run from this directory:  python escape_correction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import bounces as bo
from utils import experiment_id, load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")


def wrap(d):
    return (d + 180) % 360 - 180


def cmean(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def gather():
    rows = []
    for f in EXPERIMENTS:
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, h, sp = df.t_unix.to_numpy(), df.vrh.to_numpy(), df.speed.to_numpy()
        pref = cmean(h[sp >= 5])
        eid = experiment_id(f)

        def win(tb, lo, hi):
            m = (t >= tb + lo) & (t <= tb + hi) & (sp >= 5)
            return cmean(h[m]) if m.sum() >= 3 else np.nan

        for tb in bt:
            hpre, hpost = win(tb, -3, -1), win(tb, 1, 3)
            if np.isnan(hpre) or np.isnan(hpost):
                continue
            rows.append(dict(eid=eid,
                             dev_pre=abs(wrap(hpre - pref)),
                             dev_post=abs(wrap(hpost - pref))))
    return pd.DataFrame(rows)


def main():
    d = gather()
    print(f"{len(d)} bounces with pre+post heading")

    diff = d.dev_pre - d.dev_post                      # >0 = restoring, <0 = disrupting
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(diff, bins=np.linspace(-180, 180, 25), color="0.6", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.axvline(diff.median(), color="tab:blue", lw=2.5, label=f"median {diff.median():+.0f} deg")
    ax.set_xlabel("deviation reduced by the bounce (deg;  >0 = restored toward preferred,  "
                  "<0 = disrupted)")
    ax.set_ylabel("count")
    ax.set_title(f"does the escape correct heading? (n={len(d)} bounces)", fontsize=11)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "escape_correction.png", subdir="exploratory")
    plt.close(fig)

    print(f"  overall: dev_pre median {d.dev_pre.median():.0f} -> dev_post {d.dev_post.median():.0f} deg")
    w = wilcoxon(d.dev_pre, d.dev_post)
    print(f"  Wilcoxon pre vs post (pooled, pseudoreplicated): p={w.pvalue:.3f}")
    for eid, g in d.groupby("eid"):
        try:
            p = wilcoxon(g.dev_pre, g.dev_post).pvalue
        except ValueError:
            p = np.nan
        print(f"    {eid.split('_')[-1]}: n={len(g)} pre={g.dev_pre.median():.0f} "
              f"post={g.dev_post.median():.0f} deg  p={p:.3f}")


if __name__ == "__main__":
    main()
