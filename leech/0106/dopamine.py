#!/usr/bin/env python3
"""Dopamine before/after analysis for 1.abf (DA added at t = 1200 s = 20 min).
Burst-based: how does C3 bursting change after dopamine?

    python dopamine.py            # 1.abf, event at 1200 s
    python dopamine.py 1.abf 1200
"""
import sys, numpy as np, pandas as pd, pyabf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from crawl_cpg import rate_envelope, detect_bursts, detect_spikes, cv, rcv

fn = sys.argv[1] if len(sys.argv) > 1 else "1.abf"
EVENT = float(sys.argv[2]) if len(sys.argv) > 2 else 1200.0   # s, DA addition

a = pyabf.ABF(fn); fs = a.dataRate
a.setSweep(0, channel=0); C3 = a.sweepY
spk, _ = detect_spikes(C3, fs)
env, efs = rate_envelope(C3, fs)
bursts, _ = detect_bursts(env, efs, spk)

# per-burst table + cycle period (onset-to-onset)
on = np.array([b["onset"] for b in bursts])
df = pd.DataFrame(dict(
    onset=on,
    dur=[b["dur"] for b in bursts],
    n_spikes=[b["n"] for b in bursts],
    ifr=[b["ifr"] for b in bursts]))
df["period"] = np.r_[np.nan, np.diff(df.onset)]
df["phase"] = "pre"; df.loc[df.onset >= EVENT, "phase"] = "post"
df.to_csv("figures/dopamine_bursts.csv", index=False)

metrics = [("period", "cycle period (s)"), ("dur", "burst duration (s)"),
           ("n_spikes", "spikes / burst"), ("ifr", "intraburst IFR (Hz)")]
pre, post = df[df.phase == "pre"], df[df.phase == "post"]

# ---- console summary ----
print(f"\n{fn}: DA added at {EVENT:.0f}s.  "
      f"bursts: {len(pre)} pre / {len(post)} post")
print(f"burst rate: pre {len(pre)/EVENT*60:.1f}/min, "
      f"post {len(post)/(df.onset.max()-EVENT)*60:.1f}/min")
print(f"\n{'metric':<20}{'pre median':>12}{'post median':>13}"
      f"{'pre CVperiod':>14}{'post':>7}")
for col, lab in metrics:
    print(f"{lab:<20}{pre[col].median():>12.2f}{post[col].median():>13.2f}", end="")
    if col == "period":
        print(f"{cv(pre[col]):>14.2f}{cv(post[col]):>7.2f}")
    else:
        print()
print(f"\nRhythm regularity (CV of cycle period, lower=more regular): "
      f"pre {cv(pre.period):.2f} -> post {cv(post.period):.2f}")

# ============================ FIGURE ============================
fig, ax = plt.subplots(len(metrics), 1, figsize=(13, 9), sharex=True)
for k, (col, lab) in enumerate(metrics):
    ax[k].scatter(pre.onset, pre[col], s=8, color="C1", label="pre-DA")
    ax[k].scatter(post.onset, post[col], s=8, color="C0", label="post-DA")
    # 8-burst rolling median trend line
    ax[k].plot(df.onset, df[col].rolling(8, center=True).median(), color="k", lw=1)
    ax[k].axvline(EVENT, color="C3", lw=1.5, ls="--")
    ax[k].set_ylabel(lab)
ax[0].axvline(EVENT, color="C3", lw=1.5, ls="--", label="DA added")
ax[0].legend(loc="upper right", fontsize=8, ncol=3)
ax[0].set_title(f"{fn}: C3 burst metrics across dopamine addition (red line = 20 min)")
ax[-1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig("figures/6_dopamine_timecourse.png", dpi=130); plt.close(fig)

# before/after distributions
fig, ax = plt.subplots(1, len(metrics), figsize=(14, 4))
for k, (col, lab) in enumerate(metrics):
    parts = [pre[col].dropna().values, post[col].dropna().values]
    bp = ax[k].boxplot(parts, tick_labels=["pre", "post"], patch_artist=True, showfliers=False)
    for box, c in zip(bp["boxes"], ["C1", "C0"]):
        box.set_facecolor(c); box.set_alpha(0.5)
    ax[k].set_title(lab)
fig.suptitle(f"{fn}: C3 bursts before vs after dopamine")
fig.tight_layout(); fig.savefig("figures/7_dopamine_beforeafter.png", dpi=130); plt.close(fig)
print("\n-> figures/6_dopamine_timecourse.png, 7_dopamine_beforeafter.png, dopamine_bursts.csv")
