"""
Leech food-orientation foraging assay.
Theme: APPROACH and DISTANCE TO FOOD.

Distances are normalized by per-row arena radius (pixels): dist_norm = d / arena_r.
0 = at food, ~1 = a radius away, up to ~2 across the dish.
No cm scale exists; everything is in arena-radius units.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/food_orientation"
PLOTS = os.path.join(BASE, "plots")
METRICS = os.path.join(BASE, "metrics")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

# Short labels per video (keys without the .mp4 suffix)
LABELS = {
    "IMG_2855_dish0": "2855",
    "IMG_2857_dish0": "2857",
    "PXL_20260602_183730315.TS_dish0": "PXL1-d0",
    "PXL_20260602_183730315.TS_dish1": "PXL1-d1",
    "PXL_20260602_210739662.TS_dish1": "PXL2-d1",
}
# Videos with STABLE per-individual leech_idx (exclude 2857 from per-individual plots)
STABLE = ["IMG_2855_dish0", "PXL_20260602_183730315.TS_dish0",
          "PXL_20260602_183730315.TS_dish1", "PXL_20260602_210739662.TS_dish1"]
ORDER = ["IMG_2855_dish0", "IMG_2857_dish0", "PXL_20260602_183730315.TS_dish0",
         "PXL_20260602_183730315.TS_dish1", "PXL_20260602_210739662.TS_dish1"]

NEAR = 0.25   # "at food" threshold in arena-radius units
HALF = 0.5

# ---------------------------------------------------------------- load + prep
df = pd.read_csv(os.path.join(BASE, "food_orientation.csv"))
df["vid"] = df["video"].str.replace(".mp4", "", regex=False)

# Keep only rows with food annotated (food_x/food_y present)
df = df[df["food_x"].notna() & df["food_y"].notna()].copy()

# Recompute distances (pixels) and normalize by per-row arena radius
df["head_dist_px"] = np.hypot(df["ant_x"] - df["food_x"], df["ant_y"] - df["food_y"])
df["mid_dist_px"] = np.hypot(df["mid_x"] - df["food_x"], df["mid_y"] - df["food_y"])
df["head_norm"] = df["head_dist_px"] / df["arena_r"]
df["mid_norm"] = df["mid_dist_px"] / df["arena_r"]

# Cross-check against provided pixel head distances (different frame indexing; informational)
try:
    ref = pd.read_csv(os.path.join(METRICS, "head_distance_to_food.csv"))
    print("cross-check head_food_dist (px): provided mean %.1f, recomputed mean %.1f"
          % (ref["head_food_dist"].mean(), df["head_dist_px"].mean()))
except Exception as e:
    print("cross-check skipped:", e)

head_use = df[df["head_norm"].notna()].copy()

# food radius as fraction of arena radius: use median food blob if available else a nominal mark.
# There is no explicit food radius column, so we mark the NEAR=0.25 r "at food" band as reference.
FOOD_REF = NEAR  # reference line for "considered at food"

# ============================================================ FIGURE 1
fig, ax = plt.subplots(figsize=(10, 6))
data, ticks, labels = [], [], []
for i, v in enumerate(ORDER):
    sub = head_use[head_use["vid"] == v]["head_norm"].values
    if len(sub) == 0:
        continue
    data.append(sub)
    ticks.append(i + 1)
    labels.append("%s\n(n=%d)" % (LABELS[v], len(sub)))

parts = ax.violinplot(data, positions=ticks, showmedians=True, widths=0.8)
for pc in parts["bodies"]:
    pc.set_facecolor("#4C72B0")
    pc.set_alpha(0.6)
for k in ("cmins", "cmaxes", "cbars", "cmedians"):
    if k in parts:
        parts[k].set_color("#22324d")
ax.axhline(FOOD_REF, color="crimson", ls="--", lw=1.5,
           label="at-food band (%.2f r)" % FOOD_REF)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_ylabel("head-to-food distance (arena radii)")
ax.set_title("Per-video distribution of normalized head-to-food distance")
ax.legend(loc="upper right")
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "approach_distance_distributions.png"), dpi=130)
plt.close(fig)

# ============================================================ FIGURE 2
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
axes = axes.ravel()
for ax, v in zip(axes, STABLE):
    sub = head_use[head_use["vid"] == v]
    for lid, gg in sub.groupby("leech_idx"):
        gg = gg.sort_values("time_s")
        ax.plot(gg["time_s"], gg["head_norm"], lw=0.8, alpha=0.55, color="#888888")
    # population median over time (binned)
    s = sub.sort_values("time_s")
    if len(s) > 4:
        nb = min(12, max(4, len(s) // 8))
        s = s.copy()
        s["bin"] = pd.qcut(s["time_s"], nb, duplicates="drop")
        med = s.groupby("bin", observed=True).agg(t=("time_s", "median"),
                                                   m=("head_norm", "median")).dropna()
        ax.plot(med["t"], med["m"], lw=2.6, color="#C44E52", label="pop. median")
    ax.axhline(NEAR, color="crimson", ls=":", lw=1.0)
    ax.set_title("%s  (%d leeches)" % (LABELS[v], sub["leech_idx"].nunique()))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("head-to-food dist (arena radii)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
fig.suptitle("Individual head-to-food distance vs time (stable-id videos)", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(PLOTS, "approach_individual_trajectories.png"), dpi=130)
plt.close(fig)

# ============================================================ FIGURE 3
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

# (a) fraction within NEAR over time, per video
for v in ORDER:
    sub = head_use[head_use["vid"] == v].sort_values("time_s")
    if len(sub) < 4:
        continue
    nb = min(10, max(3, len(sub) // 10))
    sub = sub.copy()
    sub["bin"] = pd.qcut(sub["time_s"], nb, duplicates="drop")
    agg = sub.groupby("bin", observed=True).agg(
        t=("time_s", "median"),
        frac=("head_norm", lambda x: np.mean(x < NEAR))).dropna()
    axA.plot(agg["t"], agg["frac"], marker="o", ms=4, lw=1.6, label=LABELS[v])
axA.set_xlabel("time (s)")
axA.set_ylabel("fraction within %.2f r of food" % NEAR)
axA.set_title("(a) Fraction 'at food' over time")
axA.set_ylim(-0.02, 1.02)
axA.legend(fontsize=8)

# (b) grouped bar: mean normalized distance AND fraction near
labs, mean_d, frac_n = [], [], []
for v in ORDER:
    sub = head_use[head_use["vid"] == v]
    if len(sub) == 0:
        continue
    labs.append(LABELS[v])
    mean_d.append(sub["head_norm"].mean())
    frac_n.append(np.mean(sub["head_norm"] < NEAR))
x = np.arange(len(labs))
w = 0.38
axB.bar(x - w / 2, mean_d, w, color="#4C72B0", label="mean dist (arena radii)")
axB.bar(x + w / 2, frac_n, w, color="#55A868", label="fraction within %.2f r" % NEAR)
axB.set_xticks(x)
axB.set_xticklabels(labs)
axB.set_title("(b) Mean normalized distance and fraction near food")
axB.set_ylabel("value")
axB.legend(fontsize=8)
for xi, (md, fn) in enumerate(zip(mean_d, frac_n)):
    axB.text(xi - w / 2, md + 0.02, "%.2f" % md, ha="center", fontsize=8)
    axB.text(xi + w / 2, fn + 0.02, "%.2f" % fn, ha="center", fontsize=8)
fig.suptitle("Population approach summary", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(PLOTS, "approach_population_summary.png"), dpi=130)
plt.close(fig)

# ============================================================ METRICS CSV
rows = []
for v in ORDER:
    sub = head_use[head_use["vid"] == v]
    if len(sub) == 0:
        continue
    d = sub["head_norm"]
    frac_appr = np.nan
    if v in STABLE:
        dec = tot = 0
        for lid, gg in sub.groupby("leech_idx"):
            gg = gg.sort_values("time_s")
            if len(gg) < 2:
                continue
            tot += 1
            if gg["head_norm"].iloc[-1] < gg["head_norm"].iloc[0]:
                dec += 1
        frac_appr = dec / tot if tot else np.nan
    rows.append(dict(
        video=v,
        label=LABELS[v],
        n_with_food=len(sub),
        mean_headdist_norm=round(float(d.mean()), 4),
        median_headdist_norm=round(float(d.median()), 4),
        min_headdist_norm=round(float(d.min()), 4),
        frac_within_0p25r=round(float(np.mean(d < NEAR)), 4),
        frac_within_0p5r=round(float(np.mean(d < HALF)), 4),
        frac_individuals_net_approached=(round(float(frac_appr), 4)
                                         if frac_appr == frac_appr else ""),
    ))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(METRICS, "approach_metrics.csv"), index=False)
print(out.to_string(index=False))
print("\nWrote 3 PNGs to", PLOTS, "and approach_metrics.csv to", METRICS)
