"""
Leech food-orientation foraging assay.
Theme: COUPLING of ORIENTATION and DISTANCE (active-foraging signature).

Central question: Are leeches CLOSE to food also better AIMED at it
(active homing), or is orientation independent of distance (no chemotaxis
at range)?

Outputs:
  plots/couple_align_vs_distance.png
  plots/couple_polar_food_centric.png
  plots/couple_active_forager.png
  metrics/coupling_metrics.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, sem

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN = os.path.join(BASE, "food_orientation.csv")
PLOTS = os.path.join(BASE, "plots")
METRICS = os.path.join(BASE, "metrics")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

# Short labels in fixed plotting order
LABELS = {
    "IMG_2855_dish0.mp4": "2855",
    "IMG_2857_dish0.mp4": "2857",
    "PXL_20260602_183730315.TS_dish0.mp4": "PXL1-d0",
    "PXL_20260602_183730315.TS_dish1.mp4": "PXL1-d1",
    "PXL_20260602_210739662.TS_dish1.mp4": "PXL2-d1",
}
ORDER = ["2855", "2857", "PXL1-d0", "PXL1-d1", "PXL2-d1"]
COLORS = dict(zip(ORDER, plt.cm.tab10(np.linspace(0, 1, len(ORDER)))))

# Active-forager thresholds
AIM_DEG = 45.0      # aimed within 45 deg of food
STRAIGHT_DEG = 20.0  # body_align < 20 deg
NEAR_R = 0.6         # within 0.6*arena_r of food

# ----------------------------------------------------------------------
# Load and prepare
# ----------------------------------------------------------------------
df = pd.read_csv(CSV_IN)
df["label"] = df["video"].map(LABELS)

# Keep only rows with food present
df = df[df["food_x"].notna() & df["food_y"].notna()].copy()

# Normalized mid-to-food distance (in arena radii)
df["dist_norm"] = np.hypot(df["mid_x"] - df["food_x"],
                           df["mid_y"] - df["food_y"]) / df["arena_r"]

# Rows usable for alignment analysis
da = df[df["food_align_deg"].notna() & df["dist_norm"].notna()].copy()

print(f"rows with food: {len(df)}, with food_align: {len(da)}")

# ----------------------------------------------------------------------
# Figure 1: alignment vs distance (coupling)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for lab in ORDER:
    sub = da[da["label"] == lab]
    if len(sub):
        ax.scatter(sub["dist_norm"], sub["food_align_deg"], s=14,
                   color=COLORS[lab], alpha=0.5, label=lab,
                   edgecolors="none")

# Binned mean +/- SEM by distance (pooled)
bins = np.linspace(0, da["dist_norm"].max() * 1.001, 9)
da["dbin"] = pd.cut(da["dist_norm"], bins)
g = da.groupby("dbin", observed=True)["food_align_deg"]
bc = np.array([iv.mid for iv in g.mean().index])
bm = g.mean().values
bs = g.apply(lambda x: sem(x) if len(x) > 1 else np.nan).values
ax.errorbar(bc, bm, yerr=bs, color="black", lw=2.2, marker="o",
            capsize=4, zorder=5, label="binned mean +/- SEM")

rho, pval = spearmanr(da["dist_norm"], da["food_align_deg"])
ax.axhline(90, ls=":", color="gray", lw=1)
ax.set_xlabel("normalized mid-to-food distance (arena radii)")
ax.set_ylabel("food_align_deg (0 = aimed at food)")
ax.set_title(f"Alignment vs distance (pooled)\nSpearman rho={rho:.3f}, p={pval:.3g}")
ax.set_ylim(-5, 185)
ax.legend(fontsize=8, loc="upper left")

# Hexbin density version
ax = axes[1]
hb = ax.hexbin(da["dist_norm"], da["food_align_deg"], gridsize=18,
               cmap="viridis", mincnt=1)
ax.errorbar(bc, bm, yerr=bs, color="white", lw=2.2, marker="o",
            capsize=4, zorder=5)
fig.colorbar(hb, ax=ax, label="count")
ax.set_xlabel("normalized mid-to-food distance (arena radii)")
ax.set_ylabel("food_align_deg")
ax.set_title("Density (hexbin) with binned mean")
ax.set_ylim(-5, 185)

fig.suptitle("COUPLING: does aiming improve as distance shrinks?", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "couple_align_vs_distance.png"), dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: food-centric polar map (signed heading-minus-food angle)
# ----------------------------------------------------------------------
# Signed angle: heading minus food direction, wrapped to (-pi, pi].
# 0 = aimed at food. food_align_deg is the unsigned magnitude.
ddiff = da["heading_am_rad"] - da["food_dir_rad"]
da["signed_align_rad"] = np.arctan2(np.sin(ddiff), np.cos(ddiff))

n = len(ORDER)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5),
                         subplot_kw={"projection": "polar"})
for ax, lab in zip(axes, ORDER):
    sub = da[da["label"] == lab]
    ax.scatter(sub["signed_align_rad"], sub["dist_norm"], s=16,
               color=COLORS[lab], alpha=0.6, edgecolors="none")
    ax.set_theta_zero_location("N")  # 0 (food) points up
    ax.set_theta_direction(-1)
    ax.set_rmax(max(0.1, da["dist_norm"].max()))
    ax.set_title(f"{lab} (n={len(sub)})", fontsize=10)
    ax.set_xlabel("theta=0 -> aimed at food; r=dist", fontsize=7)
fig.suptitle("Food-centric polar map: clustering at theta=0 in near rings = homing",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "couple_polar_food_centric.png"), dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: active forager
# ----------------------------------------------------------------------
# Criteria on rows with food present.
af = df.copy()
af["c_aim"] = af["food_align_deg"] < AIM_DEG
af["c_straight"] = af["body_align_deg"] < STRAIGHT_DEG
af["c_near"] = af["dist_norm"] < NEAR_R
af["active"] = af["c_aim"] & af["c_straight"] & af["c_near"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) fraction active forager per video
ax = axes[0]
frac = [af[af["label"] == l]["active"].mean() for l in ORDER]
ax.bar(ORDER, frac, color=[COLORS[l] for l in ORDER])
for i, f in enumerate(frac):
    ax.text(i, f + 0.005, f"{f:.2f}", ha="center", fontsize=9)
ax.set_ylabel("fraction active foragers")
ax.set_title(f"(a) Active forager fraction\n(aim<{AIM_DEG:g}, body<{STRAIGHT_DEG:g}, dist<{NEAR_R:g}r)")
ax.set_ylim(0, max(frac) * 1.25 + 0.05 if max(frac) > 0 else 1)

# (b) fraction active over time
ax = axes[1]
for lab in ORDER:
    sub = af[af["label"] == lab].sort_values("time_s")
    if len(sub) < 2:
        continue
    tbins = np.linspace(sub["time_s"].min(), sub["time_s"].max(), 7)
    sub = sub.copy()
    sub["tb"] = pd.cut(sub["time_s"], tbins, include_lowest=True)
    gg = sub.groupby("tb", observed=True)
    tc = np.array([iv.mid for iv in gg["active"].mean().index])
    tv = gg["active"].mean().values
    ax.plot(tc, tv, marker="o", color=COLORS[lab], label=lab)
ax.set_xlabel("time_s")
ax.set_ylabel("fraction active foragers")
ax.set_title("(b) Active forager fraction over time")
ax.legend(fontsize=8)

# (c) per-criterion breakdown
ax = axes[2]
x = np.arange(len(ORDER))
w = 0.25
for j, (crit, name) in enumerate([("c_aim", f"aim<{AIM_DEG:g}"),
                                   ("c_straight", f"body<{STRAIGHT_DEG:g}"),
                                   ("c_near", f"dist<{NEAR_R:g}r")]):
    vals = [af[af["label"] == l][crit].mean() for l in ORDER]
    ax.bar(x + (j - 1) * w, vals, w, label=name)
ax.set_xticks(x)
ax.set_xticklabels(ORDER)
ax.set_ylabel("fraction of frames satisfying criterion")
ax.set_title("(c) Per-criterion breakdown")
ax.legend(fontsize=8)

fig.suptitle("ACTIVE FORAGER signature per video", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "couple_active_forager.png"), dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------
# Metrics CSV
# ----------------------------------------------------------------------
rows = []
for lab in ORDER:
    video = [k for k, v in LABELS.items() if v == lab][0]
    sub_d = da[da["label"] == lab]
    sub_af = af[af["label"] == lab]
    if len(sub_d) >= 3:
        r, p = spearmanr(sub_d["dist_norm"], sub_d["food_align_deg"])
    else:
        r, p = np.nan, np.nan
    near = sub_d[sub_d["dist_norm"] < 0.4]["food_align_deg"].mean()
    far = sub_d[sub_d["dist_norm"] > 0.8]["food_align_deg"].mean()
    rows.append({
        "video": video,
        "label": lab,
        "n_with_food": len(sub_af),
        "spearman_rho_dist_align": round(r, 4) if r == r else np.nan,
        "spearman_p": round(p, 5) if p == p else np.nan,
        "mean_align_near_lt0.4r": round(near, 2) if near == near else np.nan,
        "mean_align_far_gt0.8r": round(far, 2) if far == far else np.nan,
        "frac_active_forager": round(sub_af["active"].mean(), 4),
    })

# Pooled row
r, p = spearmanr(da["dist_norm"], da["food_align_deg"])
near = da[da["dist_norm"] < 0.4]["food_align_deg"].mean()
far = da[da["dist_norm"] > 0.8]["food_align_deg"].mean()
rows.append({
    "video": "POOLED",
    "label": "POOLED",
    "n_with_food": len(af),
    "spearman_rho_dist_align": round(r, 4),
    "spearman_p": round(p, 5),
    "mean_align_near_lt0.4r": round(near, 2),
    "mean_align_far_gt0.8r": round(far, 2),
    "frac_active_forager": round(af["active"].mean(), 4),
})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(METRICS, "coupling_metrics.csv"), index=False)
print(out.to_string(index=False))
print("\nWrote 3 PNGs + coupling_metrics.csv")
