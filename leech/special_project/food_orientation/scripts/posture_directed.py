"""Posture and directed crawling analysis for leech food-orientation assay.

Theme: does a straight, crawling posture go with aiming at food?
Hypothesis: straight-bodied leeches (body_align_deg < 20) are in directed
locomotion and should be better aimed at food (lower food_align_deg) than
bent/curled leeches.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/food_orientation"
CSV_IN = os.path.join(BASE, "food_orientation.csv")
PLOTS = os.path.join(BASE, "plots")
METRICS = os.path.join(BASE, "metrics")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

LABELS = {
    "IMG_2855_dish0.mp4": "2855",
    "IMG_2857_dish0.mp4": "2857",
    "PXL_20260602_183730315.TS_dish0.mp4": "PXL1-d0",
    "PXL_20260602_183730315.TS_dish1.mp4": "PXL1-d1",
    "PXL_20260602_210739662.TS_dish1.mp4": "PXL2-d1",
}
ORDER = ["2855", "2857", "PXL1-d0", "PXL1-d1", "PXL2-d1"]
STRAIGHT_THR = 20.0
AIMED_THR = 45.0

df = pd.read_csv(CSV_IN)
df["label"] = df["video"].map(LABELS).fillna(df["video"])
# normalize is_straight to bool independent of how it was stored
df["is_straight"] = df["body_align_deg"] < STRAIGHT_THR

food = df[df["food_align_deg"].notna() & df["body_align_deg"].notna()].copy()

colors = plt.cm.tab10(np.linspace(0, 1, len(ORDER)))
cmap_lab = dict(zip(ORDER, colors))

# ----------------------------------------------------------------------------
# Figure 1: posture_straight_fraction.png
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) fraction straight per video
fracs = []
for lab in ORDER:
    sub = df[df["label"] == lab]
    fracs.append(sub["is_straight"].mean() if len(sub) else np.nan)
axes[0].bar(ORDER, fracs, color=[cmap_lab[l] for l in ORDER])
for i, f in enumerate(fracs):
    axes[0].text(i, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=9)
axes[0].set_ylabel("fraction straight (body_align < 20 deg)")
axes[0].set_title("(a) Fraction straight per video")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=30)

# (b) body_align_deg distribution per video (box)
data_box = [df[df["label"] == lab]["body_align_deg"].dropna().values for lab in ORDER]
bp = axes[1].boxplot(data_box, tick_labels=ORDER, showfliers=False, patch_artist=True)
for patch, lab in zip(bp["boxes"], ORDER):
    patch.set_facecolor(cmap_lab[lab])
    patch.set_alpha(0.6)
axes[1].axhline(STRAIGHT_THR, color="k", ls="--", lw=1, label="straight thr (20 deg)")
axes[1].set_ylabel("body_align_deg")
axes[1].set_title("(b) Body-bend distribution per video")
axes[1].legend(fontsize=8)
axes[1].tick_params(axis="x", rotation=30)

# (c) straight fraction vs time (binned)
for lab in ORDER:
    sub = df[df["label"] == lab].sort_values("time_s")
    if len(sub) < 5:
        continue
    t = sub["time_s"].values
    nbins = min(8, max(2, len(sub) // 20))
    edges = np.linspace(t.min(), t.max(), nbins + 1)
    idx = np.digitize(t, edges[1:-1])
    bx, by = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            bx.append(t[m].mean())
            by.append(sub["is_straight"].values[m].mean())
    if bx:
        axes[2].plot(bx, by, "-o", color=cmap_lab[lab], label=lab, ms=4)
axes[2].set_xlabel("time_s")
axes[2].set_ylabel("fraction straight (binned)")
axes[2].set_title("(c) Straight fraction over time")
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "posture_straight_fraction.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2: posture_vs_orientation.png  (KEY TEST)
# ----------------------------------------------------------------------------
straight_fa = food[food["is_straight"]]["food_align_deg"].values
bent_fa = food[~food["is_straight"]]["food_align_deg"].values

U, p_mw = stats.mannwhitneyu(straight_fa, bent_fa, alternative="two-sided")
med_straight = np.median(straight_fa)
med_bent = np.median(bent_fa)

rho, p_sp = stats.spearmanr(food["body_align_deg"], food["food_align_deg"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# left: violin/box straight vs bent
parts = axes[0].violinplot([straight_fa, bent_fa], positions=[1, 2], showmedians=False, showextrema=False)
for pc, c in zip(parts["bodies"], ["#2c7fb8", "#d95f0e"]):
    pc.set_facecolor(c)
    pc.set_alpha(0.4)
bp = axes[0].boxplot([straight_fa, bent_fa], positions=[1, 2], widths=0.25,
                     showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], ["#2c7fb8", "#d95f0e"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes[0].set_xticks([1, 2])
axes[0].set_xticklabels([f"straight\n(n={len(straight_fa)})", f"bent\n(n={len(bent_fa)})"])
axes[0].set_ylabel("food_align_deg (0 = aimed at food)")
axes[0].set_title("Aiming at food: straight vs bent")
axes[0].axhline(med_straight, color="#2c7fb8", ls=":", lw=1)
axes[0].axhline(med_bent, color="#d95f0e", ls=":", lw=1)
axes[0].text(0.02, 0.97,
             f"median straight = {med_straight:.1f}\nmedian bent = {med_bent:.1f}\n"
             f"Mann-Whitney U = {U:.0f}\np = {p_mw:.3g}",
             transform=axes[0].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# right: scatter body_align vs food_align + binned mean + spearman
axes[1].scatter(food["body_align_deg"], food["food_align_deg"], s=10, alpha=0.3, color="gray")
ba = food["body_align_deg"].values
fa = food["food_align_deg"].values
edges = np.linspace(ba.min(), ba.max(), 9)
cx, cy, ce = [], [], []
idx = np.digitize(ba, edges[1:-1])
for b in range(len(edges) - 1):
    m = idx == b
    if m.sum() >= 5:
        cx.append(ba[m].mean())
        cy.append(fa[m].mean())
        ce.append(fa[m].std() / np.sqrt(m.sum()))
axes[1].errorbar(cx, cy, yerr=ce, fmt="-o", color="crimson", lw=2, label="binned mean +/- SE")
axes[1].axvline(STRAIGHT_THR, color="k", ls="--", lw=1, label="straight thr")
axes[1].set_xlabel("body_align_deg (0 = straight)")
axes[1].set_ylabel("food_align_deg (0 = aimed at food)")
axes[1].set_title("Body bend vs aiming")
axes[1].text(0.02, 0.97, f"Spearman rho = {rho:.3f}\np = {p_sp:.3g}\nn = {len(food)}",
             transform=axes[1].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
axes[1].legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "posture_vs_orientation.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 3: posture_straight_aimed_2x2.png
# ----------------------------------------------------------------------------
food["aimed"] = food["food_align_deg"] < AIMED_THR
# rows: straight(True top), bent ; cols: aimed, not aimed
n_sa = int(((food["is_straight"]) & (food["aimed"])).sum())
n_sn = int(((food["is_straight"]) & (~food["aimed"])).sum())
n_ba = int(((~food["is_straight"]) & (food["aimed"])).sum())
n_bn = int(((~food["is_straight"]) & (~food["aimed"])).sum())
table = np.array([[n_sa, n_sn], [n_ba, n_bn]])  # [straight; bent] x [aimed; not]

chi2, p_chi, dof, exp = stats.chi2_contingency(table)
odds, p_fish = stats.fisher_exact(table)
# row-normalized proportions
prop = table / table.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
row_lab = ["straight", "bent"]
col_lab = ["aimed (<45)", "not aimed"]

# left: counts table heatmap
im0 = axes[0].imshow(table, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(table[i, j]), ha="center", va="center", fontsize=16,
                     color="black")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(col_lab)
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(row_lab)
axes[0].set_title("2x2 counts (pooled)")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

# right: proportion heatmap (row-normalized)
im1 = axes[1].imshow(prop, cmap="RdYlGn", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{prop[i, j]*100:.0f}%", ha="center", va="center", fontsize=16)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(col_lab)
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(row_lab)
axes[1].set_title("Proportion within posture (row-normalized)")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

frac_aimed_straight = n_sa / (n_sa + n_sn) if (n_sa + n_sn) else np.nan
frac_aimed_bent = n_ba / (n_ba + n_bn) if (n_ba + n_bn) else np.nan
fig.suptitle(
    f"Straight & aimed = foraging signature?  "
    f"P(aimed|straight)={frac_aimed_straight:.2f}, P(aimed|bent)={frac_aimed_bent:.2f}  |  "
    f"chi2={chi2:.2f}, p={p_chi:.3g}; Fisher OR={odds:.2f}, p={p_fish:.3g}",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOTS, "posture_straight_aimed_2x2.png"), dpi=130)
plt.close(fig)

# per-video 2x2 aimed fractions note
pervid_notes = []
for lab in ORDER:
    sub = food[food["label"] == lab]
    if len(sub) == 0:
        continue
    s = sub[sub["is_straight"]]
    b = sub[~sub["is_straight"]]
    pas = (s["food_align_deg"] < AIMED_THR).mean() if len(s) else np.nan
    pab = (b["food_align_deg"] < AIMED_THR).mean() if len(b) else np.nan
    pervid_notes.append((lab, len(s), len(b), pas, pab))

# ----------------------------------------------------------------------------
# Metrics CSV
# ----------------------------------------------------------------------------
rows = []
for lab in ORDER:
    sub = df[df["label"] == lab]
    fsub = food[food["label"] == lab]
    s_fa = fsub[fsub["is_straight"]]["food_align_deg"]
    b_fa = fsub[~fsub["is_straight"]]["food_align_deg"]
    rows.append({
        "video": sub["video"].iloc[0] if len(sub) else lab,
        "label": lab,
        "n": len(sub),
        "frac_straight": round(sub["is_straight"].mean(), 4) if len(sub) else np.nan,
        "median_body_align_deg": round(sub["body_align_deg"].median(), 3),
        "median_food_align_straight": round(s_fa.median(), 3) if len(s_fa) else np.nan,
        "median_food_align_bent": round(b_fa.median(), 3) if len(b_fa) else np.nan,
        "n_food_straight": len(s_fa),
        "n_food_bent": len(b_fa),
        "pooled_mw_U": "",
        "pooled_mw_p": "",
        "spearman_rho": "",
        "spearman_p": "",
        "chi2_2x2_p": "",
        "fisher_2x2_p": "",
    })

# pooled summary row
rows.append({
    "video": "POOLED",
    "label": "POOLED",
    "n": len(food),
    "frac_straight": round(food["is_straight"].mean(), 4),
    "median_body_align_deg": round(food["body_align_deg"].median(), 3),
    "median_food_align_straight": round(med_straight, 3),
    "median_food_align_bent": round(med_bent, 3),
    "n_food_straight": len(straight_fa),
    "n_food_bent": len(bent_fa),
    "pooled_mw_U": round(U, 1),
    "pooled_mw_p": f"{p_mw:.4g}",
    "spearman_rho": round(rho, 4),
    "spearman_p": f"{p_sp:.4g}",
    "chi2_2x2_p": f"{p_chi:.4g}",
    "fisher_2x2_p": f"{p_fish:.4g}",
})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(METRICS, "posture_metrics.csv"), index=False)

print("=== Figures written ===")
print("posture_straight_fraction.png, posture_vs_orientation.png, posture_straight_aimed_2x2.png")
print("\n=== Straight fractions per video ===")
for lab, f in zip(ORDER, fracs):
    print(f"  {lab}: {f:.3f}")
print(f"\nPooled food rows: {len(food)} (straight={len(straight_fa)}, bent={len(bent_fa)})")
print(f"median food_align straight = {med_straight:.2f} deg, bent = {med_bent:.2f} deg")
print(f"Mann-Whitney U = {U:.1f}, p = {p_mw:.4g}")
print(f"Spearman rho = {rho:.4f}, p = {p_sp:.4g}")
print(f"\n2x2 [straight;bent]x[aimed;not] =\n{table}")
print(f"P(aimed|straight) = {frac_aimed_straight:.3f}, P(aimed|bent) = {frac_aimed_bent:.3f}")
print(f"chi2 = {chi2:.3f}, p = {p_chi:.4g}; Fisher OR = {odds:.3f}, p = {p_fish:.4g}")
print("\nPer-video P(aimed|straight) / P(aimed|bent):")
for lab, ns, nb, pas, pab in pervid_notes:
    print(f"  {lab}: straight {pas:.2f} (n={ns}) / bent {pab:.2f} (n={nb})")
print("\nCSV: metrics/posture_metrics.csv")
