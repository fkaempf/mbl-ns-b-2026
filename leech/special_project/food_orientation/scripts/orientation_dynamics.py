"""
Leech food-orientation foraging assay analysis.
Theme: orientation toward food over time plus circular statistics.

Main variable: food_align_deg (angle between leech heading and direction to food).
0 deg = aimed straight at food, 180 deg = pointed away, 90 deg = chance.
SIGNED alignment (wrap of heading_am_rad - food_dir_rad) used for rose plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/food_orientation"
CSV = os.path.join(BASE, "food_orientation.csv")
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
ORDER = ["IMG_2855_dish0.mp4", "IMG_2857_dish0.mp4",
         "PXL_20260602_183730315.TS_dish0.mp4", "PXL_20260602_183730315.TS_dish1.mp4",
         "PXL_20260602_210739662.TS_dish1.mp4"]
COLORS = {
    "IMG_2855_dish0.mp4": "#1f77b4",
    "IMG_2857_dish0.mp4": "#ff7f0e",
    "PXL_20260602_183730315.TS_dish0.mp4": "#2ca02c",
    "PXL_20260602_183730315.TS_dish1.mp4": "#d62728",
    "PXL_20260602_210739662.TS_dish1.mp4": "#9467bd",
}


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def signed_align_deg(df):
    """Signed alignment in degrees: 0 = at food, +/- = left/right bias."""
    return np.degrees(wrap_pi(df["heading_am_rad"].values - df["food_dir_rad"].values))


def rayleigh(theta):
    """Rayleigh test for non-uniformity. theta in radians.
    Returns (R, circular_mean_rad, p)."""
    n = len(theta)
    if n == 0:
        return np.nan, np.nan, np.nan
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.hypot(C, S)
    mu = np.arctan2(S, C)
    z = n * R * R
    p = np.exp(-z) * (1 + (2 * z - z * z) / (4 * n))
    p = min(max(p, 0.0), 1.0)
    return R, mu, p


# Load and clean
df = pd.read_csv(CSV)
df = df[df["food_align_deg"].notna()].copy()
df["signed_deg"] = signed_align_deg(df)

# -------------------------------------------------------------------
# Figure 1: food_align_deg over time, one panel per video
# -------------------------------------------------------------------
fig, axes = plt.subplots(len(ORDER), 1, figsize=(9, 13), sharex=False)
for ax, vid in zip(axes, ORDER):
    sub = df[df["video"] == vid].sort_values("time_s")
    c = COLORS[vid]
    ax.scatter(sub["time_s"], sub["food_align_deg"], s=18, alpha=0.35,
               color=c, edgecolors="none")
    # binned population mean over time (10 bins)
    t = sub["time_s"].values
    y = sub["food_align_deg"].values
    if len(t) > 3:
        nb = min(10, max(3, len(t) // 8))
        edges = np.linspace(t.min(), t.max(), nb + 1)
        idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
        bx, by = [], []
        for b in range(nb):
            m = idx == b
            if m.sum() > 0:
                bx.append(t[m].mean())
                by.append(y[m].mean())
        ax.plot(bx, by, color=c, lw=2.2, marker="o", ms=4, label="binned mean")
    ax.axhline(90, color="gray", ls="--", lw=1, label="chance, 90 deg")
    ax.set_ylim(0, 180)
    ax.set_ylabel("food align, deg")
    ax.set_title("Video " + LABELS[vid] + ", n=" + str(len(sub)))
    ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("time, s")
fig.suptitle("Orientation toward food over time, lower is better aimed", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(PLOTS, "orient_food_align_over_time.png"), dpi=130)
plt.close(fig)

# -------------------------------------------------------------------
# Figure 2: distributions + well-aimed fractions
# -------------------------------------------------------------------
fig, (axh, axb) = plt.subplots(1, 2, figsize=(14, 5.5))
bins = np.linspace(0, 180, 19)
for vid in ORDER:
    sub = df[df["video"] == vid]
    axh.hist(sub["food_align_deg"], bins=bins, density=True, histtype="step",
             lw=2, color=COLORS[vid], label=LABELS[vid])
axh.axvline(90, color="gray", ls="--", lw=1)
axh.axvline(45, color="black", ls=":", lw=1)
axh.set_xlabel("food align, deg")
axh.set_ylabel("normalized density")
axh.set_title("Distribution of food alignment per video")
axh.legend(fontsize=8)

thresholds = [30, 45, 90]
labels = [LABELS[v] for v in ORDER]
x = np.arange(len(ORDER))
w = 0.25
for i, th in enumerate(thresholds):
    fracs = [(df[df["video"] == v]["food_align_deg"] < th).mean() for v in ORDER]
    axb.bar(x + (i - 1) * w, fracs, w, label="< " + str(th) + " deg")
axb.axhline(0.5, color="gray", ls="--", lw=1, label="0.5 (chance for <90)")
axb.set_xticks(x)
axb.set_xticklabels(labels, rotation=20)
axb.set_ylabel("fraction of leeches well aimed")
axb.set_ylim(0, 1)
axb.set_title("Fraction aimed within threshold per video")
axb.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "orient_distribution_and_wellaimed.png"), dpi=130)
plt.close(fig)

# -------------------------------------------------------------------
# Figure 3: rose / polar histograms of SIGNED alignment + circular stats
# -------------------------------------------------------------------
rows = []
fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         subplot_kw={"projection": "polar"})
axes = axes.ravel()
rbins = np.linspace(-np.pi, np.pi, 25)
for ax, vid in zip(axes, ORDER):
    sub = df[df["video"] == vid]
    theta = np.radians(sub["signed_deg"].values)
    R, mu, p = rayleigh(theta)
    counts, edges = np.histogram(theta, bins=rbins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width, bottom=0.0,
           color=COLORS[vid], alpha=0.7, edgecolor="k", lw=0.3)
    # circular mean arrow, length scaled by R
    ax.set_theta_zero_location("N")  # put 0 deg (at food) at top
    ax.set_theta_direction(-1)
    rmax = counts.max() if counts.max() > 0 else 1
    ax.annotate("", xy=(mu, R * rmax), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.set_title("Video " + LABELS[vid] + "\n0 deg = at food\nR=" +
                 "{:.2f}".format(R) + ", p=" + "{:.3g}".format(p) +
                 ", n=" + str(len(sub)), fontsize=10)
    rows.append({
        "video": vid,
        "label": LABELS[vid],
        "n_with_food": int(len(sub)),
        "mean_food_align_deg": float(sub["food_align_deg"].mean()),
        "median_food_align_deg": float(sub["food_align_deg"].median()),
        "frac_within_30": float((sub["food_align_deg"] < 30).mean()),
        "frac_within_45": float((sub["food_align_deg"] < 45).mean()),
        "frac_within_90": float((sub["food_align_deg"] < 90).mean()),
        "circular_mean_deg": float(np.degrees(mu)),
        "resultant_R": float(R),
        "rayleigh_p": float(p),
    })
# hide unused polar axes
for ax in axes[len(ORDER):]:
    ax.set_visible(False)
fig.suptitle("Rose histograms of signed alignment, arrow = circular mean (length ~ R)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PLOTS, "orient_rose_circular.png"), dpi=130)
plt.close(fig)

# -------------------------------------------------------------------
# Metrics CSV
# -------------------------------------------------------------------
mdf = pd.DataFrame(rows)[["video", "label", "n_with_food",
                          "mean_food_align_deg", "median_food_align_deg",
                          "frac_within_30", "frac_within_45", "frac_within_90",
                          "circular_mean_deg", "resultant_R", "rayleigh_p"]]
mdf.to_csv(os.path.join(METRICS, "orientation_metrics.csv"), index=False)

# Time-trend report: correlation of food_align_deg with time per video
print("=== TIME TRENDS (Pearson r of food_align_deg vs time_s) ===")
for vid in ORDER:
    sub = df[df["video"] == vid]
    if len(sub) > 3:
        r = np.corrcoef(sub["time_s"], sub["food_align_deg"])[0, 1]
        print(LABELS[vid], "r=", round(r, 3), "(negative = better aimed over time)")

print("\n=== METRICS ===")
print(mdf.to_string(index=False))
print("\nWrote 3 PNGs to", PLOTS)
print("Wrote orientation_metrics.csv to", METRICS)
