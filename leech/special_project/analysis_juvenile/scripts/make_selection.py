#!/usr/bin/env python
"""Curated FINAL FIGURES set for the JUVENILE cohort.

Regenerates 7 figures into "analysis_juvenile/plots closer selection/".
Data computation + calibration are reused VERBATIM from the existing scripts
(thigmotaxis_analysis.py, space_use_exploration.py, kinematics_analysis.py,
design_2x2.py); only the presentation (grouping, colors, styling) is curated.

Colorblind-safe (Okabe-Ito) palette throughout. NO em dashes anywhere.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D


def two_shades(c):
    """Two distinguishable shades of a base color (for the 2 animals in a dish)."""
    r, g, b = to_rgb(c)
    return c, (r * 0.45, g * 0.45, b * 0.45)
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_juvenile"
CSV = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/DL/juv_all_predictions.csv"
MET = f"{BASE}/metrics"
OUT = f"{BASE}/../plots closer selection"
os.makedirs(OUT, exist_ok=True)

FPS = 29.99
DT = 1.0 / FPS
R = 1.75            # dish physical radius (cm); dish diameter 3.5 cm

# Per-video arena calibration (KNOWN; reused verbatim from the scripts).
CAL = {
    "Video1_dish0.mp4": dict(cx=274.2, cy=275.4, cmpp=0.0067609),
    "Video1_dish1.mp4": dict(cx=271.8, cy=267.0, cmpp=0.0069246),
    "Video2_dish0.mp4": dict(cx=282.6, cy=269.4, cmpp=0.0070045),
    "Video2_dish1.mp4": dict(cx=252.6, cy=243.0, cmpp=0.0075799),
}

# 8 entities, grouped into 4 dishes (each dish = one video, 2 tracked leeches).
ENTITIES = [
    ("Video1_dish0.mp4", 0, "Video1_dish0_L0"),
    ("Video1_dish0.mp4", 1, "Video1_dish0_L1"),
    ("Video1_dish1.mp4", 0, "Video1_dish1_L0"),
    ("Video1_dish1.mp4", 1, "Video1_dish1_L1"),
    ("Video2_dish0.mp4", 0, "Video2_dish0_L0"),
    ("Video2_dish0.mp4", 1, "Video2_dish0_L1"),
    ("Video2_dish1.mp4", 0, "Video2_dish1_L0"),
    ("Video2_dish1.mp4", 1, "Video2_dish1_L1"),
]

# Dish -> (treatment label, treatment color, is_low_conf). Veh+NoFood = high-conf.
# Juvenile cohort = yellow/gold shades, one shade per treatment.
DISHES = [
    ("Video1_dish0.mp4", "Video1_dish0", "Dopamine (DA+NoFood)", "#ffd92e", True),
    ("Video1_dish1.mp4", "Video1_dish1", "DA+Food",              "#e0a800", True),
    ("Video2_dish0.mp4", "Video2_dish0", "Veh+NoFood",           "#a86a00", False),
    ("Video2_dish1.mp4", "Video2_dish1", "Veh+Food",             "#5c3d00", True),
]

# Occupancy heatmap colormap for this cohort
HEATMAP_CMAP = "YlOrBr"

LOWCONF = {"Video1_dish0_L0", "Video1_dish0_L1", "Video1_dish1_L0",
           "Video1_dish1_L1", "Video2_dish1_L0", "Video2_dish1_L1"}

CELL_CM = 0.25      # occupancy metric grid (verbatim from space_use_exploration)
HM_CELL_CM = 0.10   # finer occupancy heatmap grid (verbatim)
ARENA_AREA = np.pi * R ** 2  # 9.6 cm^2

print("loading predictions...")
df = pd.read_csv(CSV)


# ----- smoothing reused verbatim from the originals -------------------------
def smooth(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


# ----- per-entity computation (centroid in cm, radial, occupancy) -----------
ent = {}  # key -> dict of computed series/metrics
for video, track, key in ENTITIES:
    g = df[(df.video == video) & (df.track == track)]
    h = g[g.node == 0].sort_values("frame")
    t = g[g.node == 1].sort_values("frame")
    m = pd.merge(h[["frame", "x", "y"]], t[["frame", "x", "y"]],
                 on="frame", suffixes=("_h", "_t"))
    if len(m) < 100:
        continue
    a = CAL[video]
    cmpp = a["cmpp"]
    hx = (smooth(m.x_h.to_numpy()) - a["cx"]) * cmpp
    hy = (smooth(m.y_h.to_numpy()) - a["cy"]) * cmpp
    tx = (smooth(m.x_t.to_numpy()) - a["cx"]) * cmpp
    ty = (smooth(m.y_t.to_numpy()) - a["cy"]) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0
    frames = m.frame.to_numpy()
    tsec = frames * DT
    valid = np.isfinite(cx) & np.isfinite(cy)
    cxv, cyv, tv = cx[valid], cy[valid], tsec[valid]

    r_cm = np.sqrt(cx ** 2 + cy ** 2)

    # cumulative unique area (0.25 cm grid) vs time -> cm^2  (verbatim)
    gx = np.floor(cxv / CELL_CM).astype(np.int64)
    gy = np.floor(cyv / CELL_CM).astype(np.int64)
    cellid = gx * 100000 + gy
    seen = set()
    cum_cells = np.empty(len(cellid), dtype=np.int64)
    for i, c in enumerate(cellid):
        seen.add(c)
        cum_cells[i] = len(seen)
    cum_area_cm2 = cum_cells * (CELL_CM * CELL_CM)
    area_grid_cm2 = float(cum_cells[-1] * CELL_CM * CELL_CM)

    ent[key] = dict(
        video=video, cx=cx, cy=cy, cxv=cxv, cyv=cyv,
        r_cm=r_cm, frames=frames,
        exp_t=tv, exp_cum=cum_area_cm2, area_grid_cm2=area_grid_cm2,
        low_conf=key in LOWCONF,
    )


def lc_tag(is_low):
    return ""


# ===========================================================================
# FIGURE 1: thigmotaxis_arena_scatter.png  (one panel PER DISH, 2 animals each)
# ===========================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
xlim = (-R - 0.5, R + 0.5)
for ax, (video, stem, treat, tcol, is_low) in zip(axes, DISHES):
    # the 2 animals of a dish: two distinguishable shades of the treatment color
    shades = two_shades(tcol)
    n_animals = 0
    for li, lname in enumerate(["L0", "L1"]):
        key = f"{stem}_{lname}"
        if key not in ent:
            continue
        v = ent[key]
        ax.scatter(v["cx"], v["cy"], s=0.3, alpha=0.16, color=shades[li])
        n_animals += 1
    # outer 3.5 cm dish boundary = solid thin black; inner annulus = dashed grey
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="black", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), 0.7 * R, fill=False,
                            color="#666666", ls="--", lw=1.0))
    ax.plot(0, 0, "x", color="black", ms=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (cm)")
    ax.set_title(f"{treat} (n={n_animals})", fontsize=10)
    if n_animals == 2:
        ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=6, color=shades[0], label="animal 1"),
                           Line2D([0], [0], marker="o", ls="", ms=6, color=shades[1], label="animal 2")],
                  loc="upper right", fontsize=7, framealpha=0.9)
axes[0].set_ylabel("y (cm)")
fig.suptitle("Centroid clouds per dish (the 2 animals shown as two shades of the treatment color)\n"
             "outer solid circle = 3.5 cm dish boundary; dashed grey = inner annulus (1.225 cm)",
             y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.97])
f1 = f"{OUT}/thigmotaxis_arena_scatter_juvenile.png"
fig.savefig(f1, dpi=130, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 2: thigmotaxis_radial_timeline.png  (one AVERAGED line per treatment)
# ===========================================================================
fig, ax = plt.subplots(figsize=(12, 6))
for i, (video, stem, treat, tcol, is_low) in enumerate(DISHES):
    keys = [f"{stem}_{ln}" for ln in ("L0", "L1") if f"{stem}_{ln}" in ent]
    if not keys:
        continue
    n = min(len(ent[k]["frames"]) for k in keys)
    ts = ent[keys[0]]["frames"][:n] / FPS / 60.0
    stack = np.vstack([ent[k]["r_cm"][:n] for k in keys])
    r_mean = np.nanmean(stack, axis=0)
    lbl = f"{treat}{lc_tag(is_low)}"
    ax.plot(ts, r_mean, color=tcol, lw=1.4, alpha=0.95, label=lbl,
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.axhline(R, color="k", ls=":", lw=0.8, label="dish wall (1.75 cm)")
ax.set_ylim(0, R + 0.3)
ax.set_xlabel("time (min)")
ax.set_ylabel("radial position r (cm)")
ax.set_title("Thigmotaxis: radial distance from arena center over time,\n"
             "averaged per treatment (mean of the 2 leeches)")
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()
f2 = f"{OUT}/thigmotaxis_radial_timeline_juvenile.png"
fig.savefig(f2, dpi=130, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 3: spaceuse_exploration_curves.png  (one AVERAGED line per treatment)
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for i, (video, stem, treat, tcol, is_low) in enumerate(DISHES):
    keys = [f"{stem}_{ln}" for ln in ("L0", "L1") if f"{stem}_{ln}" in ent]
    if not keys:
        continue
    n = min(len(ent[k]["exp_t"]) for k in keys)
    tt = ent[keys[0]]["exp_t"][:n]
    stack = np.vstack([ent[k]["exp_cum"][:n] for k in keys])
    cum_mean = np.nanmean(stack, axis=0)
    lbl = f"{treat}{lc_tag(is_low)}"
    ax.plot(tt / 60.0, cum_mean, color=tcol, lw=2, alpha=0.95, label=lbl,
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.set_xlabel("time (min)")
ax.set_ylabel("cumulative area explored (cm$^2$)")
ax.set_title("Exploration curves: cumulative unique area vs time, averaged per treatment\n"
             "(mean of the 2 leeches per treatment; centroid, 0.25 cm cells)")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
f3 = f"{OUT}/spaceuse_exploration_curves_juvenile.png"
fig.savefig(f3, dpi=130)
plt.close(fig)

# ===========================================================================
# FIGURE 4: spaceuse_area_barchart.png  (one AVERAGED bar per treatment)
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
xticks, xlabels, heights, ns, bar_colors = [], [], [], [], []
for gi, (video, stem, treat, tcol, is_low) in enumerate(DISHES):
    vals = [ent[f"{stem}_{ln}"]["area_grid_cm2"]
            for ln in ("L0", "L1") if f"{stem}_{ln}" in ent]
    heights.append(float(np.mean(vals)))
    ns.append(len(vals))
    xticks.append(gi)
    xlabels.append(f"{treat}{lc_tag(is_low)}")
    bar_colors.append(tcol)
maxv = max(heights)
ax.bar(xticks, heights, 0.6, color=bar_colors, edgecolor="black", linewidth=0.6)
for p, hgt in zip(xticks, heights):
    ax.text(p, hgt, f"{hgt:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=9)
ax.tick_params(axis="x", length=0, pad=4)
for p, n in zip(xticks, ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -52),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("area explored (cm$^2$, 0.25 cm grid)")
ax.set_ylim(0, maxv * 1.15)
ax.set_title("Area explored averaged per treatment (one bar per treatment)")
fig.tight_layout()
f4 = f"{OUT}/spaceuse_area_barchart_juvenile.png"
fig.savefig(f4, dpi=130)
plt.close(fig)

# ===========================================================================
# FIGURE 5: spaceuse_occupancy_heatmaps.png  (ONE PANEL PER TREATMENT, 4)
# ===========================================================================
edges = np.arange(-R, R + HM_CELL_CM, HM_CELL_CM)
# treatment -> mean dwell-count grid across its 2 leeches
hm_treat_hist = []
vmax = 1
for video, stem, treat, tcol, is_low in DISHES:
    keys = [f"{stem}_{ln}" for ln in ("L0", "L1") if f"{stem}_{ln}" in ent]
    grids = []
    for key in keys:
        v = ent[key]
        H, _, _ = np.histogram2d(v["cxv"], v["cyv"], bins=[edges, edges])
        grids.append(H)
    Hmean = np.mean(grids, axis=0)
    hm_treat_hist.append((treat, Hmean, is_low))
    vmax = max(vmax, Hmean.max())

fig, axs = plt.subplots(1, len(hm_treat_hist),
                        figsize=(3.2 * len(hm_treat_hist) + 1.2, 3.7))
norm = LogNorm(vmin=1, vmax=vmax)  # shared color scale (log), as before
im = None
for ax, (treat, Hmean, is_low) in zip(axs, hm_treat_hist):
    H = Hmean.T
    Hm = np.ma.masked_less(H, 1)
    im = ax.pcolormesh(edges, edges, Hm, norm=norm, cmap=HEATMAP_CMAP)
    ax.add_patch(mpatches.Circle((0, 0), R, fill=False, color="black", lw=1.0))
    ax.set_aspect("equal")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel("cm")
    ax.set_title(f"{treat}", fontsize=9)
axs[0].set_ylabel("cm")
cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
cbar.set_label("dwell (frames, log)")
fig.suptitle("Occupancy within the 3.5 cm dish: averaged per treatment, shared log "
             "color scale, 0.10 cm grid (YlOrBr)", fontsize=12)
f5 = f"{OUT}/spaceuse_occupancy_heatmaps_juvenile.png"
fig.savefig(f5, dpi=130, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 6: kinematics_total_head_path_bar.png  (per-treatment, recolored)
# ===========================================================================
kin = pd.read_csv(f"{MET}/kinematics_summary.csv")
kin["cell"] = kin["treatment"].str.rsplit(" L", n=1).str[0]
# preserve dish order, map to treatment + low-conf flag
TREAT_ORDER = [(t, c, l) for (_v, _s, t, c, l) in DISHES]
vals, labels, ns, bar_colors = [], [], [], []
for treat, tcol, is_low in TREAT_ORDER:
    # treatment label in CSV is "Dopamine", "DA+Food", "Veh+NoFood", "Veh+Food"
    csv_name = {"Dopamine (DA+NoFood)": "Dopamine"}.get(treat, treat)
    sub = kin[kin["cell"] == csv_name]
    vals.append(float(sub["total_head_path_cm"].mean()))
    ns.append(int(sub["total_head_path_cm"].notna().sum()))
    labels.append(treat + lc_tag(is_low))
    bar_colors.append(tcol)

fig, ax = plt.subplots(figsize=(9, 5))
positions = list(range(len(labels)))
bars = ax.bar(positions, vals, 0.6, color=bar_colors, edgecolor="black",
              linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax.tick_params(axis="x", pad=4)
for p, n in zip(positions, ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -64),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("total HEAD path (cm)")
ax.set_ylim(0, max(vals) * 1.15)
ax.set_title("Kinematics: total head path averaged per treatment\n"
             "(mean of the 2 leeches per treatment, whole clip)")
fig.tight_layout()
f6 = f"{OUT}/kinematics_total_head_path_bar_juvenile.png"
fig.savefig(f6, dpi=130)
plt.close(fig)

# ===========================================================================
# FIGURE 7: design_2x2_effects.png  (recolored Veh grey / DA orange)
# ===========================================================================
CELL = {
    "Dopamine":   ("DA",  "NoFood"),
    "DA+Food":    ("DA",  "Food"),
    "Veh+NoFood": ("Veh", "NoFood"),
    "Veh+Food":   ("Veh", "Food"),
}
VEH_C, DA_C = "#f2d24b", "#8c6d00"  # light gold (Veh) / dark gold (DA)


def base_treat(s):
    return str(s).rsplit(" L", 1)[0]


def cell_mean(csv, value_col, ratio_den=None):
    d = pd.read_csv(f"{MET}/{csv}")
    d["cell"] = d["treatment"].map(base_treat)
    if ratio_den is not None:
        d["_v"] = d[value_col] / d[ratio_den]
    else:
        d["_v"] = d[value_col]
    return d.groupby("cell")["_v"].mean().to_dict()


pct_moving = cell_mean("kinematics_summary.csv", "pct_time_moving")
headtail = cell_mean("kinematics_summary.csv", "total_head_path_cm", ratio_den="total_tail_path_cm")
area = cell_mean("spaceuse_metrics.csv", "area_grid_cm2")
turning = cell_mean("turning_metrics.csv", "turning_per_min_deg")

# n per cell = number of leeches (entities) per base treatment
_kn = pd.read_csv(f"{MET}/kinematics_summary.csv")
_kn["cell"] = _kn["treatment"].map(base_treat)
CELL_N = _kn.groupby("cell").size().to_dict()


def _n_for(drug, food):
    name = next(n for n, c in CELL.items() if c == (drug, food))
    return int(CELL_N.get(name, 0))

PANELS = [
    (pct_moving, "% time moving"),
    (headtail, "head:tail path (foraging probe)"),
    (area, "area explored (cm2)"),
    (turning, "turning (deg/min)"),
]
foods = ["NoFood", "Food"]
centers = np.array([0.0, 1.0])
w = 0.4

fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (data, title) in zip(axes, PANELS):
    for di, (drug, col) in enumerate([("Veh", VEH_C), ("DA", DA_C)]):
        vals = []
        for food in foods:
            name = next(n for n, c in CELL.items() if c == (drug, food))
            vals.append(data.get(name, np.nan))
        xpos = centers + (di - 0.5) * w
        ax.bar(xpos, vals, w, color=col,
               label="Vehicle" if drug == "Veh" else "Dopamine")
        for x, v in zip(xpos, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3g}", (x, v), ha="center", va="bottom",
                            fontsize=9, xytext=(0, 2),
                            textcoords="offset points")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([centers[0] - w / 2, centers[0] + w / 2,
                   centers[1] - w / 2, centers[1] + w / 2])
    # x-tick order: [NoFood-Veh, NoFood-DA, Food-Veh, Food-DA]
    ax.set_xticklabels([
        f"Veh\nn={_n_for('Veh', 'NoFood')}",
        f"DA\nn={_n_for('DA', 'NoFood')}",
        f"Veh\nn={_n_for('Veh', 'Food')}",
        f"DA\nn={_n_for('DA', 'Food')}",
    ])
    ax.tick_params(axis="x", length=0)
    ax.text(0.25, -0.26, "NoFood", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax.text(0.75, -0.26, "Food", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax.set_ylim(0, np.nanmax(list(data.values())) * 1.18)  # headroom for value labels
    ax.set_title(title, fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
fig.suptitle("2x2 design: dopamine x food effects (juveniles; each cell = mean of 2 leeches)",
             fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
f7 = f"{OUT}/design_2x2_effects_juvenile.png"
fig.savefig(f7, dpi=130)
plt.close(fig)

for p in (f1, f2, f3, f4, f5, f6, f7):
    print("wrote", p)
print("DONE")
