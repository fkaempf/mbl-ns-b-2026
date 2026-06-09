#!/usr/bin/env python
"""Curated "final figures" set for the ADULT cohort.

Regenerates 7 figures into analysis_fed_leech/plots closer selection/ using a
colorblind-safe (Okabe-Ito) palette and dish-grouped layouts. Data computation
and arena calibration are reused VERBATIM from the existing analysis scripts:
  thigmotaxis_analysis.py, space_use_exploration.py, kinematics_analysis.py,
  and the 2x2 design logic (juvenile design_2x2.py) reading metrics CSVs.

ADULT entities -> dish -> treatment:
  dish1 = Veh+Food   (1 animal)
  dish2 = DA+Food    (1 animal)
  dish3 = Veh+NoFood (1 animal)
  dish4 = DA+NoFood  (2 animals: dish4_leech0 = L0, dish4_leech1 = L1; low-conf)

No em dashes anywhere in any plot text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter


def two_shades(c):
    """Two distinguishable shades of a base color (for the 2 animals in a dish)."""
    r, g, b = to_rgb(c)
    return c, (r * 0.45, g * 0.45, b * 0.45)

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
CSV = f"{BASE}/DL/all_predictions.csv"
MET = f"{BASE}/analysis_fed_leech/metrics"
OUT = f"{BASE}/plots closer selection"
os.makedirs(OUT, exist_ok=True)

FPS = 29.99
DT = 1.0 / FPS
R = 4.5         # dish physical radius in cm
CELL_CM = 0.25  # occupancy cell size

# ---- Per-treatment blue shades (adult cohort = blue hue) ----
TREAT_COLOR = {
    "Veh+Food":   "#c6dbef",   # very light blue
    "DA+Food":    "#6baed6",   # light blue
    "Veh+NoFood": "#2171b5",   # strong blue
    "DA+NoFood":  "#08306b",   # very dark navy
}
# Occupancy heatmap colormap for this cohort
HEATMAP_CMAP = "Blues"
# design_2x2 bar colors (unchanged)
VEH_BAR = "#9ecae1"     # light blue (Vehicle)
DA_BAR = "#08519c"      # dark blue (Dopamine)

# ---- Per-dish arena calibration (KNOWN; do not re-fit) ----
ARENA = {
    1: dict(cx=303, cy=295, cmpp=0.0200),
    2: dict(cx=290, cy=311, cmpp=0.0205),
    3: dict(cx=292, cy=284, cmpp=0.0202),
    4: dict(cx=311, cy=293, cmpp=0.0206),
}
VIDEO = {1: "IMG_2859_dish1.mp4", 2: "IMG_2859_dish2.mp4",
         3: "IMG_2859_dish3.mp4", 4: "IMG_2859_dish4.mp4"}

# 5 entities: dish1, dish2, dish3, dish4_leech0, dish4_leech1
ENTITIES = [(1, 0), (2, 0), (3, 0), (4, 0), (4, 1)]
ORDER = ["dish1", "dish2", "dish3", "dish4_leech0", "dish4_leech1"]
LOWCONF = {"dish4_leech0", "dish4_leech1"}

# entity -> treatment (base, no L0/L1)
ENT_TREAT = {
    "dish1": "Veh+Food",
    "dish2": "DA+Food",
    "dish3": "Veh+NoFood",
    "dish4_leech0": "DA+NoFood",
    "dish4_leech1": "DA+NoFood",
}
# dish -> (label, treatment) for arena-scatter panel titles
DISH_TREAT = {1: "Veh+Food", 2: "DA+Food", 3: "Veh+NoFood", 4: "DA+NoFood"}


def ent_color(name):
    return TREAT_COLOR[ENT_TREAT[name]]


# ---------------------------------------------------------------------------
# DATA COMPUTATION reused verbatim from thigmotaxis_analysis.py and
# space_use_exploration.py (same smoothing, same calibration math).
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV)
df["dish"] = df["video"].str.extract(r"dish(\d)").astype(int)


def smooth_thig(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


# thigmotaxis: centroid x/y in cm, radial position
thig = {}
for dish, track in ENTITIES:
    g = df[(df.dish == dish) & (df.track == track)]
    h = g[g.node == 0].sort_values("frame")
    t = g[g.node == 1].sort_values("frame")
    m = pd.merge(h[["frame", "x", "y"]], t[["frame", "x", "y"]],
                 on="frame", suffixes=("_h", "_t"))
    if len(m) < 100:
        continue
    a = ARENA[dish]
    cmpp = a["cmpp"]
    hx = (smooth_thig(m.x_h.to_numpy()) - a["cx"]) * cmpp
    hy = (smooth_thig(m.y_h.to_numpy()) - a["cy"]) * cmpp
    tx = (smooth_thig(m.x_t.to_numpy()) - a["cx"]) * cmpp
    ty = (smooth_thig(m.y_t.to_numpy()) - a["cy"]) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0
    r_cm = np.sqrt(cx**2 + cy**2)
    key = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    thig[key] = dict(dish=dish, track=track, frames=m.frame.to_numpy(),
                     cx=cx, cy=cy, r_cm=r_cm, mean_r=float(np.mean(r_cm)))


# space use: exploration curves, occupancy, area-explored (reuse pivot pipeline)
WIN = 11


def smooth_space(a):
    a = np.asarray(a, float)
    if len(a) < WIN:
        return a
    nans = np.isnan(a)
    if nans.any():
        idx = np.arange(len(a))
        a = a.copy()
        if (~nans).any():
            a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return savgol_filter(a, WIN, 2)


dfp = df.pivot_table(index=["video", "track", "frame", "time_s"],
                     columns="node", values=["x", "y"]).reset_index()
dfp.columns = ["video", "track", "frame", "time_s", "x0", "x1", "y0", "y1"]
dfp = dfp.sort_values(["video", "track", "frame"])
dfp["cx_px"] = (dfp.x0 + dfp.x1) / 2.0
dfp["cy_px"] = (dfp.y0 + dfp.y1) / 2.0

space_results = {}
expcurves = {}
cmcoords = {}
for dish, track in ENTITIES:
    v = VIDEO[dish]
    name = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    sub = dfp[(dfp.video == v) & (dfp.track == track)].sort_values("frame")
    if len(sub) < 100:
        continue
    d = ARENA[dish]
    cx = (smooth_space(sub.cx_px.values) - d["cx"]) * d["cmpp"]
    cy = (smooth_space(sub.cy_px.values) - d["cy"]) * d["cmpp"]
    tt = sub.time_s.values
    valid = np.isfinite(cx) & np.isfinite(cy)
    cx, cy, tt = cx[valid], cy[valid], tt[valid]
    cmcoords[name] = (cx, cy)

    gx = np.floor(cx / CELL_CM).astype(np.int64)
    gy = np.floor(cy / CELL_CM).astype(np.int64)
    cellid = gx * 100000 + gy
    seen = set()
    cum_cells = np.empty(len(cellid), dtype=np.int64)
    for i, c in enumerate(cellid):
        seen.add(c)
        cum_cells[i] = len(seen)
    unique_cells = int(cum_cells[-1])
    cell_area = CELL_CM * CELL_CM
    cum_area_cm2 = cum_cells * cell_area
    area_grid_cm2 = unique_cells * cell_area
    expcurves[name] = (tt, cum_area_cm2)
    space_results[name] = dict(area_grid_cm2=float(area_grid_cm2))


# kinematics: total head path (reuse path computation)
def smooth_kin(arr):
    n = len(arr)
    if n < 11:
        return arr
    return savgol_filter(arr, 11, 2)


kin = {}
for dish, track in ENTITIES:
    name = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    sub = df[(df.dish == dish) & (df.track == track)]
    if len(sub) < 1000:
        continue
    cal = ARENA[dish]
    piv = sub.pivot_table(index="frame", columns="node",
                          values=["x", "y"]).sort_index()

    def to_cm(px, c):
        return (px - c) * cal["cmpp"]

    hx = smooth_kin(to_cm(piv[("x", 0)].to_numpy(), cal["cx"]))
    hy = smooth_kin(to_cm(piv[("y", 0)].to_numpy(), cal["cy"]))
    dx = np.diff(hx)
    dy = np.diff(hy)
    total_head_cm = float(np.sqrt(dx**2 + dy**2).sum())
    kin[name] = dict(total_head_cm=total_head_cm)


# ===========================================================================
# FIGURE 1: thigmotaxis_arena_scatter.png  -- one panel PER DISH (4 panels)
# ===========================================================================
DISHES = [1, 2, 3, 4]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
lim = R + 0.5
for ax, dish in zip(axes, DISHES):
    treat = DISH_TREAT[dish]
    col = TREAT_COLOR[treat]
    if dish == 4:
        # the 2 animals: two distinguishable shades of the treatment color
        shades = two_shades(col)
        n_animals = 0
        for li, name in enumerate(["dish4_leech0", "dish4_leech1"]):
            if name not in thig:
                continue
            v = thig[name]
            ax.scatter(v["cx"], v["cy"], s=0.3, alpha=0.16, color=shades[li])
            n_animals += 1
        if n_animals == 2:
            ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=6, color=shades[0], label="animal 1"),
                               Line2D([0], [0], marker="o", ls="", ms=6, color=shades[1], label="animal 2")],
                      loc="upper right", fontsize=7, framealpha=0.9)
    else:
        v = thig[f"dish{dish}"]
        ax.scatter(v["cx"], v["cy"], s=0.2, alpha=0.12, color=col)
        n_animals = 1
    # colorblind-safe annuli: outer dish = solid black thin; inner = dashed grey
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="black", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), 0.7 * R, fill=False, color="#666666",
                            ls="--", lw=1.0))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (cm)")
    ax.set_title(f"{treat} (n={n_animals})", fontsize=10)
axes[0].set_ylabel("y (cm)")
fig.suptitle("Centroid cloud in cm: 4.5 cm dish boundary (black), "
             "inner annulus 3.15 cm (dashed grey); one panel per treatment")
fig.tight_layout()
f1 = f"{OUT}/thigmotaxis_arena_scatter_adult.png"
fig.savefig(f1, dpi=120, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 2: thigmotaxis_radial_timeline.png  -- one AVERAGED line per treatment
# ===========================================================================
# treatment -> list of entity names (in plot order)
TREAT_ENTS = [
    ("Veh+Food",   ["dish1"]),
    ("DA+Food",    ["dish2"]),
    ("Veh+NoFood", ["dish3"]),
    ("DA+NoFood",  ["dish4_leech0", "dish4_leech1"]),
]


def mean_by_frame(series_list):
    """Mean of several (frames, values) aligned by frame index."""
    longest = max((len(s[0]) for s in series_list), default=0)
    n = min((len(s[0]) for s in series_list), default=0)
    # align by position (frame index); use shortest common length
    ts = series_list[0][0][:n] / FPS / 60.0
    stack = np.vstack([s[1][:n] for s in series_list])
    return ts, np.nanmean(stack, axis=0)


fig, ax = plt.subplots(figsize=(11, 5))
for i, (treat, names) in enumerate(TREAT_ENTS):
    names = [nm for nm in names if nm in thig]
    if not names:
        continue
    series = [(thig[nm]["frames"], thig[nm]["r_cm"]) for nm in names]
    ts, r_mean = mean_by_frame(series)
    col = TREAT_COLOR[treat]
    low = any(nm in LOWCONF for nm in names)
    leg = treat
    ax.plot(ts, r_mean, lw=1.4, color=col, label=leg,
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.axhline(R, color="k", ls=":", lw=0.8, label="dish wall (4.5 cm)")
ax.set_ylim(0, R + 0.3)
ax.set_xlabel("time (min)")
ax.set_ylabel("radial position r (cm)")
ax.set_title("Thigmotaxis: radial distance from arena center over time, "
             "averaged per treatment (cm)")
leg = ax.legend(fontsize=8, loc="upper right")
for line in leg.get_lines():
    line.set_linewidth(2.0)
fig.tight_layout()
f2 = f"{OUT}/thigmotaxis_radial_timeline_adult.png"
fig.savefig(f2, dpi=120, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 3: spaceuse_exploration_curves.png  -- one AVERAGED line per treatment
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 6))
for i, (treat, names) in enumerate(TREAT_ENTS):
    names = [nm for nm in names if nm in expcurves]
    if not names:
        continue
    series = [(expcurves[nm][0], expcurves[nm][1]) for nm in names]
    n = min(len(s[0]) for s in series)
    tt = series[0][0][:n]
    stack = np.vstack([s[1][:n] for s in series])
    cum_mean = np.nanmean(stack, axis=0)
    col = TREAT_COLOR[treat]
    low = any(nm in LOWCONF for nm in names)
    lbl = treat
    ax.plot(tt / 60.0, cum_mean, color=col, label=lbl, lw=2,
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.set_xlabel("time (min)")
ax.set_ylabel("cumulative area explored (cm$^2$)")
ax.set_title("Exploration curves: cumulative unique area vs time, "
             "averaged per treatment (centroid, 0.25 cm cells)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
f3 = f"{OUT}/spaceuse_exploration_curves_adult.png"
fig.savefig(f3, dpi=120)
plt.close(fig)

# ===========================================================================
# FIGURE 4: spaceuse_area_barchart.png  -- one AVERAGED bar per treatment
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
# 4 treatments: dish1/2/3 single animal; dish4 = mean of its 2 leeches
TREAT_BARS = [
    ("Veh+Food",   "Veh+Food",   [space_results["dish1"]["area_grid_cm2"]]),
    ("DA+Food",    "DA+Food",    [space_results["dish2"]["area_grid_cm2"]]),
    ("Veh+NoFood", "Veh+NoFood", [space_results["dish3"]["area_grid_cm2"]]),
    ("DA+NoFood",  "DA+NoFood",  [space_results["dish4_leech0"]["area_grid_cm2"],
                                  space_results["dish4_leech1"]["area_grid_cm2"]]),
]
labels = [t for t, _, _ in TREAT_BARS]
heights = [float(np.mean(v)) for _, _, v in TREAT_BARS]
ns = [len(v) for _, _, v in TREAT_BARS]
bar_colors = [TREAT_COLOR[tr] for _, tr, _ in TREAT_BARS]
positions = list(range(len(TREAT_BARS)))

ax.bar(positions, heights, width=0.6, color=bar_colors, edgecolor="black",
       linewidth=0.6)
# value labels on top
for p, hgt in zip(positions, heights):
    ax.text(p, hgt, f"{hgt:.1f}", ha="center", va="bottom", fontsize=9)
ax.axhline(np.pi * R ** 2, color="k", ls=":", lw=1, label="full arena (63.6 cm$^2$)")
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
# n= labels just below each (two-line) treatment label
ax.tick_params(axis="x", pad=4)
for p, n in zip(positions, ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -52),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("area explored (cm$^2$, 0.25 cm cells)")
ax.set_ylim(0, max(max(heights), np.pi * R ** 2) * 1.1)
ax.set_title("Space use averaged per treatment (one bar per treatment)")
ax.legend(fontsize=9)
fig.tight_layout()
f4 = f"{OUT}/spaceuse_area_barchart_adult.png"
fig.savefig(f4, dpi=120)
plt.close(fig)

# ===========================================================================
# FIGURE 5: spaceuse_occupancy_heatmaps.png  -- ONE PANEL PER TREATMENT (4)
# ===========================================================================
edges = np.arange(-R, R + CELL_CM, CELL_CM)
# treatment -> mean dwell-count grid across its animals
HM_TREAT = [
    ("Veh+Food",   ["dish1"]),
    ("DA+Food",    ["dish2"]),
    ("Veh+NoFood", ["dish3"]),
    ("DA+NoFood",  ["dish4_leech0", "dish4_leech1"]),
]
hm_treat_hist = []
vmax = 1
for treat, names in HM_TREAT:
    names = [nm for nm in names if nm in cmcoords]
    grids = []
    for nm in names:
        cx, cy = cmcoords[nm]
        H, _, _ = np.histogram2d(cx, cy, bins=[edges, edges])
        grids.append(H)
    Hmean = np.mean(grids, axis=0)
    low = any(nm in LOWCONF for nm in names)
    hm_treat_hist.append((treat, Hmean, low))
    vmax = max(vmax, Hmean.max())

fig, axs = plt.subplots(1, len(hm_treat_hist),
                        figsize=(3.2 * len(hm_treat_hist) + 1.2, 3.6))
norm = LogNorm(vmin=1, vmax=vmax)  # shared color scale
im = None
for ax, (treat, Hmean, low) in zip(axs, hm_treat_hist):
    H = Hmean.T
    Hm = np.ma.masked_less(H, 1)
    im = ax.pcolormesh(edges, edges, Hm, norm=norm, cmap=HEATMAP_CMAP)
    ax.add_patch(mpatches.Circle((0, 0), R, fill=False, color="black", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel("cm")
    ttl = treat
    ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel("cm")
cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
cbar.set_label("dwell (frames, log)")
fig.suptitle("Occupancy within the 4.5 cm dish: real cm, shared log color scale, "
             "averaged per treatment", fontsize=12)
f5 = f"{OUT}/spaceuse_occupancy_heatmaps_adult.png"
fig.savefig(f5, dpi=120, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 6: kinematics_total_head_path_bar.png  -- one AVERAGED bar per treatment
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
KIN_TREAT_BARS = [
    ("Veh+Food",   "Veh+Food",   [kin["dish1"]["total_head_cm"]]),
    ("DA+Food",    "DA+Food",    [kin["dish2"]["total_head_cm"]]),
    ("Veh+NoFood", "Veh+NoFood", [kin["dish3"]["total_head_cm"]]),
    ("DA+NoFood",  "DA+NoFood",  [kin["dish4_leech0"]["total_head_cm"],
                                  kin["dish4_leech1"]["total_head_cm"]]),
]
labels6 = [t for t, _, _ in KIN_TREAT_BARS]
vals6 = [float(np.mean(v)) for _, _, v in KIN_TREAT_BARS]
ns6 = [len(v) for _, _, v in KIN_TREAT_BARS]
bar_colors6 = [TREAT_COLOR[tr] for _, tr, _ in KIN_TREAT_BARS]
positions6 = list(range(len(KIN_TREAT_BARS)))
ax.bar(positions6, vals6, width=0.6, color=bar_colors6, edgecolor="black",
       linewidth=0.6)
for i, v in enumerate(vals6):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(positions6)
ax.set_xticklabels(labels6, fontsize=9)
ax.tick_params(axis="x", pad=4)
for p, n in zip(positions6, ns6):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -52),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("total HEAD path (cm)")
ax.set_ylim(0, max(vals6) * 1.1)
ax.set_title("Kinematics: total head path averaged per treatment (whole clip, cm)")
fig.tight_layout()
f6 = f"{OUT}/kinematics_total_head_path_bar_adult.png"
fig.savefig(f6, dpi=120)
plt.close(fig)

# ===========================================================================
# FIGURE 7: design_2x2_effects.png  -- adult 2x2, recolored
# ===========================================================================
CELL = {
    "DA+NoFood": ("DA",  "NoFood"),
    "DA+Food":   ("DA",  "Food"),
    "Veh+NoFood":("Veh", "NoFood"),
    "Veh+Food":  ("Veh", "Food"),
}


def base_treat(s):
    # treatment strings in CSVs are e.g. "DA+NoFood L0"; strip trailing " L0"/" L1"
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
headtail = cell_mean("kinematics_summary.csv", "total_head_path_cm",
                     ratio_den="total_tail_path_cm")
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
    for di, (drug, col) in enumerate([("Veh", VEH_BAR), ("DA", DA_BAR)]):
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
    ax.text(0.25, -0.26, "NoFood", transform=ax.transAxes, ha="center",
            va="top", fontsize=11)
    ax.text(0.75, -0.26, "Food", transform=ax.transAxes, ha="center",
            va="top", fontsize=11)
    ax.set_ylim(0, np.nanmax(list(data.values())) * 1.18)  # headroom for value labels
    ax.set_title(title, fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
fig.suptitle("2x2 design: dopamine x food effects (adults; dish4 = mean of 2 leeches)",
             fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
f7 = f"{OUT}/design_2x2_effects_adult.png"
fig.savefig(f7, dpi=130)
plt.close(fig)

for f in [f1, f2, f3, f4, f5, f6, f7]:
    print("wrote", f)
print("DONE")
