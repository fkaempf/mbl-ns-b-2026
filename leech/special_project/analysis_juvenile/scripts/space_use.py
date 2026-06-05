import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = f"{BASE}/analysis_juvenile/plots"
MET = f"{BASE}/analysis_juvenile/metrics"
CSV = f"{BASE}/DL/juv_all_predictions.csv"
CAL = f"{BASE}/DL/juv_calibration.json"
FPS = 29.99
WIN = 11
CELL_CM = 0.25  # occupancy cell size
DISH_RADIUS_CM = 1.75  # 3.5 cm dish

with open(CAL) as fh:
    cal = json.load(fh)

# 4 videos x 2 tracks = 8 entities. Video keys in CSV carry ".mp4".
VIDEOS = ["Video1_dish0", "Video1_dish1", "Video2_dish0", "Video2_dish1"]
TREATMENT = {
    "Video1_dish0": "Dopamine",
    "Video1_dish1": "Dopamine+Food",
    "Video2_dish0": "Food",
    "Video2_dish1": "Veh+Food",
}
# treatment colour grouping
TREAT_COLOR = {
    "Dopamine": "C3",
    "Dopamine+Food": "C1",
    "Food": "C2",
    "Veh+Food": "C0",
}

print("loading...")
df = pd.read_csv(CSV)
print(df.video.unique(), df.shape)
df["vid"] = df.video.str.replace(".mp4", "", regex=False)

df = df.pivot_table(index=["vid", "track", "frame", "time_s"],
                    columns="node", values=["x", "y"]).reset_index()
df.columns = ["vid", "track", "frame", "time_s", "x0", "x1", "y0", "y1"]
df = df.sort_values(["vid", "track", "frame"])

# pixel centroid = midpoint anterior(node0)/posterior(node1)
df["cx_px"] = (df.x0 + df.x1) / 2.0
df["cy_px"] = (df.y0 + df.y1) / 2.0


def smooth(a):
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


groups = []  # (vid, track, name)
for v in VIDEOS:
    for trk in (0, 1):
        groups.append((v, trk, f"{v}_L{trk}"))
order = [g[2] for g in groups]

DISPLAY = {}
LOW = set()
for v, trk, name in groups:
    treat = TREATMENT[v]
    lc = cal[v]["pck"] < 60
    DISPLAY[name] = f"{treat} L{trk}" + (" (low-conf)" if lc else "")
    if lc:
        LOW.add(name)

results = {}
expcurves = {}
cmcoords = {}
entity_treat = {}

for v, trk, name in groups:
    sub = df[(df.vid == v) & (df.track == trk)].sort_values("frame")
    if len(sub) < 100:
        continue
    c = cal[v]
    cmpp = c["cmpp"]
    # smooth pixel coords, convert to cm relative to arena center
    cx = (smooth(sub.cx_px.values) - c["cx"]) * cmpp
    cy = (smooth(sub.cy_px.values) - c["cy"]) * cmpp
    t = sub.time_s.values
    valid = np.isfinite(cx) & np.isfinite(cy)
    cx, cy, t = cx[valid], cy[valid], t[valid]
    cmcoords[name] = (cx, cy)
    entity_treat[name] = TREATMENT[v]

    # arena area from this video's calibration (dish radius in cm)
    arena_area = np.pi * (cmpp * c["r_px"]) ** 2

    # occupancy grid in cm (fixed 0.25 cm cells)
    gx = np.floor(cx / CELL_CM).astype(np.int64)
    gy = np.floor(cy / CELL_CM).astype(np.int64)
    cellid = gx * 100000 + gy

    seen = set()
    cum_cells = np.empty(len(cellid), dtype=np.int64)
    for i, cid in enumerate(cellid):
        seen.add(cid)
        cum_cells[i] = len(seen)
    unique_cells = int(cum_cells[-1])
    cell_area = CELL_CM * CELL_CM
    cum_area_cm2 = cum_cells * cell_area
    area_grid_cm2 = unique_cells * cell_area
    expcurves[name] = (t, cum_area_cm2)

    pts = np.column_stack([cx, cy])
    try:
        hull = ConvexHull(pts)
        hull_area_cm2 = float(hull.volume)
    except Exception:
        hull_area_cm2 = np.nan

    results[name] = dict(
        video=v,
        treatment=TREATMENT[v],
        leech=f"L{trk}",
        low_conf=name in LOW,
        n_frames=int(len(cx)),
        area_grid_cm2=float(area_grid_cm2),
        hull_area_cm2=float(hull_area_cm2),
        arena_area_cm2=float(arena_area),
        pct_arena_grid=float(100.0 * area_grid_cm2 / arena_area),
        pct_arena_hull=float(100.0 * hull_area_cm2 / arena_area),
        range_x_cm=float(cx.max() - cx.min()),
        range_y_cm=float(cy.max() - cy.min()),
    )
    print(name, {k: results[name][k] for k in
                 ("area_grid_cm2", "hull_area_cm2", "pct_arena_grid")})

names = [n for n in order if n in results]
arena_full = np.pi * DISH_RADIUS_CM ** 2  # nominal 9 cm dish

# ---------------------------------------------------------------------------
# FIGURE 1: exploration curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
seen_treat = set()
for name in names:
    t, cum = expcurves[name]
    treat = entity_treat[name]
    col = TREAT_COLOR[treat]
    ls = "--" if name in LOW else "-"
    lbl = DISPLAY[name]
    ax.plot(t / 60.0, cum, ls, color=col, lw=2, label=lbl)
ax.axhline(arena_full, color="k", ls=":", lw=1.2,
           label=f"full arena ({arena_full:.1f} cm$^2$)")
ax.set_xlabel("time (min)")
ax.set_ylabel("cumulative area explored (cm$^2$)")
ax.set_title("Juvenile exploration: cumulative unique area vs time (centroid, 0.25 cm cells)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
f1 = f"{OUT}/space_exploration_curves.png"
fig.savefig(f1, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2: grouped bars per entity, grid + hull area
# ---------------------------------------------------------------------------
disp = [DISPLAY[n].replace(" (low-conf)", "\n(low-conf)") for n in names]
bar_colors = [TREAT_COLOR[entity_treat[n]] for n in names]
grid_vals = [results[n]["area_grid_cm2"] for n in names]
hull_vals = [results[n]["hull_area_cm2"] for n in names]
x = np.arange(len(names))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w / 2, grid_vals, w, color=bar_colors, label="occupancy-grid area")
b2 = ax.bar(x + w / 2, hull_vals, w, color=bar_colors, alpha=0.5,
            hatch="//", label="convex hull area")
ax.axhline(arena_full, color="k", ls=":", lw=1.2,
           label=f"full arena ({arena_full:.1f} cm$^2$)")
ax.set_xticks(x)
ax.set_xticklabels(disp, fontsize=8)
ax.set_ylabel("area (cm$^2$)")
ax.set_title("Juvenile space use per leech: occupancy-grid (solid) vs convex hull (hatched)")
for rects, vals in ((b1, grid_vals), (b2, hull_vals)):
    for r, val in zip(rects, vals):
        ax.text(r.get_x() + r.get_width() / 2, val, f"{val:.1f}",
                ha="center", va="bottom", fontsize=7)
ax.legend(fontsize=9)
fig.tight_layout()
f2 = f"{OUT}/space_area_bars.png"
fig.savefig(f2, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 3: occupancy heatmaps, real cm, 8 panels, shared log colorbar
# ---------------------------------------------------------------------------
edges = np.arange(-DISH_RADIUS_CM, DISH_RADIUS_CM + CELL_CM, CELL_CM)
hists = {}
vmax = 1
for n in names:
    cx, cy = cmcoords[n]
    H, _, _ = np.histogram2d(cx, cy, bins=[edges, edges])
    hists[n] = H
    vmax = max(vmax, H.max())

ncol = 4
nrow = int(np.ceil(len(names) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow))
axs = np.atleast_2d(axs).ravel()
norm = LogNorm(vmin=1, vmax=vmax)
im = None
for ax, n in zip(axs, names):
    H = hists[n].T
    Hm = np.ma.masked_less(H, 1)
    im = ax.pcolormesh(edges, edges, Hm, norm=norm, cmap="magma")
    ax.add_patch(mpatches.Circle((0, 0), DISH_RADIUS_CM, fill=False, color="c", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xlim(-DISH_RADIUS_CM, DISH_RADIUS_CM)
    ax.set_ylim(-DISH_RADIUS_CM, DISH_RADIUS_CM)
    ax.set_xlabel("cm")
    ax.set_ylabel("cm")
    ax.set_title(DISPLAY[n], fontsize=9)
for ax in axs[len(names):]:
    ax.axis("off")
cbar = fig.colorbar(im, ax=axs.tolist(), fraction=0.025, pad=0.02)
cbar.set_label("dwell (frames, log)")
fig.suptitle("Juvenile occupancy within 3.5 cm dish: real cm scale, shared colorbar (8 leeches)",
             fontsize=13)
f3 = f"{OUT}/space_occupancy_heatmaps.png"
fig.savefig(f3, dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------
rows = []
for n in names:
    r = results[n]
    rows.append(dict(
        entity=n,
        video=r["video"],
        treatment=r["treatment"],
        leech=r["leech"],
        low_conf=r["low_conf"],
        n_frames=r["n_frames"],
        area_grid_cm2=round(r["area_grid_cm2"], 4),
        hull_area_cm2=round(r["hull_area_cm2"], 4),
        pct_arena_grid=round(r["pct_arena_grid"], 3),
        pct_arena_hull=round(r["pct_arena_hull"], 3),
        range_x_cm=round(r["range_x_cm"], 4),
        range_y_cm=round(r["range_y_cm"], 4),
    ))
mdf = pd.DataFrame(rows)
mcsv = f"{MET}/space_use_metrics.csv"
mdf.to_csv(mcsv, index=False)

print("FIGS", f1, f2, f3)
print("CSV", mcsv)
print(mdf.to_string(index=False))
print("DONE")
