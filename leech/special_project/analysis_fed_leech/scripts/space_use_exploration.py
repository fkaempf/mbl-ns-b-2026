import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter

OUT = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_fed_leech/plots"
MET = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_fed_leech/metrics"
CSV = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/DL/all_predictions.csv"
FPS = 29.99

# Per-dish arena calibration (KNOWN; do not re-fit)
DISH = {
    "IMG_2859_dish1.mp4": dict(cx=303.0, cy=295.0, cmpp=0.0200),
    "IMG_2859_dish2.mp4": dict(cx=290.0, cy=311.0, cmpp=0.0205),
    "IMG_2859_dish3.mp4": dict(cx=292.0, cy=284.0, cmpp=0.0202),
    "IMG_2859_dish4.mp4": dict(cx=311.0, cy=293.0, cmpp=0.0206),
}
RADIUS_CM = 4.5
CELL_CM = 0.25  # occupancy cell size

print("loading...")
df = pd.read_csv(CSV)
print(df.video.unique(), df.shape)

df = df.pivot_table(index=["video", "track", "frame", "time_s"],
                    columns="node", values=["x", "y"]).reset_index()
df.columns = ["video", "track", "frame", "time_s", "x0", "x1", "y0", "y1"]
df = df.sort_values(["video", "track", "frame"])

# pixel centroid = midpoint head/tail
df["cx_px"] = (df.x0 + df.x1) / 2.0
df["cy_px"] = (df.y0 + df.y1) / 2.0

WIN = 11

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

# 5 entities (each leech separate)
groups = [
    ("IMG_2859_dish1.mp4", 0, "dish1"),
    ("IMG_2859_dish2.mp4", 0, "dish2"),
    ("IMG_2859_dish3.mp4", 0, "dish3"),
    ("IMG_2859_dish4.mp4", 0, "dish4_leech0"),
    ("IMG_2859_dish4.mp4", 1, "dish4_leech1"),
]
order = [g[2] for g in groups]

results = {}
expcurves = {}
cmcoords = {}  # name -> (cx, cy) in cm, for occupancy heatmaps

for v, trk, name in groups:
    sub = df[(df.video == v) & (df.track == trk)].sort_values("frame")
    if len(sub) < 100:
        continue
    d = DISH[v]
    # smooth pixel coords first, then convert to cm relative to known center
    cx = (smooth(sub.cx_px.values) - d["cx"]) * d["cmpp"]
    cy = (smooth(sub.cy_px.values) - d["cy"]) * d["cmpp"]
    t = sub.time_s.values
    valid = np.isfinite(cx) & np.isfinite(cy)
    cx, cy, t = cx[valid], cy[valid], t[valid]
    cmcoords[name] = (cx, cy)

    # occupancy grid in cm (fixed 0.25 cm cells, global arena frame)
    gx = np.floor(cx / CELL_CM).astype(np.int64)
    gy = np.floor(cy / CELL_CM).astype(np.int64)
    cellid = gx * 100000 + gy

    # cumulative unique cells visited vs time -> cm^2
    seen = set()
    cum_cells = np.empty(len(cellid), dtype=np.int64)
    for i, c in enumerate(cellid):
        seen.add(c)
        cum_cells[i] = len(seen)
    unique_cells = int(cum_cells[-1])
    cell_area = CELL_CM * CELL_CM  # cm^2 per cell
    cum_area_cm2 = cum_cells * cell_area
    area_grid_cm2 = unique_cells * cell_area
    expcurves[name] = (t, cum_area_cm2)

    # convex hull area in cm^2
    pts = np.column_stack([cx, cy])
    try:
        hull = ConvexHull(pts)
        hull_area_cm2 = float(hull.volume)
    except Exception:
        hull_area_cm2 = np.nan

    arena_area = np.pi * RADIUS_CM ** 2  # 63.6 cm^2
    half = len(cum_cells) // 2
    sat_50 = float(cum_cells[half] / cum_cells[-1])

    results[name] = dict(
        n_frames=int(len(cx)),
        unique_cells=unique_cells,
        cell_cm=CELL_CM,
        area_grid_cm2=float(area_grid_cm2),
        hull_area_cm2=float(hull_area_cm2),
        pct_arena_grid=float(100.0 * area_grid_cm2 / arena_area),
        pct_arena_hull=float(100.0 * hull_area_cm2 / arena_area),
        sat_50=sat_50,
        range_x_cm=float(cx.max() - cx.min()),
        range_y_cm=float(cy.max() - cy.min()),
    )
    print(name, results[name])

low = {"dish4_leech0", "dish4_leech1"}
DISPLAY = {"dish1": "Veh+Food", "dish2": "DA+Food", "dish3": "Veh+NoFood",
           "dish4_leech0": "DA+NoFood L0", "dish4_leech1": "DA+NoFood L1"}
colors = {"dish1": "C0", "dish2": "C1", "dish3": "C2",
          "dish4_leech0": "C3", "dish4_leech1": "C4"}

# FIGURE 1: exploration curves (cm^2 vs time)
fig, ax = plt.subplots(figsize=(9, 6))
for name in order:
    if name not in expcurves:
        continue
    t, cum = expcurves[name]
    ls = "--" if name in low else "-"
    lbl = DISPLAY[name] + (" (low-conf)" if name in low else "")
    ax.plot(t / 60.0, cum, ls, color=colors[name], label=lbl, lw=2)
ax.axhline(np.pi * RADIUS_CM ** 2, color="k", ls=":", lw=1,
           label="full arena (63.6 cm$^2$)")
ax.set_xlabel("time (min)")
ax.set_ylabel("cumulative area explored (cm$^2$)")
ax.set_title("Exploration curves: cumulative unique area vs time (centroid, 0.25 cm cells)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
f1 = f"{OUT}/spaceuse_exploration_curves.png"
fig.savefig(f1, dpi=120)
plt.close(fig)

# FIGURE 2: bar chart (cm^2) grid + hull
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
names = [n for n in order if n in results]
bar_colors = [colors[n] for n in names]
disp = [DISPLAY[n] + ("\n(low-conf)" if n in low else "") for n in names]
grid_vals = [results[n]["area_grid_cm2"] for n in names]
hull_vals = [results[n]["hull_area_cm2"] for n in names]
axs[0].bar(disp, grid_vals, color=bar_colors)
axs[0].set_ylabel("area covered (cm$^2$)")
axs[0].set_title("Occupancy-grid area covered (0.25 cm cells)")
axs[0].tick_params(axis="x", rotation=20)
for i, val in enumerate(grid_vals):
    axs[0].text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
axs[1].bar(disp, hull_vals, color=bar_colors)
axs[1].set_ylabel("convex hull area (cm$^2$)")
axs[1].set_title("Convex hull area")
axs[1].tick_params(axis="x", rotation=20)
for i, val in enumerate(hull_vals):
    axs[1].text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
for ax in axs:
    ax.axhline(np.pi * RADIUS_CM ** 2, color="k", ls=":", lw=1)
fig.suptitle("Space use per leech (cm$^2$); dotted line = full arena 63.6 cm$^2$", fontsize=12)
fig.tight_layout()
f2 = f"{OUT}/spaceuse_area_barchart.png"
fig.savefig(f2, dpi=120)
plt.close(fig)

# CSV
rows = []
for n in names:
    r = {"entity": n, "treatment": DISPLAY[n], "low_confidence": n in low}
    r.update(results[n])
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{MET}/spaceuse_metrics.csv", index=False)

# FIGURE 3: occupancy heatmaps in real cm, each leech its own panel, shared log colorbar
from matplotlib.colors import LogNorm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

hm_names = [n for n in order if n in cmcoords]
edges = np.arange(-RADIUS_CM, RADIUS_CM + CELL_CM, CELL_CM)
hists = {}
vmax = 1
for n in hm_names:
    cx, cy = cmcoords[n]
    H, _, _ = np.histogram2d(cx, cy, bins=[edges, edges])
    hists[n] = H
    vmax = max(vmax, H.max())

fig, axs = plt.subplots(1, len(hm_names), figsize=(3.2 * len(hm_names) + 1.2, 3.6))
norm = LogNorm(vmin=1, vmax=vmax)
im = None
for ax, n in zip(axs, hm_names):
    H = hists[n].T  # transpose so x->columns, y->rows
    Hm = np.ma.masked_less(H, 1)
    im = ax.pcolormesh(edges, edges, Hm, norm=norm, cmap="magma")
    ax.add_patch(mpatches.Circle((0, 0), RADIUS_CM, fill=False, color="c", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xlim(-RADIUS_CM, RADIUS_CM); ax.set_ylim(-RADIUS_CM, RADIUS_CM)
    ax.set_xlabel("cm")
    ttl = DISPLAY[n] + ("\n(low-conf)" if n in low else "")
    ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel("cm")
cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
cbar.set_label("dwell (frames, log)")
fig.suptitle("Occupancy within the 9 cm dish: real cm scale, shared colorbar (each leech separate)",
             fontsize=12)
f3 = f"{OUT}/spaceuse_occupancy_heatmaps.png"
fig.savefig(f3, dpi=120, bbox_inches="tight")
plt.close(fig)

print("FIGS", f1, f2, f3)
print("DONE")
