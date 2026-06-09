import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
CSV = f"{BASE}/DL/all_predictions.csv"
OUT = f"{BASE}/analysis_fed_leech/plots"
MET = f"{BASE}/analysis_fed_leech/metrics"
FPS = 29.99
R = 4.5  # dish physical radius in cm

# KNOWN arena: per-dish center (px) and cm/px (do NOT re-fit)
ARENA = {
    1: dict(cx=303, cy=295, cmpp=0.0200),
    2: dict(cx=290, cy=311, cmpp=0.0205),
    3: dict(cx=292, cy=284, cmpp=0.0202),
    4: dict(cx=311, cy=293, cmpp=0.0206),
}
BL_CM = {1: 1.14, 2: 1.13, 3: 0.88, 4: 0.50}  # median body length (cm)

# 5 entities: dish1, dish2, dish3, dish4_leech0, dish4_leech1
ENTITIES = [(1, 0), (2, 0), (3, 0), (4, 0), (4, 1)]

df = pd.read_csv(CSV)
df["dish"] = df["video"].str.extract(r"dish(\d)").astype(int)


def smooth(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


results = {}
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
    # smooth pixels then convert to cm relative to known arena center
    hx = (smooth(m.x_h.to_numpy()) - a["cx"]) * cmpp
    hy = (smooth(m.y_h.to_numpy()) - a["cy"]) * cmpp
    tx = (smooth(m.x_t.to_numpy()) - a["cx"]) * cmpp
    ty = (smooth(m.y_t.to_numpy()) - a["cy"]) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0  # centroid in cm relative to center

    r_cm = np.sqrt(cx**2 + cy**2)          # radial distance from center (cm)
    rn = r_cm / R                          # normalized r/4.5
    wall_dist = R - r_cm                    # distance from wall (cm)

    OUTER = 0.7 * R   # 3.15 cm
    INNER = 0.5 * R   # 2.25 cm
    frac_outer = np.mean(r_cm > OUTER)
    frac_center = np.mean(r_cm < INNER)

    # center crossings: entries into central zone (r < 2.25 cm), debounced
    inside = r_cm < INNER
    crossings = int(np.sum(np.diff(inside.astype(int)) == 1))

    key = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    results[key] = dict(
        dish=dish, track=track, frames=m.frame.to_numpy(),
        cx=cx, cy=cy, r_cm=r_cm, rn=rn, wall_dist=wall_dist,
        mean_r=np.mean(r_cm), median_r=np.median(r_cm),
        mean_rn=np.mean(rn), mean_wall=np.mean(wall_dist),
        frac_outer=frac_outer, frac_center=frac_center,
        crossings=crossings, n=len(r_cm),
    )

keys = list(results.keys())
n = len(keys)
DISPLAY = {"dish1": "Veh+Food", "dish2": "DA+Food", "dish3": "Veh+NoFood",
           "dish4_leech0": "DA+NoFood L0", "dish4_leech1": "DA+NoFood L1"}
COL = {"dish1": "tab:blue", "dish2": "tab:orange", "dish3": "tab:green",
       "dish4_leech0": "tab:red", "dish4_leech1": "tab:purple"}

# Figure 1: radial position over time (r in cm), 5 series stacked panels
fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=True)
for ax, k in zip(axes, keys):
    v = results[k]
    ts = v["frames"] / FPS / 60.0
    ax.plot(ts, v["r_cm"], lw=0.4, color=COL[k])
    ax.axhline(0.7 * R, color="red", ls="--", lw=0.8, label="wall zone (3.15 cm)")
    ax.axhline(0.5 * R, color="green", ls="--", lw=0.8, label="center zone (2.25 cm)")
    ax.axhline(R, color="k", ls=":", lw=0.8)
    ax.set_ylabel("r (cm)")
    ax.set_ylim(0, R + 0.3)
    lc = "  [LOW CONF]" if v["dish"] == 4 else ""
    ax.set_title(f"{DISPLAY[k]}{lc}   mean r={v['mean_r']:.2f} cm   "
                 f"%wall={100*v['frac_outer']:.0f}   %center={100*v['frac_center']:.0f}",
                 fontsize=9)
axes[0].legend(fontsize=7, loc="upper right", ncol=2)
axes[-1].set_xlabel("time (min)")
fig.suptitle("Thigmotaxis: radial distance from arena center over time (cm)", y=1.0)
fig.tight_layout()
f1 = f"{OUT}/thigmotaxis_radial_timeline.png"
fig.savefig(f1, dpi=120, bbox_inches="tight"); plt.close(fig)

# Figure 2: bar chart %time wall vs center
fig, ax = plt.subplots(figsize=(9, 5))
xw = np.arange(n)
wall = [100 * results[k]["frac_outer"] for k in keys]
cen = [100 * results[k]["frac_center"] for k in keys]
ax.bar(xw - 0.2, wall, 0.4, label="% time outer annulus (r>3.15 cm)", color="firebrick")
ax.bar(xw + 0.2, cen, 0.4, label="% time central zone (r<2.25 cm)", color="seagreen")
labels = [DISPLAY[k] + ("\n[low conf]" if results[k]["dish"] == 4 else "") for k in keys]
ax.set_xticks(xw); ax.set_xticklabels(labels, rotation=15, fontsize=8)
ax.set_ylabel("% of time")
ax.set_title("Wall-hugging vs center use per leech (known 4.5 cm arena)")
ax.legend()
fig.tight_layout()
f2 = f"{OUT}/thigmotaxis_wall_vs_center_bar.png"
fig.savefig(f2, dpi=120, bbox_inches="tight"); plt.close(fig)

# Figure 3: centroid scatter in cm with 4.5 cm dish circle, one panel per entity
fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4))
for ax, k in zip(axes, keys):
    v = results[k]
    ax.scatter(v["cx"], v["cy"], s=0.2, alpha=0.12, color=COL[k])
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="black", lw=1.4))
    ax.add_patch(plt.Circle((0, 0), 0.7 * R, fill=False, color="red", ls="--", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), 0.5 * R, fill=False, color="green", ls="--", lw=1.0))
    ax.plot(0, 0, "x", color="orange")
    ax.set_xlim(-R - 0.5, R + 0.5); ax.set_ylim(-R - 0.5, R + 0.5)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("x (cm)")
    lc = "\n[low conf]" if v["dish"] == 4 else ""
    ax.set_title(f"{DISPLAY[k]}{lc}\nmean r={v['mean_r']:.2f} cm", fontsize=9)
axes[0].set_ylabel("y (cm)")
fig.suptitle("Centroid cloud in cm with 4.5 cm dish (red=3.15 cm, green=2.25 cm)")
fig.tight_layout()
f3 = f"{OUT}/thigmotaxis_arena_scatter.png"
fig.savefig(f3, dpi=120, bbox_inches="tight"); plt.close(fig)

# CSV
rows = []
for k in keys:
    v = results[k]
    rows.append(dict(
        entity=k, treatment=DISPLAY[k], dish=v["dish"], track=v["track"], n_frames=v["n"],
        low_conf=(v["dish"] == 4),
        mean_radial_cm=round(v["mean_r"], 3),
        median_radial_cm=round(v["median_r"], 3),
        mean_r_norm=round(v["mean_rn"], 3),
        mean_wall_dist_cm=round(v["mean_wall"], 3),
        pct_time_outer_annulus=round(100 * v["frac_outer"], 1),
        pct_time_central_zone=round(100 * v["frac_center"], 1),
        center_crossings=v["crossings"],
        body_length_cm=BL_CM[v["dish"]],
    ))
pd.DataFrame(rows).to_csv(f"{MET}/thigmotaxis_metrics.csv", index=False)

for r_ in rows:
    print(r_)
print(f1); print(f2); print(f3)
