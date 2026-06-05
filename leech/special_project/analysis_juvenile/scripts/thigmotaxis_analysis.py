import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
CSV = f"{BASE}/DL/juv_all_predictions.csv"
OUT = f"{BASE}/analysis_juvenile/plots"
MET = f"{BASE}/analysis_juvenile/metrics"
FPS = 29.99
R = 1.75  # dish physical radius in cm

# KNOWN arena: per-video center (px) and cm/px (do NOT re-fit)
ARENA = {
    "Video1_dish0.mp4": dict(cx=274.2, cy=275.4, cmpp=0.0067609),
    "Video1_dish1.mp4": dict(cx=271.8, cy=267.0, cmpp=0.0069246),
    "Video2_dish0.mp4": dict(cx=282.6, cy=269.4, cmpp=0.0070045),
    "Video2_dish1.mp4": dict(cx=252.6, cy=243.0, cmpp=0.0075799),
}

# 8 entities keyed "<videostem>_L<track>", grouped by video.
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
LOWCONF = {"Video1_dish0_L0", "Video1_dish0_L1", "Video1_dish1_L0",
           "Video1_dish1_L1", "Video2_dish1_L0", "Video2_dish1_L1"}
DISPLAY = {"Video1_dish0_L0": "Dopamine L0", "Video1_dish0_L1": "Dopamine L1",
           "Video1_dish1_L0": "DA+Food L0", "Video1_dish1_L1": "DA+Food L1",
           "Video2_dish0_L0": "Veh+NoFood L0", "Video2_dish0_L1": "Veh+NoFood L1",
           "Video2_dish1_L0": "Veh+Food L0", "Video2_dish1_L1": "Veh+Food L1"}
COL = {"Video1_dish0_L0": "tab:blue", "Video1_dish0_L1": "tab:orange",
       "Video1_dish1_L0": "tab:green", "Video1_dish1_L1": "tab:red",
       "Video2_dish0_L0": "tab:purple", "Video2_dish0_L1": "tab:brown",
       "Video2_dish1_L0": "tab:pink", "Video2_dish1_L1": "tab:olive"}

df = pd.read_csv(CSV)


def smooth(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


results = {}
for video, track, key in ENTITIES:
    g = df[(df.video == video) & (df.track == track)]
    h = g[g.node == 0].sort_values("frame")
    t = g[g.node == 1].sort_values("frame")
    m = pd.merge(h[["frame", "x", "y"]], t[["frame", "x", "y"]],
                 on="frame", suffixes=("_h", "_t"))
    if len(m) < 100:
        continue
    a = ARENA[video]
    cmpp = a["cmpp"]
    # smooth pixels then convert to cm relative to known arena center
    hx = (smooth(m.x_h.to_numpy()) - a["cx"]) * cmpp
    hy = (smooth(m.y_h.to_numpy()) - a["cy"]) * cmpp
    tx = (smooth(m.x_t.to_numpy()) - a["cx"]) * cmpp
    ty = (smooth(m.y_t.to_numpy()) - a["cy"]) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0  # centroid in cm relative to center

    # median body length in cm from the data (median |ant-post| * cmpp)
    bl_cm = float(np.nanmedian(np.hypot(hx - tx, hy - ty)))

    r_cm = np.sqrt(cx**2 + cy**2)          # radial distance from center (cm)
    rn = r_cm / R                          # normalized r/1.75
    wall_dist = R - r_cm                    # distance from wall (cm)

    OUTER = 0.7 * R   # 1.225 cm
    INNER = 0.5 * R   # 0.875 cm
    frac_outer = np.mean(r_cm > OUTER)
    frac_center = np.mean(r_cm < INNER)

    # center crossings: entries into central zone (r < 0.875 cm), debounced
    inside = r_cm < INNER
    crossings = int(np.sum(np.diff(inside.astype(int)) == 1))

    results[key] = dict(
        video=video, track=track, frames=m.frame.to_numpy(),
        cx=cx, cy=cy, r_cm=r_cm, rn=rn, wall_dist=wall_dist,
        mean_r=np.mean(r_cm), median_r=np.median(r_cm),
        mean_rn=np.mean(rn), mean_wall=np.mean(wall_dist),
        frac_outer=frac_outer, frac_center=frac_center,
        crossings=crossings, n=len(r_cm), low_conf=key in LOWCONF,
        body_length_cm=bl_cm,
    )

keys = list(results.keys())
n = len(keys)

# Figure 1: radial position over time (r in cm), stacked panels
fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=True)
for ax, k in zip(axes, keys):
    v = results[k]
    ts = v["frames"] / FPS / 60.0
    ax.plot(ts, v["r_cm"], lw=0.4, color=COL[k])
    ax.axhline(0.7 * R, color="red", ls="--", lw=0.8, label="wall zone (1.225 cm)")
    ax.axhline(0.5 * R, color="green", ls="--", lw=0.8, label="center zone (0.875 cm)")
    ax.axhline(R, color="k", ls=":", lw=0.8)
    ax.set_ylabel("r (cm)")
    ax.set_ylim(0, R + 0.3)
    lc = "  [LOW CONF]" if v["low_conf"] else ""
    ax.set_title(f"{DISPLAY[k]}{lc}   mean r={v['mean_r']:.2f} cm   "
                 f"%wall={100*v['frac_outer']:.0f}   %center={100*v['frac_center']:.0f}",
                 fontsize=9)
axes[0].legend(fontsize=7, loc="upper right", ncol=2)
axes[-1].set_xlabel("time (min)")
fig.suptitle("Thigmotaxis: radial distance from arena center over time (cm)", y=1.0)
fig.tight_layout()
f1 = f"{OUT}/thigmotaxis_radial_timeline.png"
fig.savefig(f1, dpi=120, bbox_inches="tight"); plt.close(fig)

# Figure 2: bar chart %time wall vs center, one bar PER TREATMENT
# (mean of that treatment's 2 leeches). Treatment name = DISPLAY without
# the trailing " L0"/" L1" leech suffix.
def treatment_name(k):
    return DISPLAY[k].rsplit(" L", 1)[0]

# preserve first-seen treatment order from keys
treatments = []
for k in keys:
    tn = treatment_name(k)
    if tn not in treatments:
        treatments.append(tn)

tr_keys = {tn: [k for k in keys if treatment_name(k) == tn] for tn in treatments}
tr_lowconf = {tn: all(results[k]["low_conf"] for k in tr_keys[tn]) for tn in treatments}

wall = [float(np.mean([100 * results[k]["frac_outer"] for k in tr_keys[tn]]))
        for tn in treatments]
cen = [float(np.mean([100 * results[k]["frac_center"] for k in tr_keys[tn]]))
       for tn in treatments]

fig, ax = plt.subplots(figsize=(9, 5))
xw = np.arange(len(treatments))
bw = ax.bar(xw - 0.2, wall, 0.4, label="% time outer annulus (r>1.225 cm)", color="firebrick")
bc = ax.bar(xw + 0.2, cen, 0.4, label="% time central zone (r<0.875 cm)", color="seagreen")
for bars in (bw, bc):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
labels = [tn + (" (low-conf)" if tr_lowconf[tn] else "") for tn in treatments]
ax.set_xticks(xw); ax.set_xticklabels(labels, rotation=15, fontsize=8)
ax.set_ylabel("% of time")
ax.set_title("Wall-hugging vs center use per treatment (known 1.75 cm arena)\n"
             "bars = mean of the 2 leeches per treatment")
ax.legend()
fig.tight_layout()
f2 = f"{OUT}/thigmotaxis_wall_vs_center_bar.png"
fig.savefig(f2, dpi=120, bbox_inches="tight"); plt.close(fig)

# Figure 3: centroid scatter in cm with 1.75 cm dish circle, one panel per entity
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
    lc = "\n[low conf]" if v["low_conf"] else ""
    ax.set_title(f"{DISPLAY[k]}{lc}\nmean r={v['mean_r']:.2f} cm", fontsize=9)
axes[0].set_ylabel("y (cm)")
fig.suptitle("Centroid cloud in cm with 1.75 cm dish (red=1.225 cm, green=0.875 cm)")
fig.tight_layout()
f3 = f"{OUT}/thigmotaxis_arena_scatter.png"
fig.savefig(f3, dpi=120, bbox_inches="tight"); plt.close(fig)

# CSV
rows = []
for k in keys:
    v = results[k]
    rows.append(dict(
        entity=k, treatment=DISPLAY[k], dish=v["video"], track=v["track"], n_frames=v["n"],
        low_conf=v["low_conf"],
        mean_radial_cm=round(v["mean_r"], 3),
        median_radial_cm=round(v["median_r"], 3),
        mean_r_norm=round(v["mean_rn"], 3),
        mean_wall_dist_cm=round(v["mean_wall"], 3),
        pct_time_outer_annulus=round(100 * v["frac_outer"], 1),
        pct_time_central_zone=round(100 * v["frac_center"], 1),
        center_crossings=v["crossings"],
        body_length_cm=round(v["body_length_cm"], 3),
    ))
pd.DataFrame(rows).to_csv(f"{MET}/thigmotaxis_metrics.csv", index=False)

for r_ in rows:
    print(r_)
print(f1); print(f2); print(f3)
