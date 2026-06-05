import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import savgol_filter

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
CSV = f"{BASE}/DL/juv_all_predictions.csv"
CALIB = f"{BASE}/DL/juv_calibration.json"
OUT = f"{BASE}/analysis_juvenile/plots"
MET = f"{BASE}/analysis_juvenile/metrics"
FPS = 29.99

# video stem -> exact treatment label (authoritative)
TREAT = {
    "Video1_dish0": "Dopamine",
    "Video1_dish1": "Dopamine+Food",
    "Video2_dish0": "Food",
    "Video2_dish1": "Veh+Food",
}

OUTER = 0.7  # outer annulus threshold (normalized rho)
CENTRAL = 0.5  # central zone threshold

with open(CALIB) as fh:
    calib = json.load(fh)

df = pd.read_csv(CSV)
df["stem"] = df["video"].str.replace(".mp4", "", regex=False)


def smooth(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


# 8 entities: each video stem x track 0/1
entities = []
for stem in TREAT:
    for track in (0, 1):
        entities.append((stem, track))

results = {}
for stem, track in entities:
    g = df[(df["stem"] == stem) & (df.track == track)]
    ant = g[g.node == 0].sort_values("frame")[["frame", "x", "y"]]
    post = g[g.node == 1].sort_values("frame")[["frame", "x", "y"]]
    m = pd.merge(ant, post, on="frame", suffixes=("_a", "_p"))
    if len(m) < 100:
        continue
    cal = calib[stem]
    cx0, cy0, r_px, cmpp = cal["cx"], cal["cy"], cal["r_px"], cal["cmpp"]

    ax_ = smooth(m.x_a.to_numpy())
    ay_ = smooth(m.y_a.to_numpy())
    px_ = smooth(m.x_p.to_numpy())
    py_ = smooth(m.y_p.to_numpy())
    # centroid = midpoint anterior/posterior, in pixels
    cx = (ax_ + px_) / 2.0
    cy = (ay_ + py_) / 2.0

    dist_px = np.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2)
    rho = dist_px / r_px  # 0 = center, 1 = wall
    wall_dist_cm = (r_px - dist_px) * cmpp

    frac_outer = float(np.mean(rho > OUTER))
    frac_central = float(np.mean(rho < CENTRAL))
    # center crossings: entries into central zone (rho goes below 0.5)
    inside = rho < CENTRAL
    crossings = int(np.sum(np.diff(inside.astype(int)) == 1))

    key = f"{stem}_L{track}"
    results[key] = dict(
        stem=stem, track=track, treatment=TREAT[stem],
        low_conf=cal["pck"] < 60,
        # store centroid relative to center, normalized by r_px for maps
        nx=(cx - cx0) / r_px, ny=(cy - cy0) / r_px,
        rho=rho, wall_dist_cm=wall_dist_cm,
        mean_rho=float(np.mean(rho)), median_rho=float(np.median(rho)),
        mean_wall=float(np.mean(wall_dist_cm)),
        frac_outer=frac_outer, frac_central=frac_central,
        crossings=crossings, n=len(rho),
        frames=m.frame.to_numpy(),
    )

keys = list(results.keys())
n = len(keys)

COL = {k: c for k, c in zip(keys, plt.cm.tab10(np.linspace(0, 1, 10)))}


def label(k):
    v = results[k]
    lc = " (low-conf)" if v["low_conf"] else ""
    return f"{k}\n{v['treatment']}{lc}"


# ---------------------------------------------------------------------------
# Figure 1: radial distribution (violin) of rho per entity
fig, ax = plt.subplots(figsize=(12, 6))
data = [results[k]["rho"] for k in keys]
parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
for i, b in enumerate(parts["bodies"]):
    b.set_facecolor(COL[keys[i]])
    b.set_alpha(0.6)
parts["cmeans"].set_color("black")
ax.axhline(OUTER, color="red", ls="--", lw=1.2, label="wall zone (rho=0.7)")
ax.axhline(CENTRAL, color="green", ls="--", lw=1.0, label="central zone (rho=0.5)")
ax.set_xticks(np.arange(1, n + 1))
ax.set_xticklabels([label(k) for k in keys], fontsize=8)
ax.set_ylabel("radial position rho (0=center, 1=wall)")
ax.set_ylim(0, 1.05)
ax.set_title("Juvenile thigmotaxis: distribution of normalized radial position per leech")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
f1 = f"{OUT}/thig_radial_distributions.png"
fig.savefig(f1, dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: per-entity 2D occupancy maps (normalized coords) in dish circle
ncol = 4
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.7 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax, k in zip(axes, keys):
    v = results[k]
    # 2D histogram occupancy inside unit circle
    h = ax.hist2d(v["nx"], v["ny"], bins=60, range=[[-1.05, 1.05], [-1.05, 1.05]],
                  cmap="inferno", cmin=1)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="white", lw=1.6))
    ax.add_patch(Circle((0, 0), OUTER, fill=False, color="red", ls="--", lw=1.0))
    ax.add_patch(Circle((0, 0), CENTRAL, fill=False, color="cyan", ls="--", lw=0.8))
    ax.plot(0, 0, "+", color="white", ms=8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_facecolor("black")
    lc = " (low-conf)" if v["low_conf"] else ""
    ax.set_title(f"{k}\n{v['treatment']}{lc}  mean rho={v['mean_rho']:.2f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes[n:]:
    ax.axis("off")
fig.suptitle("Juvenile position occupancy within dish (white=wall, red=0.7 annulus, cyan=0.5 zone)",
             y=1.0)
fig.tight_layout()
f2 = f"{OUT}/thig_radial_maps.png"
fig.savefig(f2, dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: grouped bars mean rho and percent time in outer annulus
fig, ax = plt.subplots(figsize=(12, 6))
xw = np.arange(n)
mean_rho = [results[k]["mean_rho"] for k in keys]
pct_outer = [100 * results[k]["frac_outer"] for k in keys]
ax2 = ax.twinx()
b1 = ax.bar(xw - 0.2, mean_rho, 0.4, color="steelblue", label="mean rho")
b2 = ax2.bar(xw + 0.2, pct_outer, 0.4, color="firebrick", label="% time outer annulus (rho>0.7)")
ax.axhline(OUTER, color="green", ls=":", lw=1.0)
ax.set_ylabel("mean radial position rho", color="steelblue")
ax2.set_ylabel("% time in outer annulus", color="firebrick")
ax.set_ylim(0, 1.05)
ax2.set_ylim(0, 105)
ax.set_xticks(xw)
ax.set_xticklabels([label(k) for k in keys], fontsize=8)
ax.set_title("Juvenile thigmotaxis summary: mean radial position and wall-zone occupancy per leech")
lines = [b1, b2]
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
fig.tight_layout()
f3 = f"{OUT}/thig_summary_bars.png"
fig.savefig(f3, dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Metrics CSV
rows = []
for k in keys:
    v = results[k]
    rows.append(dict(
        entity=k,
        video=v["stem"],
        treatment=v["treatment"],
        leech=v["track"],
        low_conf=v["low_conf"],
        n_frames=v["n"],
        mean_radial_norm=round(v["mean_rho"], 4),
        median_radial_norm=round(v["median_rho"], 4),
        mean_wall_dist_cm=round(v["mean_wall"], 4),
        pct_time_outer_annulus=round(100 * v["frac_outer"], 2),
        pct_time_central_zone=round(100 * v["frac_central"], 2),
        center_crossings=v["crossings"],
    ))
out_df = pd.DataFrame(rows)
out_df.to_csv(f"{MET}/thigmotaxis_metrics.csv", index=False)

print(out_df.to_string(index=False))
print(f1)
print(f2)
print(f3)
