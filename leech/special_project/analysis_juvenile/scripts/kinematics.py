"""Locomotor kinematics of juvenile leeches from DL keypoint predictions.

4 videos x 2 leeches = 8 entities. Mirrors the adult analysis.
node 0 = anterior (head), node 1 = posterior (tail).
Coordinates converted px -> cm via per-video cmpp (3.5 cm dish).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------------------------------------------- config
BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
PRED_CSV = os.path.join(BASE, "DL", "juv_all_predictions.csv")
CALIB_JSON = os.path.join(BASE, "DL", "juv_calibration.json")
OUT_PLOTS = os.path.join(BASE, "analysis_juvenile", "plots")
OUT_METRICS = os.path.join(BASE, "analysis_juvenile", "metrics")
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)

FPS = 29.99
STILL_THRESH = 0.05  # cm/s, below this = not moving
SG_WIN = 11
SG_POLY = 2
MIN_BOUT_FRAMES = 3  # min consecutive moving frames to count a bout

# display labels per video
VIDEO_TREATMENT = {
    "Video1_dish0.mp4": "Dopamine",
    "Video1_dish1.mp4": "Dopamine+Food",
    "Video2_dish0.mp4": "Food",
    "Video2_dish1.mp4": "Veh+Food",
}
# low-conf flag from pck (<60)
LOW_CONF = {
    "Video1_dish0": True,   # pck22
    "Video1_dish1": True,   # pck32
    "Video2_dish0": False,  # pck90
    "Video2_dish1": True,   # pck55
}
# consistent treatment ordering / colors
TREAT_ORDER = ["Dopamine", "Dopamine+Food", "Food", "Veh+Food"]
TREAT_COLORS = {
    "Dopamine": "#d62728",
    "Dopamine+Food": "#ff7f0e",
    "Food": "#2ca02c",
    "Veh+Food": "#1f77b4",
}


def vstem(video):
    return video.replace(".mp4", "")


# ---------------------------------------------------------------- load
print("Loading predictions ...")
df = pd.read_csv(PRED_CSV)
with open(CALIB_JSON) as f:
    calib = json.load(f)

# pivot to ant/post x,y per (video,track,frame)
piv = df.pivot_table(
    index=["video", "track", "frame", "time_s"],
    columns="node",
    values=["x", "y"],
).reset_index()
# flatten columns
piv.columns = [
    f"{a}_{b}" if b != "" else a for a, b in piv.columns
]
# node columns become x_0,y_0 (ant) and x_1,y_1 (post)
piv = piv.rename(
    columns={"x_0": "ax", "y_0": "ay", "x_1": "px", "y_1": "py"}
)

entities = []  # metric rows
traces = {}    # entity -> (time, speed)

for video in VIDEO_TREATMENT:
    stem = vstem(video)
    cmpp = calib[stem]["cmpp"]
    treat = VIDEO_TREATMENT[video]
    low = LOW_CONF[stem]
    sub_v = piv[piv["video"] == video]
    for track in sorted(sub_v["track"].unique()):
        s = sub_v[sub_v["track"] == track].sort_values("frame").copy()
        # drop frames with any missing keypoint
        s = s.dropna(subset=["ax", "ay", "px", "py"])
        if len(s) < SG_WIN:
            continue
        leech = f"L{int(track)}"
        entity = f"{stem}_{leech}"

        # px -> cm
        ax = s["ax"].to_numpy() * cmpp
        ay = s["ay"].to_numpy() * cmpp
        px = s["px"].to_numpy() * cmpp
        py = s["py"].to_numpy() * cmpp
        t = s["time_s"].to_numpy()

        # smooth coordinates before derivatives
        def sm(arr):
            return savgol_filter(arr, SG_WIN, SG_POLY)

        ax_s, ay_s = sm(ax), sm(ay)
        px_s, py_s = sm(px), sm(py)

        # centroid = midpoint of ant/post
        cx = 0.5 * (ax_s + px_s)
        cy = 0.5 * (ay_s + py_s)

        # framewise displacement
        d_cent = np.hypot(np.diff(cx), np.diff(cy))
        d_head = np.hypot(np.diff(ax_s), np.diff(ay_s))
        d_tail = np.hypot(np.diff(px_s), np.diff(py_s))

        speed = d_cent * FPS  # cm/s, length n-1
        speed_t = t[1:]

        # body length cm
        body_len = np.hypot(ax_s - px_s, ay_s - py_s)

        # moving mask
        moving = speed > STILL_THRESH
        pct_moving = 100.0 * moving.mean()

        # movement bouts (runs of moving frames >= MIN_BOUT_FRAMES)
        n_bouts = 0
        bout_durs = []
        run = 0
        for m in moving:
            if m:
                run += 1
            else:
                if run >= MIN_BOUT_FRAMES:
                    n_bouts += 1
                    bout_durs.append(run / FPS)
                run = 0
        if run >= MIN_BOUT_FRAMES:
            n_bouts += 1
            bout_durs.append(run / FPS)
        mean_bout = float(np.mean(bout_durs)) if bout_durs else 0.0

        total_head = float(d_head.sum())
        total_tail = float(d_tail.sum())
        ht_ratio = total_head / total_tail if total_tail > 0 else np.nan

        entities.append(
            dict(
                entity=entity,
                video=video,
                treatment=treat,
                leech=leech,
                low_conf=low,
                total_head_path_cm=total_head,
                total_tail_path_cm=total_tail,
                mean_speed_cms=float(np.mean(speed)),
                median_speed_cms=float(np.median(speed)),
                pct_time_moving=float(pct_moving),
                n_bouts=int(n_bouts),
                mean_bout_dur_s=mean_bout,
                body_length_cm=float(np.median(body_len)),
                head_tail_path_ratio=float(ht_ratio),
            )
        )
        traces[entity] = (speed_t, speed, treat, low)

met = pd.DataFrame(entities)
# order entities by treatment then leech
met["_to"] = met["treatment"].map({t: i for i, t in enumerate(TREAT_ORDER)})
met = met.sort_values(["_to", "leech"]).drop(columns="_to").reset_index(drop=True)
csv_path = os.path.join(OUT_METRICS, "kinematics_metrics.csv")
met.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}  ({len(met)} entities)")


def disp_label(row):
    tag = " (low-conf)" if row["low_conf"] else ""
    return f"{row['treatment']} {row['leech']}{tag}"


met["label"] = met.apply(disp_label, axis=1)
ordered_entities = met["entity"].tolist()

# =============================================================== FIG 1
# speed vs time per entity, 8 panels
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
axes = axes.ravel()
for i, ent in enumerate(ordered_entities):
    row = met[met["entity"] == ent].iloc[0]
    t, sp, treat, low = traces[ent]
    ax = axes[i]
    ax.plot(t, sp, lw=0.6, color=TREAT_COLORS[treat])
    ax.axhline(STILL_THRESH, color="gray", ls=":", lw=0.8)
    ax.set_title(row["label"], fontsize=10)
    ax.set_ylabel("speed (cm/s)", fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[-2:]:
    ax.set_xlabel("time (s)", fontsize=9)
fig.suptitle(
    "Juvenile leech centroid speed over time (smoothed)\n"
    "dotted line = 0.05 cm/s still threshold",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(OUT_PLOTS, "kin_speed_traces.png")
fig.savefig(p1, dpi=130)
plt.close(fig)
print("Wrote", p1)

# =============================================================== FIG 2
# grouped bars across 8 entities for 4 metrics
metrics_bar = [
    ("pct_time_moving", "percent time moving (%)"),
    ("mean_speed_cms", "mean speed (cm/s)"),
    ("total_head_path_cm", "total head path (cm)"),
    ("head_tail_path_ratio", "head:tail path ratio"),
]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()
xpos = np.arange(len(met))
colors = [TREAT_COLORS[t] for t in met["treatment"]]
hatches = ["//" if lc else "" for lc in met["low_conf"]]
for ax, (col, ylab) in zip(axes, metrics_bar):
    bars = ax.bar(xpos, met[col].to_numpy(), color=colors, edgecolor="black")
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(met["label"], rotation=45, ha="right", fontsize=8)
    if col == "head_tail_path_ratio":
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
fig.suptitle(
    "Juvenile leech kinematics by entity (hatched bars = low-conf tracking)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
p2 = os.path.join(OUT_PLOTS, "kin_summary_bars.png")
fig.savefig(p2, dpi=130)
plt.close(fig)
print("Wrote", p2)

# =============================================================== FIG 3
# speed distributions per treatment (violin) + pct moving bar
fig, (axv, axb) = plt.subplots(1, 2, figsize=(15, 6))

# gather speed arrays per treatment (pool the 2 leeches)
treat_speeds = {t: [] for t in TREAT_ORDER}
for ent, (t_, sp, treat, low) in traces.items():
    treat_speeds[treat].append(sp)
violin_data = []
violin_labels = []
violin_colors = []
for treat in TREAT_ORDER:
    if treat_speeds[treat]:
        pooled = np.concatenate(treat_speeds[treat])
        # clip extreme outliers for readability of the violin
        pooled = pooled[pooled < np.percentile(pooled, 99.5)]
        violin_data.append(pooled)
        violin_labels.append(treat)
        violin_colors.append(TREAT_COLORS[treat])
parts = axv.violinplot(violin_data, showmedians=True, showextrema=False)
for pc, c in zip(parts["bodies"], violin_colors):
    pc.set_facecolor(c)
    pc.set_alpha(0.7)
axv.set_xticks(np.arange(1, len(violin_labels) + 1))
axv.set_xticklabels(violin_labels, rotation=20, ha="right")
axv.set_ylabel("centroid speed (cm/s)")
axv.axhline(STILL_THRESH, color="gray", ls=":", lw=0.8)
axv.set_title("Speed distribution per treatment\n(2 leeches pooled, capped at 99.5 pct)")

# pct time moving bar per treatment (mean of 2 leeches, dots = individuals)
treat_pct = met.groupby("treatment")["pct_time_moving"].mean().reindex(TREAT_ORDER)
bx = np.arange(len(TREAT_ORDER))
axb.bar(
    bx,
    treat_pct.to_numpy(),
    color=[TREAT_COLORS[t] for t in TREAT_ORDER],
    edgecolor="black",
    alpha=0.85,
)
for i, treat in enumerate(TREAT_ORDER):
    vals = met[met["treatment"] == treat]["pct_time_moving"].to_numpy()
    axb.scatter([i] * len(vals), vals, color="black", zorder=3, s=30)
axb.set_xticks(bx)
axb.set_xticklabels(TREAT_ORDER, rotation=20, ha="right")
axb.set_ylabel("percent time moving (%)")
axb.set_title("Percent time moving per treatment\n(bar = mean of 2 leeches, dots = individuals)")

fig.suptitle("Juvenile leech speed distributions and activity", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p3 = os.path.join(OUT_PLOTS, "kin_distributions.png")
fig.savefig(p3, dpi=130)
plt.close(fig)
print("Wrote", p3)

# ---------------------------------------------------------------- summary
print("\n=== SUMMARY (per entity) ===")
cols = [
    "entity", "treatment", "low_conf", "mean_speed_cms", "pct_time_moving",
    "total_head_path_cm", "head_tail_path_ratio", "body_length_cm", "n_bouts",
]
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(met[cols].to_string(index=False))
print("\n=== per treatment means ===")
print(
    met.groupby("treatment")[
        ["mean_speed_cms", "pct_time_moving", "total_head_path_cm", "head_tail_path_ratio"]
    ].mean().reindex(TREAT_ORDER).to_string()
)
print("\nDone.")
