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
DT = 1.0 / FPS

# Per-video arena calibration (KNOWN, do not refit). center px, cm_per_px.
CAL = {
    "Video1_dish0.mp4": {"cx": 274.2, "cy": 275.4, "cmpp": 0.0067609},
    "Video1_dish1.mp4": {"cx": 271.8, "cy": 267.0, "cmpp": 0.0069246},
    "Video2_dish0.mp4": {"cx": 282.6, "cy": 269.4, "cmpp": 0.0070045},
    "Video2_dish1.mp4": {"cx": 252.6, "cy": 243.0, "cmpp": 0.0075799},
}

STILL_THRESH_CMS = 0.05  # cm/s; centroid speed below = still
MIN_BOUT_FRAMES = 5      # min contiguous moving frames to count as a bout

# 8 entities keyed "<videostem>_L<track>", grouped by video.
ENTITIES = [
    ("Video1_dish0.mp4", 0, "Video1_dish0_L0", "Dopamine L0"),
    ("Video1_dish0.mp4", 1, "Video1_dish0_L1", "Dopamine L1"),
    ("Video1_dish1.mp4", 0, "Video1_dish1_L0", "DA+Food L0"),
    ("Video1_dish1.mp4", 1, "Video1_dish1_L1", "DA+Food L1"),
    ("Video2_dish0.mp4", 0, "Video2_dish0_L0", "Veh+NoFood L0"),
    ("Video2_dish0.mp4", 1, "Video2_dish0_L1", "Veh+NoFood L1"),
    ("Video2_dish1.mp4", 0, "Video2_dish1_L0", "Veh+Food L0"),
    ("Video2_dish1.mp4", 1, "Video2_dish1_L1", "Veh+Food L1"),
]
LOWCONF = {"Video1_dish0_L0", "Video1_dish0_L1", "Video1_dish1_L0",
           "Video1_dish1_L1", "Video2_dish1_L0", "Video2_dish1_L1"}
DISPLAY = {e[2]: e[3] for e in ENTITIES}

df = pd.read_csv(CSV)


def smooth(arr):
    # Savitzky-Golay, window 11 frames, order 2 (applied per coordinate, pre-derivative)
    n = len(arr)
    if n < 11:
        return arr
    return savgol_filter(arr, 11, 2)


def analyze_track(sub, video):
    cal = CAL[video]
    cx0, cy0, cmpp = cal["cx"], cal["cy"], cal["cmpp"]
    piv = sub.pivot_table(index="frame", columns="node", values=["x", "y"]).sort_index()

    # px -> cm relative to arena center
    def to_cm(px, c):
        return (px - c) * cmpp

    hx = smooth(to_cm(piv[("x", 0)].to_numpy(), cx0))
    hy = smooth(to_cm(piv[("y", 0)].to_numpy(), cy0))
    tx = smooth(to_cm(piv[("x", 1)].to_numpy(), cx0))
    ty = smooth(to_cm(piv[("y", 1)].to_numpy(), cy0))
    t = piv.index.to_numpy() * DT

    # median body length in cm from the data (median |ant-post| * cmpp)
    bl_cm = float(np.nanmedian(np.hypot(hx - tx, hy - ty)))

    def path_speed(x, y):
        dx = np.diff(x); dy = np.diff(y)
        disp = np.sqrt(dx**2 + dy**2)   # cm per frame
        speed = disp * FPS              # cm/s
        return disp, speed

    hdisp, hspeed = path_speed(hx, hy)
    tdisp, tspeed = path_speed(tx, ty)
    total_head_cm = hdisp.sum()
    total_tail_cm = tdisp.sum()

    cx = (hx + tx) / 2.0; cy = (hy + ty) / 2.0
    cdisp, cspeed = path_speed(cx, cy)

    mean_speed = float(np.mean(cspeed))
    median_speed = float(np.median(cspeed))

    moving = cspeed > STILL_THRESH_CMS
    frac_moving = float(moving.mean())

    # movement bouts: contiguous moving frames >= MIN_BOUT_FRAMES
    bouts = []
    i, n = 0, len(moving)
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if (j - i) >= MIN_BOUT_FRAMES:
                bouts.append((i, j))
            i = j
        else:
            i += 1
    n_bouts = len(bouts)
    bout_durs = np.array([(b[1] - b[0]) * DT for b in bouts]) if bouts else np.array([])
    mean_bout_dur = float(bout_durs.mean()) if len(bout_durs) else 0.0
    ibis = [(bouts[k][0] - bouts[k - 1][1]) * DT for k in range(1, len(bouts))]
    mean_ibi = float(np.mean(ibis)) if ibis else 0.0

    return {
        "total_head_cm": total_head_cm, "total_tail_cm": total_tail_cm,
        "mean_speed_cms": mean_speed, "median_speed_cms": median_speed,
        "frac_moving": frac_moving, "n_bouts": n_bouts,
        "mean_bout_dur_s": mean_bout_dur, "mean_ibi_s": mean_ibi,
        "speed_series": cspeed, "t": t[1:], "n_frames": len(piv),
        "body_length_cm": bl_cm,
    }


# 8 entities: each leech (video, track) separate
results = {}
order = []
for video, trk, key, label in ENTITIES:
    sub = df[(df["video"] == video) & (df["track"] == trk)]
    if len(sub) < 1000:
        continue
    results[key] = analyze_track(sub, video)
    order.append(key)
    print("done", key, "rows", len(sub))

labels = order
colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 3)))[:len(labels)]


def disp_label(l):
    return DISPLAY[l] + " (low-conf)" if l in LOWCONF else DISPLAY[l]


# 1. Speed over time, binned (60s mean), 8 series
plt.figure(figsize=(11, 5))
for lab, c in zip(labels, colors):
    r = results[lab]
    t, s = r["t"], r["speed_series"]
    bins = np.arange(0, 2200 + 60, 60)
    idx = np.digitize(t, bins)
    binned = [s[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(bins))]
    centers = (bins[:-1] + bins[1:]) / 2
    ls = "--" if lab in LOWCONF else "-"
    plt.plot(centers / 60.0, binned, label=disp_label(lab), color=c, lw=1.3, ls=ls)
plt.xlabel("Time (min)"); plt.ylabel("Centroid speed (cm/s, 60 s mean)")
plt.title("Kinematics: activity over time per leech (cm/s)")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/kinematics_speed_over_time.png", dpi=130); plt.close()

# 2. Speed distribution (cm/s)
plt.figure(figsize=(10, 5))
for lab, c in zip(labels, colors):
    s = results[lab]["speed_series"]
    s = s[s < np.percentile(s, 99.5)]
    ls = "--" if lab in LOWCONF else "-"
    plt.hist(s, bins=120, histtype="step", density=True, label=disp_label(lab), color=c, lw=1.4, ls=ls)
plt.axvline(STILL_THRESH_CMS, color="k", ls=":", lw=1.2,
            label=f"still thresh {STILL_THRESH_CMS} cm/s")
plt.xlabel("Centroid speed (cm/s)"); plt.ylabel("Density (log)")
plt.yscale("log"); plt.title("Kinematics: centroid speed distribution per leech (cm/s)")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/kinematics_speed_dist.png", dpi=130); plt.close()

# 3. Bar: total head path (cm), 4 bars = mean of the 2 leeches per treatment
# Treatment = base name (strip trailing " L0"/" L1" from DISPLAY label).
def treatment_name(lab):
    return DISPLAY[lab].rsplit(" L", 1)[0]

treat_order = []
treat_vals = {}
treat_lowconf = {}
for lab in labels:
    tn = treatment_name(lab)
    if tn not in treat_vals:
        treat_vals[tn] = []
        treat_order.append(tn)
    treat_vals[tn].append(results[lab]["total_head_cm"])
    # a treatment is low-conf if its leeches are flagged low-conf
    treat_lowconf[tn] = treat_lowconf.get(tn, False) or (lab in LOWCONF)

def treat_disp(tn):
    return tn + " (low-conf)" if treat_lowconf[tn] else tn

plt.figure(figsize=(9, 5))
vals = [float(np.mean(treat_vals[tn])) for tn in treat_order]
tcolors = plt.cm.tab10(np.linspace(0, 1, max(len(treat_order), 3)))[:len(treat_order)]
bcolors = [("0.6" if treat_lowconf[tn] else c) for tn, c in zip(treat_order, tcolors)]
bars = plt.bar([treat_disp(tn) for tn in treat_order], vals, color=bcolors)
plt.ylabel("Total HEAD path (cm)")
plt.title("Kinematics: total head path length per treatment (mean of 2 leeches, whole clip, cm)")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT}/kinematics_total_head_path_bar.png", dpi=130); plt.close()

# CSV summary
rows = []
for lab in labels:
    r = results[lab]
    bl = r["body_length_cm"]
    rows.append({
        "entity": lab,
        "treatment": DISPLAY[lab],
        "low_confidence": lab in LOWCONF,
        "total_head_path_cm": round(r["total_head_cm"], 1),
        "total_tail_path_cm": round(r["total_tail_cm"], 1),
        "mean_speed_cms": round(r["mean_speed_cms"], 4),
        "median_speed_cms": round(r["median_speed_cms"], 4),
        "pct_time_moving": round(100 * r["frac_moving"], 1),
        "n_bouts": r["n_bouts"],
        "mean_bout_dur_s": round(r["mean_bout_dur_s"], 2),
        "mean_ibi_s": round(r["mean_ibi_s"], 2),
        "body_length_cm": round(bl, 3),
        "total_head_path_BL": round(r["total_head_cm"] / bl, 1),
        "still_thresh_cms": STILL_THRESH_CMS,
        "smoothing": "savgol w11 o2",
    })
summ = pd.DataFrame(rows)
summ.to_csv(f"{MET}/kinematics_summary.csv", index=False)
print(summ.to_string(index=False))
