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

FPS = 149784 / 4994.0  # 29.99
DT = 1.0 / FPS

# Per-dish arena calibration (KNOWN, do not refit). center px, cm_per_px.
CAL = {
    "dish1": {"cx": 303, "cy": 295, "cmpp": 0.0200},
    "dish2": {"cx": 290, "cy": 311, "cmpp": 0.0205},
    "dish3": {"cx": 292, "cy": 284, "cmpp": 0.0202},
    "dish4": {"cx": 311, "cy": 293, "cmpp": 0.0206},
}
# Median body length per dish (cm) for reference notes only.
BL_CM = {"dish1": 1.14, "dish2": 1.13, "dish3": 0.88, "dish4": 0.50}

STILL_THRESH_CMS = 0.05  # cm/s; centroid speed below = still
MIN_BOUT_FRAMES = 5      # min contiguous moving frames to count as a bout

df = pd.read_csv(CSV)
df["dish"] = df["video"].str.extract(r"(dish\d)")


def smooth(arr):
    # Savitzky-Golay, window 11 frames, order 2 (applied per coordinate, pre-derivative)
    n = len(arr)
    if n < 11:
        return arr
    return savgol_filter(arr, 11, 2)


def analyze_track(sub, dish):
    cal = CAL[dish]
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
    }


# 5 entities: dish1, dish2, dish3, dish4_leech0, dish4_leech1
results = {}
order = []
for dish in ["dish1", "dish2", "dish3", "dish4"]:
    d = df[df["dish"] == dish]
    for trk in sorted(d["track"].unique()):
        sub = d[d["track"] == trk]
        if len(sub) < 1000:
            continue
        label = dish if dish != "dish4" else f"dish4_leech{int(trk)}"
        results[label] = analyze_track(sub, dish)
        order.append(label)
        print("done", label, "rows", len(sub))

labels = order
LOWCONF = {"dish4_leech0", "dish4_leech1"}
DISPLAY = {"dish1": "Veh+Food", "dish2": "DA+Food", "dish3": "Veh+NoFood",
           "dish4_leech0": "DA+NoFood L0", "dish4_leech1": "DA+NoFood L1"}
colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 3)))[:len(labels)]


def disp_label(l):
    return DISPLAY[l] + " (low-conf)" if l in LOWCONF else DISPLAY[l]


# 1. Speed over time, binned (60s mean), 5 series
plt.figure(figsize=(11, 5))
for lab, c in zip(labels, colors):
    r = results[lab]
    t, s = r["t"], r["speed_series"]
    bins = np.arange(0, 5000 + 60, 60)
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

# 3. Bar: total head path (cm), 5 bars
plt.figure(figsize=(9, 5))
vals = [results[l]["total_head_cm"] for l in labels]
bcolors = [("0.6" if l in LOWCONF else c) for l, c in zip(labels, colors)]
bars = plt.bar([disp_label(l) for l in labels], vals, color=bcolors)
plt.ylabel("Total HEAD path (cm)")
plt.title("Kinematics: total head path length per leech (whole clip, cm)")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT}/kinematics_total_head_path_bar.png", dpi=130); plt.close()

# CSV summary
rows = []
for lab in labels:
    r = results[lab]
    dish = lab.split("_")[0]
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
        "body_length_cm": BL_CM[dish],
        "total_head_path_BL": round(r["total_head_cm"] / BL_CM[dish], 1),
        "still_thresh_cms": STILL_THRESH_CMS,
        "smoothing": "savgol w11 o2",
    })
summ = pd.DataFrame(rows)
summ.to_csv(f"{MET}/kinematics_summary.csv", index=False)
print(summ.to_string(index=False))
