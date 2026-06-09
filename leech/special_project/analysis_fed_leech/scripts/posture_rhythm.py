import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, welch, find_peaks

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = BASE + "/analysis_fed_leech/plots"
MET = BASE + "/analysis_fed_leech/metrics"
FPS = 149784 / 4994.0  # 29.99

# Per-dish arena calibration (center px, cm per px). Dish = 9 cm diameter, radius 4.5 cm.
CAL = {
    "dish1": dict(cx=303, cy=295, cmpp=0.0200),
    "dish2": dict(cx=290, cy=311, cmpp=0.0205),
    "dish3": dict(cx=292, cy=284, cmpp=0.0202),
    "dish4": dict(cx=311, cy=293, cmpp=0.0206),
}
# Reported median body lengths (cm) for sanity / labels
BL_REF = {"dish1": 1.14, "dish2": 1.13, "dish3": 0.88, "dish4": 0.50}

df = pd.read_csv(BASE + "/DL/all_predictions.csv")
df["dish"] = df["video"].str.extract(r"(dish\d)")


def get_series(sub):
    p = sub.pivot_table(index="frame", columns="node", values=["x", "y", "time_s"])
    t = p[("time_s", 0)].values
    hx = p[("x", 0)].values; hy = p[("y", 0)].values
    tx = p[("x", 1)].values; ty = p[("y", 1)].values
    return t, hx, hy, tx, ty


def smooth(a, win=11):
    # interpolate gaps, rolling-median despike, then Savitzky-Golay (win~11)
    a = pd.Series(a).interpolate(limit_direction="both").values
    a = pd.Series(a).rolling(win, center=True, min_periods=1).median().values
    if len(a) > 11:
        a = savgol_filter(a, 11, 3)
    return a


# Build the 5 entities: dish1, dish2, dish3 (track 0); dish4 tracks 0 and 1
units = []
for dish in ["dish1", "dish2", "dish3"]:
    units.append((dish, 0, f"{dish}"))
d4 = df[df["dish"] == "dish4"]
for tr in sorted(d4["track"].unique()):
    units.append(("dish4", int(tr), f"dish4_leech{int(tr)}"))

results = {}
for dish, tr, label in units:
    sub = df[(df["dish"] == dish) & (df["track"] == tr)]
    t, hx, hy, tx, ty = get_series(sub)
    cal = CAL[dish]; cmpp = cal["cmpp"]; cx = cal["cx"]; cy = cal["cy"]
    # to cm (centered on arena)
    hxc = (hx - cx) * cmpp; hyc = (hy - cy) * cmpp
    txc = (tx - cx) * cmpp; tyc = (ty - cy) * cmpp
    hxc, hyc, txc, tyc = [smooth(a) for a in (hxc, hyc, txc, tyc)]
    bl_cm = np.hypot(hxc - txc, hyc - tyc)
    head_path = np.nansum(np.hypot(np.diff(hxc), np.diff(hyc)))  # cm
    tail_path = np.nansum(np.hypot(np.diff(txc), np.diff(tyc)))  # cm
    results[label] = dict(t=t, bl_cm=bl_cm, head_path=head_path,
                          tail_path=tail_path, dish=dish, lowconf=(dish == "dish4"))

# ---- metrics ----
metrics = {}
SPEC = {}
for label, r in results.items():
    bl = r["bl_cm"]
    med = np.nanmedian(bl)
    bl_min = np.nanmin(bl); bl_max = np.nanmax(bl)
    frac_elong = np.nanmean(bl > med)
    cv = np.nanstd(bl) / np.nanmean(bl)
    # elongation / contraction events: peaks above/below own median, >=1 s apart,
    # prominence = 0.10 * median body length (cm)
    sig = bl - med
    prom = 0.10 * med
    pk_e, _ = find_peaks(sig, prominence=prom, distance=int(FPS * 1))
    pk_c, _ = find_peaks(-sig, prominence=prom, distance=int(FPS * 1))
    dur_min = (r["t"][-1] - r["t"][0]) / 60.0
    elong_rate = len(pk_e) / dur_min
    contr_rate = len(pk_c) / dur_min
    # FFT/Welch on body length (cm, detrended)
    x = pd.Series(bl).interpolate(limit_direction="both").values
    x = x - np.nanmean(x)
    nper = int(FPS * 120)  # 2-min segments
    f, Pxx = welch(x, fs=FPS, nperseg=min(nper, len(x)))
    band = (f > 0.01) & (f < 2.0)  # plausible peristalsis / crawl band
    fb, Pb = f[band], Pxx[band]
    peak_f = fb[np.argmax(Pb)]
    rel_pow = Pb.max() / np.sum(Pb)  # fraction of in-band power at dominant peak
    ht_ratio = r["head_path"] / r["tail_path"] if r["tail_path"] > 0 else np.nan
    metrics[label] = dict(
        dish=r["dish"], low_conf=int(r["lowconf"]),
        median_bl_cm=med, bl_min_cm=bl_min, bl_max_cm=bl_max,
        bl_range_cm=bl_max - bl_min, cv_bl=cv, pct_time_elongated=100 * frac_elong,
        elong_rate_per_min=elong_rate, contr_rate_per_min=contr_rate,
        dom_freq_hz=peak_f, dom_rel_power=rel_pow,
        head_path_cm=r["head_path"], tail_path_cm=r["tail_path"], head_tail_path_ratio=ht_ratio)
    SPEC[label] = (f, Pxx)

DISPLAY = {"dish1": "Veh+Food", "dish2": "DA+Food", "dish3": "Veh+NoFood",
           "dish4_leech0": "DA+NoFood L0", "dish4_leech1": "DA+NoFood L1"}

mdf = pd.DataFrame(metrics).T
mdf.insert(0, "treatment", [DISPLAY[i] for i in mdf.index])
mdf.index.name = "entity"
mdf.to_csv(MET + "/posture_rhythm_metrics.csv")
print(mdf.round(4).to_string())

COLORS = {"dish1": "#1f77b4", "dish2": "#2ca02c", "dish3": "#d62728",
          "dish4_leech0": "#9467bd", "dish4_leech1": "#8c564b"}

# ---- Figure 1: body length over a representative 5-min mid-clip window (cm) ----
n = len(results)
fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=True)
if n == 1: axes = [axes]
for ax, (label, r) in zip(axes, results.items()):
    t = r["t"]; bl = r["bl_cm"]
    m0 = (t >= 2400) & (t <= 2700)  # 40-45 min window
    ax.plot(t[m0] / 60.0, bl[m0], lw=0.6, color=COLORS[label])
    ax.axhline(np.nanmedian(bl), color="grey", ls="--", lw=0.7)
    tag = " (low conf)" if r["lowconf"] else ""
    ax.set_ylabel(f"{DISPLAY[label]}{tag}\nbody length (cm)", fontsize=8)
axes[-1].set_xlabel("time (min)")
fig.suptitle("Body length |head-tail| over time: 40-45 min window, smoothed (SavGol win 11)")
fig.tight_layout()
fig.savefig(OUT + "/posture_rhythm_bodylength_timeseries.png", dpi=130)
plt.close(fig)

# ---- Figure 2: power spectra ----
fig, ax = plt.subplots(figsize=(9, 6))
for label, (f, Pxx) in SPEC.items():
    m = (f > 0.005) & (f < 1.0)
    ls = "--" if results[label]["lowconf"] else "-"
    lbl = DISPLAY[label] + (" (low conf)" if results[label]["lowconf"] else "")
    ax.semilogy(f[m], Pxx[m], ls, lw=1.2, color=COLORS[label], label=lbl)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD (cm^2 / Hz)")
ax.set_title("Welch power spectra of body length (crawling rhythm)")
ax.legend(fontsize=8); ax.set_xlim(0, 0.6)
fig.tight_layout(); fig.savefig(OUT + "/posture_rhythm_powerspectra.png", dpi=130); plt.close(fig)

# ---- Figure 3: bar charts ----
labels = list(metrics.keys())
disp_labels = [DISPLAY[l] for l in labels]
cols = [COLORS[l] for l in labels]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))

def bar(ax, vals, title, ylab, hline=None):
    ax.bar(disp_labels, vals, color=cols)
    ax.set_title(title); ax.set_ylabel(ylab)
    ax.tick_params(axis='x', rotation=45)
    if hline is not None: ax.axhline(hline, color="k", ls="--", lw=0.8)

bar(axs[0, 0], [metrics[l]["median_bl_cm"] for l in labels], "Median body length", "cm")
bar(axs[0, 1], [100 * metrics[l]["cv_bl"] for l in labels],
    "Body-length variability (CV)", "% of median (higher = more length change)")
bar(axs[0, 2], [metrics[l]["dom_freq_hz"] for l in labels], "Dominant rhythm frequency", "Hz")
bar(axs[1, 0], [metrics[l]["dom_rel_power"] for l in labels], "Dominant-peak relative power", "fraction of in-band power")
bar(axs[1, 1], [metrics[l]["elong_rate_per_min"] for l in labels], "Elongation event rate", "events / min")
bar(axs[1, 2], [metrics[l]["head_tail_path_ratio"] for l in labels], "Head : tail path ratio", "ratio", hline=1.0)
fig.suptitle("Posture and crawling rhythm metrics per leech (cm units); DA+NoFood = low confidence", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "/posture_rhythm_barcharts.png", dpi=130); plt.close(fig)
print("DONE")
