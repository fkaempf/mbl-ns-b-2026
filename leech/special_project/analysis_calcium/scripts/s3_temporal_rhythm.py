"""
Stage 3: High-level temporal rhythm characterization of the leech ganglion.

This is a 10x (low-mag), PRELIMINARY transgenic line. Signal is at the GANGLION
level, not reliable single neurons. We therefore characterize ganglion-wide and
coarse regional temporal dynamics only. The dominant phenomenon is a strong,
regular, sustained calcium oscillation across the whole active band.

Builds a whole-ganglion mean dF/F from a tissue mask, then:
  1. ganglion signal vs time (+ zoom inset)
  2. Welch PSD (ganglion + patch-population mean), dominant frequency
  3. spectrogram + instantaneous dominant-band power (stability vs drift)
  4. autocorrelation + inter-peak interval distribution (regularity)
  5. coarse regional comparison (dom freq + phase lag / traveling wave)

Run with:
  /Users/fkampf/.../.venv/bin/python analysis_calcium/scripts/s3_temporal_rhythm.py
"""

import os
import json

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal, ndimage as ndi
from skimage import filters

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
DERIVED = os.path.join(ROOT, "analysis_calcium", "derived")
PLOTS = os.path.join(ROOT, "analysis_calcium", "plots")
METRICS = os.path.join(ROOT, "analysis_calcium", "metrics")
MOVIE = os.path.join(
    ROOT, "data", "calcium", "2026-06-04",
    "helobdella_LeechNo2_elavgcamp6m-17.tif",
)
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

with open(os.path.join(DERIVED, "meta.json")) as f:
    meta = json.load(f)
FS = float(meta["fs"])
NYQ = FS / 2.0
DT = 1.0 / FS

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def rolling_percentile_f0(F, win, pct=10.0):
    """F0 as rolling percentile (centered). F is 1D length T."""
    T = F.shape[0]
    half = win // 2
    f0 = np.empty(T, dtype=np.float64)
    for t in range(T):
        a = max(0, t - half)
        b = min(T, t + half + 1)
        f0[t] = np.percentile(F[a:b], pct)
    return f0


def dff_from_meanF(meanF, win):
    f0 = rolling_percentile_f0(meanF, win, 10.0)
    f0 = np.maximum(f0, 1e-6)
    return (meanF - f0) / f0


# ----------------------------------------------------------------------------
# Load static images, build tissue mask (Otsu on smoothed corr/mean band)
# ----------------------------------------------------------------------------
corr = np.load(os.path.join(DERIVED, "correlation_image.npy")).astype(np.float64)
mean_img = np.load(os.path.join(DERIVED, "mean_image.npy")).astype(np.float64)

# combine correlation (functional activity) and mean (tissue brightness)
def norm01(x):
    x = x - np.nanmin(x)
    m = np.nanmax(x)
    return x / m if m > 0 else x

det = norm01(ndi.gaussian_filter(corr, 2.0)) * norm01(ndi.gaussian_filter(mean_img, 2.0))
det_s = ndi.gaussian_filter(det, 3.0)
thr = filters.threshold_otsu(det_s)
mask = det_s > thr
# keep largest connected component (the bright band)
lbl, n = ndi.label(mask)
if n > 1:
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    mask = lbl == (np.argmax(sizes) + 1)
mask = ndi.binary_fill_holes(mask)
mask = ndi.binary_closing(mask, iterations=2)
n_mask_px = int(mask.sum())
print(f"tissue mask: {n_mask_px} px ({100*n_mask_px/mask.size:.1f}% of frame)")

ys, xs = np.where(mask)

# ----------------------------------------------------------------------------
# Whole-ganglion mean raw F per frame over the mask (chunked memmap read)
# ----------------------------------------------------------------------------
tf = tifffile.TiffFile(MOVIE)
T = tf.series[0].shape[0]
mm = tf.series[0].asarray(out="memmap")

meanF = np.empty(T, dtype=np.float64)
CHUNK = 64
for s in range(0, T, CHUNK):
    e = min(T, s + CHUNK)
    block = np.asarray(mm[s:e]).astype(np.float64)  # (chunk, 700, 700)
    meanF[s:e] = block[:, ys, xs].mean(axis=1)
del mm

WIN60 = int(round(60.0 * FS))  # ~60 s window for F0
ganglion = dff_from_meanF(meanF, WIN60).astype(np.float64)
np.save(os.path.join(DERIVED, "ganglion_signal.npy"), ganglion.astype(np.float32))

t = np.arange(T) * DT
duration_s = (T - 1) * DT

# cross-check: mean across all 180 patch dffs
patch = np.load(os.path.join(DERIVED, "traces_dff.npy")).astype(np.float64)
patch_mean = patch.mean(axis=0)

ganglion_mean_dff = float(np.mean(ganglion))
ganglion_peak_dff = float(np.max(ganglion))
print(f"ganglion mean dF/F={ganglion_mean_dff:.3f}, peak={ganglion_peak_dff:.3f}")

# ----------------------------------------------------------------------------
# Spectral analysis (Welch). Detrend the signal first.
# ----------------------------------------------------------------------------
def welch_psd(x, fs):
    x = x - np.mean(x)
    nperseg = min(256, len(x))
    f, P = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                        detrend="linear", scaling="density")
    return f, P

fG, PG = welch_psd(ganglion, FS)
fP, PP = welch_psd(patch_mean, FS)

# dominant frequency: ignore the lowest (DC/drift) bins below 0.02 Hz
valid = fG > 0.02
idx = np.argmax(PG[valid])
dom_freq = float(fG[valid][idx])
dom_power = float(PG[valid][idx])
dom_period = float(1.0 / dom_freq)

# rhythm prominence: peak power vs median (broadband) power
broadband = float(np.median(PG[valid]))
prominence_ratio = float(dom_power / broadband) if broadband > 0 else float("nan")
near_nyquist = bool(dom_freq > 0.8 * NYQ)
print(f"dominant freq={dom_freq:.4f} Hz, period={dom_period:.3f} s, "
      f"prominence={prominence_ratio:.1f}x, near_nyquist={near_nyquist}")

# ----------------------------------------------------------------------------
# Spectrogram: is the dominant frequency stable or drifting?
# ----------------------------------------------------------------------------
nps = min(128, T // 4)
f_sg, t_sg, Sxx = signal.spectrogram(
    ganglion - np.mean(ganglion), fs=FS, nperseg=nps,
    noverlap=int(nps * 0.85), detrend="linear", scaling="density")

# peak frequency per time column (above 0.02 Hz)
vsg = f_sg > 0.02
peak_f_t = f_sg[vsg][np.argmax(Sxx[vsg, :], axis=0)]
freq_drift = float(np.std(peak_f_t))
# stable if the spectrogram peak frequency stays within ~10% of the global dom freq
freq_stable = bool(np.median(np.abs(peak_f_t - dom_freq)) < 0.10 * dom_freq)
print(f"spectrogram peak-freq std={freq_drift:.4f} Hz, freq_stable={freq_stable}")

# instantaneous power in the dominant band
band = (f_sg >= dom_freq * 0.7) & (f_sg <= dom_freq * 1.3)
inst_power = Sxx[band, :].sum(axis=0)

# ----------------------------------------------------------------------------
# Autocorrelation + regularity (peak detection)
# ----------------------------------------------------------------------------
x = ganglion - np.mean(ganglion)
ac = np.correlate(x, x, mode="full")[T - 1:]
ac = ac / ac[0]
lags_s = np.arange(T) * DT

# first autocorr side peak (skip the zero-lag region up to ~0.3 of expected period)
min_lag = max(1, int((dom_period * 0.4) / DT))
ac_peaks, _ = signal.find_peaks(ac[min_lag:], height=0.0)
if len(ac_peaks) > 0:
    first_ac_lag_idx = ac_peaks[0] + min_lag
    ac_first_peak_lag_s = float(lags_s[first_ac_lag_idx])
    ac_first_peak_height = float(ac[first_ac_lag_idx])
else:
    ac_first_peak_lag_s = float("nan")
    ac_first_peak_height = float("nan")

# detect oscillation cycles on the signal itself
min_dist = max(1, int((dom_period * 0.5) / DT))
sig_peaks, _ = signal.find_peaks(ganglion, distance=min_dist,
                                prominence=0.3 * np.std(ganglion))
n_cycles = int(len(sig_peaks))
if n_cycles >= 2:
    interpeak = np.diff(sig_peaks) * DT
    median_interpeak = float(np.median(interpeak))
    interpeak_cv = float(np.std(interpeak) / np.mean(interpeak))
else:
    interpeak = np.array([])
    median_interpeak = float("nan")
    interpeak_cv = float("nan")
print(f"n_cycles={n_cycles}, median interpeak={median_interpeak:.3f} s, "
      f"CV={interpeak_cv:.3f}, ac first-peak height={ac_first_peak_height:.3f}")

# ----------------------------------------------------------------------------
# Coarse regional comparison (anterior/posterior + left/right via mask split)
# Use the long axis of the mask (PCA) to define anterior-posterior.
# ----------------------------------------------------------------------------
pts = np.column_stack([xs, ys]).astype(np.float64)
ctr = pts.mean(axis=0)
ptsc = pts - ctr
cov = np.cov(ptsc.T)
evals, evecs = np.linalg.eigh(cov)
long_axis = evecs[:, np.argmax(evals)]  # principal (longest) direction
proj = ptsc @ long_axis  # coordinate along long axis

# 3 regions along the long axis (terciles): R0, R1, R2
qs = np.quantile(proj, [1/3, 2/3])
reg_assign = np.digitize(proj, qs)  # 0,1,2

# region mean raw F -> dF/F (reuse the same per-pixel reads)
tf2 = tifffile.TiffFile(MOVIE)
mm2 = tf2.series[0].asarray(out="memmap")
n_regions = 3
regF = np.zeros((n_regions, T), dtype=np.float64)
counts = np.array([(reg_assign == r).sum() for r in range(n_regions)])
for s in range(0, T, CHUNK):
    e = min(T, s + CHUNK)
    block = np.asarray(mm2[s:e]).astype(np.float64)
    vals = block[:, ys, xs]  # (chunk, n_mask_px)
    for r in range(n_regions):
        regF[r, s:e] = vals[:, reg_assign == r].mean(axis=1)
del mm2

reg_dff = np.array([dff_from_meanF(regF[r], WIN60) for r in range(n_regions)])

# per-region dominant freq + phase lag vs region 0 (cross-correlation peak lag)
def xcorr_lag_ms(a, b, fs, max_lag_s):
    a = a - a.mean(); b = b - b.mean()
    a = a / (np.std(a) + 1e-9); b = b / (np.std(b) + 1e-9)
    n = len(a)
    cc = np.correlate(a, b, mode="full") / n
    lags = np.arange(-n + 1, n)
    ml = int(max_lag_s * fs)
    mid = n - 1
    sl = slice(mid - ml, mid + ml + 1)
    seg = cc[sl]; lseg = lags[sl]
    k = int(np.argmax(seg))
    # parabolic sub-sample interpolation for sub-frame lag resolution
    lag_samp = float(lseg[k])
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            lag_samp += 0.5 * (y0 - y2) / denom
    return float(lag_samp / fs * 1000.0), float(seg[k])

regional = []
for r in range(n_regions):
    fr, Pr = welch_psd(reg_dff[r], FS)
    vr = fr > 0.02
    dfr = float(fr[vr][np.argmax(Pr[vr])])
    lag_ms, cc_peak = xcorr_lag_ms(reg_dff[0], reg_dff[r], FS,
                                   max_lag_s=dom_period)
    regional.append({
        "region": int(r),
        "n_px": int(counts[r]),
        "dom_freq_hz": round(dfr, 5),
        "phase_lag_ms_vs_region0": round(lag_ms, 2),
        "xcorr_peak": round(cc_peak, 3),
    })
    print(f"region {r}: dom_freq={dfr:.4f} Hz, lag={lag_ms:.1f} ms, "
          f"xcorr={cc_peak:.3f}")

max_abs_lag = max(abs(rr["phase_lag_ms_vs_region0"]) for rr in regional)
traveling_wave = bool(max_abs_lag > 1000.0 * (dom_period * 0.05))

# ============================================================================
# FIGURE 1: ganglion signal vs time + zoom inset
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t, ganglion, color="#1b6f3b", lw=0.8, label="whole-ganglion mean dF/F")
ax.plot(t, patch_mean, color="#999999", lw=0.6, alpha=0.6,
        label="patch-population mean (cross-check)")
ax.axhline(ganglion_mean_dff, color="#1b6f3b", ls=":", lw=0.8)
ax.set_xlabel("time (s)")
ax.set_ylabel("dF/F")
ax.set_title("Stage 3: whole-ganglion mean dF/F over full recording "
             f"(fs={FS:.3f} Hz, {duration_s:.0f} s)")
ax.text(0.01, 0.97,
        f"mean dF/F = {ganglion_mean_dff:.3f}\npeak dF/F = {ganglion_peak_dff:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
ax.legend(loc="upper right", fontsize=8)

# zoom inset ~20 s, centered in a high-power region
z0 = duration_s * 0.4
z1 = z0 + 20.0
sel = (t >= z0) & (t <= z1)
axin = ax.inset_axes([0.62, 0.08, 0.35, 0.42])
axin.plot(t[sel], ganglion[sel], color="#1b6f3b", lw=1.0)
zp = sig_peaks[(t[sig_peaks] >= z0) & (t[sig_peaks] <= z1)]
axin.plot(t[zp], ganglion[zp], "v", color="#cc3311", ms=5)
axin.set_title(f"zoom {z0:.0f}-{z1:.0f} s (period ~{dom_period:.2f} s)",
               fontsize=8)
axin.tick_params(labelsize=7)
ax.indicate_inset_zoom(axin, edgecolor="black")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "s3_ganglion_signal.png"), dpi=140)
plt.close(fig)

# ============================================================================
# FIGURE 2: Welch PSD
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.semilogy(fG, PG, color="#1b6f3b", lw=1.4, label="ganglion")
ax.semilogy(fP, PP, color="#999999", lw=1.0, alpha=0.8,
            label="patch-population mean")
ax.axvline(dom_freq, color="#cc3311", ls="--", lw=1.2)
ax.axvline(NYQ, color="black", ls=":", lw=1.0)
ax.text(NYQ, ax.get_ylim()[1], f" Nyquist={NYQ:.3f} Hz", color="black",
        va="top", ha="right", fontsize=8, rotation=90)
ax.annotate(
    f"dominant\n{dom_freq:.3f} Hz\nperiod {dom_period:.2f} s\n"
    f"prominence {prominence_ratio:.0f}x"
    + ("\nNEAR NYQUIST" if near_nyquist else ""),
    xy=(dom_freq, dom_power), xytext=(dom_freq + 0.15, dom_power),
    color="#cc3311", fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#cc3311"))
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (dF/F^2 / Hz)")
ax.set_title("Stage 3: Welch power spectral density of ganglion signal")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "s3_spectrum.png"), dpi=140)
plt.close(fig)

# ============================================================================
# FIGURE 3: spectrogram + instantaneous dominant-band power
# ============================================================================
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
pcm = a1.pcolormesh(t_sg, f_sg, 10 * np.log10(Sxx + 1e-12),
                    shading="gouraud", cmap="magma")
a1.axhline(dom_freq, color="cyan", ls="--", lw=1.0)
a1.plot(t_sg, peak_f_t, ".", color="white", ms=3, alpha=0.7,
        label="per-window peak freq")
a1.set_ylabel("frequency (Hz)")
a1.set_ylim(0, min(NYQ, dom_freq * 3))
a1.set_title("Stage 3: spectrogram of ganglion signal "
             f"(freq_stable={freq_stable}, peak-freq std={freq_drift:.4f} Hz)")
a1.legend(loc="upper right", fontsize=8)
fig.colorbar(pcm, ax=a1, label="power (dB)")
a2.plot(t_sg, inst_power, color="#1b6f3b", lw=1.2)
a2.set_xlabel("time (s)")
a2.set_ylabel("dominant-band\npower")
a2.set_title(f"instantaneous power in {dom_freq*0.7:.3f}-{dom_freq*1.3:.3f} Hz band")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "s3_spectrogram.png"), dpi=140)
plt.close(fig)

# ============================================================================
# FIGURE 4: autocorrelation + inter-peak interval distribution
# ============================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
maxlag = min(T, int(30.0 / DT))
a1.plot(lags_s[:maxlag], ac[:maxlag], color="#1b6f3b", lw=1.0)
a1.axhline(0, color="gray", lw=0.6)
if np.isfinite(ac_first_peak_lag_s):
    a1.plot(ac_first_peak_lag_s, ac_first_peak_height, "o", color="#cc3311")
    a1.annotate(f"1st peak\nlag={ac_first_peak_lag_s:.2f} s\n"
                f"h={ac_first_peak_height:.2f}",
                xy=(ac_first_peak_lag_s, ac_first_peak_height),
                xytext=(ac_first_peak_lag_s + 2, ac_first_peak_height),
                fontsize=8, color="#cc3311")
a1.set_xlabel("lag (s)")
a1.set_ylabel("autocorrelation")
a1.set_title("autocorrelation of ganglion signal")

if len(interpeak) > 0:
    a2.hist(interpeak, bins=min(20, max(5, len(interpeak) // 2)),
            color="#1b6f3b", alpha=0.8, edgecolor="white")
    a2.axvline(median_interpeak, color="#cc3311", ls="--",
               label=f"median={median_interpeak:.2f} s")
    a2.legend(fontsize=9)
a2.set_xlabel("inter-peak interval (s)")
a2.set_ylabel("count")
a2.set_title(f"cycle intervals: n={n_cycles}, CV={interpeak_cv:.3f}")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "s3_autocorr_regularity.png"), dpi=140)
plt.close(fig)

# ============================================================================
# FIGURE 5: regional comparison
# ============================================================================
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2)
axm = fig.add_subplot(gs[:, 0])
region_rgb = np.zeros((*mask.shape, 3))
cols = [(0.85, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.4, 0.85)]
for r in range(n_regions):
    rr, cc2 = ys[reg_assign == r], xs[reg_assign == r]
    region_rgb[rr, cc2] = cols[r]
axm.imshow(mean_img, cmap="gray")
axm.imshow(region_rgb, alpha=(mask * 0.5))
axm.set_title("coarse regions along ganglion long axis")
axm.axis("off")

axt = fig.add_subplot(gs[0, 1])
zsel = (t >= z0) & (t <= z0 + 20)
for r in range(n_regions):
    axt.plot(t[zsel], reg_dff[r][zsel], color=cols[r], lw=1.0,
             label=f"R{r} ({regional[r]['dom_freq_hz']:.3f} Hz)")
axt.set_xlabel("time (s)")
axt.set_ylabel("dF/F")
axt.set_title("regional dF/F (20 s zoom)")
axt.legend(fontsize=8)

axl = fig.add_subplot(gs[1, 1])
labels = [f"R{rr['region']}" for rr in regional]
lags = [rr["phase_lag_ms_vs_region0"] for rr in regional]
axl.bar(labels, lags, color=cols)
axl.axhline(0, color="gray", lw=0.6)
axl.set_ylabel("phase lag vs R0 (ms)")
axl.set_title(f"cross-correlation lag (traveling_wave={traveling_wave})")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "s3_regional_comparison.png"), dpi=140)
plt.close(fig)

# ============================================================================
# Metrics
# ============================================================================
notes = (
    "10x low-mag preliminary elav-GCaMP6m line; signal is ganglion/regional, "
    "NOT single neurons. Recording annotated 'after dopamine pharyngals' but "
    "NO pre-dopamine baseline here, so no causal claim. "
    f"Nyquist={NYQ:.3f} Hz. "
    + ("Dominant peak is near Nyquist: possible temporal aliasing of a faster "
       "true rhythm; treat exact frequency with caution. "
       if near_nyquist else
       "Dominant peak is well below Nyquist; aliasing unlikely. ")
    + ("Regional cross-correlation lags suggest a phase gradient / possible "
       "traveling wave." if traveling_wave else
       "Regions are essentially synchronous (no significant phase gradient).")
)

metrics = {
    "fs": FS,
    "duration_s": round(duration_s, 3),
    "n_frames": int(T),
    "tissue_mask_px": n_mask_px,
    "ganglion_mean_dff": round(ganglion_mean_dff, 4),
    "ganglion_mean_dff_peak": round(ganglion_peak_dff, 4),
    "dominant_freq_hz": round(dom_freq, 5),
    "dominant_period_s": round(dom_period, 4),
    "peak_prominence_ratio": round(prominence_ratio, 2),
    "nyquist_hz": round(NYQ, 4),
    "near_nyquist": near_nyquist,
    "freq_stable": freq_stable,
    "spectrogram_peakfreq_std_hz": round(freq_drift, 5),
    "median_interpeak_s": (round(median_interpeak, 4)
                           if np.isfinite(median_interpeak) else None),
    "interpeak_cv": (round(interpeak_cv, 4)
                     if np.isfinite(interpeak_cv) else None),
    "n_cycles_detected": n_cycles,
    "autocorr_first_peak_lag_s": (round(ac_first_peak_lag_s, 4)
                                  if np.isfinite(ac_first_peak_lag_s) else None),
    "autocorr_first_peak_height": (round(ac_first_peak_height, 4)
                                   if np.isfinite(ac_first_peak_height) else None),
    "traveling_wave": traveling_wave,
    "regional": regional,
    "notes": notes,
}
with open(os.path.join(METRICS, "s3_temporal.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== DONE ===")
print(json.dumps(metrics, indent=2))
