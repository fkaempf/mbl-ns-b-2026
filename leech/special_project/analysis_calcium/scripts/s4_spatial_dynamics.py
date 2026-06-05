"""
Stage 4: Spatial dynamics of the ganglion-wide calcium oscillation.

HIGH-LEVEL framing: 10x low-mag, PRELIMINARY line, signal at GANGLION level.
No single-neuron claims. We characterize the SPATIAL structure of the
dominant ganglion-wide calcium oscillation: where it lives, its phase
structure (in-phase vs traveling wave), and whether the ganglion is one
coherent unit or split into territories.

Outputs:
  plots/s4_activity_maps.png
  plots/s4_pixelwise_rhythm_power.png
  plots/s4_phase_map.png
  plots/s4_regional_correlation.png
  plots/s4_peak_trough_frames.png
  metrics/s4_spatial.json
  derived/rhythm_power_map.npy
  derived/phase_map.npy
"""

import os
import json
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

ROOT = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
MOVIE = os.path.join(ROOT, "data/calcium/2026-06-04/helobdella_LeechNo2_elavgcamp6m-17.tif")
DERIVED = os.path.join(ROOT, "analysis_calcium/derived")
PLOTS = os.path.join(ROOT, "analysis_calcium/plots")
METRICS = os.path.join(ROOT, "analysis_calcium/metrics")
for d in (PLOTS, METRICS, DERIVED):
    os.makedirs(d, exist_ok=True)

FS = 2.882          # Hz
DS = 2              # spatial downsample factor for pixelwise FFT (700 -> 350)
H = W = 700
HD = WD = H // DS   # 350

rng = np.random.default_rng(0)


def pclip(img, lo=1, hi=99):
    a, b = np.nanpercentile(img, [lo, hi])
    if b <= a:
        b = a + 1e-9
    return np.clip((img - a) / (b - a), 0, 1)


# ---------------------------------------------------------------------------
# Load static images and ROI products
# ---------------------------------------------------------------------------
mean_img = np.load(os.path.join(DERIVED, "mean_image.npy")).astype(np.float32)
corr_img = np.load(os.path.join(DERIVED, "correlation_image.npy")).astype(np.float32)
roi_masks = np.load(os.path.join(DERIVED, "roi_masks.npy"))  # label = cell_id+1
dff = np.load(os.path.join(DERIVED, "traces_dff.npy")).astype(np.float32)  # (180,692)

import csv
centroids = []
with open(os.path.join(DERIVED, "roi_centroids.csv")) as f:
    r = csv.DictReader(f)
    for row in r:
        centroids.append((int(row["cell_id"]), float(row["y"]), float(row["x"])))
centroids = sorted(centroids, key=lambda t: t[0])
cent_yx = np.array([[c[1], c[2]] for c in centroids])  # (N,2)

n_frames = dff.shape[1]
print(f"loaded: mean{mean_img.shape} corr{corr_img.shape} dff{dff.shape} N_roi={len(centroids)}")

# ---------------------------------------------------------------------------
# Tissue outline / mask. Build from where signal lives: combine correlation
# image and std-over-time proxy. We compute std proj while streaming movie.
# ---------------------------------------------------------------------------
print("streaming movie for projections (chunked)...")
tif = tifffile.TiffFile(MOVIE)
n_pages = len(tif.pages)
n_frames = min(n_frames, n_pages)

# running stats for full-res projections
sum_img = np.zeros((H, W), np.float64)
sumsq_img = np.zeros((H, W), np.float64)
max_img = np.zeros((H, W), np.float32)

# downsampled movie cube for FFT (350x350x T) float32 ~ 339 MB, acceptable
cube = np.empty((n_frames, HD, WD), np.float32)
# spatial-mean signal over the whole FOV (use it for dom-freq + peak/trough)
spatial_mean = np.zeros(n_frames, np.float64)

CHUNK = 64
mm = tif.asarray(out="memmap")  # (T,H,W) memmap
for s in range(0, n_frames, CHUNK):
    e = min(s + CHUNK, n_frames)
    blk = np.asarray(mm[s:e]).astype(np.float32)  # (b,H,W)
    sum_img += blk.sum(0)
    sumsq_img += (blk.astype(np.float64) ** 2).sum(0)
    max_img = np.maximum(max_img, blk.max(0))
    spatial_mean[s:e] = blk.reshape(blk.shape[0], -1).mean(1)
    # downsample by 2x block-mean
    bd = blk.reshape(blk.shape[0], HD, DS, WD, DS).mean(axis=(2, 4))
    cube[s:e] = bd.astype(np.float32)
    print(f"  frames {s}-{e}", end="\r")
print()

mean_proj = (sum_img / n_frames).astype(np.float32)
var_proj = sumsq_img / n_frames - (sum_img / n_frames) ** 2
std_proj = np.sqrt(np.clip(var_proj, 0, None)).astype(np.float32)

# Tissue mask: std-over-time is the best "where does signal live" proxy at low mag.
std_ds = std_proj.reshape(HD, DS, WD, DS).mean(axis=(1, 3))
thr = np.percentile(std_ds, 80)
tissue_ds = std_ds > thr
tissue_ds = ndi.binary_opening(tissue_ds, iterations=1)
tissue_ds = ndi.binary_closing(tissue_ds, iterations=2)
# keep largest connected component(s) -> ganglion body
lbl, nl = ndi.label(tissue_ds)
if nl > 0:
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, nl + 1))
    keep = np.argsort(sizes)[::-1]
    big = sizes.max()
    keepmask = np.zeros_like(tissue_ds)
    for k in keep:
        if sizes[k] >= 0.05 * big:
            keepmask |= (lbl == (k + 1))
    tissue_ds = keepmask
tissue_ds = ndi.binary_fill_holes(tissue_ds)
tissue_full = np.kron(tissue_ds, np.ones((DS, DS), bool))
frac_tissue = tissue_ds.mean()
print(f"tissue fraction of FOV (ds): {frac_tissue:.3f}")

# ---------------------------------------------------------------------------
# Dominant frequency. Prefer s3 metrics / ganglion_signal if present.
# ---------------------------------------------------------------------------
dom_freq = None
dom_src = None
s3_path = os.path.join(METRICS, "s3_temporal.json")
gsig_path = os.path.join(DERIVED, "ganglion_signal.npy")

ganglion_sig = None
if os.path.exists(gsig_path):
    try:
        ganglion_sig = np.load(gsig_path).astype(np.float64)
        if ganglion_sig.shape[0] != n_frames:
            ganglion_sig = None
    except Exception:
        ganglion_sig = None

if os.path.exists(s3_path):
    try:
        with open(s3_path) as f:
            s3 = json.load(f)
        for k in ("dom_freq_hz", "dominant_freq_hz", "dom_freq", "peak_freq_hz"):
            if k in s3 and s3[k]:
                dom_freq = float(s3[k]); dom_src = f"s3:{k}"; break
    except Exception:
        pass

# signal used for spectral analysis: ganglion mean within tissue mask, detrended
# Build ganglion mean dF/F-style signal from the downsampled cube over tissue.
tis_cube = cube[:, tissue_ds]  # (T, npix_tissue)
gmean_raw = tis_cube.mean(1)
if ganglion_sig is None:
    ganglion_sig = gmean_raw.copy()

# detrend (remove slow bleach/baseline) with a linear+mean removal for FFT
t = np.arange(n_frames)
def detrend(x):
    x = np.asarray(x, float)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef

sig_for_fft = detrend(ganglion_sig)
win = np.hanning(n_frames)
freqs = np.fft.rfftfreq(n_frames, d=1.0 / FS)
sp = np.fft.rfft(sig_for_fft * win)
psd = np.abs(sp) ** 2

if dom_freq is None:
    # search physiological band: above slow drift, below Nyquist.
    band = (freqs > 0.02) & (freqs < 1.2)
    idx = np.argmax(psd * band)
    dom_freq = float(freqs[idx])
    dom_src = "computed:spatial_mean_psd"
print(f"dominant freq = {dom_freq:.4f} Hz  (source {dom_src})")

# index of dom freq bin
dom_bin = int(np.argmin(np.abs(freqs - dom_freq)))
dom_freq_used = float(freqs[dom_bin])

# ---------------------------------------------------------------------------
# Pixelwise FFT at dom freq (vectorized over downsampled cube).
# ---------------------------------------------------------------------------
print("pixelwise FFT (downsampled cube)...")
flat = cube.reshape(n_frames, -1).astype(np.float64)  # (T, HD*WD)
# detrend each pixel: subtract linear trend
A = np.vstack([t, np.ones_like(t)]).T
coef, *_ = np.linalg.lstsq(A, flat, rcond=None)  # (2, P)
flat_dt = flat - A @ coef
flat_dt *= win[:, None]
F = np.fft.rfft(flat_dt, axis=0)  # (Nf, P)
P_all = np.abs(F) ** 2
total_power = P_all[1:].sum(0)            # exclude DC
dom_power = P_all[dom_bin]                # (P,)
# normalized: fraction of variance at dom freq
norm_power = dom_power / (total_power + 1e-12)
phase = np.angle(F[dom_bin])             # (P,)

power_map = dom_power.reshape(HD, WD).astype(np.float32)
norm_map = norm_power.reshape(HD, WD).astype(np.float32)
phase_map = phase.reshape(HD, WD).astype(np.float32)

# Noise floor for "rhythmic": median power across neighboring bins (off-peak)
off = np.ones(P_all.shape[0], bool)
off[max(0, dom_bin - 2):dom_bin + 3] = False
off[0] = False
noise_floor = np.median(P_all[off], axis=0)  # (P,) per-pixel local noise level
rhythmic = (dom_power > 3.0 * noise_floor).reshape(HD, WD)
rhythmic_tissue = rhythmic & tissue_ds
frac_tissue_rhythmic = float(rhythmic_tissue.sum() / max(1, tissue_ds.sum()))
print(f"frac tissue pixels rhythmic (power>3x local noise): {frac_tissue_rhythmic:.3f}")

np.save(os.path.join(DERIVED, "rhythm_power_map.npy"), power_map)
np.save(os.path.join(DERIVED, "phase_map.npy"), phase_map)

# ---------------------------------------------------------------------------
# Phase structure within tissue (high-power pixels).
# ---------------------------------------------------------------------------
# restrict to tissue pixels that actually carry the rhythm
hp = tissue_ds & (dom_power.reshape(HD, WD) > np.percentile(dom_power.reshape(HD, WD)[tissue_ds], 50))
ph_vals = phase_map[hp]
# circular std
Cbar = np.mean(np.exp(1j * ph_vals))
R = np.abs(Cbar)
circ_std = float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))  # radians
mean_phase = float(np.angle(Cbar))
print(f"phase circular std (tissue, high-power) = {circ_std:.3f} rad ; R={R:.3f}")

# Phase gradient along band long axis: PCA on tissue coords, project, regress phase.
ys, xs = np.where(hp)
coords = np.column_stack([xs, ys]).astype(float)
coords_c = coords - coords.mean(0)
u, s_, vt = np.linalg.svd(coords_c, full_matrices=False)
long_axis = vt[0]
proj = coords_c @ long_axis  # position along long axis
# unwrap phase relative to mean before regressing (handle wrap)
ph_rel = np.angle(np.exp(1j * (ph_vals - mean_phase)))
# robust linear fit phase vs position
Apos = np.vstack([proj, np.ones_like(proj)]).T
slope, intercept = np.linalg.lstsq(Apos, ph_rel, rcond=None)[0]
pred = Apos @ np.array([slope, intercept])
ss_res = np.sum((ph_rel - pred) ** 2)
ss_tot = np.sum((ph_rel - ph_rel.mean()) ** 2) + 1e-12
r2 = 1 - ss_res / ss_tot
# total phase change across the band (rad)
span = proj.max() - proj.min()
total_phase_change = float(slope * span)
# wave detected if gradient explains meaningful variance AND phase spread non-trivial
phase_gradient_detected = bool((abs(r2) > 0.15) and (abs(total_phase_change) > 0.5))
grad_desc = (
    f"phase vs long-axis position: slope={slope:.4f} rad/px, "
    f"R2={r2:.3f}, total phase change across band={total_phase_change:.2f} rad "
    f"({np.degrees(total_phase_change):.0f} deg over {span:.0f} ds-px)"
)
print(grad_desc, "-> wave" if phase_gradient_detected else "-> in-phase")

# ---------------------------------------------------------------------------
# Regional correlation: spatial k-means on ROI centroids into ~6 clusters,
# mean dF/F per region, region x region correlation matrix.
# ---------------------------------------------------------------------------
N_REG = 6
km = KMeans(n_clusters=N_REG, n_init=10, random_state=0).fit(cent_yx)
reg_lab = km.labels_  # (N,) per ROI
reg_traces = np.zeros((N_REG, dff.shape[1]), np.float32)
reg_centers = np.zeros((N_REG, 2))
for k in range(N_REG):
    sel = reg_lab == k
    reg_traces[k] = dff[sel].mean(0)
    reg_centers[k] = cent_yx[sel].mean(0)
corr_mat = np.corrcoef(reg_traces)
iu = np.triu_indices(N_REG, k=1)
mean_inter = float(corr_mat[iu].mean())
min_inter = float(corr_mat[iu].min())
max_inter = float(corr_mat[iu].max())
print(f"inter-region corr: mean={mean_inter:.3f} min={min_inter:.3f} max={max_inter:.3f}")

# ---------------------------------------------------------------------------
# Peak / trough frames from ganglion signal.
# ---------------------------------------------------------------------------
# bandpass-ish: use detrended ganglion signal smoothed at dom-freq scale
gs = detrend(ganglion_sig)
period_fr = max(3, int(round(FS / dom_freq_used)))
sm = ndi.uniform_filter1d(gs, size=max(3, period_fr // 3))
# avoid edges
edge = period_fr
peak_t = edge + int(np.argmax(sm[edge:n_frames - edge]))
trough_t = edge + int(np.argmin(sm[edge:n_frames - edge]))
print(f"peak frame {peak_t}, trough frame {trough_t}")

peak_frame = np.asarray(mm[peak_t]).astype(np.float32)
trough_frame = np.asarray(mm[trough_t]).astype(np.float32)
diff_frame = peak_frame - trough_frame

# ===========================================================================
# FIGURES
# ===========================================================================
print("rendering figures...")

def draw_outline(ax, mask_full, color="cyan", lw=1.2):
    ax.contour(mask_full, levels=[0.5], colors=color, linewidths=lw)

# ---- Fig 1: activity maps ----
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
panels = [
    (mean_proj, "Robust mean projection"),
    (std_proj, "Std-over-time projection"),
    (max_img, "Max projection"),
    (corr_img, "Local correlation image"),
]
for ax, (img, title) in zip(axes.ravel(), panels):
    ax.imshow(pclip(img, 1, 99), cmap="magma")
    draw_outline(ax, tissue_full, "cyan", 1.2)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
fig.suptitle("Stage 4: Activity maps (where signal lives) with ganglion outline\n"
             "10x low-mag, preliminary, ganglion-level", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PLOTS, "s4_activity_maps.png"), dpi=130)
plt.close(fig)

# ---- Fig 2: pixelwise rhythm power (KEY) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
pm = power_map.copy()
pm_disp = np.where(tissue_ds, pm, np.nan)
vmax = np.nanpercentile(pm_disp, 99)
im0 = axes[0].imshow(pm_disp, cmap="inferno", vmax=vmax)
draw_outline(axes[0], tissue_ds, "cyan", 1.0)
axes[0].set_title(f"Oscillation POWER at {dom_freq_used:.3f} Hz (tissue)", fontsize=11)
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="power (a.u.)")

nm_disp = np.where(tissue_ds, norm_map, np.nan)
im1 = axes[1].imshow(nm_disp, cmap="viridis",
                     vmax=np.nanpercentile(nm_disp, 99))
draw_outline(axes[1], tissue_ds, "cyan", 1.0)
axes[1].set_title("Normalized: power at dom freq / total power", fontsize=11)
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="fraction")
fig.suptitle("Stage 4: Pixelwise rhythm power (2x downsampled, 350x350) - "
             "which ganglion parts carry the rhythm", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOTS, "s4_pixelwise_rhythm_power.png"), dpi=130)
plt.close(fig)

# ---- Fig 3: phase map ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
ph_disp = np.where(hp, phase_map, np.nan)
im = axes[0].imshow(ph_disp, cmap="twilight", vmin=-np.pi, vmax=np.pi)
draw_outline(axes[0], tissue_ds, "k", 0.8)
axes[0].set_title(f"Phase at {dom_freq_used:.3f} Hz (high-power tissue pixels)\n"
                  f"circular std = {circ_std:.2f} rad", fontsize=11)
axes[0].axis("off")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="phase (rad)")

# phase vs long-axis position scatter
axes[1].scatter(proj, np.degrees(ph_rel), s=4, alpha=0.25, color="steelblue")
xx = np.linspace(proj.min(), proj.max(), 50)
axes[1].plot(xx, np.degrees(slope * xx + intercept), "r-", lw=2,
             label=f"fit R2={r2:.2f}\n{np.degrees(total_phase_change):.0f} deg across band")
axes[1].set_xlabel("position along ganglion long axis (ds-px)")
axes[1].set_ylabel("phase rel. to mean (deg)")
axes[1].set_title("Phase gradient along long axis\n"
                  + ("traveling wave" if phase_gradient_detected else "in-phase (no wave)"),
                  fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
fig.suptitle("Stage 4: Phase structure of the ganglion oscillation", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOTS, "s4_phase_map.png"), dpi=130)
plt.close(fig)

# ---- Fig 4: regional correlation ----
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
cmap_reg = plt.get_cmap("tab10")
axes[0].imshow(pclip(corr_img, 1, 99), cmap="gray")
for k in range(N_REG):
    sel = reg_lab == k
    axes[0].scatter(cent_yx[sel, 1], cent_yx[sel, 0], s=22,
                    color=cmap_reg(k), edgecolor="k", linewidth=0.3,
                    label=f"R{k}")
    axes[0].text(reg_centers[k, 1], reg_centers[k, 0], str(k),
                 color="white", fontsize=13, fontweight="bold",
                 ha="center", va="center")
axes[0].set_title(f"{N_REG} spatial regions (k-means on ROI centroids)", fontsize=11)
axes[0].axis("off")
axes[0].legend(fontsize=8, loc="upper right", framealpha=0.6)

im = axes[1].imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_xticks(range(N_REG)); axes[1].set_yticks(range(N_REG))
axes[1].set_title(f"Region x region dF/F correlation\nmean inter={mean_inter:.2f}",
                  fontsize=11)
for i in range(N_REG):
    for j in range(N_REG):
        axes[1].text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=8, color="k")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Pearson r")
fig.suptitle("Stage 4: Regional coherence - one unit vs distinct territories",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOTS, "s4_regional_correlation.png"), dpi=130)
plt.close(fig)

# ---- Fig 5: peak / trough frames ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
# contrast-match peak & trough using shared percentiles
both = np.concatenate([peak_frame[tissue_full], trough_frame[tissue_full]])
lo, hi = np.percentile(both, [1, 99])
def show_matched(ax, img, title):
    ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="magma")
    draw_outline(ax, tissue_full, "cyan", 1.0)
    ax.set_title(title, fontsize=11); ax.axis("off")
show_matched(axes[0], peak_frame, f"PEAK frame (t={peak_t}, {peak_t/FS:.1f}s)")
show_matched(axes[1], trough_frame, f"TROUGH frame (t={trough_t}, {trough_t/FS:.1f}s)")
dlim = np.percentile(np.abs(diff_frame[tissue_full]), 99)
im = axes[2].imshow(diff_frame, cmap="RdBu_r", vmin=-dlim, vmax=dlim)
draw_outline(axes[2], tissue_full, "k", 0.8)
axes[2].set_title("Difference (peak - trough)", fontsize=11); axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="dF (a.u.)")
fig.suptitle("Stage 4: Peak vs trough raw frames - rhythm is real fluorescence "
             "modulation (motion already ruled out)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOTS, "s4_peak_trough_frames.png"), dpi=130)
plt.close(fig)

# ---------------------------------------------------------------------------
# Metrics JSON
# ---------------------------------------------------------------------------
metrics = {
    "dom_freq_hz_used": round(dom_freq_used, 5),
    "dom_freq_source": dom_src,
    "dom_period_s": round(1.0 / dom_freq_used, 3),
    "frac_tissue_pixels_rhythmic": round(frac_tissue_rhythmic, 4),
    "tissue_fraction_of_fov": round(float(frac_tissue), 4),
    "phase_circular_std_rad": round(circ_std, 4),
    "phase_resultant_R": round(float(R), 4),
    "phase_gradient_detected": phase_gradient_detected,
    "phase_gradient_description": grad_desc,
    "phase_gradient_total_rad_across_band": round(total_phase_change, 4),
    "phase_gradient_r2": round(float(r2), 4),
    "n_regions": N_REG,
    "mean_interregion_corr": round(mean_inter, 4),
    "min_interregion_corr": round(min_inter, 4),
    "max_interregion_corr": round(max_inter, 4),
    "peak_frame": int(peak_t),
    "trough_frame": int(trough_t),
    "fs_hz": FS,
    "n_frames": int(n_frames),
    "downsample_factor": DS,
    "notes": (
        "10x low-mag preliminary line; signal characterized at GANGLION level, "
        "ROIs treated as active patches not single neurons. Pixelwise FFT done "
        "on 2x-downsampled (350x350) movie for speed. Tissue mask from "
        "std-over-time 80th pctile + morphology, largest components. Motion "
        "ruled out in Stage 1 (drift=0px), so peak/trough modulation is real "
        "fluorescence, not registration artifact. No single-neuron claims."
    ),
}
with open(os.path.join(METRICS, "s4_spatial.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("done. metrics:")
print(json.dumps(metrics, indent=2))
