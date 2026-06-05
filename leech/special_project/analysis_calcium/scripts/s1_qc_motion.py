"""
Stage 1: Quality control and motion-correction decision for a leech GCaMP6m
calcium-imaging movie.

Input movie (target):
  data/calcium/2026-06-04/helobdella_LeechNo2_elavgcamp6m-17.tif
  shape (T=692, 700, 700), uint16, single channel, fs=2.882 Hz.

Outputs:
  plots/   s1_projections.png, s1_correlation_image.png,
           s1_brightness_bleach.png, s1_motion.png
  metrics/ s1_qc.json
  derived/ correlation_image.npy

No em dashes anywhere. matplotlib Agg backend. Memmap the movie.
"""

import os
import json
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
from skimage.transform import downscale_local_mean

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
MOVIE = os.path.join(
    ROOT, "data/calcium/2026-06-04/helobdella_LeechNo2_elavgcamp6m-17.tif"
)
PLOT_DIR = os.path.join(ROOT, "analysis_calcium/plots")
METRIC_DIR = os.path.join(ROOT, "analysis_calcium/metrics")
DERIVED_DIR = os.path.join(ROOT, "analysis_calcium/derived")
for d in (PLOT_DIR, METRIC_DIR, DERIVED_DIR):
    os.makedirs(d, exist_ok=True)

FS = 2.882          # Hz
FRAME_DT = 0.34702  # s

# ----------------------------------------------------------------------
# Load (memmap)
# ----------------------------------------------------------------------
mov = tifffile.memmap(MOVIE)
T, H, W = mov.shape
duration_s = T * FRAME_DT
t = np.arange(T) * FRAME_DT
print(f"Loaded memmap shape={mov.shape} dtype={mov.dtype} duration={duration_s:.1f}s")


def clip_norm(img, lo=1, hi=99):
    """Robust percentile contrast normalization to [0,1]."""
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        b = a + 1.0
    return np.clip((img - a) / (b - a), 0, 1)


# ----------------------------------------------------------------------
# Pass 1: per-frame mean intensity + running stats for projections
# Iterate in chunks to keep RAM bounded.
# ----------------------------------------------------------------------
print("Pass 1: per-frame mean + projections (chunked)...")
frame_mean = np.zeros(T, dtype=np.float64)
sum_img = np.zeros((H, W), dtype=np.float64)
sumsq_img = np.zeros((H, W), dtype=np.float64)
max_img = np.full((H, W), -np.inf, dtype=np.float32)

# per-frame saturation / clipping checks
frame_max = np.zeros(T, dtype=np.float64)
frame_min = np.zeros(T, dtype=np.float64)

CHUNK = 32
for s in range(0, T, CHUNK):
    e = min(s + CHUNK, T)
    blk = np.asarray(mov[s:e], dtype=np.float32)
    frame_mean[s:e] = blk.reshape(blk.shape[0], -1).mean(axis=1)
    frame_max[s:e] = blk.reshape(blk.shape[0], -1).max(axis=1)
    frame_min[s:e] = blk.reshape(blk.shape[0], -1).min(axis=1)
    sum_img += blk.sum(axis=0)
    sumsq_img += (blk.astype(np.float64) ** 2).sum(axis=0)
    max_img = np.maximum(max_img, blk.max(axis=0))

mean_img = (sum_img / T).astype(np.float32)
var_img = np.maximum(sumsq_img / T - (sum_img / T) ** 2, 0.0)
std_img = np.sqrt(var_img).astype(np.float32)

global_max_val = float(frame_max.max())
print(f"  global max pixel value = {global_max_val}")

# ----------------------------------------------------------------------
# Figure 1: projections
# ----------------------------------------------------------------------
print("Figure 1: projections...")
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
for a, img, ttl in zip(
    ax,
    [mean_img, max_img, std_img],
    ["Mean projection", "Max projection", "Std-over-time projection"],
):
    a.imshow(clip_norm(img), cmap="gray")
    a.set_title(ttl)
    a.axis("off")
fig.suptitle(
    "S1 projections: helobdella LeechNo2 elavgcamp6m-17 "
    "(std localizes active neurons)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "s1_projections.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: brightness / photobleaching (compute before motion for clarity)
# ----------------------------------------------------------------------
print("Figure 3: brightness / bleaching...")


def exp_decay(x, a, tau, c):
    return a * np.exp(-x / tau) + c


bleach_pct = np.nan
fit_ok = False
try:
    p0 = [frame_mean[0] - frame_mean[-1], duration_s / 2.0, frame_mean[-1]]
    popt, _ = curve_fit(exp_decay, t, frame_mean, p0=p0, maxfev=20000)
    fit_curve = exp_decay(t, *popt)
    f0 = exp_decay(t[0], *popt)
    fend = exp_decay(t[-1], *popt)
    bleach_pct = float((f0 - fend) / f0 * 100.0)
    fit_ok = True
except Exception as ex:  # noqa
    print("  exp fit failed:", ex)
    fit_curve = None
    # fallback: linear endpoints
    bleach_pct = float(
        (frame_mean[:30].mean() - frame_mean[-30:].mean())
        / frame_mean[:30].mean() * 100.0
    )

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t, frame_mean, lw=0.8, color="C0", label="mean frame intensity")
if fit_ok:
    ax.plot(
        t, fit_curve, lw=2.0, color="C3",
        label=f"exp fit (tau={popt[1]:.1f}s, bleach={bleach_pct:.2f}%)",
    )
ax.set_xlabel("time (s)")
ax.set_ylabel("mean intensity (a.u.)")
ax.set_title("S1 brightness vs time and photobleaching fit")
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "s1_brightness_bleach.png"), dpi=130)
plt.close(fig)
print(f"  bleach_pct = {bleach_pct:.3f}%")

# ----------------------------------------------------------------------
# Figure 4: motion assessment
#   (a) full rigid drift trace via phase_cross_correlation on 2x downsampled
#   (b) 3x3 per-block drift on a subset of frames (non-rigid check)
#   (c) frame-to-frame correlation vs time
# ----------------------------------------------------------------------
print("Figure 4: motion assessment...")

DS = 2  # spatial downsample for rigid registration


def ds2(img):
    return downscale_local_mean(img.astype(np.float32), (DS, DS))


# reference = mean of first 30 frames
ref_full = np.asarray(mov[:30], dtype=np.float32).mean(axis=0)
ref_ds = ds2(ref_full)

dx = np.zeros(T)
dy = np.zeros(T)
for i in range(T):
    fr = ds2(np.asarray(mov[i], dtype=np.float32))
    shift, _, _ = phase_cross_correlation(
        ref_ds, fr, upsample_factor=4, normalization=None
    )
    # shift is (row, col) in downsampled px; scale back to full-res px
    dy[i] = shift[0] * DS
    dx[i] = shift[1] * DS

drift_mag = np.sqrt(dx ** 2 + dy ** 2)
rigid_max = float(drift_mag.max())
rigid_mean = float(drift_mag.mean())
rigid_p95 = float(np.percentile(drift_mag, 95))
print(f"  rigid drift px max={rigid_max:.3f} mean={rigid_mean:.3f} p95={rigid_p95:.3f}")

# (b) non-rigid: 3x3 block per-frame drift on subset of frames
n_sub = 60
sub_idx = np.linspace(0, T - 1, n_sub).astype(int)
gy = np.linspace(0, H, 4).astype(int)
gx = np.linspace(0, W, 4).astype(int)

# block references from mean of first 30 frames
ref30 = np.asarray(mov[:30], dtype=np.float32).mean(axis=0)
block_refs = []
block_slices = []
for bi in range(3):
    for bj in range(3):
        ys, ye = gy[bi], gy[bi + 1]
        xs, xe = gx[bj], gx[bj + 1]
        block_slices.append((ys, ye, xs, xe))
        block_refs.append(ref30[ys:ye, xs:xe])

block_drift = np.zeros((len(sub_idx), 9))
for k, fi in enumerate(sub_idx):
    frame = np.asarray(mov[fi], dtype=np.float32)
    for b, (ys, ye, xs, xe) in enumerate(block_slices):
        blk = frame[ys:ye, xs:xe]
        try:
            sh, _, _ = phase_cross_correlation(
                block_refs[b], blk, upsample_factor=4, normalization=None
            )
            block_drift[k, b] = np.sqrt(sh[0] ** 2 + sh[1] ** 2)
        except Exception:
            block_drift[k, b] = np.nan

nonrigid_max = float(np.nanmax(block_drift))
# residual non-rigid = how much blocks disagree with each other per frame
block_residual = np.nanstd(block_drift, axis=1)
nonrigid_residual_max = float(np.nanmax(block_residual))
print(f"  nonrigid (per-block) drift max={nonrigid_max:.3f} "
      f"inter-block-residual max={nonrigid_residual_max:.3f}")

# (c) frame-to-frame correlation
print("  frame-to-frame correlation...")
framecorr = np.full(T, np.nan)
prev = np.asarray(mov[0], dtype=np.float32).ravel()
prev = prev - prev.mean()
prev_norm = np.sqrt((prev ** 2).sum())
for i in range(1, T):
    cur = np.asarray(mov[i], dtype=np.float32).ravel()
    cur = cur - cur.mean()
    cur_norm = np.sqrt((cur ** 2).sum())
    denom = prev_norm * cur_norm
    framecorr[i] = (prev @ cur) / denom if denom > 0 else np.nan
    prev, prev_norm = cur, cur_norm

framecorr_valid = framecorr[1:]
framecorr_mean = float(np.nanmean(framecorr_valid))
framecorr_min = float(np.nanmin(framecorr_valid))
print(f"  framecorr mean={framecorr_mean:.4f} min={framecorr_min:.4f}")

# plot motion
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].plot(t, dx, lw=0.8, label="dx (px)")
ax[0, 0].plot(t, dy, lw=0.8, label="dy (px)")
ax[0, 0].set_title("Rigid shift dx, dy vs time (full-res px)")
ax[0, 0].set_xlabel("time (s)")
ax[0, 0].set_ylabel("shift (px)")
ax[0, 0].legend(fontsize=9)

ax[0, 1].plot(t, drift_mag, lw=0.8, color="C2")
ax[0, 1].axhline(rigid_p95, color="C3", ls="--", lw=1,
                 label=f"p95={rigid_p95:.2f}px")
ax[0, 1].set_title(f"Rigid drift magnitude (max={rigid_max:.2f}px)")
ax[0, 1].set_xlabel("time (s)")
ax[0, 1].set_ylabel("drift |shift| (px)")
ax[0, 1].legend(fontsize=9)

for b in range(9):
    ax[1, 0].plot(sub_idx * FRAME_DT, block_drift[:, b], lw=0.7, alpha=0.8)
ax[1, 0].set_title(f"Per-block (3x3) drift, subset (max={nonrigid_max:.2f}px)")
ax[1, 0].set_xlabel("time (s)")
ax[1, 0].set_ylabel("block drift (px)")

ax[1, 1].plot(t[1:], framecorr_valid, lw=0.8, color="C4")
ax[1, 1].axhline(framecorr_mean, color="C3", ls="--", lw=1,
                 label=f"mean={framecorr_mean:.3f}")
ax[1, 1].set_title(f"Frame-to-frame correlation (min={framecorr_min:.3f})")
ax[1, 1].set_xlabel("time (s)")
ax[1, 1].set_ylabel("Pearson r")
ax[1, 1].legend(fontsize=9)

fig.suptitle("S1 motion assessment: rigid drift, non-rigid block check, frame corr",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "s1_motion.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: local correlation image Cn (8-neighbor), on 2x downsampled movie
# ----------------------------------------------------------------------
print("Figure 2: local correlation image Cn...")

# Build a downsampled movie stack in RAM (2x => 350x350x692 float32 ~ 340 MB).
Hd, Wd = (H + DS - 1) // DS, (W + DS - 1) // DS
# use downscale_local_mean to get consistent shape
ref_shape = ds2(np.asarray(mov[0], dtype=np.float32)).shape
Hd, Wd = ref_shape
stack = np.empty((T, Hd, Wd), dtype=np.float32)
for i in range(T):
    stack[i] = ds2(np.asarray(mov[i], dtype=np.float32))

# z-score each pixel over time
mu = stack.mean(axis=0)
sd = stack.std(axis=0)
sd[sd == 0] = 1.0
z = (stack - mu) / sd  # (T,Hd,Wd)

# 8-neighbor local correlation: mean over neighbors of (1/T) sum_t z*z_neighbor
def shifted_sum(zz, dyy, dxx):
    """Sum over time of z * z shifted by (dyy,dxx), aligned, with valid mask."""
    src = zz
    # build product of z and neighbor; pad to keep shape
    prod = np.zeros((Hd, Wd), dtype=np.float64)
    # overlapping region
    ys0, ys1 = max(0, dyy), Hd + min(0, dyy)
    xs0, xs1 = max(0, dxx), Wd + min(0, dxx)
    yn0, yn1 = max(0, -dyy), Hd + min(0, -dyy)
    xn0, xn1 = max(0, -dxx), Wd + min(0, -dxx)
    a = src[:, ys0:ys1, xs0:xs1]
    b = src[:, yn0:yn1, xn0:xn1]
    corr = (a * b).mean(axis=0)  # mean over time => correlation (z-scored)
    prod[ys0:ys1, xs0:xs1] = corr
    return prod, (ys0, ys1, xs0, xs1)

neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]
corr_sum = np.zeros((Hd, Wd), dtype=np.float64)
count = np.zeros((Hd, Wd), dtype=np.float64)
for (dyy, dxx) in neighbors:
    c, (ys0, ys1, xs0, xs1) = shifted_sum(z, dyy, dxx)
    corr_sum[ys0:ys1, xs0:xs1] += c[ys0:ys1, xs0:xs1]
    count[ys0:ys1, xs0:xs1] += 1.0
count[count == 0] = 1.0
Cn = (corr_sum / count).astype(np.float32)

np.save(os.path.join(DERIVED_DIR, "correlation_image.npy"), Cn)
print(f"  Cn shape={Cn.shape} min={Cn.min():.3f} max={Cn.max():.3f} "
      f"mean={Cn.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
im0 = ax[0].imshow(clip_norm(Cn, 2, 99.5), cmap="magma")
ax[0].set_title("Local correlation image Cn (2x downsampled, 8-neighbor)")
ax[0].axis("off")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(clip_norm(std_img), cmap="magma")
ax[1].set_title("Std-over-time projection (full-res, for comparison)")
ax[1].axis("off")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
fig.suptitle("S1 correlation image: cell bodies visible as bright blobs",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "s1_correlation_image.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------
needs_rigid_mc = bool(rigid_p95 > 1.0 or rigid_max > 2.0)
needs_nonrigid_mc = bool(nonrigid_residual_max > 1.0 and nonrigid_max > 2.0)
needs_detrend = bool(abs(bleach_pct) > 5.0)

reasons = []
reasons.append(
    f"rigid drift p95={rigid_p95:.2f}px max={rigid_max:.2f}px "
    f"({'above' if needs_rigid_mc else 'below'} sub-pixel threshold)"
)
reasons.append(
    f"non-rigid inter-block residual max={nonrigid_residual_max:.2f}px "
    f"({'present' if needs_nonrigid_mc else 'negligible'})"
)
reasons.append(
    f"bleaching={bleach_pct:.2f}% "
    f"({'detrend' if needs_detrend else 'no detrend needed'})"
)
reason_str = "; ".join(reasons)

metrics = {
    "movie": os.path.relpath(MOVIE, ROOT),
    "fs": FS,
    "nframes": int(T),
    "duration_s": round(duration_s, 3),
    "frame_dt_s": FRAME_DT,
    "bleach_pct": round(float(bleach_pct), 4),
    "bleach_fit_ok": fit_ok,
    "rigid_drift_px_max": round(rigid_max, 4),
    "rigid_drift_px_mean": round(rigid_mean, 4),
    "rigid_drift_px_p95": round(rigid_p95, 4),
    "nonrigid_drift_px_max": round(nonrigid_max, 4),
    "nonrigid_interblock_residual_px_max": round(nonrigid_residual_max, 4),
    "framecorr_mean": round(framecorr_mean, 4),
    "framecorr_min": round(framecorr_min, 4),
    "global_max_pixel_value": global_max_val,
    "saturation_suspected": bool(global_max_val >= 4094),
    "needs_rigid_mc": needs_rigid_mc,
    "needs_nonrigid_mc": needs_nonrigid_mc,
    "needs_detrend": needs_detrend,
    "reason": reason_str,
}
with open(os.path.join(METRIC_DIR, "s1_qc.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== METRICS ===")
print(json.dumps(metrics, indent=2))
print("\nDone. All outputs written.")
