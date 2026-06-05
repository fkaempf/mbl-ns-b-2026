"""
Stage 2b: Independent (non-suite2p) segmentation + signal extraction.

Suite2p (Stage 2) under-segmented this dense confocal ganglion (24 cells).
Here we build a correlation-image blob + watershed segmentation that captures
the visible somata along the ganglionic band, extract raw traces, neuropil,
and dF/F, and compare to suite2p. caiman/CNMF is not installed.

Method: det = normalize(Cn) * normalize(std_proj), tissue-masked, seeded by
peak_local_max, split by watershed, area/border/intensity filtered.

Run with:
  /Users/fkampf/.../.venv/bin/python analysis_calcium/scripts/s2b_segment_corr.py
"""

import os
import json
import shutil

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, exposure
from skimage.feature import peak_local_max

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
for d in (DERIVED, PLOTS, METRICS):
    os.makedirs(d, exist_ok=True)

FS = 2.882
NEUROPIL_COEF = 0.7
SOMA_RADIUS = 6          # full-res soma radius in px (~11 px median diameter)
MIN_DISTANCE = 5         # peak_local_max separation ~ soma radius
DET_SMOOTH_SIGMA = 1.0   # light Gaussian on detection image
AREA_MIN = 25            # px^2 lower soma area bound
AREA_MAX = 500           # px^2 upper soma area bound
F0_WINDOW_S = 60.0       # rolling baseline window (~173 frames)


def norm01(a):
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    if hi <= lo:
        hi = a.max()
        lo = a.min()
    out = (a - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 0. Backup suite2p canonicals to *_s2p before we overwrite the primaries
# ----------------------------------------------------------------------------
def backup_s2p():
    pairs = [
        ("traces_F.npy", "traces_F_s2p.npy"),
        ("traces_Fneu.npy", "traces_Fneu_s2p.npy"),
        ("traces_dff.npy", "traces_dff_s2p.npy"),
        ("roi_centroids.csv", "roi_centroids_s2p.csv"),
        ("roi_masks.npy", "roi_masks_s2p.npy"),
    ]
    for src, dst in pairs:
        sp = os.path.join(DERIVED, src)
        dp = os.path.join(DERIVED, dst)
        if os.path.exists(dp):
            print("  [keep] existing backup", dst)
            continue
        if os.path.exists(sp):
            shutil.copy2(sp, dp)
            print("  [backup]", src, "->", dst)
        else:
            print("  [warn] missing suite2p canonical", src)
    # also stash suite2p meta for n_cells reference
    return None


# ----------------------------------------------------------------------------
# 1. Load detection inputs and compute std projection over time
# ----------------------------------------------------------------------------
def load_detection_inputs():
    cn = np.load(os.path.join(DERIVED, "correlation_image.npy")).astype(np.float64)
    mean_img = np.load(os.path.join(DERIVED, "mean_image.npy")).astype(np.float64)

    H, W = mean_img.shape
    # Upsample Cn to full res if needed.
    if cn.shape != (H, W):
        zoom = (H / cn.shape[0], W / cn.shape[1])
        cn = ndi.zoom(cn, zoom, order=1)
        print("  upsampled Cn to", cn.shape)

    # Std projection over time (welford-style two-pass, chunked to bound RAM).
    mm = tifffile.memmap(MOVIE)
    T = mm.shape[0]
    ssum = np.zeros((H, W), np.float64)
    ssq = np.zeros((H, W), np.float64)
    chunk = 64
    for i in range(0, T, chunk):
        block = np.asarray(mm[i:i + chunk], dtype=np.float64)
        ssum += block.sum(axis=0)
        ssq += (block * block).sum(axis=0)
    mean_t = ssum / T
    var_t = np.maximum(ssq / T - mean_t * mean_t, 0.0)
    std_proj = np.sqrt(var_t)
    del mm
    return cn, mean_img, std_proj, T


# ----------------------------------------------------------------------------
# 2. Detection image + tissue mask
# ----------------------------------------------------------------------------
def build_detection(cn, mean_img, std_proj):
    cn_n = norm01(cn)
    std_n = norm01(std_proj)
    det = cn_n * std_n
    det = filters.gaussian(det, sigma=DET_SMOOTH_SIGMA, preserve_range=True)

    # Tissue mask: the active ganglionic band spans more than one cluster, so we
    # use the Cn-based Otsu threshold alone (which selects the high-correlation
    # somata region, ~7-14% of FOV) rather than intersecting with a mean
    # threshold that clips the dimmer secondary band. We keep all sizeable
    # connected components, not just the single largest.
    cn_sm = filters.gaussian(cn, sigma=2, preserve_range=True)
    t_cn = filters.threshold_otsu(cn_sm)
    tissue = cn_sm > t_cn
    tissue = morphology.closing(tissue, morphology.disk(3))
    tissue = morphology.remove_small_holes(tissue, max_size=150)
    tissue = morphology.remove_small_objects(tissue, min_size=150)
    return det, tissue, cn_n, std_n


# ----------------------------------------------------------------------------
# 3-5. Seeds, watershed, ROI filtering
# ----------------------------------------------------------------------------
def segment(det, tissue, mean_img):
    det_t = det.copy()
    det_t[~tissue] = 0.0

    coords = peak_local_max(
        det_t,
        min_distance=MIN_DISTANCE,
        threshold_rel=0.03,
        exclude_border=False,
        labels=tissue.astype(int),
    )
    seeds = np.zeros(det.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        seeds[y, x] = k
    print("  raw seeds:", len(coords))

    # Watershed on inverted detection image so basins sit on bright somata.
    elevation = -det_t
    labels_ws = segmentation.watershed(elevation, markers=seeds, mask=tissue)

    # Filter ROIs.
    bg_level = np.percentile(mean_img[tissue], 30)
    keep = []
    H, W = det.shape
    props = measure.regionprops(labels_ws, intensity_image=mean_img)
    for p in props:
        area = p.area
        if area < AREA_MIN or area > AREA_MAX:
            continue
        miny, minx, maxy, maxx = p.bbox
        if miny == 0 or minx == 0 or maxy == H or maxx == W:
            continue  # touches image border
        if p.intensity_mean <= bg_level:
            continue
        keep.append(p.label)

    # Relabel kept ROIs 1..N contiguous.
    new = np.zeros_like(labels_ws)
    centroids = []
    for new_id, old in enumerate(keep, start=1):
        m = labels_ws == old
        new[m] = new_id
        ys, xs = np.where(m)
        cy, cx = ys.mean(), xs.mean()
        rad = np.sqrt(m.sum() / np.pi)
        centroids.append((new_id - 1, cy, cx, rad))
    print("  kept ROIs:", len(keep))
    return new, centroids


# ----------------------------------------------------------------------------
# 6. Signal extraction (raw movie) with local neuropil annulus
# ----------------------------------------------------------------------------
def extract_traces(masks, T):
    n = int(masks.max())
    all_rois = masks > 0

    # Precompute per-cell pixel indices and neuropil annulus indices.
    roi_px = []
    neu_px = []
    for cid in range(1, n + 1):
        m = masks == cid
        roi_px.append(np.where(m.ravel())[0])
        dil = morphology.dilation(m, morphology.disk(8))
        annulus = dil & (~all_rois)  # exclude all ROIs (self + neighbors)
        neu_px.append(np.where(annulus.ravel())[0])

    F = np.zeros((n, T), np.float32)
    Fneu = np.zeros((n, T), np.float32)
    mm = tifffile.memmap(MOVIE)
    chunk = 64
    for i in range(0, T, chunk):
        block = np.asarray(mm[i:i + chunk], dtype=np.float32)
        nf = block.shape[0]
        flat = block.reshape(nf, -1)
        for c in range(n):
            F[c, i:i + nf] = flat[:, roi_px[c]].mean(axis=1)
            if neu_px[c].size > 0:
                Fneu[c, i:i + nf] = flat[:, neu_px[c]].mean(axis=1)
            else:
                Fneu[c, i:i + nf] = 0.0
    del mm
    return F, Fneu


# ----------------------------------------------------------------------------
# 7. dF/F with rolling 10th-percentile baseline
# ----------------------------------------------------------------------------
def compute_dff(F_corr):
    n, T = F_corr.shape
    win = int(round(F0_WINDOW_S * FS))
    win = max(win, 11)
    if win % 2 == 0:
        win += 1
    f0_method = "rolling 10th percentile, window %d frames (~%.0f s)" % (
        win, win / FS)
    dff = np.zeros_like(F_corr)
    half = win // 2
    use_rolling = win < T
    for c in range(n):
        x = F_corr[c]
        if use_rolling:
            f0 = ndi.percentile_filter(x, 10, size=win, mode="nearest")
            # Guard against tiny/negative baselines.
            g = np.percentile(x, 10)
            f0 = np.where(f0 <= 1e-3, max(g, 1e-3), f0)
        else:
            f0 = np.full_like(x, max(np.percentile(x, 10), 1e-3))
        dff[c] = (x - f0) / f0
    return dff, f0_method


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_rois_on_corr(cn_n, std_n, masks, centroids):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    boundaries = segmentation.find_boundaries(masks, mode="outer")
    for ax, base, title in (
        (axes[0], cn_n, "ROIs on correlation image"),
        (axes[1], std_n, "ROIs on std projection"),
    ):
        ax.imshow(base, cmap="gray")
        overlay = np.zeros((*base.shape, 4))
        overlay[boundaries] = [1, 0.3, 0, 1]
        ax.imshow(overlay)
        for cid, cy, cx, rad in centroids:
            ax.text(cx, cy, str(cid), color="cyan", fontsize=4,
                    ha="center", va="center")
        ax.set_title("%s (n=%d)" % (title, len(centroids)))
        ax.axis("off")
    fig.suptitle("Stage 2b correlation-blob watershed ROIs", fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOTS, "s2b_rois_on_corr.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_method_comparison(cn_n, masks_mine, n_mine, masks_s2p, n_s2p):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, masks, n, title in (
        (axes[0], masks_s2p, n_s2p, "suite2p"),
        (axes[1], masks_mine, n_mine, "correlation blob + watershed"),
    ):
        ax.imshow(cn_n, cmap="gray")
        if masks is not None:
            b = segmentation.find_boundaries(masks, mode="outer")
            ov = np.zeros((*cn_n.shape, 4))
            ov[b] = [1, 0.3, 0, 1]
            ax.imshow(ov)
        ax.set_title("%s: n_cells=%d" % (title, n))
        ax.axis("off")
    fig.suptitle("Under-segmentation: suite2p vs Stage 2b "
                 "(same correlation image)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS, "s2b_method_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_example_traces(dff, n_show=15):
    n, T = dff.shape
    t = np.arange(T) / FS
    # Pick most active cells by 95th-pct dF/F.
    score = np.percentile(dff, 95, axis=1)
    order = np.argsort(score)[::-1][:min(n_show, n)]
    fig, ax = plt.subplots(figsize=(12, 9))
    offset = 0.0
    step = 1.5 * np.median(np.percentile(dff[order], 99, axis=1)) + 1e-3
    for rank, c in enumerate(order):
        ax.plot(t, dff[c] + offset, lw=0.6)
        ax.text(-8, offset, "cell %d" % c, fontsize=7, va="center", ha="right")
        offset += step
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F (offset stacked)")
    ax.set_title("Stage 2b dF/F, %d most active cells" % len(order))
    ax.set_xlim(-12, t[-1])
    ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(PLOTS, "s2b_example_traces.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("== Stage 2b segmentation ==")
    print("Backing up suite2p canonicals...")
    backup_s2p()

    print("Loading detection inputs + std projection...")
    cn, mean_img, std_proj, T = load_detection_inputs()

    print("Building detection image + tissue mask...")
    det, tissue, cn_n, std_n = build_detection(cn, mean_img, std_proj)
    print("  tissue fraction of FOV: %.3f" % (tissue.mean()))

    print("Segmenting...")
    masks, centroids = segment(det, tissue, mean_img)
    n_cells = int(masks.max())
    if n_cells == 0:
        raise RuntimeError("No ROIs survived filtering; loosen thresholds.")

    print("Extracting traces from raw movie...")
    F, Fneu = extract_traces(masks, T)
    F_corr = F - NEUROPIL_COEF * Fneu

    print("Computing dF/F...")
    dff, f0_method = compute_dff(F_corr)

    # ---- write canonical outputs (primary = my method) ----
    np.save(os.path.join(DERIVED, "traces_F.npy"), F.astype(np.float32))
    np.save(os.path.join(DERIVED, "traces_Fneu.npy"), Fneu.astype(np.float32))
    np.save(os.path.join(DERIVED, "traces_dff.npy"), dff.astype(np.float32))
    np.save(os.path.join(DERIVED, "roi_masks.npy"), masks.astype(np.int32))

    import csv
    with open(os.path.join(DERIVED, "roi_centroids.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "y", "x", "radius_px"])
        for cid, cy, cx, rad in centroids:
            w.writerow([cid, round(cy, 2), round(cx, 2), round(rad, 3)])

    meta = {
        "fs": FS,
        "n_cells": n_cells,
        "n_frames": int(T),
        "method": "correlation_blob_watershed",
        "neuropil_coef": NEUROPIL_COEF,
        "f0_method": f0_method,
        "registration_used": False,
        "source_file": MOVIE,
        "n_cells_s2p": 24,
        "detection_params": {
            "min_distance": MIN_DISTANCE,
            "det_smooth_sigma": DET_SMOOTH_SIGMA,
            "soma_radius_px": SOMA_RADIUS,
            "area_min_px2": AREA_MIN,
            "area_max_px2": AREA_MAX,
            "neuropil_annulus_dilate_px": 8,
        },
        "label_offset_note": "roi_masks label = cell_id + 1 (0 = background)",
    }
    with open(os.path.join(DERIVED, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- figures ----
    print("Writing figures...")
    masks_s2p = np.load(os.path.join(DERIVED, "roi_masks_s2p.npy"))
    n_s2p = int(masks_s2p.max())
    p1 = fig_rois_on_corr(cn_n, std_n, masks, centroids)
    p2 = fig_method_comparison(cn_n, masks, n_cells, masks_s2p, n_s2p)
    p3 = fig_example_traces(dff)

    # ---- quick metrics ----
    areas = np.array([np.sum(masks == c) for c in range(1, n_cells + 1)])
    peak = np.percentile(dff, 99, axis=1)
    active = int(np.sum(peak > 0.10))
    metrics = {
        "n_cells": n_cells,
        "n_cells_s2p": n_s2p,
        "median_area_px2": float(np.median(areas)),
        "median_radius_px": float(np.median(np.sqrt(areas / np.pi))),
        "fraction_active_dff_gt_0.1": active / n_cells,
        "dff_global_min": float(dff.min()),
        "dff_global_max": float(dff.max()),
        "dff_median_99pct": float(np.median(peak)),
        "tissue_fraction_fov": float(tissue.mean()),
    }
    with open(os.path.join(METRICS, "s2b_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("== DONE ==")
    print(json.dumps(metrics, indent=2))
    print("figures:", p1, p2, p3)


if __name__ == "__main__":
    main()
