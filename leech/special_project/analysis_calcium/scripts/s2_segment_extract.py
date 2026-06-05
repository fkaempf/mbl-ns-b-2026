"""Stage 2: segmentation (ROI detection) and signal extraction for a leech
GCaMP6m calcium movie.

Runs suite2p (registration + ROI detection + extraction), and a fallback
blob-based ROI extractor. Picks whichever gives the more plausible set of
cells and writes a canonical output contract consumed by downstream stages.

No em dashes anywhere. matplotlib Agg backend.
"""

import os
import json
import copy

import numpy as np
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian

import suite2p


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
TIF = os.path.join(
    ROOT, "data/calcium/2026-06-04/helobdella_LeechNo2_elavgcamp6m-17.tif"
)
RUN_DIR = os.path.join(
    ROOT, "data/calcium/runs/helobdella_LeechNo2_elavgcamp6m-17"
)
S2P_PLANE = os.path.join(RUN_DIR, "suite2p", "plane0")

ANA = os.path.join(ROOT, "analysis_calcium")
PLOTS = os.path.join(ANA, "plots")
METRICS = os.path.join(ANA, "metrics")
DERIVED = os.path.join(ANA, "derived")
for d in (PLOTS, METRICS, DERIVED):
    os.makedirs(d, exist_ok=True)

FS = 2.882
TAU = 1.0
NEUROPIL_COEF = 0.7
DIAMETER = 15  # leech soma estimate ~12-20 px


# ----------------------------------------------------------------------------
# dF/F helper. F0 = 10th percentile of the (global) per-cell baseline.
# ----------------------------------------------------------------------------
def compute_dff(F_corr, percentile=10.0):
    """dF/F per cell, F0 = nth percentile of that cell's fluorescence."""
    F0 = np.percentile(F_corr, percentile, axis=1, keepdims=True)
    F0 = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    return (F_corr - F0) / F0


# ----------------------------------------------------------------------------
# 1. suite2p
# ----------------------------------------------------------------------------
def run_suite2p(force=False):
    """Run suite2p 1.1.0 via run_s2p(db, settings). Returns the loaded plane0
    outputs dict or None on failure."""
    have_outputs = all(
        os.path.exists(os.path.join(S2P_PLANE, f))
        for f in ("F.npy", "Fneu.npy", "stat.npy", "iscell.npy", "ops.npy")
    )
    if have_outputs and not force:
        print("suite2p outputs already present, loading existing run")
        return load_s2p()

    db = {
        "data_path": [os.path.dirname(TIF)],
        "tiff_list": [os.path.basename(TIF)],
        "file_list": [TIF],
        "input_format": "tif",
        "save_path0": RUN_DIR,
        "fast_disk": RUN_DIR,
        "save_folder": "suite2p",
        "nplanes": 1,
        "nchannels": 1,
        "functional_chan": 1,
        "look_one_level_down": False,
    }

    settings = copy.deepcopy(suite2p.default_settings())
    settings["tau"] = TAU
    settings["fs"] = FS
    settings["diameter"] = [float(DIAMETER), float(DIAMETER)]
    settings["torch_device"] = "cpu"
    settings["run"]["do_registration"] = 1
    settings["run"]["do_detection"] = True
    settings["run"]["do_deconvolution"] = True
    settings["registration"]["nonrigid"] = True
    # let sparsery detect; relax threshold a touch to find more cells
    settings["detection"]["algorithm"] = "sparsery"
    settings["detection"]["threshold_scaling"] = 0.8
    settings["detection"]["sparsery_settings"]["spatial_scale"] = 0
    settings["extraction"]["neuropil_coefficient"] = NEUROPIL_COEF

    print("launching suite2p run_s2p ...")
    suite2p.run_s2p(db=db, settings=settings)
    return load_s2p()


def load_s2p():
    out = {}
    out["F"] = np.load(os.path.join(S2P_PLANE, "F.npy"))
    out["Fneu"] = np.load(os.path.join(S2P_PLANE, "Fneu.npy"))
    out["stat"] = np.load(os.path.join(S2P_PLANE, "stat.npy"), allow_pickle=True)
    out["iscell"] = np.load(os.path.join(S2P_PLANE, "iscell.npy"))
    out["ops"] = np.load(
        os.path.join(S2P_PLANE, "ops.npy"), allow_pickle=True
    ).item()
    return out


# ----------------------------------------------------------------------------
# 2. Fallback ROI extraction (blob detection on a projection image)
# ----------------------------------------------------------------------------
def fallback_segmentation(mov, base_img, soma_radius=7):
    """Peak-detect + watershed on a normalized projection image to get round
    cell ROIs, then extract mean F per ROI per frame plus an annulus neuropil.

    mov: (T, Ly, Lx). base_img: 2D projection (correlation or std).
    Returns F, Fneu, centroids(list y,x,r), labels(2D int).
    """
    img = base_img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    sm = gaussian(img, sigma=1.5)

    # adaptive threshold: keep brighter-than-background structure
    thr = np.percentile(sm, 80)
    mask = sm > thr

    min_dist = max(3, int(round(soma_radius)))
    coords = peak_local_max(
        sm, min_distance=min_dist, threshold_abs=thr, labels=mask
    )
    if coords.shape[0] == 0:
        return None

    markers = np.zeros(sm.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    labels = watershed(-sm, markers, mask=mask)

    Ly, Lx = sm.shape
    yy, xx = np.mgrid[0:Ly, 0:Lx]

    centroids = []
    masks_keep = []
    npix_min = int(np.pi * (soma_radius * 0.45) ** 2)
    npix_max = int(np.pi * (soma_radius * 2.2) ** 2)
    for lab in range(1, labels.max() + 1):
        m = labels == lab
        n = int(m.sum())
        if n < npix_min or n > npix_max:
            continue
        cy, cx = ndi.center_of_mass(m)
        r = float(np.sqrt(n / np.pi))
        centroids.append((cy, cx, r))
        masks_keep.append(m)

    if not masks_keep:
        return None

    # relabel kept ROIs contiguously
    new_labels = np.zeros(sm.shape, dtype=np.int32)
    for i, m in enumerate(masks_keep, start=1):
        new_labels[m] = i

    n_cells = len(masks_keep)
    T = mov.shape[0]
    F = np.zeros((n_cells, T), dtype=np.float32)
    Fneu = np.zeros((n_cells, T), dtype=np.float32)

    cell_union = new_labels > 0
    movf = mov.reshape(T, -1)
    for i, (m, (cy, cx, r)) in enumerate(zip(masks_keep, centroids)):
        idx = np.flatnonzero(m.ravel())
        F[i] = movf[:, idx].mean(axis=1)
        # annulus neuropil: ring from r+2 to r+2+6, excluding any cell pixels
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        inner = (r + 2.0) ** 2
        outer = (r + 8.0) ** 2
        ring = (d2 >= inner) & (d2 <= outer) & (~cell_union)
        ridx = np.flatnonzero(ring.ravel())
        if ridx.size >= 20:
            Fneu[i] = movf[:, ridx].mean(axis=1)
    return F, Fneu, centroids, new_labels


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_rois_on_mean(base_img, centroids, masks2d, path, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(base_img, [1, 99])
    ax.imshow(base_img, cmap="gray", vmin=vmin, vmax=vmax)
    if masks2d is not None:
        from skimage.segmentation import find_boundaries

        bnd = find_boundaries(masks2d, mode="outer")
        overlay = np.zeros((*masks2d.shape, 4))
        overlay[bnd] = [1, 0.5, 0, 1]
        ax.imshow(overlay)
    for i, (cy, cx, r) in enumerate(centroids):
        circ = plt.Circle((cx, cy), r, color="cyan", fill=False, lw=0.8)
        ax.add_patch(circ)
        ax.text(cx, cy, str(i), color="yellow", fontsize=6, ha="center",
                va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_example_traces(dff, path, fs, n_show=12):
    t = np.arange(dff.shape[1]) / fs
    # pick the most active cells by dF/F std
    order = np.argsort(-dff.std(axis=1))
    sel = order[: min(n_show, dff.shape[0])]
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 0.0
    step = np.percentile(dff[sel], 99) * 1.2 + 0.5
    for k, ci in enumerate(sel):
        ax.plot(t, dff[ci] + offset, lw=0.7)
        ax.text(t[-1] * 1.005, offset, f"#{ci}", fontsize=7, va="center")
        offset += step
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F (offset per cell)")
    ax.set_title("Stage 2 example dF/F traces (most active cells)")
    ax.set_xlim(0, t[-1] * 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_registration_check(ops, path):
    xoff = np.asarray(ops.get("xoff", []))
    yoff = np.asarray(ops.get("yoff", []))
    fig, ax = plt.subplots(figsize=(10, 4))
    if xoff.size:
        t = np.arange(xoff.size) / FS
        ax.plot(t, xoff, label="xoff (px)", lw=0.8)
        ax.plot(t, yoff, label="yoff (px)", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rigid shift (px)")
        ax.legend()
        ax.set_title(
            "Stage 2 registration check: rigid offsets "
            "(|x|max=%g |y|max=%g px)"
            % (np.abs(xoff).max(), np.abs(yoff).max())
        )
    else:
        ax.text(0.5, 0.5, "no offsets available", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("loading movie ...")
    mov = tifffile.imread(TIF)
    if mov.ndim == 4:
        mov = mov[:, 0] if mov.shape[1] < mov.shape[-1] else mov[..., 0]
    T = mov.shape[0]
    print("movie", mov.shape, mov.dtype)

    force = os.environ.get("S2_FORCE_S2P", "1") == "1"
    s2p = run_suite2p(force=force)
    ops = s2p["ops"]
    meanImg = np.asarray(ops["meanImg"])
    corr_img = np.asarray(ops.get("Vcorr", meanImg))

    # save a correlation image for downstream / fallback reuse
    np.save(os.path.join(DERIVED, "correlation_image.npy"), corr_img)
    np.save(os.path.join(DERIVED, "mean_image.npy"), meanImg)

    iscell = s2p["iscell"][:, 0].astype(bool)
    n_s2p = int(iscell.sum())
    print("suite2p iscell count:", n_s2p, "of", len(iscell), "ROIs")

    # ---- suite2p cell traces / centroids / masks ----
    stat = s2p["stat"]
    Ly, Lx = ops["Ly"], ops["Lx"]
    sel = np.flatnonzero(iscell)
    F_s2p = s2p["F"][sel]
    Fneu_s2p = s2p["Fneu"][sel]
    cent_s2p = []
    labels_s2p = np.zeros((Ly, Lx), dtype=np.int32)
    for new_i, si in enumerate(sel, start=1):
        st = stat[si]
        ys, xs = st["ypix"], st["xpix"]
        labels_s2p[ys, xs] = new_i
        cy, cx = float(st["med"][0]), float(st["med"][1])
        r = float(st.get("radius", np.sqrt(len(ys) / np.pi)))
        cent_s2p.append((cy, cx, r))

    # ---- fallback ----
    print("running fallback blob segmentation ...")
    fb = fallback_segmentation(
        mov, corr_img, soma_radius=DIAMETER / 2.0
    )
    n_fb = 0 if fb is None else fb[0].shape[0]
    print("fallback cell count:", n_fb)

    # ---- choose method ----
    # Prefer suite2p unless it yields too few/poor cells AND fallback finds
    # meaningfully more plausible cells.
    use_fallback = (n_s2p < 15) and (fb is not None) and (n_fb > n_s2p)
    if use_fallback:
        method = "fallback"
        F, Fneu, centroids, labels = fb
        F_corr = F - NEUROPIL_COEF * Fneu
        f0_method = "10th percentile of per-cell global fluorescence"
        base_for_fig = corr_img
    else:
        method = "suite2p"
        F, Fneu = F_s2p, Fneu_s2p
        F_corr = F - NEUROPIL_COEF * Fneu
        centroids, labels = cent_s2p, labels_s2p
        f0_method = "10th percentile of per-cell global fluorescence"
        base_for_fig = meanImg

    n_cells = F.shape[0]
    print("CHOSEN method:", method, "n_cells:", n_cells)

    dff = compute_dff(F_corr, percentile=10.0)

    # ---- write canonical outputs ----
    np.save(os.path.join(DERIVED, "traces_F.npy"), F.astype(np.float32))
    np.save(os.path.join(DERIVED, "traces_Fneu.npy"), Fneu.astype(np.float32))
    np.save(os.path.join(DERIVED, "traces_dff.npy"), dff.astype(np.float32))
    np.save(os.path.join(DERIVED, "roi_masks.npy"), labels.astype(np.int32))

    import csv

    with open(os.path.join(DERIVED, "roi_centroids.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "y", "x", "radius_px"])
        for i, (cy, cx, r) in enumerate(centroids):
            w.writerow([i, round(cy, 3), round(cx, 3), round(r, 3)])

    reg_used = bool(ops.get("xoff") is not None and len(np.asarray(ops["xoff"])))
    meta = {
        "fs": FS,
        "n_cells": int(n_cells),
        "n_frames": int(T),
        "method": method,
        "neuropil_coef": NEUROPIL_COEF,
        "f0_method": f0_method,
        "registration_used": reg_used,
        "source_file": TIF,
        "suite2p_n_cells": int(n_s2p),
        "fallback_n_cells": int(n_fb),
        "tau": TAU,
        "diameter_px": DIAMETER,
    }
    with open(os.path.join(DERIVED, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- metrics ----
    xoff = np.asarray(ops.get("xoff", []))
    yoff = np.asarray(ops.get("yoff", []))
    dff_amp = dff.max(axis=1)
    active_frac = float(np.mean(dff_amp > 0.10))
    metrics = {
        "n_cells": int(n_cells),
        "method": method,
        "reg_xoff_absmax": float(np.abs(xoff).max()) if xoff.size else None,
        "reg_yoff_absmax": float(np.abs(yoff).max()) if yoff.size else None,
        "dff_p99": float(np.percentile(dff, 99)),
        "dff_max": float(dff.max()),
        "median_peak_dff": float(np.median(dff_amp)),
        "fraction_active_dff_gt_0.1": active_frac,
        "median_radius_px": float(np.median([c[2] for c in centroids])),
    }
    with open(os.path.join(METRICS, "s2_segmentation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- figures ----
    fig_rois_on_mean(
        base_for_fig, centroids, labels,
        os.path.join(PLOTS, "s2_rois_on_mean.png"),
        "Stage 2 ROIs (%s, n=%d)" % (method, n_cells),
    )
    fig_example_traces(
        dff, os.path.join(PLOTS, "s2_example_traces.png"), FS, n_show=14
    )
    fig_registration_check(
        ops, os.path.join(PLOTS, "s2_registration_check.png")
    )

    print("DONE")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
