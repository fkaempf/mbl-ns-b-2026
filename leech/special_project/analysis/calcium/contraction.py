"""Body-wall contraction vs ganglion rhythm — timing analysis.

Minimal annotation: one line drawn across the body-wall edge. We sample the RAW
(un-registered) movie along that line every frame to build a kymograph, track the
edge position over time (= contraction trace), then compare its timing to the
ganglion rhythm using peak-to-peak intervals and a cross-correlation lag.

  # 1. draw one line across the body-wall edge:
  python -m analysis.calcium.contraction annotate <run_dir>
  # 2. run the timing analysis:
  python -m analysis.calcium.contraction analyze <run_dir>

Uses the RAW movie because suite2p registration removes the very motion we want.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import find_peaks

from . import signals


# ------------------------------------------------------------------ raw I/O ---
def raw_movie_path(run_dir: str | Path) -> Path:
    """Find the original TIFF for a run (registration removed the motion)."""
    run_dir = Path(run_dir)
    plane = run_dir / "suite2p" / "plane0"
    if not (plane / "db.npy").exists():
        plane = run_dir if (run_dir / "db.npy").exists() else run_dir / "plane0"
    db = np.load(plane / "db.npy", allow_pickle=True).item()  # our own run output
    for key in ("file_list", "tiff_list"):
        fl = db.get(key)
        if fl:
            p = Path(fl[0])
            if p.exists():
                return p
    # fallback: search by stem
    stem = run_dir.name
    for cand in Path("data/calcium").rglob(f"{stem}.tif"):
        return cand
    raise FileNotFoundError(f"raw TIFF for {run_dir} not found")


def load_raw(run_dir: str | Path) -> np.ndarray:
    import tifffile
    path = raw_movie_path(run_dir)
    try:
        return tifffile.memmap(path)            # contiguous ImageJ tiff -> memmap
    except (ValueError, MemoryError):
        return tifffile.imread(path)


# ----------------------------------------------------------- pure functions ---
def kymograph(movie: np.ndarray, p0, p1, n: int | None = None) -> np.ndarray:
    """Sample intensity along the line p0->p1 (each (row, col)) every frame.

    Returns (T, n): rows = frames, cols = position along the line.
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    if n is None:
        n = int(round(np.hypot(*(p1 - p0)))) + 1
    rows = np.linspace(p0[0], p1[0], n)
    cols = np.linspace(p0[1], p1[1], n)
    T = movie.shape[0]
    kymo = np.empty((T, n), dtype="float32")
    for t in range(T):
        kymo[t] = map_coordinates(movie[t], [rows, cols], order=1, mode="nearest")
    return kymo


def kymograph_perp(movie: np.ndarray, p0, p1, half_width: int = 14,
                   n_along: int | None = None) -> np.ndarray:
    """Kymograph for a line drawn ALONG a (moving) edge.

    At each point along p0->p1 we sample a short profile PERPENDICULAR to the
    line and average those profiles, then track the edge's perpendicular
    position over time. This measures the wall's displacement perpendicular to
    itself (real movement), averaged along its length for SNR — not brightness.

    Returns (T, 2*half_width+1): rows = frames, cols = perpendicular position.
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    if n_along is None:
        n_along = max(2, int(round(L)))
    d = d / (L + 1e-9)
    nrm = np.array([-d[1], d[0]])               # unit normal to the line
    along = np.linspace(p0, p1, n_along)        # (n_along, 2) in (row, col)
    perp = np.arange(-half_width, half_width + 1)
    # grid of sample coords: (n_along, n_perp, 2)
    coords = along[:, None, :] + perp[None, :, None] * nrm[None, None, :]
    rr = coords[..., 0].ravel(); cc = coords[..., 1].ravel()
    T = movie.shape[0]
    kymo = np.empty((T, len(perp)), dtype="float32")
    for t in range(T):
        prof = map_coordinates(movie[t], [rr, cc], order=1, mode="nearest")
        kymo[t] = prof.reshape(n_along, len(perp)).mean(axis=0)
    return kymo


def track_shift_ncc(kymo: np.ndarray, max_shift: int | None = None) -> np.ndarray:
    """Brightness-INVARIANT perpendicular displacement per frame.

    Each frame's profile is normalised (z-scored) and cross-correlated against
    the median profile; the sub-pixel peak gives the translation. Because both
    profiles are z-scored, a pure brightness oscillation (uniform scaling) does
    NOT move the peak — only real movement does. This distinguishes genuine
    wall displacement from the brightness rhythm.
    """
    k = kymo - kymo.mean(axis=1, keepdims=True)
    k = k / (k.std(axis=1, keepdims=True) + 1e-9)
    ref = np.median(k, axis=0)
    n = k.shape[1]
    if max_shift is None:
        max_shift = n // 3
    lags = np.arange(-max_shift, max_shift + 1)
    shift = np.zeros(kymo.shape[0])
    for t in range(kymo.shape[0]):
        cc = np.array([np.dot(k[t], np.roll(ref, L)) for L in lags])
        i = int(np.argmax(cc))
        if 0 < i < len(cc) - 1:
            a, b, c = cc[i - 1], cc[i], cc[i + 1]
            d = a - 2 * b + c
            sub = 0.5 * (a - c) / d if d != 0 else 0.0
        else:
            sub = 0.0
        shift[t] = lags[i] + sub
    return shift


def phase_at(x: np.ndarray, freq: float, fs: float) -> float:
    """Phase (rad) of signal x at frequency `freq` (single-bin DFT)."""
    x = np.asarray(x, float); x = x - x.mean()
    t = np.arange(len(x))
    return float(np.angle(np.sum(x * np.exp(-2j * np.pi * freq * t / fs))))


def track_edge(kymo: np.ndarray, smooth: float = 1.5) -> np.ndarray:
    """Sub-pixel edge position (in samples along the line) per frame.

    The edge = location of steepest intensity gradient along the line; refined
    to sub-pixel by a parabolic fit around the gradient peak.
    """
    k = gaussian_filter1d(kymo, smooth, axis=1)
    grad = np.abs(np.gradient(k, axis=1))
    pos = np.empty(kymo.shape[0], dtype="float64")
    for t in range(kymo.shape[0]):
        i = int(np.argmax(grad[t]))
        if 0 < i < grad.shape[1] - 1:
            a, b, c = grad[t, i - 1], grad[t, i], grad[t, i + 1]
            denom = (a - 2 * b + c)
            i = i + (0.5 * (a - c) / denom if denom != 0 else 0.0)
        pos[t] = i
    return pos


def peak_intervals(x: np.ndarray, fs: float, min_sep_s: float = 1.0,
                   prominence: float | None = None):
    """Detect peaks and return (peak_indices, median_interval_s).

    This is the 'delta between peaks' frequency estimate (no FFT).
    """
    x = np.asarray(x, float)
    if prominence is None:
        prominence = 0.5 * np.std(x)
    peaks, _ = find_peaks(x, distance=max(1, int(round(min_sep_s * fs))),
                          prominence=prominence)
    if len(peaks) < 2:
        return peaks, float("nan")
    return peaks, float(np.median(np.diff(peaks)) / fs)


def peak_delays(ref_peaks: np.ndarray, tgt_peaks: np.ndarray, fs: float,
                max_s: float = 3.0):
    """For each ref peak, signed delay to the nearest tgt peak, in seconds.

    Positive => tgt peak comes AFTER the ref peak (tgt lags ref).
    """
    out = []
    for p in ref_peaks:
        if len(tgt_peaks) == 0:
            continue
        d = tgt_peaks - p
        j = np.argmin(np.abs(d))
        if abs(d[j]) <= max_s * fs:
            out.append(d[j] / fs)
    return np.array(out)


# ---------------------------------------------------------- line annotation ---
def _load_lines(path: str | Path) -> list[dict]:
    """Read the line file; supports old single-line and new multi-line formats."""
    d = json.loads(Path(path).read_text())
    if "lines" in d:
        return d["lines"]
    return [{"name": "wall1", "p0": d["p0"], "p1": d["p1"]}]


def annotate_line(run_dir: str | Path, out_path: str | Path | None = None) -> None:
    import napari
    from magicgui.widgets import Container, PushButton, Label

    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "bodywall_line.json"
    run = signals.load_run(run_dir)

    viewer = napari.Viewer(title=f"body-wall line — {run_dir.name}")
    viewer.add_image(run.mean_img, name="mean image", colormap="gray")
    viewer.add_image(run.movie, name="movie (registered)", colormap="gray",
                     visible=False)
    lines = viewer.add_shapes(name="bodywall line", shape_type="line",
                              edge_color="cyan", edge_width=3)
    if out_path.exists():
        for seg in _load_lines(out_path):
            lines.add([np.array([seg["p0"], seg["p1"]])], shape_type="line")

    status = Label(value="draw a line ALONG each body wall (one or both); it "
                         "tracks perpendicular movement. Then Save.")
    btn = PushButton(text="Save lines")

    def _save():
        if not len(lines.data):
            status.value = "no line drawn"; return
        segs = [np.asarray(s) for s in lines.data]      # save ALL lines
        out_path.write_text(json.dumps(
            {"lines": [{"name": f"wall{i+1}", "p0": s[0].tolist(),
                        "p1": s[1].tolist()} for i, s in enumerate(segs)],
             "run_dir": str(run_dir)}, indent=2))
        status.value = f"saved {len(segs)} line(s) -> {out_path}"
        print(status.value)

    btn.clicked.connect(_save)
    viewer.window.add_dock_widget(Container(widgets=[btn, status]),
                                  name="bodywall", area="right")
    napari.run()


# ----------------------------------------------------------------- analysis ---
def _wall_signal(raw, seg, fs):
    """Per-wall contraction trace + its timing features.

    Displacement is the brightness-invariant NCC shift (not the gradient edge),
    so a pure brightness rhythm does not masquerade as movement.
    """
    kymo = kymograph_perp(raw, seg["p0"], seg["p1"])
    shift = track_shift_ncc(kymo)
    contract = -(shift - np.median(shift))         # up = wall moved inward
    edge = shift - shift.min() + 1                 # for kymograph overlay
    peaks, period = peak_intervals(contract, fs)
    return {"kymo": kymo, "edge": edge, "contract": contract,
            "peaks": peaks, "period": period}


def analyze(run_dir: str | Path, line_path: str | Path | None = None,
            out_dir: str | Path | None = None) -> dict:
    run_dir = Path(run_dir)
    line_path = Path(line_path) if line_path else run_dir / "bodywall_line.json"
    out = Path(out_dir) if out_dir else run_dir / "contraction"
    out.mkdir(parents=True, exist_ok=True)
    walls_in = _load_lines(line_path)

    raw = load_raw(run_dir)
    reg = signals.load_run(run_dir)
    fs = reg.fs

    # ganglion rhythm = global-mean dF/F of the registered movie
    gm = reg.movie.reshape(reg.movie.shape[0], -1).mean(axis=1)
    dff = signals.dff(gm[None], fs)[0]
    r_peaks, r_period = peak_intervals(dff, fs)

    from .ganglion_activity import crosscorr_lag
    from .phase_analysis import dominant_frequency
    f0, _ = dominant_frequency(dff, fs)               # shared rhythm frequency

    walls = []
    for seg in walls_in:
        w = _wall_signal(raw, seg, fs)
        c = w["contract"]
        # delay vs rhythm: phase (at f0), peak-matched, and cross-correlation
        dphase = np.angle(np.exp(1j * (phase_at(c, f0, fs) - phase_at(dff, f0, fs))))
        lag, lag_r = crosscorr_lag(dff, c, max_lag=int(round(3 * fs)))
        delays = peak_delays(r_peaks, w["peaks"], fs)
        w.update(name=seg.get("name", "wall"),
                 phase_delay_ms=float(dphase / (2 * np.pi * f0) * 1000),
                 xcorr_lag_s=float(lag / fs), xcorr_r=float(lag_r),
                 median_peak_delay_ms=float(np.median(delays) * 1000) if len(delays) else float("nan"),
                 excursion_px=float(np.ptp(w["edge"])), delays=delays)
        walls.append(w)

    # inter-wall relationship (bilateral coordination), if two walls
    inter = None
    if len(walls) >= 2:
        a, b = walls[0]["contract"], walls[1]["contract"]
        dph = np.angle(np.exp(1j * (phase_at(b, f0, fs) - phase_at(a, f0, fs))))
        inter = {"phase_diff_ms": float(dph / (2 * np.pi * f0) * 1000),
                 "corr": float(np.corrcoef(a, b)[0, 1])}

    _plots(walls, dff, fs, r_peaks, out)

    result = {
        "shared_freq_hz": f0, "period_s": 1.0 / f0, "rhythm_period_s": r_period,
        "n_walls": len(walls),
        "walls": [{"name": w["name"], "period_s": w["period"],
                   "phase_delay_ms": w["phase_delay_ms"],
                   "median_peak_delay_ms": w["median_peak_delay_ms"],
                   "xcorr_lag_s": w["xcorr_lag_s"], "xcorr_r": w["xcorr_r"],
                   "excursion_px": w["excursion_px"]} for w in walls],
        "inter_wall": inter, "out": str(out),
    }
    np.savez(out / "contraction.npz", dff=dff, fs=fs, f0=f0,
             **{f"{w['name']}_contract": w["contract"] for w in walls},
             **{f"{w['name']}_kymo": w["kymo"] for w in walls})
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def _plots(walls, dff, fs, r_peaks, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(dff); t = np.arange(T) / fs
    nw = len(walls)

    # 1) per-wall kymographs + tracked edge (the diagnostic)
    fig, axes = plt.subplots(nw, 1, figsize=(11, 3.2 * nw), squeeze=False)
    for ax, w in zip(axes[:, 0], walls):
        ax.imshow(w["kymo"].T, aspect="auto", cmap="gray",
                  extent=[0, t[-1], w["kymo"].shape[1], 0])
        ax.plot(t, w["edge"], color="cyan", lw=1)
        ax.set_ylabel(f"{w['name']}\nperp pos (px)")
        ax.set_title(f"{w['name']} kymograph + tracked edge "
                     f"(wavy = moving; flat = still)")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out / "kymograph.png", dpi=120); plt.close(fig)

    # 2) rhythm + each wall's contraction, peaks marked
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t, (dff - dff.mean()) / (dff.std() + 1e-9), color="k", lw=1,
            label="ganglion rhythm (z)")
    ax.plot(t[r_peaks], ((dff - dff.mean()) / (dff.std() + 1e-9))[r_peaks], "k^", ms=5)
    for i, w in enumerate(walls):
        c = w["contract"]; z = (c - c.mean()) / (c.std() + 1e-9)
        ax.plot(t, z - 4 * (i + 1), lw=0.9,
                label=f"{w['name']} contraction (Δφ={w['phase_delay_ms']:+.0f} ms)")
        ax.plot(t[w["peaks"]], (z - 4 * (i + 1))[w["peaks"]], "v", ms=4)
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.set_title("ganglion rhythm vs body-wall contraction (z-scored; markers = peaks)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "contraction_vs_rhythm.png", dpi=120)
    plt.close(fig)

    # 3) delay histogram(s): rhythm peak -> contraction peak
    fig, ax = plt.subplots(figsize=(7, 4))
    for w in walls:
        if len(w["delays"]):
            ax.hist(w["delays"] * 1000, bins=20, alpha=0.6, label=w["name"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("delay: contraction peak − rhythm peak (ms)  (+ = contraction after rhythm)")
    ax.set_ylabel("count"); ax.set_title("rhythm → contraction delay"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "delay_histogram.png", dpi=120); plt.close(fig)


# ----------------------------------------- from-TIFF (2-channel) variant ------
def load_tiff(path):
    """Load a (T,C,Y,X) or (T,Y,X) tiff -> (gcamp, struct, fs, (g_ch, s_ch)).

    For 2-channel GCaMP+mCherry: GCaMP = higher temporal variance (functional),
    struct (mCherry) = lower variance. Single-channel returns the same array twice.
    """
    import tifffile
    arr = np.asarray(tifffile.imread(path), dtype="float32")
    with tifffile.TiffFile(path) as tf:
        md = tf.imagej_metadata or {}
    fs = round(1.0 / md["finterval"], 4) if md.get("finterval") else 1.0
    if arr.ndim == 4:                              # (T, C, Y, X)
        v = [arr[:, c].var(0).mean() for c in range(arr.shape[1])]
        g, s = int(np.argmax(v)), int(np.argmin(v))
        return arr[:, g], arr[:, s], fs, (g, s)
    return arr, arr, fs, (0, 0)


def annotate_line_tiff(tiff_path, out_path=None) -> None:
    """Draw body-wall line(s) on a raw tiff (struct channel as background)."""
    import napari
    from magicgui.widgets import Container, PushButton, Label
    tiff_path = Path(tiff_path)
    out_path = Path(out_path) if out_path else tiff_path.with_suffix(".bodywall_line.json")
    g, s, fs, chans = load_tiff(tiff_path)
    viewer = napari.Viewer(title=f"body-wall line — {tiff_path.name}")
    viewer.add_image(s.mean(0), name="struct mean (mCherry)", colormap="gray")
    viewer.add_image(s, name="struct movie", colormap="gray", visible=False)
    viewer.add_image(g, name="GCaMP movie", colormap="green", visible=False)
    lines = viewer.add_shapes(name="bodywall line", shape_type="line",
                              edge_color="cyan", edge_width=3)
    if out_path.exists():
        for seg in _load_lines(out_path):
            lines.add([np.array([seg["p0"], seg["p1"]])], shape_type="line")
    status = Label(value="draw a line ALONG each body wall, then Save")
    btn = PushButton(text="Save lines")

    def _save():
        if not len(lines.data):
            status.value = "no line drawn"; return
        segs = [np.asarray(x) for x in lines.data]
        out_path.write_text(json.dumps(
            {"lines": [{"name": f"wall{i+1}", "p0": x[0].tolist(), "p1": x[1].tolist()}
                       for i, x in enumerate(segs)], "tiff": str(tiff_path)}, indent=2))
        status.value = f"saved {len(segs)} line(s) -> {out_path}"; print(status.value)

    btn.clicked.connect(_save)
    viewer.window.add_dock_widget(Container(widgets=[btn, status]), name="bodywall",
                                  area="right")
    napari.run()


def analyze_tiff(tiff_path, line_path=None, out_dir=None) -> dict:
    """Contraction (struct channel) vs ganglion rhythm (GCaMP) from a raw tiff."""
    from .ganglion_activity import crosscorr_lag
    from .phase_analysis import dominant_frequency
    tiff_path = Path(tiff_path)
    line_path = Path(line_path) if line_path else tiff_path.with_suffix(".bodywall_line.json")
    out = Path(out_dir) if out_dir else tiff_path.parent / (tiff_path.stem + "_contraction")
    out.mkdir(parents=True, exist_ok=True)

    g, s, fs, chans = load_tiff(tiff_path)
    # rhythm = global-mean dF/F of GCaMP; contraction tracked on struct (mCherry)
    dff = signals.dff(g.reshape(len(g), -1).mean(1)[None], fs)[0]
    r_peaks, r_period = peak_intervals(dff, fs)
    f0, _ = dominant_frequency(dff, fs)

    walls = []
    for seg in _load_lines(line_path):
        kymo = kymograph_perp(s, seg["p0"], seg["p1"])
        shift = track_shift_ncc(kymo)
        contract = -(shift - np.median(shift))
        edge = shift - shift.min() + 1
        peaks, period = peak_intervals(contract, fs)
        dphase = np.angle(np.exp(1j * (phase_at(contract, f0, fs) - phase_at(dff, f0, fs))))
        lag, lag_r = crosscorr_lag(dff, contract, max_lag=int(round(3 * fs)))
        delays = peak_delays(r_peaks, peaks, fs)
        walls.append({"kymo": kymo, "edge": edge, "contract": contract, "peaks": peaks,
                      "period": period, "name": seg.get("name", "wall"),
                      "phase_delay_ms": float(dphase / (2 * np.pi * f0) * 1000),
                      "xcorr_lag_s": float(lag / fs), "xcorr_r": float(lag_r),
                      "median_peak_delay_ms": float(np.median(delays) * 1000) if len(delays) else float("nan"),
                      "excursion_px": float(np.ptp(edge)), "delays": delays})
    _plots(walls, dff, fs, r_peaks, out)
    result = {"tiff": str(tiff_path), "shared_freq_hz": f0, "fs": fs,
              "gcamp_ch": chans[0], "struct_ch": chans[1],
              "walls": [{k: w[k] for k in ("name", "phase_delay_ms", "xcorr_lag_s",
                         "xcorr_r", "median_peak_delay_ms", "excursion_px")} for w in walls],
              "out": str(out)}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Body-wall contraction vs ganglion rhythm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate"); a.add_argument("run_dir"); a.add_argument("--line", default=None)
    z = sub.add_parser("analyze")
    z.add_argument("run_dir"); z.add_argument("--line", default=None); z.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "annotate":
        annotate_line(args.run_dir, args.line)
    else:
        r = analyze(args.run_dir, args.line, args.out)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
