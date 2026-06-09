"""Render the registered movie to an MP4 with ganglion ROI contours burned in.

  python -m analysis.calcium.render_overlay <run_dir> <regions_base> [--fps 15]

regions_base is the per-run regions folder; every set subfolder that has a
masks.npy is overlaid (full -> red, small -> green, others cycle).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import signals

# BGR colours per set name (OpenCV uses BGR).
SET_COLORS = {"full": (0, 0, 255), "small": (0, 255, 0)}
FALLBACK = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]


def _contours(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def render(run_dir, regions_base, out_path, fps: float = 15.0,
           gain: float = 1.0, draw_rois: bool = True,
           color_map: dict | None = None, upscale: int = 1,
           title: str = "elav-GCaMP6m") -> Path:
    """Render the registered movie to MP4.

    gain        : brightness multiplier.
    draw_rois   : if False, render the bare movie (no contours/labels).
    color_map   : {set_name: (B,G,R)} overriding SET_COLORS (BGR).
    upscale     : integer pixel-replication factor (2 => each pixel -> 2x2 block,
                  700->1400). The movie is nearest-neighbour scaled (stays crisp);
                  text/contours are drawn AFTER upscaling, at native high-res, so
                  the font is genuinely higher-resolved (not blurred).
    """
    run = signals.load_run(run_dir)
    mov = run.movie
    lo, hi = np.percentile(mov[::20], (1, 99.5))           # global contrast
    scale = 255.0 / max(hi - lo, 1e-6)
    colors = {**SET_COLORS, **(color_map or {})}

    regions_base = Path(regions_base)
    sets, ref_masks = [], None
    if draw_rois:
        for i, sub in enumerate(sorted(p for p in regions_base.iterdir() if p.is_dir())):
            mpath = sub / "masks.npy"
            if not mpath.exists():
                continue
            masks = np.load(mpath)
            color = colors.get(sub.name, FALLBACK[i % len(FALLBACK)])
            cnts = [c for m in masks for c in _contours(m)]
            sets.append((sub.name, color, cnts))
            if sub.name == "full" or ref_masks is None:
                ref_masks = masks                  # prefer 'full' for label geometry
        if not sets:
            raise SystemExit(f"no region sets with masks.npy under {regions_base}")
        cent = np.array([[np.where(m)[1].mean(), np.where(m)[0].mean()]
                         for m in ref_masks])
        order = np.argsort(cent[:, 0])
        labels = np.empty(len(order), int)
        labels[order] = np.arange(1, len(order) + 1)

    T, Ly, Lx = mov.shape
    up = max(1, int(upscale))
    W, H = Lx * up, Ly * up
    # scale ROI geometry to the upscaled canvas
    if draw_rois:
        sets = [(n, c, [cnt * up for cnt in cnts]) for n, c, cnts in sets]
        cent = cent * up
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        g = np.clip((mov[t] - lo) * scale * gain, 0, 255).astype("uint8")
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if up > 1:                                  # pixel-replicate the MOVIE only
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        # HUD (always shown, with or without the ROI overlay): title + timer
        cv2.putText(img, title, (10, 24 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * up,
                    (255, 255, 255), max(1, up), cv2.LINE_AA)
        cv2.putText(img, f"t={t/run.fs:5.1f}s  ({t+1}/{T})", (10, 46 * up),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * up, (255, 255, 255),
                    max(1, up), cv2.LINE_AA)
        if draw_rois:
            for _name, color, cnts in sets:
                cv2.drawContours(img, cnts, -1, color, max(1, up))
            for (cx, cy), lab in zip(cent, labels):
                org = (int(cx) - 6 * up, int(cy) + 6 * up)
                cv2.putText(img, str(lab), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * up,
                            (0, 0, 0), 3 * up, cv2.LINE_AA)
                cv2.putText(img, str(lab), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * up,
                            (255, 255, 255), max(1, up), cv2.LINE_AA)
            y = 68 * up
            for name, color, _c in sets:
                cv2.putText(img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * up,
                            color, 2 * up, cv2.LINE_AA)
                y += 22 * up
        vw.write(img)
    vw.release()
    return out_path


def render_bare(movie, fs, out_path, fps: float = 15.0, gain: float = 1.0,
                upscale: int = 2, title: str = "elav-GCaMP6m") -> Path:
    """Render a bare (T, Y, X) movie array to MP4 with HUD title + timer.

    No suite2p run / ROIs needed — works directly from a raw movie.
    """
    mov = np.asarray(movie, dtype="float32")
    lo, hi = np.percentile(mov[::20], (1, 99.5))
    scale = 255.0 / max(hi - lo, 1e-6)
    T, Ly, Lx = mov.shape
    up = max(1, int(upscale)); W, H = Lx * up, Ly * up
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        g = np.clip((mov[t] - lo) * scale * gain, 0, 255).astype("uint8")
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if up > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, title, (10, 24 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * up,
                    (255, 255, 255), max(1, up), cv2.LINE_AA)
        cv2.putText(img, f"t={t/fs:5.1f}s  ({t+1}/{T})", (10, 46 * up),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * up, (255, 255, 255),
                    max(1, up), cv2.LINE_AA)
        vw.write(img)
    vw.release()
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render ROI-overlay MP4")
    ap.add_argument("run_dir")
    ap.add_argument("regions_base", help="per-run regions folder (has set subdirs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args(argv)
    out = args.out or str(Path(args.run_dir) / "overlay.mp4")
    p = render(args.run_dir, args.regions_base, out, fps=args.fps)
    print("wrote", p)


if __name__ == "__main__":
    main()
