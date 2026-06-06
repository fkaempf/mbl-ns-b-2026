"""Fragment a raw multi-dish recording into single-dish MP4 clips.

Dishes are found by Hough circle detection on a representative frame (or taken from
manual ROIs in config when detection is unreliable). Each dish is written as a square
crop bounding its circle. The annotator only ever runs on these single-dish clips.

CLI: leech-arena-split --video raw.mp4 --out clips/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import Config


def detect_dishes(gray: np.ndarray, hough, max_dishes: int = 2, min_dist: float = 100.0):
    """Detect up to max_dishes circles; returns list of (cx, cy, r) in pixel x,y."""
    import cv2

    img = gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=hough.dp, minDist=min_dist,
        param1=hough.param1, param2=hough.param2,
        minRadius=int(hough.min_radius), maxRadius=int(hough.max_radius),
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0][:max_dishes]]


def circle_to_square_roi(cx, cy, r, width, height, pad=1.05):
    """Square crop (x, y, w, h) bounding a circle, clamped to the frame."""
    side = int(2 * r * pad)
    x = int(round(cx - side / 2))
    y = int(round(cy - side / 2))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(side, width - x)
    h = min(side, height - y)
    return x, y, w, h


def circles_to_rois(circles, width: int, height: int, pad: float = 1.05) -> list[list[int]]:
    """Convert placed circles [(cx, cy, r), ...] into clamped square ROIs [[x,y,w,h],...]
    as plain ints, ready to write to config or hand to split_video."""
    return [list(circle_to_square_roi(cx, cy, r, width, height, pad)) for cx, cy, r in circles]


def save_split_rois(config_path: str | Path, rois: list[list[int]]) -> None:
    """Write ``split.rois`` into the YAML config, preserving every other key."""
    import yaml

    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text()) or {}
    data.setdefault("split", {})["rois"] = [list(map(int, r)) for r in rois]
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))


def split_video(video_path: str | Path, out_dir: str | Path, rois: list[tuple]) -> list[Path]:
    """Write one cropped MP4 per ROI. ROIs are (x, y, w, h) in pixels."""
    import cv2

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers, outs = [], []
    for i, (_x, _y, w, h) in enumerate(rois):
        out = out_dir / f"{video_path.stem}_dish{i}.mp4"
        writers.append(cv2.VideoWriter(str(out), fourcc, fps, (w, h)))
        outs.append(out)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for (x, y, w, h), wr in zip(rois, writers, strict=False):
            wr.write(frame[y:y + h, x:x + w])
    cap.release()
    for wr in writers:
        wr.release()
    return outs


def main(argv: list[str] | None = None) -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Split a multi-dish recording into single-dish clips.")
    parser.add_argument("--config", default="arena_config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default=None, help="output dir (default: config clips_dir)")
    parser.add_argument("--max-dishes", type=int, default=2)
    parser.add_argument("--compress", action="store_true",
                        help="re-encode each clip with H.264 (overrides config); off by default")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    out_dir = Path(args.out) if args.out else cfg.clips_dir

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {args.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, sample = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read a sample frame")

    if cfg.split_rois:
        rois = [tuple(r) for r in cfg.split_rois]
        print(f"Using {len(rois)} manual ROI(s) from config.")
    else:
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        dishes = detect_dishes(gray, cfg.hough, max_dishes=args.max_dishes)
        if not dishes:
            raise RuntimeError("no dishes detected; set split.rois in config and retry")
        rois = [circle_to_square_roi(cx, cy, r, width, height) for cx, cy, r in dishes]
        print(f"Detected {len(rois)} dish(es): {rois}")

    outs = split_video(args.video, out_dir, rois)
    for o in outs:
        print("wrote", o)

    if args.compress or cfg.compression.enabled:
        from .compress import compress_in_place

        c = cfg.compression
        for o in outs:
            before = o.stat().st_size
            compress_in_place(o, crf=c.crf, preset=c.preset, scale=c.scale)
            after = o.stat().st_size
            print(f"compressed {o.name}: {before/1e6:.1f} -> {after/1e6:.1f} MB "
                  f"({100*after/before:.0f}%)")


if __name__ == "__main__":
    main()
