"""Compress MP4 recordings with H.264 (libx264) via ffmpeg.

OpenCV's mp4v writer produces large, weakly-compressed files; this re-encodes to
H.264 with a quality target (CRF) and optional downscaling, which typically shrinks
raw arena recordings several-fold for storage and sharing. The ffmpeg binary is
located through imageio-ffmpeg, so no system ffmpeg install is required.

CLI: leech-arena-compress --video in.mp4 --out out.mp4 --crf 28 [--scale 0.5]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    """Path to an ffmpeg executable (bundled via imageio-ffmpeg)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def scale_filter(factor: float) -> str:
    """ffmpeg scale filter for a resize factor, keeping even dimensions (libx264 needs even)."""
    return f"scale=trunc(iw*{factor}/2)*2:trunc(ih*{factor}/2)*2"


def build_ffmpeg_cmd(
    ffmpeg: str,
    in_path: str | Path,
    out_path: str | Path,
    crf: int = 28,
    preset: str = "medium",
    scale: float | None = None,
    drop_audio: bool = True,
) -> list[str]:
    """Build the ffmpeg argument list (pure; no side effects)."""
    cmd = [ffmpeg, "-y", "-i", str(in_path)]
    if drop_audio:
        cmd.append("-an")
    cmd += ["-c:v", "libx264", "-crf", str(int(crf)), "-preset", preset]
    if scale is not None and scale != 1.0:
        cmd += ["-vf", scale_filter(scale)]
    cmd += ["-pix_fmt", "yuv420p", str(out_path)]
    return cmd


def compress_video(
    in_path: str | Path,
    out_path: str | Path,
    crf: int = 28,
    preset: str = "medium",
    scale: float | None = None,
) -> Path:
    """Re-encode in_path to out_path with H.264. Raises if ffmpeg fails."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_cmd(ffmpeg_exe(), in_path, out_path, crf, preset, scale)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_path


def compress_in_place(path: str | Path, crf: int = 28, preset: str = "medium",
                      scale: float | None = None) -> Path:
    """Compress a file and atomically replace it with the compressed version."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".cmp.mp4")
    compress_video(path, tmp, crf=crf, preset=preset, scale=scale)
    os.replace(tmp, path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compress an MP4 with H.264 (libx264).")
    parser.add_argument("--video", required=True, help="input MP4")
    parser.add_argument("--out", required=True, help="output MP4")
    parser.add_argument("--crf", type=int, default=28, help="quality 18 (high) - 32 (small); default 28")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--scale", type=float, default=None, help="resize factor, e.g. 0.5")
    args = parser.parse_args(argv)

    in_size = Path(args.video).stat().st_size
    out = compress_video(args.video, args.out, crf=args.crf, preset=args.preset, scale=args.scale)
    out_size = out.stat().st_size
    print(f"{args.video} ({in_size/1e6:.1f} MB) -> {out} ({out_size/1e6:.1f} MB, "
          f"{100*out_size/in_size:.0f}% of original)")


if __name__ == "__main__":
    main()
