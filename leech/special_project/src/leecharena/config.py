"""Load arena_config.yaml into a typed dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class HoughParams:
    dp: float = 1.2
    min_dist: float = 1000.0
    param1: float = 100.0
    param2: float = 30.0
    min_radius: int = 0
    max_radius: int = 0


@dataclass(frozen=True)
class Compression:
    enabled: bool = False
    crf: int = 28
    preset: str = "medium"
    scale: float | None = None


@dataclass(frozen=True)
class Config:
    clips_dir: Path
    annotations_path: Path
    n_sample: int
    seed: int
    hough: HoughParams = field(default_factory=HoughParams)
    # Optional manual ROIs for the splitter, each [x, y, w, h]; empty -> auto-detect.
    split_rois: list[list[int]] = field(default_factory=list)
    compression: Compression = field(default_factory=Compression)

    @classmethod
    def load(cls, path: str | Path = "arena_config.yaml") -> "Config":
        root = Path(path).resolve().parent
        data = yaml.safe_load(Path(path).read_text()) or {}

        def resolve(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (root / q)

        paths = data.get("paths", {})
        samp = data.get("sampling", {})
        h = data.get("hough", {})
        return cls(
            clips_dir=resolve(paths.get("clips_dir", "clips")),
            annotations_path=resolve(paths.get("annotations", "annotations/annotations.csv")),
            n_sample=int(samp.get("n_sample", 30)),
            seed=int(samp.get("seed", 0)),
            hough=HoughParams(
                dp=float(h.get("dp", 1.2)),
                min_dist=float(h.get("min_dist", 1000.0)),
                param1=float(h.get("param1", 100.0)),
                param2=float(h.get("param2", 30.0)),
                min_radius=int(h.get("min_radius", 0)),
                max_radius=int(h.get("max_radius", 0)),
            ),
            split_rois=data.get("split", {}).get("rois", []),
            compression=Compression(
                enabled=bool(data.get("compression", {}).get("enabled", False)),
                crf=int(data.get("compression", {}).get("crf", 28)),
                preset=data.get("compression", {}).get("preset", "medium"),
                scale=data.get("compression", {}).get("scale", None),
            ),
        )
