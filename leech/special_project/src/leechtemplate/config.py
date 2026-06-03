"""Load and validate config.yaml into a typed dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    figures_dir: Path
    template_dir: Path
    name_list: Path
    midline_tol: float
    default_confidence: str
    sides: list[str] = field(default_factory=lambda: ["L", "R", "M"])
    aspects: list[str] = field(default_factory=lambda: ["ventral", "dorsal"])
    confidences: list[str] = field(default_factory=lambda: ["high", "medium", "low"])

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        root = Path(path).resolve().parent
        data = yaml.safe_load(Path(path).read_text()) or {}
        paths = data.get("paths", {})
        ann = data.get("annotation", {})

        def resolve(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (root / q)

        return cls(
            figures_dir=resolve(paths["figures_dir"]),
            template_dir=resolve(paths["template_dir"]),
            name_list=resolve(paths["name_list"]),
            midline_tol=float(ann.get("midline_tol", 0.02)),
            default_confidence=ann.get("default_confidence", "high"),
            sides=ann.get("sides", ["L", "R", "M"]),
            aspects=ann.get("aspects", ["ventral", "dorsal"]),
            confidences=ann.get("confidences", ["high", "medium", "low"]),
        )

    def template_csv(self, aspect: str) -> Path:
        return self.template_dir / f"{aspect}.csv"

    def calibration_json(self, aspect: str) -> Path:
        return self.template_dir / f"{aspect}_calibration.json"
