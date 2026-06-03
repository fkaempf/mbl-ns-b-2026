"""Read/write the template CSV files and calibration sidecars, with schema checks.

One CSV per aspect (ventral.csv, dorsal.csv); one calibration JSON per aspect.
Depends on pandas only — no napari, no coordinate math.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .coordinates import Calibration

TEMPLATE_COLUMNS = ["name", "side", "aspect", "x", "y", "confidence", "notes"]
VALID_SIDES = {"L", "R", "M"}
VALID_ASPECTS = {"ventral", "dorsal"}
VALID_CONFIDENCES = {"high", "medium", "low"}


def validate_template(df: pd.DataFrame) -> None:
    """Raise ValueError if the template DataFrame violates the schema."""
    missing = [c for c in TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"template is missing required columns: {missing}")

    bad_side = set(df["side"]) - VALID_SIDES
    if bad_side:
        raise ValueError(f"invalid side value(s): {sorted(bad_side)}; allowed {sorted(VALID_SIDES)}")

    bad_aspect = set(df["aspect"]) - VALID_ASPECTS
    if bad_aspect:
        raise ValueError(f"invalid aspect value(s): {sorted(bad_aspect)}; allowed {sorted(VALID_ASPECTS)}")

    bad_conf = set(df["confidence"]) - VALID_CONFIDENCES
    if bad_conf:
        raise ValueError(f"invalid confidence value(s): {sorted(bad_conf)}; allowed {sorted(VALID_CONFIDENCES)}")

    if not pd.api.types.is_numeric_dtype(df["x"]) or not pd.api.types.is_numeric_dtype(df["y"]):
        raise ValueError("x and y must be numeric")


def save_template(df: pd.DataFrame, path: str | Path) -> None:
    """Validate then write a template CSV (columns ordered, index dropped)."""
    validate_template(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[TEMPLATE_COLUMNS].to_csv(path, index=False)


def load_template(path: str | Path) -> pd.DataFrame:
    """Read and validate a template CSV."""
    df = pd.read_csv(path, dtype={"notes": str}).fillna({"notes": ""})
    validate_template(df)
    return df[TEMPLATE_COLUMNS]


# --- calibration sidecar ----------------------------------------------------


@dataclass(frozen=True)
class CalibrationRecord:
    """A Calibration plus provenance, persisted alongside a template CSV."""

    aspect: str
    image_path: str
    image_sha256: str
    calibration: Calibration

    def to_dict(self) -> dict:
        c = self.calibration
        return {
            "aspect": self.aspect,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "anterior_midline": list(c.anterior_midline),
            "posterior_midline": list(c.posterior_midline),
            "right_ref": list(c.right_ref),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationRecord":
        return cls(
            aspect=d["aspect"],
            image_path=d["image_path"],
            image_sha256=d["image_sha256"],
            calibration=Calibration(
                anterior_midline=tuple(d["anterior_midline"]),
                posterior_midline=tuple(d["posterior_midline"]),
                right_ref=tuple(d["right_ref"]),
            ),
        )


def sha256_of_file(path: str | Path) -> str:
    """SHA-256 of a file, used to detect if a saved template's source image changed."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def save_calibration(record: CalibrationRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2))


def load_calibration(path: str | Path) -> CalibrationRecord:
    return CalibrationRecord.from_dict(json.loads(Path(path).read_text()))
