"""Read/write annotations.csv: one row per leech per frame.

A fully-annotated frame with no leeches still writes one placeholder row
(leech_idx == -1, empty point columns) so that "annotated but empty" is distinct from
"not yet annotated" when resuming.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "video", "frame", "time_s",
    "arena_cx", "arena_cy", "arena_r",
    "food_x", "food_y",
    "leech_idx",
    "ant_x", "ant_y", "post_x", "post_y", "mid_x", "mid_y",
]


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"annotations missing required columns: {missing}")


def load(path: str | Path) -> pd.DataFrame:
    """Load the store, or an empty correctly-typed frame if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    validate(df)
    return df[COLUMNS]


def append_rows(path: str | Path, rows: list[dict]) -> None:
    """Append rows (list of dicts keyed by COLUMNS) to the store, creating it if needed."""
    path = Path(path)
    new = pd.DataFrame(rows)
    missing = [c for c in COLUMNS if c not in new.columns]
    if missing:
        raise ValueError(f"rows missing required columns: {missing}")
    existing = load(path)
    out = pd.concat([existing, new[COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def upsert_frame(path: str | Path, video: str, frame: int, rows: list[dict]) -> None:
    """Replace all stored rows for (video, frame) with `rows`, creating the file if
    needed. Used when re-annotating a revisited frame so it updates rather than
    appending duplicates. Other frames/videos are left untouched."""
    path = Path(path)
    new = pd.DataFrame(rows)
    missing = [c for c in COLUMNS if c not in new.columns]
    if missing:
        raise ValueError(f"rows missing required columns: {missing}")
    existing = load(path)
    keep = existing[~((existing["video"] == video) & (existing["frame"] == int(frame)))]
    out = pd.concat([keep, new[COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def annotated_frames(path: str | Path, video: str) -> set[int]:
    """Set of frame indices already annotated for a given clip."""
    df = load(path)
    if df.empty:
        return set()
    return set(int(f) for f in df.loc[df["video"] == video, "frame"].unique())
