"""Read/write the dense per-frame leech-position tracks CSV.

One row per track per frame: tracks_<video-stem>.csv next to the annotations file.
Distinct from store.py (the sparse hand-annotation file) so dense predictions never
mix into the hand labels. `corrected` flags rows edited by hand in the GUI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRACK_COLUMNS = ["frame", "time_s", "track", "node", "x", "y", "held", "corrected"]


def tracks_path_for(annotations_path: str | Path, video_id: str) -> Path:
    """annotations/tracks_<video-stem>.csv beside the annotations file."""
    stem = Path(video_id).stem
    return Path(annotations_path).with_name(f"tracks_{stem}.csv")


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "held" not in df.columns:
        df["held"] = 0
    if "corrected" not in df.columns:
        df["corrected"] = 0
    return df[TRACK_COLUMNS]


def save_tracks(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(rows).to_csv(path, index=False)


def load_tracks(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=TRACK_COLUMNS)
    df = pd.read_csv(path)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    return df[TRACK_COLUMNS]


def upsert_track_frame(path: str | Path, frame: int, rows: list[dict]) -> None:
    """Replace all rows for `frame` with `rows` (flagged corrected=1), creating the
    file if needed. Other frames are left intact."""
    path = Path(path)
    new = _frame([{**r, "corrected": 1} for r in rows]) if rows else _frame([])
    existing = load_tracks(path)
    keep = existing[existing["frame"] != int(frame)]
    out = pd.concat([keep, new], ignore_index=True).sort_values(["frame", "track", "node"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
