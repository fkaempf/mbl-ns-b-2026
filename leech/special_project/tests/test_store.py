import numpy as np

from leecharena.store import COLUMNS, annotated_frames, append_rows, load


def _row(video, frame, leech_idx, **kw):
    base = {c: np.nan for c in COLUMNS}
    base.update(video=video, frame=frame, leech_idx=leech_idx, time_s=frame / 30.0,
                arena_cx=100, arena_cy=100, arena_r=50)
    base.update(kw)
    return base


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "annotations.csv"
    rows = [
        _row("clipA.mp4", 10, 0, ant_x=1, ant_y=2, post_x=3, post_y=4, mid_x=2, mid_y=3),
        _row("clipA.mp4", 10, 1, ant_x=5, ant_y=6, post_x=7, post_y=8, mid_x=6, mid_y=7),
    ]
    append_rows(path, rows)
    df = load(path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert set(df["leech_idx"]) == {0, 1}


def test_append_accumulates(tmp_path):
    path = tmp_path / "annotations.csv"
    append_rows(path, [_row("clipA.mp4", 10, 0)])
    append_rows(path, [_row("clipA.mp4", 22, -1)])
    df = load(path)
    assert len(df) == 2
    assert set(df["frame"]) == {10, 22}


def test_annotated_frames_per_video(tmp_path):
    path = tmp_path / "annotations.csv"
    append_rows(path, [
        _row("clipA.mp4", 10, 0),
        _row("clipA.mp4", 22, -1),
        _row("clipB.mp4", 5, 0),
    ])
    assert annotated_frames(path, "clipA.mp4") == {10, 22}
    assert annotated_frames(path, "clipB.mp4") == {5}
    assert annotated_frames(path, "missing.mp4") == set()


def test_load_missing_returns_empty(tmp_path):
    df = load(tmp_path / "nope.csv")
    assert df.empty
    assert list(df.columns) == COLUMNS
