import os
import tempfile

from leecharena.tracks_store import (
    TRACK_COLUMNS,
    load_tracks,
    save_tracks,
    tracks_path_for,
    upsert_track_frame,
)


def _csv():
    return os.path.join(tempfile.mkdtemp(), "tracks.csv")


def test_save_load_roundtrip():
    path = _csv()
    rows = [
        {"frame": 0, "time_s": 0.0, "track": 0, "node": 0, "x": 1.0, "y": 2.0, "held": 0},
        {"frame": 0, "time_s": 0.0, "track": 0, "node": 1, "x": 3.0, "y": 4.0, "held": 0},
        {"frame": 1, "time_s": 0.03, "track": 0, "node": 0, "x": 1.5, "y": 2.5, "held": 0},
    ]
    save_tracks(path, rows)
    df = load_tracks(path)
    assert list(df.columns) == TRACK_COLUMNS
    assert len(df) == 3
    assert set(df["node"]) == {0, 1}
    assert int(df["corrected"].sum()) == 0          # defaults to 0


def test_upsert_replaces_only_that_frame_and_flags_corrected():
    path = _csv()
    save_tracks(path, [
        {"frame": 0, "time_s": 0.0, "track": 0, "node": 0, "x": 1.0, "y": 2.0, "held": 0},
        {"frame": 1, "time_s": 0.03, "track": 0, "node": 0, "x": 9.0, "y": 9.0, "held": 0},
    ])
    upsert_track_frame(path, frame=1, rows=[
        {"frame": 1, "time_s": 0.03, "track": 0, "node": 0, "x": 5.0, "y": 6.0},
        {"frame": 1, "time_s": 0.03, "track": 0, "node": 1, "x": 7.0, "y": 8.0},
    ])
    df = load_tracks(path)
    f0 = df[df["frame"] == 0]
    f1 = df[df["frame"] == 1]
    assert len(f0) == 1 and float(f0["x"].iloc[0]) == 1.0      # untouched
    assert sorted(int(nd) for nd in f1["node"]) == [0, 1]     # replaced (2 nodes)
    assert float(f1[f1["node"] == 0]["x"].iloc[0]) == 5.0
    assert int(f1["corrected"].min()) == 1                    # upserted rows flagged


def test_upsert_creates_file_when_absent():
    path = _csv()
    upsert_track_frame(path, frame=3, rows=[
        {"frame": 3, "time_s": 0.1, "track": 0, "node": 0, "x": 1.0, "y": 1.0}])
    assert os.path.exists(path)
    assert len(load_tracks(path)) == 1


def test_tracks_path_for_uses_video_stem():
    p = tracks_path_for("/data/annotations/annotations.csv", "IMG_2859_dish1.mp4")
    assert p.name == "tracks_IMG_2859_dish1.csv"
    assert p.parent.name == "annotations"
