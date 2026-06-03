import os
import tempfile

from leecharena.annotate import _build_rows, annotation_from_rows
from leecharena.store import append_rows, load, upsert_frame


def _csv():
    return os.path.join(tempfile.mkdtemp(), "annotations.csv")


def test_upsert_replaces_only_that_frame():
    path = _csv()
    append_rows(path, _build_rows("v", 1, 0.0, (10, 10, 5), (1, 2), {0: {"anterior": (3, 4)}}))
    append_rows(path, _build_rows("v", 2, 0.1, (10, 10, 5), None, {0: {"anterior": (5, 6)}}))

    # re-annotate frame 1 with two leeches
    upsert_frame(path, "v", 1, _build_rows(
        "v", 1, 0.0, (11, 11, 6), (7, 8),
        {0: {"anterior": (1, 1)}, 1: {"anterior": (2, 2)}},
    ))

    df = load(path)
    f1 = df[df["frame"] == 1]
    f2 = df[df["frame"] == 2]
    assert sorted(int(x) for x in f1["leech_idx"]) == [0, 1]   # replaced, not appended
    assert float(f1["arena_cx"].iloc[0]) == 11.0               # new values kept
    assert len(f2) == 1                                        # other frame untouched


def test_upsert_creates_file_when_absent():
    path = _csv()
    upsert_frame(path, "v", 3, _build_rows("v", 3, 0.0, None, None, {0: {"anterior": (1, 1)}}))
    assert os.path.exists(path)
    assert len(load(path)) == 1


def test_annotation_from_rows_roundtrips():
    arena = (10.0, 20.0, 5.0)
    food = (3.0, 4.0)
    leeches = {0: {"anterior": (1.0, 2.0), "posterior": (3.0, 4.0), "middle": (5.0, 6.0)},
               1: {"anterior": (7.0, 8.0)}}
    rows = _build_rows("v", 5, 0.1, arena, food, leeches)

    a, f, l = annotation_from_rows(rows)
    assert a == arena
    assert f == food
    assert l == leeches


def test_annotation_from_rows_handles_empty_placeholder():
    rows = _build_rows("v", 5, 0.1, (1, 2, 3), None, {})  # no leeches -> placeholder row
    a, f, l = annotation_from_rows(rows)
    assert a == (1.0, 2.0, 3.0)
    assert f is None
    assert l == {}
