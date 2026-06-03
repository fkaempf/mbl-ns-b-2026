import os
import tempfile

from leecharena.config import Config
from leecharena.split import circles_to_rois, save_split_rois


def test_circles_to_rois_makes_clamped_square_boxes():
    rois = circles_to_rois([(100.0, 100.0, 50.0)], width=640, height=480)
    assert len(rois) == 1
    x, y, w, h = rois[0]
    # square crop bounding the circle (pad 1.05 -> side ~105), centered on (100, 100)
    assert all(isinstance(v, int) for v in (x, y, w, h))
    assert w == h
    assert x <= 100 <= x + w
    assert y <= 100 <= y + h


def test_circles_to_rois_clamps_to_frame():
    # circle near the top-left corner must not produce negative origin
    rois = circles_to_rois([(10.0, 10.0, 40.0)], width=640, height=480)
    x, y, w, h = rois[0]
    assert x >= 0 and y >= 0
    assert x + w <= 640 and y + h <= 480


def test_save_split_rois_roundtrips_and_preserves_other_config():
    d = tempfile.mkdtemp()
    cfg_path = os.path.join(d, "arena_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(
            "sampling:\n  n_sample: 42\n  seed: 7\n"
            "split:\n  rois: []\n"
        )

    save_split_rois(cfg_path, [[1, 2, 3, 4], [5, 6, 7, 8]])

    cfg = Config.load(cfg_path)
    assert cfg.split_rois == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert cfg.n_sample == 42      # unrelated keys preserved
    assert cfg.seed == 7


def test_save_split_rois_creates_split_section_when_absent():
    d = tempfile.mkdtemp()
    cfg_path = os.path.join(d, "arena_config.yaml")
    with open(cfg_path, "w") as f:
        f.write("sampling:\n  n_sample: 30\n  seed: 0\n")

    save_split_rois(cfg_path, [[0, 0, 10, 10]])

    cfg = Config.load(cfg_path)
    assert cfg.split_rois == [[0, 0, 10, 10]]
