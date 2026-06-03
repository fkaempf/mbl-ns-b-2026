"""Regression test for the Annotate tab's click-to-advance role stamping.

The pure cycle logic lives in test_annotate_sequence.py; this guards the napari
wiring (a placed point must actually receive the advancing role/leech), which broke
once because feature_defaults changed inside an add are reverted by napari.

Runs headless via the offscreen Qt platform; skipped if Qt/napari aren't importable.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_placed_points_get_advancing_roles():
    pytest.importorskip("napari")
    pytest.importorskip("magicgui")
    from napari.layers import Points, Shapes

    from leecharena.annotate_panel import build_annotate_panel
    from leecharena.config import Config

    created = {}

    class FakeViewer:
        def add_shapes(self, *a, **k):
            layer = Shapes()
            created["arena"] = layer
            return layer

        def add_points(self, data=None, *a, **k):
            layer = Points(np.empty((0, 2)) if data is None else data,
                           features=k.get("features"))
            created[k.get("name", "pts")] = layer
            return layer

    class Ctx:
        viewer = FakeViewer()
        config = Config.load(os.path.join(os.path.dirname(__file__), "..", "arena_config.yaml"))
        config_path = "arena_config.yaml"
        state = {"video": None, "reader": None, "width": 0, "height": 0}
        image_layer = type("I", (), {"data": np.zeros((4, 4, 3), np.uint8)})()

        def status(self, msg):
            pass

        def show_frame(self, idx):
            pass

        def on_video_loaded(self, fn):
            pass

    build_annotate_panel(Ctx())
    leech = created["leeches"]

    for k in range(6):
        leech.add(np.array([[float(k), float(k)]]))

    roles = list(leech.features["role"])
    leeches = [int(x) for x in leech.features["leech_idx"]]
    assert roles == ["anterior", "posterior", "middle", "anterior", "posterior", "middle"]
    assert leeches == [0, 0, 0, 1, 1, 1]
