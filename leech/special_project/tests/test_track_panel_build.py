"""Headless construction smoke test for the Track tab (mirrors the annotate one)."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_build_track_panel_constructs():
    pytest.importorskip("napari")
    pytest.importorskip("magicgui")
    from napari.layers import Points, Shapes

    from leecharena.config import Config
    from leecharena.track_panel import build_track_panel

    created = {}

    class FakeViewer:
        def add_points(self, data=None, *a, **k):
            layer = Points(np.empty((0, 2)) if data is None else data,
                           features=k.get("features"))
            created[k.get("name", "pts")] = layer
            return layer

        def add_shapes(self, *a, **k):
            layer = Shapes()
            created[k.get("name", "shapes")] = layer
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

        def on_frame_changed(self, fn):
            pass

    panel = build_track_panel(Ctx())
    assert len(panel) > 0
    assert "tracks" in created and "track seeds" in created
