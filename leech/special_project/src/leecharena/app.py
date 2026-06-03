"""Unified napari app shell: one viewer + shared video, three tabs.

This module imports cleanly without Qt: all napari / magicgui / qtpy / cv2
imports live inside functions. Top-level imports are limited to stdlib, numpy,
pathlib, dataclasses, and sibling pure leecharena modules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .video import VideoReader


@dataclass
class AppContext:
    viewer: object              # napari.Viewer
    config: object              # leecharena.config.Config
    config_path: str
    state: dict                 # {"video": Path|None, "reader": VideoReader|None, "width": int, "height": int}
    image_layer: object         # napari.layers.Image (main canvas)
    _status: object             # magicgui Label
    _listeners: list = field(default_factory=list)  # callables(path) run after a load

    def on_video_loaded(self, fn) -> None:
        """Register a callback fired (with the new path) whenever a video is loaded.
        Panels use this to auto-react when the user opens/switches a clip."""
        self._listeners.append(fn)

    def load_video(self, path) -> object:
        """Open a VideoReader, record state, show frame 0, notify listeners, return it."""
        p = Path(path)
        reader = VideoReader(p)
        self.state["video"] = p
        self.state["reader"] = reader
        self.state["width"] = reader.width
        self.state["height"] = reader.height
        if reader.n_frames > 0:
            self.show_frame(0)
        self.status(f"loaded {p.name}: {reader.n_frames} frames {reader.width}x{reader.height}")
        for fn in self._listeners:
            try:
                fn(p)
            except Exception as exc:  # noqa: BLE001 — a panel listener must not block loading
                self.status(f"video-load listener error: {exc}")
        return reader

    def show_frame(self, idx: int) -> None:
        """Display frame `idx` from the loaded reader on the image layer."""
        reader = self.state.get("reader")
        if reader is None:
            return
        self.image_layer.data = reader.frame(int(idx))

    def set_image(self, img) -> None:
        """Set the image layer data to an arbitrary RGB uint8 array (e.g. a composite)."""
        self.image_layer.data = img

    def status(self, msg: str) -> None:
        """Write a message into the shared status Label."""
        self._status.value = msg


def build_app(config, config_path, video=None):
    """Build the unified napari viewer with a shared session dock and three tabs.

    Returns the napari.Viewer.
    """
    import napari
    from magicgui.widgets import Container, FileEdit, Label, Slider
    from qtpy.QtWidgets import QScrollArea, QTabWidget

    def _scrollable(widget):
        """Wrap a Qt widget in a resizable scroll area so its dock can be freely
        resized/dragged and overflow scrolls instead of pinning the pane width."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    viewer = napari.Viewer(title="leech arena")

    image_layer = viewer.add_image(
        np.zeros((1, 1, 3), dtype=np.uint8),
        name="video",
        rgb=True,
    )

    status_label = Label(label="status", value="no video loaded")

    ctx = AppContext(
        viewer=viewer,
        config=config,
        config_path=config_path,
        state={"video": None, "reader": None, "width": 0, "height": 0},
        image_layer=image_layer,
        _status=status_label,
    )

    # --- shared session dock ---
    video_edit = FileEdit(
        label="video", mode="r",
        filter="Videos (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.m4v);;All files (*)",
    )
    frame_slider = Slider(label="frame", min=0, max=0)

    def _on_frame_changed(value):
        ctx.show_frame(int(value))

    frame_slider.changed.connect(_on_frame_changed)

    def _open(path):
        reader = ctx.load_video(path)
        frame_slider.max = max(0, reader.n_frames - 1)
        frame_slider.value = 0

    def _on_video_changed(path):
        if path and Path(str(path)).exists():
            try:
                _open(path)
            except Exception as exc:  # noqa: BLE001
                ctx.status(f"failed to load video: {exc}")

    video_edit.changed.connect(_on_video_changed)

    session = Container(widgets=[video_edit, frame_slider, status_label])

    # --- tabs (built before the initial open so their video-loaded listeners fire) ---
    from .compress_panel import build_compress_panel
    from .split_panel import build_split_panel
    from .annotate_panel import build_annotate_panel

    tabs = QTabWidget()

    def _add_tab(factory, name):
        try:
            panel = factory(ctx)
            tabs.addTab(_scrollable(panel.native), name)
        except Exception as exc:  # noqa: BLE001
            placeholder = Label(value=f"{name} panel failed to load:\n{type(exc).__name__}: {exc}")
            tabs.addTab(placeholder.native, f"{name} (error)")

    _add_tab(build_compress_panel, "Compress")
    _add_tab(build_split_panel, "Split")
    _add_tab(build_annotate_panel, "Annotate")

    viewer.window.add_dock_widget(_scrollable(session.native), area="right", name="session")
    viewer.window.add_dock_widget(tabs, area="right", name="tools")

    if video is not None:
        try:
            video_edit.value = str(video)
            _open(video)
        except Exception as exc:  # noqa: BLE001
            ctx.status(f"failed to load video: {exc}")

    return viewer


def main(argv=None):
    parser = argparse.ArgumentParser(description="leech arena unified app")
    parser.add_argument("--config", default="arena_config.yaml")
    parser.add_argument("--video", default=None)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    build_app(cfg, args.config, args.video)

    import napari

    napari.run()


if __name__ == "__main__":
    main()
