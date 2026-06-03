"""napari hub to place dish circles on a composite image and split a recording.

The whole splitting workflow lives in this one window: pick a raw multi-dish video,
build a mean/median composite (static dishes stay sharp, moving leeches wash out),
place one editable ellipse per dish (pre-seeded from Hough auto-detect), then either
save the circles to config for batch runs or split the recording into single-dish
clips right here.

It is a thin GUI over the tested modules (compositing, arena geometry, split). napari
and magicgui are imported lazily so the package imports without Qt.

CLI: leech-arena-place [--video raw.mp4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .arena import circle_to_corners, corners_to_circle
from .compositing import composite
from .config import Config
from .split import circles_to_rois, detect_dishes, save_split_rois, split_video

METHODS = ["mean", "median"]


def launch(config: Config, config_path: str | Path, video: str | Path | None = None) -> None:
    import napari
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        FileEdit,
        Label,
        PushButton,
        SpinBox,
    )

    state: dict = {"video": Path(video) if video else None, "width": 0, "height": 0}

    viewer = napari.Viewer(title="leech arena — dish placement")
    image_layer = viewer.add_image(np.zeros((1, 1, 3), np.uint8), name="composite")
    dish_layer = viewer.add_shapes(
        name="dishes", edge_color="yellow", face_color="transparent", edge_width=4
    )

    # --- dock widgets --------------------------------------------------------
    video_edit = FileEdit(
        label="video", mode="r",
        filter="Videos (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.m4v);;All files (*)",
        value=str(state["video"]) if state["video"] else "")
    method_combo = ComboBox(label="composite", choices=METHODS, value="mean")
    sample_spin = SpinBox(label="sample frames (0=all)", value=0, min=0, max=100000)
    build_btn = PushButton(text="Build composite")
    max_dish_spin = SpinBox(label="max dishes (auto)", value=2, min=1, max=12)
    detect_btn = PushButton(text="Auto-detect dishes")
    compress_cb = CheckBox(label="compress clips (H.264)", value=config.compression.enabled)
    save_cfg_btn = PushButton(text="Save circles to config")
    split_btn = PushButton(text="Split now")
    status = Label(value="Pick a video and build a composite.")

    def set_dishes(circles):
        dish_layer.data = []
        for cx, cy, r in circles:
            dish_layer.add(circle_to_corners(cx, cy, r), shape_type="ellipse")

    def current_circles():
        return [corners_to_circle(c) for c in dish_layer.data]

    def build_composite(*_):
        path = video_edit.value
        if not path or not Path(path).exists():
            status.value = "Choose an existing video file first."
            return
        state["video"] = Path(path)
        sample = int(sample_spin.value) or None
        status.value = f"Building {method_combo.value} composite… (this can take a moment)"
        try:
            img = composite(state["video"], method=method_combo.value, sample=sample)
        except Exception as exc:
            status.value = f"Composite failed: {exc}"
            return
        state["height"], state["width"] = img.shape[0], img.shape[1]
        image_layer.data = img
        viewer.reset_view()
        auto_detect()

    def auto_detect(*_):
        if state["width"] == 0:
            status.value = "Build a composite first."
            return
        circles = detect_dishes(image_layer.data, config.hough,
                                max_dishes=int(max_dish_spin.value))
        set_dishes(circles)
        if circles:
            status.value = (f"Detected {len(circles)} dish(es). Drag/resize/add/delete "
                            "ellipses, then save or split.")
        else:
            status.value = "No dishes detected — add ellipses manually (one per dish)."

    def save_to_config(*_):
        circles = current_circles()
        if not circles:
            status.value = "No dishes placed — nothing to save."
            return
        rois = circles_to_rois(circles, state["width"], state["height"])
        save_split_rois(config_path, rois)
        status.value = f"Saved {len(rois)} ROI(s) to {Path(config_path).name} (split.rois)."

    def split_now(*_):
        if state["video"] is None:
            status.value = "Choose a video first."
            return
        circles = current_circles()
        if not circles:
            status.value = "No dishes placed — nothing to split."
            return
        rois = circles_to_rois(circles, state["width"], state["height"])
        status.value = f"Splitting into {len(rois)} clip(s)…"
        try:
            outs = split_video(state["video"], config.clips_dir, rois)
            if compress_cb.value:
                from .compress import compress_in_place

                c = config.compression
                for o in outs:
                    compress_in_place(o, crf=c.crf, preset=c.preset, scale=c.scale)
            status.value = f"Wrote {len(outs)} clip(s) to {config.clips_dir}: " + \
                ", ".join(o.name for o in outs)
        except Exception as exc:
            status.value = f"Split failed: {exc}"

    build_btn.changed.connect(build_composite)
    detect_btn.changed.connect(auto_detect)
    save_cfg_btn.changed.connect(save_to_config)
    split_btn.changed.connect(split_now)

    widget = Container(widgets=[
        Label(value="— 1. composite —"), video_edit, method_combo, sample_spin, build_btn,
        Label(value="— 2. dishes (one ellipse each) —"), max_dish_spin, detect_btn,
        Label(value="— 3. output —"), compress_cb, save_cfg_btn, split_btn,
        status,
    ])
    viewer.window.add_dock_widget(widget, name="place dishes", area="right")

    if state["video"] is not None and state["video"].exists():
        # Don't auto-build: a mean over every frame of a multi-GB clip would freeze the
        # window for minutes. Let the user pick a sample first, then click Build.
        status.value = (f"Loaded {state['video'].name}. Set 'sample frames' (e.g. 200 for "
                        "a quick look on long clips), then click Build composite.")
    napari.run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Place dish circles on a composite image and split a recording.")
    parser.add_argument("--config", default="arena_config.yaml")
    parser.add_argument("--video", default=None, help="raw multi-dish MP4 (optional)")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    launch(config, args.config, args.video)


if __name__ == "__main__":
    main()
