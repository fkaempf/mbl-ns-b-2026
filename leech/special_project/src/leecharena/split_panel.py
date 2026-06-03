"""Split tab for the unified leech-arena napari app.

A port of ``place.py`` into a panel factory driven by a shared ``AppContext``.
Build a mean/median composite of the loaded multi-dish video (static dishes stay
sharp, moving leeches wash out), place one editable ellipse per dish (pre-seeded
from Hough auto-detect), then save the circles to config or split the recording
into single-dish clips.

Thin GUI over the tested modules (compositing, arena geometry, split). napari and
magicgui are imported lazily so this module imports without Qt.
"""

from __future__ import annotations

from .arena import circle_to_corners, corners_to_circle
from .compositing import composite
from .split import circles_to_rois, detect_dishes, save_split_rois, split_video

METHODS = ["mean", "median"]


def build_split_panel(ctx) -> "object":
    from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, SpinBox

    method_combo = ComboBox(label="composite", choices=METHODS, value="mean")
    sample_spin = SpinBox(label="sample frames (0=all)", value=0, min=0, max=100000)
    build_btn = PushButton(text="Build composite")
    max_dish_spin = SpinBox(label="max dishes (auto)", value=2, min=1, max=12)
    detect_btn = PushButton(text="Auto-detect dishes")
    compress_cb = CheckBox(label="compress clips (H.264)", value=ctx.config.compression.enabled)
    save_cfg_btn = PushButton(text="Save circles to config")
    split_btn = PushButton(text="Split now")

    # One persistent 'dishes' Shapes layer on the shared viewer.
    def dishes_layer():
        for lyr in ctx.viewer.layers:
            if lyr.name == "dishes":
                return lyr
        return ctx.viewer.add_shapes(
            name="dishes", edge_color="yellow", face_color="transparent", edge_width=4
        )

    def set_dishes(circles):
        lyr = dishes_layer()
        lyr.data = []
        for cx, cy, r in circles:
            lyr.add(circle_to_corners(cx, cy, r), shape_type="ellipse")

    def current_circles():
        return [corners_to_circle(c) for c in dishes_layer().data]

    def auto_detect():
        if ctx.state.get("width", 0) == 0:
            ctx.status("Build a composite first.")
            return
        try:
            circles = detect_dishes(
                ctx.image_layer.data, ctx.config.hough, max_dishes=int(max_dish_spin.value)
            )
        except Exception as exc:
            ctx.status(f"Auto-detect failed: {exc}")
            return
        set_dishes(circles)
        if circles:
            ctx.status(
                f"Detected {len(circles)} dish(es). Drag/resize/add/delete ellipses, "
                "then save or split."
            )
        else:
            ctx.status("No dishes detected — add ellipses manually (one per dish).")

    def build_composite(*_):
        video = ctx.state.get("video")
        if video is None:
            ctx.status("Load a video first.")
            return
        sample = int(sample_spin.value) or None
        ctx.status(f"Building {method_combo.value} composite… (this can take a moment)")
        try:
            img = composite(video, method=method_combo.value, sample=sample)
        except Exception as exc:
            ctx.status(f"Composite failed: {exc}")
            return
        ctx.set_image(img)
        ctx.state["height"], ctx.state["width"] = int(img.shape[0]), int(img.shape[1])
        auto_detect()

    def save_to_config(*_):
        circles = current_circles()
        if not circles:
            ctx.status("No dishes placed — nothing to save.")
            return
        from pathlib import Path

        rois = circles_to_rois(circles, ctx.state["width"], ctx.state["height"])
        save_split_rois(ctx.config_path, rois)
        ctx.status(f"Saved {len(rois)} ROI(s) to {Path(ctx.config_path).name} (split.rois).")

    def split_now(*_):
        video = ctx.state.get("video")
        if video is None:
            ctx.status("Load a video first.")
            return
        circles = current_circles()
        if not circles:
            ctx.status("No dishes placed — nothing to split.")
            return
        rois = circles_to_rois(circles, ctx.state["width"], ctx.state["height"])
        ctx.status(f"Splitting into {len(rois)} clip(s)…")
        try:
            outs = split_video(video, ctx.config.clips_dir, rois)
            if compress_cb.value:
                from .compress import compress_in_place

                c = ctx.config.compression
                for o in outs:
                    compress_in_place(o, crf=c.crf, preset=c.preset, scale=c.scale)
            ctx.status(
                f"Wrote {len(outs)} clip(s) to {ctx.config.clips_dir}: "
                + ", ".join(o.name for o in outs)
            )
        except Exception as exc:
            ctx.status(f"Split failed: {exc}")

    build_btn.changed.connect(build_composite)
    detect_btn.changed.connect(lambda *_: auto_detect())
    save_cfg_btn.changed.connect(save_to_config)
    split_btn.changed.connect(split_now)

    return Container(widgets=[
        Label(value="— 1. composite —"), method_combo, sample_spin, build_btn,
        Label(value="— 2. dishes (one ellipse each) —"), max_dish_spin, detect_btn,
        Label(value="— 3. output —"), compress_cb, save_cfg_btn, split_btn,
    ])
