"""Compress tab for the unified leecharena napari app.

Builds a magicgui Container that re-encodes the currently loaded video with
H.264 via leecharena.compress.compress_video. All Qt/magicgui imports are lazy
so this module imports cleanly without a Qt environment.
"""

from __future__ import annotations

from pathlib import Path

from .compress import compress_video

PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]


def build_compress_panel(ctx) -> "object":
    """Return a magicgui Container for the Compress tab."""
    from magicgui.widgets import (
        ComboBox,
        Container,
        FloatSpinBox,
        Label,
        PushButton,
        SpinBox,
    )

    comp = ctx.config.compression

    crf = SpinBox(name="crf", label="CRF", value=int(comp.crf), min=0, max=51)
    preset_default = comp.preset if comp.preset in PRESETS else "medium"
    preset = ComboBox(name="preset", label="Preset", choices=PRESETS, value=preset_default)
    scale = FloatSpinBox(name="scale", label="Scale", value=1.0, min=0.0, step=0.1)
    info = Label(name="info", value="No video loaded.")

    def _output_path(src: Path) -> Path:
        return src.with_name(src.stem + "_small.mp4")

    def _refresh_info() -> None:
        src = ctx.state.get("video")
        if not src:
            info.value = "No video loaded."
            return
        src = Path(src)
        try:
            size_mb = src.stat().st_size / 1e6
            size_txt = f"{size_mb:.1f} MB"
        except OSError:
            size_txt = "unknown size"
        out = _output_path(src)
        info.value = f"Source: {src.name} ({size_txt})\nOutput: {out.name}"

    def _on_compress() -> None:
        src = ctx.state.get("video")
        if not src:
            ctx.status("Compress: no video loaded.")
            return
        src = Path(src)
        out = _output_path(src)
        s = float(scale.value)
        scale_arg = None if s in (0.0, 1.0) else s
        try:
            in_mb = src.stat().st_size / 1e6
            ctx.status(f"Compressing {src.name} ({in_mb:.1f} MB)...")
            result = compress_video(
                src,
                out,
                crf=int(crf.value),
                preset=str(preset.value),
                scale=scale_arg,
            )
            out_mb = Path(result).stat().st_size / 1e6
            pct = (100 * out_mb / in_mb) if in_mb else 0.0
            ctx.status(
                f"Compressed: {src.name} ({in_mb:.1f} MB) -> "
                f"{Path(result).name} ({out_mb:.1f} MB, {pct:.0f}% of original)"
            )
            _refresh_info()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            ctx.status(f"Compress failed: {exc}")

    button = PushButton(name="compress", text="Compress")
    button.changed.connect(lambda *_: _on_compress())

    _refresh_info()

    container = Container(widgets=[crf, preset, scale, info, button])
    # Expose a refresh hook so the shell can update info when a video loads.
    container.refresh_info = _refresh_info  # type: ignore[attr-defined]
    return container
