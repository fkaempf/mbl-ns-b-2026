"""Interactive napari tool to digitize identified-cell positions from a figure panel.

Workflow per figure (one aspect at a time):
  1. Calibrate: place exactly three points and tag each with its role
     (anterior_midline, posterior_midline, right_ref).
  2. Annotate: click each identified soma; set its name / side / confidence / notes
     in the dock widget BEFORE clicking (new points inherit the current values).
  3. Save: pixel positions are converted to canonical coordinates and written to
     template/<aspect>.csv plus a calibration sidecar.
  4. Load: re-open a previously saved aspect to resume or edit.

This module is a thin GUI layer over the tested pure modules (coordinates,
template_io, naming). napari is imported lazily so the package can be imported
without a Qt environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from .config import Config
from .coordinates import (
    Calibration,
    canonical_to_pixel,
    infer_side,
    pixel_to_canonical,
    rowcol_from_xy,
    xy_from_rowcol,
)
from .naming import NameList
from .template_io import (
    CalibrationRecord,
    load_calibration,
    load_template,
    save_calibration,
    save_template,
    sha256_of_file,
)

CALIB_ROLES = ["anterior_midline", "posterior_midline", "right_ref"]
SIDE_CHOICES = ["auto", "L", "R", "M"]


def _calibration_from_layer(calib_layer) -> Calibration:
    """Build a Calibration from the calibration Points layer, by role."""
    roles = list(calib_layer.features["role"])
    data = calib_layer.data  # (N, 2) in (row, col)
    by_role: dict[str, tuple[float, float]] = {}
    for role, rowcol in zip(roles, data):
        if role in by_role:
            raise ValueError(f"duplicate calibration role: {role!r}")
        x, y = xy_from_rowcol(rowcol)[0]
        by_role[role] = (float(x), float(y))
    missing = [r for r in CALIB_ROLES if r not in by_role]
    if missing:
        raise ValueError(f"calibration incomplete; missing roles: {missing}")
    return Calibration(
        anterior_midline=by_role["anterior_midline"],
        posterior_midline=by_role["posterior_midline"],
        right_ref=by_role["right_ref"],
    )


def launch(image_path: str | Path, aspect: str, config: Config) -> None:
    import napari
    from magicgui.widgets import (
        ComboBox,
        Container,
        Label,
        LineEdit,
        PushButton,
    )

    image_path = Path(image_path)
    if aspect not in config.aspects:
        raise ValueError(f"aspect must be one of {config.aspects}, got {aspect!r}")

    names = NameList.from_yaml(config.name_list)
    image = iio.imread(image_path)

    viewer = napari.Viewer(title=f"leech template — {aspect} — {image_path.name}")
    viewer.add_image(image, name=f"{aspect} figure")

    calib_layer = viewer.add_points(
        np.empty((0, 2)),
        name="calibration",
        features={"role": np.array([], dtype=object)},
        face_color="yellow",
        size=14,
        symbol="cross",
    )
    calib_layer.feature_defaults["role"] = CALIB_ROLES[0]

    cells_layer = viewer.add_points(
        np.empty((0, 2)),
        name="cells",
        features={
            "name": np.array([], dtype=object),
            "side": np.array([], dtype=object),
            "confidence": np.array([], dtype=object),
            "notes": np.array([], dtype=object),
        },
        face_color="red",
        size=10,
        text={"string": "{name}", "color": "white", "size": 8},
    )
    cells_layer.feature_defaults["name"] = names.names[0] if names.names else ""
    cells_layer.feature_defaults["side"] = "auto"
    cells_layer.feature_defaults["confidence"] = config.default_confidence
    cells_layer.feature_defaults["notes"] = ""

    # --- dock widget ---------------------------------------------------------
    role_combo = ComboBox(label="calibration role", choices=CALIB_ROLES)
    name_combo = ComboBox(
        label="cell name", choices=names.names or [""], value=(names.names or [""])[0]
    )
    name_free = LineEdit(label="…or free name", value="")
    side_combo = ComboBox(label="side", choices=SIDE_CHOICES, value="auto")
    conf_combo = ComboBox(
        label="confidence", choices=config.confidences, value=config.default_confidence
    )
    notes_edit = LineEdit(label="notes", value="")
    status = Label(value="Place 3 calibration points, then annotate cells.")

    def current_name() -> str:
        return name_free.value.strip() or name_combo.value

    def push_calib_default(*_):
        calib_layer.feature_defaults["role"] = role_combo.value

    def push_cell_defaults(*_):
        cells_layer.feature_defaults["name"] = current_name()
        cells_layer.feature_defaults["side"] = side_combo.value
        cells_layer.feature_defaults["confidence"] = conf_combo.value
        cells_layer.feature_defaults["notes"] = notes_edit.value

    role_combo.changed.connect(push_calib_default)
    for w in (name_combo, name_free, side_combo, conf_combo, notes_edit):
        w.changed.connect(push_cell_defaults)

    def do_save(*_):
        try:
            calib = _calibration_from_layer(calib_layer)
            n = len(cells_layer.data)
            if n == 0:
                status.value = "No cells to save."
                return
            xy = xy_from_rowcol(cells_layer.data)
            canon = pixel_to_canonical(xy, calib)
            feats = cells_layer.features
            rows = []
            unknown = []
            for i in range(n):
                cname = str(feats["name"].iloc[i]) or "UNNAMED"
                if not names.is_known(cname):
                    unknown.append(cname)
                side = str(feats["side"].iloc[i])
                if side == "auto":
                    side = infer_side(float(canon[i, 0]), config.midline_tol)
                rows.append(
                    {
                        "name": cname,
                        "side": side,
                        "aspect": aspect,
                        "x": float(canon[i, 0]),
                        "y": float(canon[i, 1]),
                        "confidence": str(feats["confidence"].iloc[i]),
                        "notes": str(feats["notes"].iloc[i] or ""),
                    }
                )
            df = pd.DataFrame(rows)
            save_template(df, config.template_csv(aspect))
            record = CalibrationRecord(
                aspect=aspect,
                image_path=str(image_path),
                image_sha256=sha256_of_file(image_path),
                calibration=calib,
            )
            save_calibration(record, config.calibration_json(aspect))
            msg = f"Saved {n} cells -> {config.template_csv(aspect)}"
            if unknown:
                msg += f"  (flagged unknown names: {sorted(set(unknown))})"
            status.value = msg
        except Exception as exc:  # surface the reason in the GUI, don't crash
            status.value = f"Save failed: {exc}"

    def do_load(*_):
        try:
            calib_path = config.calibration_json(aspect)
            csv_path = config.template_csv(aspect)
            if not calib_path.exists() or not csv_path.exists():
                status.value = f"Nothing saved yet for aspect {aspect!r}."
                return
            record = load_calibration(calib_path)
            if record.image_sha256 != sha256_of_file(image_path):
                status.value = (
                    "WARNING: current image differs from the one used to calibrate "
                    "this saved template; coordinates may not match. Loaded anyway."
                )
            calib = record.calibration
            calib_layer.data = rowcol_from_xy(
                np.array(
                    [calib.anterior_midline, calib.posterior_midline, calib.right_ref]
                )
            )
            calib_layer.features = pd.DataFrame({"role": CALIB_ROLES})

            df = load_template(csv_path)
            pix = canonical_to_pixel(df[["x", "y"]].to_numpy(), calib)
            cells_layer.data = rowcol_from_xy(pix)
            cells_layer.features = df[["name", "side", "confidence", "notes"]].copy()
            if status.value.startswith("WARNING"):
                return
            status.value = f"Loaded {len(df)} cells from {csv_path}."
        except Exception as exc:
            status.value = f"Load failed: {exc}"

    save_btn = PushButton(text="Save template")
    load_btn = PushButton(text="Load existing")
    save_btn.changed.connect(do_save)
    load_btn.changed.connect(do_load)

    widget = Container(
        widgets=[
            Label(value=f"Aspect: {aspect}"),
            role_combo,
            Label(value="— cell defaults (set before clicking) —"),
            name_combo,
            name_free,
            side_combo,
            conf_combo,
            notes_edit,
            save_btn,
            load_btn,
            status,
        ]
    )
    viewer.window.add_dock_widget(widget, name="annotate", area="right")
    push_calib_default()
    push_cell_defaults()
    napari.run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Annotate leech identified cells on a figure panel.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--aspect", required=True, choices=["ventral", "dorsal"])
    parser.add_argument("--image", required=True, help="path to the figure-panel image")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    launch(args.image, args.aspect, config)


if __name__ == "__main__":
    main()
