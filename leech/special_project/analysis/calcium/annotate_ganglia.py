"""Stage 2: draw one polygon per ganglion in napari.

Background defaults to the local-correlation image (active ganglia lit up),
with the mean image and the playable registered movie as toggleable layers.
Draw polygons in the 'ganglia' Shapes layer, then click Save.

  python -m analysis.calcium.annotate_ganglia <run_dir> [--regions <out_dir>]

Saves to <out_dir> (default analysis/calcium/regions/<run-stem>/):
  masks.npy     (n_regions, Ly, Lx) bool
  ganglia.json  {ganglia:[{id, name, vertices}], run_dir, shape}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import signals


def _default_regions_dir(run_dir: Path) -> Path:
    stem = run_dir.name
    if stem in ("plane0", "suite2p"):
        stem = run_dir.parent.name if stem == "plane0" else run_dir.name
    # walk up to the run folder name under runs/
    for part in run_dir.parts[::-1]:
        if part not in ("plane0", "suite2p"):
            stem = part
            break
    root = Path(__file__).resolve().parents[2]  # project root
    return root / "analysis" / "calcium" / "regions" / stem


def launch(run_dir: str | Path, regions_dir: str | Path | None = None,
           set_name: str = "default") -> None:
    import napari
    from magicgui.widgets import Container, LineEdit, PushButton, Label

    run_dir = Path(run_dir)
    base = Path(regions_dir) if regions_dir else _default_regions_dir(run_dir)
    # Each annotation *set* is its own subfolder, so you can keep several
    # (e.g. 'full' big hulls vs 'small' edge-shrunk) without overwriting.
    set_dir = base / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    run = signals.load_run(run_dir)
    print("Computing local-correlation background ...")
    local_corr = signals.local_correlation_image(run.movie)
    Ly, Lx = run.mean_img.shape

    viewer = napari.Viewer(title=f"ganglia — {run_dir.name}")
    viewer.add_image(local_corr, name="local correlation", colormap="inferno",
                     contrast_limits=(0, float(np.percentile(local_corr, 99))))
    viewer.add_image(run.mean_img, name="mean image", colormap="gray",
                     visible=False)
    viewer.add_image(run.movie, name="movie", colormap="gray", visible=False)

    shapes = viewer.add_shapes(name="ganglia", shape_type="polygon",
                               edge_color="cyan", face_color="transparent",
                               edge_width=3)

    # Re-load this set's regions if present, so annotation is resumable.
    existing = set_dir / "ganglia.json"
    if existing.exists():
        meta = json.loads(existing.read_text())
        polys = [np.array(g["vertices"]) for g in meta["ganglia"]]
        if polys:
            shapes.add(polys, shape_type="polygon")
        print(f"Loaded {len(polys)} existing region(s) from {existing}")

    set_field = LineEdit(value=set_name, label="set name")
    name_field = LineEdit(value="G", label="name prefix")
    status = Label(value=f"set '{set_name}': draw polygons, then Save")
    save_btn = PushButton(text="Save regions")

    def _save():
        polys = [np.asarray(p) for p in shapes.data]
        if not polys:
            status.value = "no polygons drawn"
            return
        # Save into base/<set name>; typing a new set name = save-as a new set.
        out = base / (set_field.value or "default")
        out.mkdir(parents=True, exist_ok=True)
        masks = np.stack([m for m in shapes.to_masks((Ly, Lx))]).astype(bool)
        prefix = name_field.value or "G"
        ganglia = [{"id": i + 1, "name": f"{prefix}{i + 1}",
                    "vertices": polys[i].tolist()} for i in range(len(polys))]
        np.save(out / "masks.npy", masks)
        (out / "ganglia.json").write_text(json.dumps(
            {"ganglia": ganglia, "run_dir": str(run_dir),
             "shape": [int(Ly), int(Lx)]}, indent=2))
        status.value = f"saved {len(ganglia)} region(s) -> {out}"
        print(status.value)

    save_btn.clicked.connect(_save)
    viewer.window.add_dock_widget(
        Container(widgets=[set_field, name_field, save_btn, status]),
        name="ganglia", area="right")
    napari.run()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Annotate leech ganglia in napari")
    ap.add_argument("run_dir", help="suite2p run dir (…/runs/<stem> or its plane0)")
    ap.add_argument("--regions", default=None, help="base regions dir")
    ap.add_argument("--set", dest="set_name", default="default",
                    help="annotation set name (subfolder); e.g. full, small")
    args = ap.parse_args(argv)
    launch(args.run_dir, args.regions, set_name=args.set_name)


if __name__ == "__main__":
    main()
