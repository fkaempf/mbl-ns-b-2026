"""Run the suite2p pipeline on a single calcium TIFF.

Usage:
    python scripts/run_suite2p.py <path/to/movie.tif>

Frame rate (fs) is read from the TIFF's own ImageJ `finterval` metadata.
tau is fixed at 1.0 s for GCaMP6m. Single plane / single channel
(these CZI->TIFF exports contain only the green/GCaMP channel). Runs on
CPU (no CUDA on macOS).

Results are written to  data/calcium/runs/<file-stem>/suite2p/plane0/ .
Load them in the GUI via File -> Load processed data -> stat.npy, or:
    python -c "from suite2p.gui import gui2p; gui2p.run(statfile='<.../stat.npy>')"
"""
import sys
from pathlib import Path
import tifffile
import suite2p

ROOT = Path(__file__).resolve().parents[1]

# Default to the previously-run movie if no arg is given.
DEFAULT = ROOT / "data" / "calcium" / "helobdella_elavgcamp6m+palmmcherry-02.tif"


def derive_fs(tiff_path: Path) -> float:
    """Frame rate from ImageJ finterval (s/frame); fall back to 1.0 Hz."""
    with tifffile.TiffFile(tiff_path) as tf:
        md = tf.imagej_metadata or {}
    fi = md.get("finterval")
    return round(1.0 / fi, 4) if fi else 1.0


def main(tiff: Path) -> None:
    tiff = tiff.resolve()
    if not tiff.exists():
        sys.exit(f"No such file: {tiff}")

    fs = derive_fs(tiff)
    out = ROOT / "data" / "calcium" / "runs" / tiff.stem
    out.mkdir(parents=True, exist_ok=True)

    db = {
        **suite2p.default_db(),
        "data_path": [str(tiff.parent)],
        "file_list": [str(tiff)],   # restrict to THIS file (folder has many tiffs)
        "save_path0": str(out),
        "input_format": "tif",
        "nplanes": 1,
        "nchannels": 1,
        "functional_chan": 1,
    }
    settings = suite2p.default_settings()
    settings["fs"] = fs
    settings["tau"] = 1.0          # GCaMP6m
    settings["torch_device"] = "cpu"

    print(f"Movie : {tiff.name}")
    print(f"fs    : {fs} Hz   tau: 1.0 s")
    print(f"Out   : {out/'suite2p'/'plane0'}")
    suite2p.run_s2p(db=db, settings=settings)
    print("DONE ->", out / "suite2p" / "plane0")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    main(target)
