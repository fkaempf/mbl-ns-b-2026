# Leech tools (special_project)

Two independent hand-annotation tools for the leech (*Helobdella austinensis*) work,
each its own package under `src/`:

- **`leechtemplate`** — build a canonical 2D template of identified neurons (this file, below).
- **`leecharena`** — annotate leech orientation in a radial arena from video, and
  auto-track leech positions (see **Arena orientation tool** at the end).

`uv run pytest` runs both suites.

---

# Install on another computer

The project is fully reproducible via [uv](https://docs.astral.sh/uv/) (a `pyproject.toml`
+ `uv.lock` pin every dependency and the Python version). On a fresh machine:

```bash
# 1. install uv (macOS/Linux):
curl -LsSf https://astral.sh/uv/install.sh | sh        # Windows: see uv docs

# 2. get the code and enter this folder:
git clone https://github.com/fkaempf/mbl-ns-b-2026.git
cd mbl-ns-b-2026/leech/special_project

# 3. create the environment (fetches Python 3.11 + napari/Qt/OpenCV/ffmpeg, all pinned):
uv sync

# 4. run the app:
uv run leech-arena                               # then pick a video in the window
uv run leech-arena --video path/to/clip.mp4
uv run pytest                                    # optional: run the test suite
```

No system ffmpeg or OpenCV install is needed — they come via `imageio-ffmpeg` and
`opencv-python-headless` inside the uv environment. If `uv run` ever resolves the wrong
Python, call the venv directly: `.venv/bin/python -m leecharena.app`.

---

# Leech canonical identified-cell template

Tooling to build a **canonical 2D template** of identified neurons in *Helobdella
austinensis* segmental ganglia, digitized by hand from figure panels in
Kuo et al. 2024 (*J Exp Biol* 227:jeb247419, PMC11418187).

A separate calcium-imaging pipeline (not in this repo) will register each prep's
segmented cell centroids to this template via point-set registration (pycpd), so
identified neurons can be tracked across preps and conditions.

See `docs/superpowers/specs/2026-06-02-leech-canonical-template-design.md` for the
full design and the coordinate convention.

## What gets produced

- `template/ventral.csv`, `template/dorsal.csv` — named cell centroids, one row per
  cell: `name, side (L/R/M), aspect, x, y, confidence, notes`. Ventral and dorsal are
  **separate files**; left and right are digitized **independently** (never mirrored).
- `template/<aspect>_calibration.json` — the 3 calibration landmarks + source image
  hash, so each template is reproducible and re-openable.

## Coordinate convention

Normalized, since absolute scale varies by prep. Each figure is calibrated with three
clicks: **anterior midline** (origin), **posterior midline** (sets the A–P axis and the
unit scale), and a **right-side reference** (fixes the L/R sign). `x` is medial–lateral
(right = +), `y` is anterior–posterior (posterior = +), scale is isotropic in
ganglion-length units.

## Setup

```bash
uv sync            # creates .venv with Python 3.11 and all deps (incl. napari + Qt)
```

## Annotate a figure

```bash
uv run leech-annotate --aspect ventral --image data/figures/your_panel.png
# or: uv run python -m leechtemplate.annotate --aspect dorsal --image ...
```

In the napari window:
1. With the **calibration** layer active, set *calibration role* and click the three
   landmarks (anterior_midline → posterior_midline → right_ref).
2. With the **cells** layer active, set name/side/confidence/notes in the dock widget,
   then click each soma. New points inherit the current values; `side = auto` resolves
   from position at save time.
3. Click **Save template**. Use **Load existing** to resume an aspect later.

## Eyeball the result

```bash
uv run python scripts/view_template.py            # both aspects
uv run python scripts/view_template.py --aspect ventral
```

## Tests

```bash
uv run pytest
```

Tests cover the pure logic (coordinate transforms, schema I/O, naming). The napari GUI
is intentionally not unit-tested; it is a thin layer over those tested modules.

---

# Arena orientation tool (`leecharena`)

Measure how leeches are oriented in a circular arena relative to food, by
hand-annotating a randomized sample of frames from single-dish MP4 clips. See
`docs/superpowers/specs/2026-06-02-leech-arena-orientation-design.md`.

Config: `arena_config.yaml`.

## Unified app (recommended): one window, four tabs

The whole pipeline is also available as a single napari app that shares **one
viewer + one loaded video + one frame slider** across four tabs:

- **Compress** — re-encode the loaded video with H.264 (CRF / preset / scale).
- **Split** — build a mean/median composite, place one ellipse per dish (Hough
  auto-detect or by hand), then save ROIs to config or split into clips.
- **Annotate** — orientation annotation on sampled frames (arena ellipse, food,
  per-leech points). Choose `ant/post/mid` or `ant/post` nodes; random or even-interval
  (by fps) sampling; click-to-advance roles; **Space = save & next**; per-frame arena
  auto-detect. Saved to the annotations CSV.
- **Track** — automatic leech-position tracking: seed each leech on a frame, tune
  params, **Verify 10 s**, **Predict full video** (background), then scrub/▶Play and
  drag to correct. Two body-end nodes per leech; a central food deadzone; the dish
  is re-detected periodically to follow drift. Writes `annotations/tracks_<video>.csv`.

```bash
uv run leech-arena --video data/videos/raw.mp4
# or, offline / without uv:
.venv/bin/python -m leecharena.app --video data/videos/raw.mp4
# launch bare and pick a video in the session dock:
uv run leech-arena
```

Load (or switch) the video once in the **session** dock; every tab operates on
that same video and the shared frame slider. The per-CLI tools below still work
unchanged if you prefer a single-purpose command.

## 1. Split multi-dish recordings into single-dish clips (if needed)

### Recommended: place dishes on a composite image in napari

```bash
uv run leech-arena-place --video data/videos/raw.mp4    # or launch bare and pick in-GUI
```

Does the whole split workflow in one window:
1. **Composite** — pick the video, choose `mean`/`median` and an optional frame sample
   (blank/0 = every frame), and click **Build composite**. Dishes are static so they
   stay sharp while moving leeches wash out — a far better target than any single frame.
2. **Dishes** — one editable ellipse per dish, pre-seeded by **Auto-detect dishes**
   (Hough). Drag/resize/add/delete as needed.
3. **Output** — **Save circles to config** writes `split.rois` to `arena_config.yaml`
   for batch reuse, or **Split now** writes the cropped `*_dishN.mp4` clips immediately
   (tick "compress" for H.264).

### Headless / batch

```bash
uv run leech-arena-split --video data/videos/raw.mp4 --out clips/
```

Uses `split.rois` from `arena_config.yaml` if set (e.g. saved from the placement tool),
otherwise auto-detects dishes (Hough circles) on a sample frame. Writes one cropped
`*_dishN.mp4` per dish. Add `--compress` to re-encode the clips with H.264 as written.

## 1b. Compress recordings (H.264)

```bash
uv run leech-arena-compress --video data/videos/raw.mp4 --out data/videos/raw_small.mp4 --crf 28
# optional: --scale 0.5 to halve resolution
```

Re-encodes MP4s with libx264 (ffmpeg located via imageio-ffmpeg — no system ffmpeg
needed). `--crf` trades quality for size (18 high … 32 small; 28 default). Or set
`compression.enabled: true` in `arena_config.yaml` to compress splitter output
automatically.

## 2. Annotate orientation on a single-dish clip

```bash
uv run leech-arena-annotate --video clips/raw_dish0.mp4
```

The app serves randomized frames (seeded, resumable — already-annotated frames are
skipped). Per frame:
- **Arena**: auto-detected as an editable ellipse; drag/resize it if wrong.
- **Leeches** (multiple, no IDs): set *role* (anterior/posterior/middle) and *leech #*
  in the dock, then click. Use "New leech #" for the next animal.
- **Food**: click in the `food` layer, or untick "food present" if absent.
- **Save & next** appends rows to `annotations/annotations.csv` and advances.

## 3. Derive metrics and eyeball

```bash
uv run python scripts/view_orientations.py
```

Writes `annotations/metrics.csv` (heading, body length, bend, orientation relative to
food, radial position, arena bearing) and plots headings within the arena plus the
distribution of orientation relative to food.

## Outputs

- `annotations/annotations.csv` — one row per leech per frame (raw clicked points +
  arena circle + food).
- `annotations/metrics.csv` — derived orientation metrics.
