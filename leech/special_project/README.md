# Leech tools (special_project)

Tools for the leech (*Helobdella austinensis*) work:

- **`leecharena`** — annotate leech orientation in a radial arena from video, and
  auto-track leech positions (see **Arena orientation tool** at the end).
- **`analysis/calcium`** — calcium-imaging analysis for whole-leech GCaMP recordings.

`uv run pytest` runs the test suite.

---

# Install on another computer

The project is fully reproducible via [uv](https://docs.astral.sh/uv/) (a `pyproject.toml`
+ `uv.lock` pin every dependency and the Python version). You only need **git** and **uv**;
Python, napari/Qt, OpenCV and ffmpeg are all installed by `uv sync` (no system ffmpeg or
OpenCV needed — they come via `imageio-ffmpeg` and `opencv-python-headless`).

## macOS / Linux

```bash
# 1. install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    (git is preinstalled on macOS via `xcode-select --install`; Linux: `sudo apt install git`)

# 2. get the code and enter this folder:
git clone https://github.com/fkaempf/mbl-ns-b-2026.git
cd mbl-ns-b-2026/leech/special_project

# 3. create the environment (fetches Python 3.11 + all deps, pinned):
uv sync

# 4. run the app:
uv run leech-arena                               # then pick a video in the window
uv run leech-arena --video path/to/clip.mp4
uv run pytest                                    # optional: run the test suite
```

If `uv run` ever resolves the wrong Python, call the venv directly:
`.venv/bin/python -m leecharena.app`.

## Windows (step by step)

Use **PowerShell** (search Start menu → "PowerShell"). Windows 10/11 already include
`winget` and `curl`.

```powershell
# 1. install git and uv:
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
#    (if winget is unavailable, install uv with:
#     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" )

# 2. CLOSE and reopen PowerShell so git and uv are on PATH, then verify:
git --version
uv --version

# 3. get the code and enter this folder:
git clone https://github.com/fkaempf/mbl-ns-b-2026.git
cd mbl-ns-b-2026\leech\special_project

# 4. create the environment (downloads Python 3.11 + napari/Qt/OpenCV/ffmpeg, pinned):
uv sync

# 5. run the app:
uv run leech-arena                               # then pick a video in the window
uv run leech-arena --video path\to\clip.mp4
uv run pytest                                    # optional: run the test suite
```

Notes for Windows:
- Use backslashes in paths (`path\to\clip.mp4`), or quote paths that contain spaces.
- napari opens a normal desktop window; if it fails to start, update your GPU drivers
  (napari needs OpenGL). On a remote/headless box use a real display, not RDP software GL.
- If `uv run` resolves the wrong Python, run `.venv\Scripts\python -m leecharena.app`.

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
