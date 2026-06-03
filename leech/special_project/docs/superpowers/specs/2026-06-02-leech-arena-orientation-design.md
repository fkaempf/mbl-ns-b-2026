# Leech arena-orientation annotation tool — design

Date: 2026-06-02
Package: `leecharena` (sits alongside `leechtemplate` in the same repo)

## Goal

Measure how leeches are oriented inside a circular ("radial") arena relative to food,
by hand-annotating a randomized sample of frames from MP4 recordings. Output is raw
clicked coordinates per frame; orientation metrics are derived from them.

## Decisions (from the user)

- **One dish per clip for annotation.** Raw recordings may contain two dishes; a
  separate **splitter** fragments them into single-dish clips first. The annotator
  assumes exactly one arena.
- **Multiple leeches per dish, no identity tracking.** Per frame, annotate each
  leech's anterior/posterior/middle; `leech_idx` is a within-frame ordinal only.
- **Food annotated per frame** (with an "absent" option).
- **MP4 (H.264)**; random frame access via OpenCV `VideoCapture` + `CAP_PROP_POS_FRAMES`.

## Two tools

### 1. Splitter (`split.py`)
Turn a raw multi-dish MP4 into single-dish clips.
- Detect circular dishes on a representative frame (`cv2.HoughCircles`), OR use
  manual ROIs from config when detection is unreliable.
- For each dish, write a cropped MP4 (square crop bounding the circle) via
  `cv2.VideoWriter`.
- CLI: `leech-arena-split --video raw.mp4 --out clips/`.

### 2. Annotator (`annotate.py`) — the main tool
- Open a single-dish clip; sample **N randomized frames** (seeded, resumable).
- Per frame, in napari:
  - **Arena**: auto-detect the circle (`HoughCircles`); shown as an editable ellipse
    the user can move/resize when detection is wrong. Recorded as (cx, cy, r).
  - **Food**: one point (or marked absent).
  - **Leeches**: a Points layer; for each leech set role
    (anterior/posterior/middle) and leech index in the dock, then click. "New leech"
    increments the index and resets role to anterior.
  - **Save & next** appends the frame's rows and loads the next unannotated sampled
    frame. Progress shows k/N.
- CLI: `leech-arena-annotate --video clips/dish0.mp4`.

## Coordinate convention

Pixel coordinates in the clip's frame, `x = column`, `y = row` (y down, image
convention). napari uses (row, col); converted at the boundary. Arena center and food
provide the reference frame for derived metrics; nothing is normalized at annotation
time (the arena radius gives scale for metrics).

## Output schema — `annotations.csv` (one row per leech per frame)

| column | meaning |
|--------|---------|
| video | clip identifier (filename) |
| frame | frame index in the clip |
| time_s | frame / fps |
| arena_cx, arena_cy, arena_r | detected/adjusted arena circle (px) |
| food_x, food_y | food point, or empty if absent |
| leech_idx | within-frame ordinal (-1 if frame has no leech) |
| ant_x, ant_y | anterior point |
| post_x, post_y | posterior point |
| mid_x, mid_y | middle point |

A fully annotated frame with no leeches still writes one placeholder row
(`leech_idx = -1`, point columns empty) so "annotated but empty" is distinct from
"not yet annotated" for resume.

## Derived metrics (`metrics.py` -> `metrics.csv`)

Computed from raw points, never stored during annotation:
- `heading_rad`: direction posterior→anterior, standard math convention (y up).
- `body_length_px`: |anterior − posterior|.
- `bend_px`: perpendicular distance of middle from the anterior–posterior line.
- `food_dir_rad`: direction middle→food.
- `orient_rel_food_rad`: wrapped(heading − food_dir); 0 = facing the food.
- `radial_pos`: |middle − arena_center| / arena_r (0 center, 1 wall).
- `arena_angle_rad`: bearing of the leech middle from arena center.

## Module layout

```
src/leecharena/
  __init__.py
  config.py      # load arena_config.yaml -> dataclass
  video.py       # VideoReader over cv2.VideoCapture (len, fps, frame(i))
  arena.py       # circle<->ellipse-bbox helpers (pure) + detect_circle (cv2)
  sampling.py    # seeded random frame plan, resumable
  store.py       # annotations.csv schema, append, annotated-frame lookup
  metrics.py     # derive orientation metrics
  annotate.py    # napari + magicgui app (thin over the above)
  split.py       # multi-dish -> single-dish clips (optional H.264 compression)
  compress.py    # H.264 (libx264) re-encode via ffmpeg (imageio-ffmpeg binary)
scripts/
  view_orientations.py   # quick plot of headings within the arena, vs food
tests/
  test_arena.py     # bbox<->circle round trip; HoughCircles on a synthetic disk
  test_sampling.py  # seeded determinism; resume skips annotated; count/uniqueness
  test_store.py     # schema round trip; append; annotated-frame set
  test_metrics.py   # heading/relative-to-food/radial on known geometry
```

## Testing

Pure logic (arena bbox math, sampling, store schema, metrics geometry) is unit-tested;
`detect_circle` is tested on a synthetic drawn disk with OpenCV-headless. The napari GUI
is a thin layer over tested modules and is not unit-tested (consistent with the
`leechtemplate` tool).

## Anti-scope

No automatic leech tracking/pose estimation — annotation is manual by design. No
multi-dish annotation (splitter handles that upstream). No metric is computed inside
the annotation save path.
