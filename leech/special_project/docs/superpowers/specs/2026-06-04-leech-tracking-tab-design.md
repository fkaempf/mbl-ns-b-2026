# Leech position-tracking tab — design

Add automatic leech **position** tracking to the unified napari app, with tuneable
parameters, a fast 10-second verification pass, full-clip background prediction, and
per-frame manual correction by scrubbing + dragging in napari.

Scope is **position only** (no anterior/posterior, no head/tail). Method is classical
detect-then-track (no ML, no labels, no GPU), validated on IMG_2859 dishes: over the
full 86-min dish1 clip both tracks were present in ~99.9% of frames with only 6–7% of
frames coasting (held), and a central food deadzone keeps tracks off the food clump.

## Components

### 1. `leecharena/tracking.py` — engine (pure, testable)
- `TrackParams` dataclass with validated-good defaults:
  `n_tracks=2, deadzone_frac=0.32, motion_thresh=14, dark_thresh=22,
   area_min=30, area_max=6000, max_jump=90, max_hold=90`.
- `annulus_mask(shape, cx, cy, r, deadzone_frac) -> np.ndarray[bool]` — dish interior
  (≤ 0.90·r) minus the central food deadzone (≥ deadzone_frac·r). Pure.
- `detect_blobs(gray, bg_gray, mask, params) -> list[(area, x, y)]` — foreground =
  `(|gray-bg| > motion_thresh) OR ((bg-gray) > dark_thresh)`, within mask, morph
  open(3)+close(9), connected components filtered to `area_min..area_max`, sorted by
  area desc. Pure (feed arrays). cv2 imported lazily.
- `associate(tracks, held, dets, params) -> (tracks, held)` — greedy nearest-neighbour
  of live tracks to detections within `max_jump`; unmatched tracks increment `held`;
  leftover detections re-acquire empty or stale (`held > max_hold`) tracks. Pure.
- `track_clip(video, params, frame_range=None, progress=None) -> list[dict]` — builds
  the median background (`compositing.composite`), finds the dish (`arena.detect_circle`,
  centered fallback), reads frames sequentially, rejects contaminated frames (median
  brightness far from background), runs detect+associate, returns rows
  `{frame, time_s, track, x, y, held}`. Calls `progress(done, total)` if given.

### 2. `leecharena/tracks_store.py` — dense per-frame CSV
- Columns: `frame, time_s, track, x, y, corrected`.
- `save_tracks(path, rows)`, `load_tracks(path) -> DataFrame`,
  `upsert_track_frame(path, frame, rows)` (replace one frame's rows for corrections).
- Path convention: `annotations/tracks_<video-stem>.csv`. Pure, tested.

### 3. `leecharena/track_panel.py` — new "Track" tab
- One widget per `TrackParams` field (spinboxes), seeded with defaults.
- **Verify 10 s**: run `track_clip` on ~`10*fps` frames from the current session frame
  slider position (synchronous). Load results into a `tracks` Points layer; user scrubs
  to review. Status reports track count and held %.
- **Predict full video**: run `track_clip` over the whole clip in a
  `napari.qt.threading.thread_worker`; on finish, `save_tracks` to
  `tracks_<video>.csv`; status shows progress %.
- **Manual correction**: the `tracks` Points layer shows the **current frame's**
  positions, synced to the frame slider. Drag/add/delete a point → `upsert_track_frame`
  for that frame with `corrected=1`; other frames keep their predictions. Existing
  `tracks_<video>.csv` is loaded on open.

### 4. `AppContext.on_frame_changed(fn)` hook
Mirror the existing `on_video_loaded`. The session frame slider already calls
`show_frame`; fire registered frame listeners there so the Track tab can refresh the
points layer when scrubbing.

## Tests
- `tracking.py`: `annulus_mask` geometry; `detect_blobs` on a synthetic frame (a dark
  square inside, a food blob in the deadzone that must be excluded, a speck below
  area_min); `associate` for nearest-neighbour match, hold-last on a missed detection,
  and re-acquire after `max_hold`.
- `tracks_store.py`: save/load round-trip; `upsert_track_frame` replaces one frame and
  leaves others intact.
- GUI untested per existing convention, plus one headless `build_track_panel`
  construction smoke test (offscreen Qt, mock viewer).

## Out of scope (YAGNI)
- No ML / learned keypoints. No anterior/posterior. No identity beyond N nearest-neighbour
  tracks. No re-track-on-drag (corrections are per-frame only). No full overlay-video
  export from the GUI (the 579 MB render confirmed CSV + scrubbing is the right path).
