# Canonical identified-cell template for leech segmental ganglia — design

Date: 2026-06-02
Species: *Helobdella austinensis*
Identity source: Kuo et al. 2024, *J Exp Biol* 227:jeb247419 (PMC11418187)

## Scope

This project builds **only** the canonical template and the tooling to create it.
It does **not** build the calcium-imaging pipeline. A separate pipeline will later
register each experimental prep's segmented cell centroids to this template via
point-set registration (pycpd), so identified neurons can be tracked across preps
and conditions.

There is no machine-readable atlas. Cell identities and positions are digitized by
hand from figure-panel images of the Kuo et al. ganglion maps, which the user
provides as cropped images.

## What the template is

A set of **named cell centroids** in a defined canonical 2D coordinate frame.

- **Separate maps per aspect.** Ventral and dorsal surfaces are distinct layers and
  distinct files — never merged into one plane.
- **Independent L/R digitization.** Left and right cells are each clicked from the
  figures. We do **not** mirror one side across the midline to synthesize the other:
  real left/right positional asymmetry is part of the organization and must be
  preserved. The midline is still recorded so that side and medial–lateral position
  are meaningful.

### Template entry schema (one row per cell)

| column      | type   | values / meaning                                  |
|-------------|--------|---------------------------------------------------|
| name        | str    | canonical cell name (e.g. `Retzius`, `DE-3`)      |
| side        | str    | `L`, `R`, or `M` (midline)                         |
| aspect      | str    | `ventral` or `dorsal`                              |
| x           | float  | medial–lateral, canonical units (right = +)        |
| y           | float  | anterior–posterior, canonical units (posterior = +)|
| confidence  | str    | `high`, `medium`, `low`                             |
| notes       | str    | free text (e.g. "ID uncertain", "partially occluded")|

One file per aspect: `template/ventral.csv`, `template/dorsal.csv`.

## Coordinate convention (explicit)

Absolute scale varies by prep, so coordinates are **normalized**. Each figure panel is
calibrated with **three clicked landmarks**:

1. **Anterior midline point** — origin `(0, 0)`.
2. **Posterior midline point** — defines the A–P axis direction and the scale unit.
3. **Right-side reference point** — disambiguates the L/R sign (ventral vs dorsal
   views and image flips can invert handedness, so this is recorded per figure, never
   assumed).

From these:

- **Origin**: anterior midline point.
- **y axis (anterior–posterior)**: unit vector from anterior to posterior midline
  point; `y` increases posteriorly. `y = 1` at the posterior midline landmark.
- **x axis (medial–lateral)**: perpendicular to the A–P axis, signed so the right-side
  reference has `x > 0`. Left cells are `x < 0`, right cells `x > 0`, midline `x ≈ 0`.
- **Scale**: isotropic, equal to the anterior→posterior midline distance (coords are in
  "ganglion-length units"). Isotropic scaling preserves the figure's aspect ratio and
  therefore real asymmetry.
- **Side inference**: `|x| <= midline_tol` → `M`; else sign of `x`. The user can
  override side per cell (the tool defers to an explicit choice).

The pixel→canonical transform and its inverse live in `coordinates.py` as pure
functions and are unit-tested (round-trip, sign of x, midline tolerance).

Each aspect also writes a **calibration sidecar** (`template/<aspect>_calibration.json`)
recording the three landmark pixel coordinates, the source image path and its SHA-256,
and the A–P length in pixels — so any template file is reproducible and re-openable.

## Cell scope (first pass)

Start with the **crawling-relevant identified cells**, expanding later without rework:

`Retzius`, `P` (pressure), `N` (nociceptive), `T` (touch), `AE`, `CV`, `DE-3`, `NS/151`.

These seed a controlled vocabulary in `naming.py` / `cell_names.yaml`. The annotation
tool offers these names but allows free-text names, flagging any name not in the list.
Counts per cell (e.g. multiple P/N/T subtypes) are **not** hardcoded — the user clicks
whatever the figure shows and labels each point.

## Repo layout

```
special_project/
  pyproject.toml            # uv-managed, Python 3.11
  config.yaml               # paths, aspect, name-list path, midline tolerance
  README.md
  src/leechtemplate/
    __init__.py
    config.py               # load + validate config.yaml -> dataclass
    naming.py               # load/validate canonical name list
    coordinates.py          # PURE: Calibration + pixel<->canonical transforms
    template_io.py          # read/write template CSV + calibration JSON, schema
    annotate.py             # napari + magicgui interactive annotation tool
  data/figures/             # input figure panels (gitignored)
  template/                 # outputs: ventral.csv, dorsal.csv, *_calibration.json
  scripts/
    view_template.py        # matplotlib sanity plot (midline, L/R, by aspect)
  tests/
    test_coordinates.py
    test_template_io.py
    test_naming.py
```

## Components

- **coordinates.py** (pure, tested): `Calibration` dataclass;
  `pixel_to_canonical`, `canonical_to_pixel`, `infer_side`. No I/O, no napari.
- **template_io.py**: schema validation, CSV read/write, calibration JSON read/write,
  image hashing. Depends on pandas only.
- **naming.py**: load the controlled vocabulary; `is_known(name)` for the tool to flag
  unknown names. No hard counts.
- **config.py**: parse `config.yaml` into a dataclass; resolve paths.
- **annotate.py**: thin napari/magicgui layer over the tested modules. A dock widget
  sets the *current* name/side/confidence/notes; new points inherit these via the
  Points layer `feature_defaults`. A calibration Points layer holds the 3 landmarks
  tagged by role. Save computes canonical coords and writes CSV + calibration JSON;
  Load repopulates an existing aspect for resume/editing.
- **view_template.py**: load template CSV(s), scatter in canonical frame with the
  midline drawn, L/R/M colored, ventral/dorsal as separate axes — a fast eyeball check.

## Data flow

```
figure panel image
   └─(napari: 3 calibration clicks)─> Calibration
   └─(napari: cell clicks + labels)─> points (pixel) + features
        └─ pixel_to_canonical ─> (x, y) canonical
             └─ template_io ─> template/<aspect>.csv  +  <aspect>_calibration.json
                  └─ view_template.py ─> sanity plot
                  └─ (later, separate project) pycpd registration
```

## Error handling

- Calibration requires exactly the 3 roles before Save; missing/duplicate roles raise a
  clear error surfaced in the napari notification area.
- Schema validation on read and write: bad `side`/`aspect`/`confidence` values raise.
- Unknown cell names are allowed but flagged (warning + `notes`), never silently dropped.
- Loading an aspect whose calibration image hash differs from the current image warns,
  so coordinates are not silently mixed across different source crops.

## Testing

- `coordinates.py`: round-trip pixel→canonical→pixel; x-sign follows the right
  reference; midline tolerance classification; isotropic scale invariance.
- `template_io.py`: CSV and JSON round-trip; schema rejection of bad values.
- `naming.py`: known/unknown classification against the seed list.
- The napari GUI is not unit-tested by design; logic it depends on is isolated and
  tested in the modules above.

## Out of scope (this session)

Calcium-imaging pipeline, motion correction, segmentation, dF/F, rhythmicity,
and the actual pycpd registration — all live in the separate pipeline project.
