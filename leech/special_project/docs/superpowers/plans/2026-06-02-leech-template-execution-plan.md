# Execution plan — leech canonical-template tooling

Date: 2026-06-02
Each phase is self-contained and runnable in a fresh chat context. Tasks are framed
to COPY from verified docs/code, not to transform from memory.

Design reference: `docs/superpowers/specs/2026-06-02-leech-canonical-template-design.md`
Scope boundary: this repo is template tooling ONLY (not the imaging pipeline).

---

## Phase 0 — Documentation Discovery output (DONE; verified with sources)

Two Documentation Discovery subagents verified every external-library API used in
`src/leechtemplate/annotate.py` against authoritative sources (napari stable API ref +
napari v0.4.19 source; magicgui official docs + source `main`). Result: **no invented
APIs; nothing to fix for correctness.**

### Allowed APIs (verified — safe to rely on)

napari 0.4.19+ (sources: napari.org/stable API ref; napari `layers/points/points.py`,
`layers/utils/layer_utils.py` @ v0.4.19; context7 `/napari/napari` llms.txt):
- `napari.Viewer(title=...)`, `napari.run()`.
- `viewer.add_image(image, name=...)`.
- `viewer.add_points(data, name=, features={col: array}, face_color=, size=, symbol=, text={...})`.
  - `features=` accepts a dict of arrays (or DataFrame).
  - `text=` dict keys are `string`, `size`, `color`, `anchor`, `translation` (also
    `visible`/`rotation`/`blending`). The code uses `string`/`color`/`size` — all valid.
    `string="{name}"` resolves against the `name` feature column.
- `layer.feature_defaults["col"] = value` — and **newly added points inherit these
  defaults** (`_FeatureTable.resize()` fills new rows from `_defaults`; `currents()`
  derives from `defaults`). This is the core mechanism of the tool and it holds.
- `layer.features` getter returns a DataFrame; `layer.features = df` replaces all.
- `layer.data = np.ndarray` replaces all points; point order is **(row, col) = (y, x)**.
- `viewer.window.add_dock_widget(widget, name=, area="right")`; valid `area` ∈
  {left, right, top, bottom}; a magicgui `Container` is an accepted widget.

magicgui 0.8+ (sources: pyapp-kit.github.io/magicgui docs; magicgui source `_concrete.py`,
`bases/_button_widget.py`, `_categorical_widget.py`, `_value_widget.py`,
`_container_widget.py`):
- From `magicgui.widgets`: `ComboBox`, `Container`, `Label`, `LineEdit`, `PushButton`.
- `ComboBox(label=, choices=[...], value=)`; `LineEdit(label=, value=)`;
  `Label(value=)`; `PushButton(text=)`.
- `.value` is a read/write property on value widgets.
- `widget.changed.connect(callback)` registers a handler. For `PushButton`,
  `.changed` works for clicks; `.clicked` is a documented alias (more readable).
- `Container(widgets=[...])` is itself a dockable widget.

imageio: `imageio.v3.imread(path)` (already used; standard).

### Anti-patterns to avoid (DO NOT do these)
- Do NOT replace the `feature_defaults` mechanism with `current_properties` string
  juggling or manual per-point feature writes — the defaults-inheritance path is the
  documented, verified approach.
- Do NOT add `text=` keys outside the verified set above.
- Do NOT pass an `area` to `add_dock_widget` other than left/right/top/bottom.
- Do NOT "fix" the verified API calls in `annotate.py` for correctness — they are
  confirmed correct. Only change code if a real runtime error appears in Phase 1.

---

## Phase 1 — Install full deps and smoke-test the GUI (THE real gap)

The pure modules are unit-tested (18 passing), but the napari GUI has never been
launched (no Qt display in the build environment). This phase proves it runs.

**What to do (copy these documented commands, do not improvise APIs):**
1. `uv sync` — installs Python 3.11 + numpy/pandas/matplotlib/pyyaml/imageio +
   `napari[all]` (pulls a Qt backend). Expect a large download.
2. Place any test image at `data/figures/test_panel.png` (a screenshot is fine for a
   smoke test; a real Kuo et al. panel for real work).
3. Launch: `uv run leech-annotate --aspect ventral --image data/figures/test_panel.png`
4. In the napari window, exercise the documented flow from `README.md`:
   - calibration layer active → set role → click anterior_midline, posterior_midline,
     right_ref (3 points).
   - cells layer active → set name/side/confidence in the dock → click a few somata.
   - click **Save template**, then **Load existing**.

**Documentation references:** `README.md` (Annotate section); Phase 0 Allowed APIs;
`src/leechtemplate/annotate.py` (`launch`, lines ~78–245).

**Verification checklist:**
- [ ] napari window opens with the image and two points layers (`calibration`, `cells`).
- [ ] After Save: `template/ventral.csv` exists and `uv run python -c "from leechtemplate.template_io import load_template; print(load_template('template/ventral.csv'))"` prints rows with valid schema.
- [ ] `template/ventral_calibration.json` exists with 3 landmarks + image sha256.
- [ ] Load existing repopulates both layers (points reappear at the same spots).
- [ ] `uv run python scripts/view_template.py --aspect ventral` renders midline + L/R/M.

**Anti-pattern guards:**
- If a widget callback errors, FIRST re-check against Phase 0 Allowed APIs before
  editing — the APIs are verified, so a real failure is more likely an env/Qt issue.
- Do not stub out the GUI or replace napari interaction with synthetic data to "pass"
  this phase; the point is to confirm the interactive path.

---

## Phase 2 — Optional polish from Phase 0 findings (small, doc-cited)

Only readability/robustness; no behavior change required.

**What to do:**
1. (Readability) For the two `PushButton`s in `annotate.py`, copy the documented
   idiom: `save_btn.clicked.connect(do_save)` / `load_btn.clicked.connect(do_load)`
   instead of `.changed.connect(...)`. Source: magicgui overview example
   (`PushButton(...)` + `@button.clicked.connect`); `.clicked` is the alias for
   `.changed`, so behavior is identical.

**Documentation references:** Phase 0 magicgui findings (item 5); magicgui docs
overview example.

**Verification checklist:**
- [ ] `uv run pytest -q` still 18 passing (annotate.py is import-only in tests).
- [ ] GUI Save/Load buttons still fire (repeat the Phase 1 click test).

**Anti-pattern guards:** Do not refactor the `feature_defaults` flow; it is verified.

---

## Phase 3 — Digitize the crawling subset (data production)

Produce the first real template once Kuo et al. figure panels are available.

**What to do:**
1. Drop the ventral and dorsal figure panels into `data/figures/`.
2. Run `leech-annotate` once per aspect; digitize the crawling seed cells
   (Retzius, P, N, T, AE, CV, DE-3, NS/151) per `src/leechtemplate/cell_names.yaml`,
   clicking BOTH sides independently (do not mirror).
3. Set `confidence`/`notes` honestly for uncertain IDs.
4. Eyeball with `view_template.py` for both aspects.

**Documentation references:** spec doc (cell scope, coordinate convention);
`cell_names.yaml`; `README.md`.

**Verification checklist:**
- [ ] `template/ventral.csv` and `template/dorsal.csv` both exist and validate.
- [ ] L and R cells have independent (non-mirrored) coordinates — verify a pair like
      Retzius L/R do NOT have exactly negated x with identical y.
- [ ] Midline cells (if any) carry `side == M`.
- [ ] Each aspect has a matching `*_calibration.json` whose `image_sha256` matches the
      panel actually used.

**Anti-pattern guards:**
- Do NOT synthesize one side by mirroring the other (the whole point of independent
  L/R digitization).
- Do NOT merge ventral and dorsal into one file.

---

## Phase 4 — Downstream interface note (handoff, NOT built here)

The pycpd registration that consumes this template lives in the SEPARATE imaging
pipeline project (see memory: project-scope-template-vs-pipeline). This phase only
records the contract so that project can consume the template without rework:
- Input it will read: `template/<aspect>.csv` with columns
  `name, side, aspect, x, y, confidence, notes` (canonical normalized frame).
- The pipeline registers prep centroids → these `(x, y)` per aspect.
No code in this repo. Do not implement registration here.

---

## Final Phase — Verification

Draft commands (run from repo root):

1. **Unit tests:** `uv run pytest -q` → expect `18 passed`.
2. **No invented-API regressions:** confirm only verified text keys / areas are used:
   - `grep -n "text=" src/leechtemplate/annotate.py` → keys ⊆ {string,size,color,anchor,translation,visible,rotation,blending}.
   - `grep -n "add_dock_widget" src/leechtemplate/annotate.py` → `area` ∈ {left,right,top,bottom}.
3. **Schema integrity of any produced templates:**
   `uv run python -c "from leechtemplate.template_io import load_template; load_template('template/ventral.csv')"` (raises on bad schema).
4. **Both aspects + matching calibration present** (after Phase 3):
   - `ls template/ventral.csv template/dorsal.csv template/ventral_calibration.json template/dorsal_calibration.json`
5. **Independent L/R (no mirroring) spot check:** load a template and confirm a known
   bilateral pair does not have exactly mirrored coordinates.
6. **Sanity plot renders:** `uv run python scripts/view_template.py`.

Map back to phases: step 1 covers all code phases; steps 2 guards Phase 2/anti-patterns;
steps 3–6 cover Phase 3 data production.
