# Leech Ganglion Calcium Imaging — Analysis Report

**Recording:** `helobdella_LeechNo2_elavgcamp6m-17.tif`
**Indicator:** pan-neuronal `elav`-GCaMP6m, *Helobdella* ventral nerve cord
**Acquisition:** 692 frames · 700×700 px · 1.3 µm/px · single plane · single channel ·
frame interval 0.347 s → **fs ≈ 2.88 Hz** · ~**240 s** total
**Tools:** suite2p 1.1.0 (registration), custom `analysis/calcium/` module (everything else)

---

## 0. What I did (pipeline overview)

1. **Installed suite2p 1.1.0** (from `Mouseland/suite2p`) into the project's `uv` venv.
2. **Registered** the movie with suite2p (motion correction). Measured drift was
   ~0 px rigid / <0.5 px non-rigid — the prep is essentially still. All traces below
   come from the **registered** binary (`data.bin`).
3. **Annotated 11 ganglia** by hand in a napari tool I built
   (`annotate_ganglia.py`), drawing one polygon per ganglion on the
   local-correlation background. Saved as two ROI **sets**:
   - **`full`** (100% polygons) and **`small`** (each polygon scaled to **65%**
     about its centroid, to avoid edge/background contamination).
   - Both **renumbered `G1…G11` left-to-right** (G1 = anterior, G11 = posterior).
4. **Extracted per-ganglion ΔF/F** and computed correlation, raster, lag, events,
   and pixel-level correlation maps (`ganglion_activity.py`).
5. **Removed the shared global signal** and re-ran the analysis to expose
   ganglion-specific structure (`--remove-global`).
6. **Characterised the global rhythm**: power spectrum + per-pixel phase map
   (`phase_analysis.py`).
7. **Batched** suite2p + phase analysis over **15 LeechNo2 recordings** to test
   whether the rhythm replicates (`scripts/batch_phase.py`).

Everything is reproducible — commands are in the last section. The math
(ΔF/F, correlation, lag, events, common-mode removal) has unit tests in
`tests/test_calcium_analysis.py` (10 passing).

---

## 1. Field of view & finding the ganglia

![mean and local-correlation](figures/01_mean_localcorr.png)

**What it is.** *Left:* the time-averaged image (raw anatomy). *Right:* the
**local-correlation image** — for every pixel, the average temporal correlation
with its 4 nearest neighbours.

**How to read it.** In the right panel, a pixel is **bright** when it fluctuates
*in sync with its neighbours over time* — which is what real active neural tissue
does, while shot noise does not. So bright = coherent activity.

**Interpretation.** The mean image is murky, but the correlation image cleanly
reveals the **chain of segmental ganglia** running diagonally across the FOV.
This is what guided the annotation.

![local correlation full-res](figures/05_local_correlation.png)

*(Same map, full resolution, as produced by the analysis pipeline.)*

---

## 2. The annotated ganglia

![ROIs on mean image](figures/02_rois_on_mean.png)

**What it is.** The 11 hand-drawn ganglion ROIs (the `full` set) over the mean image.

**How to read it.** Each coloured patch is one ganglion, numbered along the cord.

![full vs small](figures/03_full_vs_small.png)

**What it is.** **Red = full** (100%) outlines; **green = small** (65%, shrunk to
each ROI's centroid). **How to read it.** The green outline nests inside the red,
pulled off the bright edges. **Why.** Widefield ROIs catch out-of-focus
background at their borders; shrinking tests whether results depend on that edge.

![overlay video frame](figures/04_overlay_frame.png)

**What it is.** A single frame from the rendered overlay video
(`…/runs/…-17/overlay.mp4`), with both ROI sets + left-to-right numbers burned in.
**How to read it.** Play the MP4 to watch activity under each ganglion over time;
the timestamp (top-left) and legend show frame/time and which colour is which set.

---

## 3. Per-ganglion activity (raw ΔF/F)

![per-ganglion traces](figures/07_per_ganglion_traces.png)

**What it is.** Each row is one ganglion's ΔF/F (G1 anterior → G11 posterior).
ΔF/F = (F − F₀)/F₀ with F₀ a rolling 8th-percentile baseline. Red bar = ΔF/F scale.

**How to read it.** Vertical position is just an offset per ganglion; look at the
*wiggles*. Up = brighter (more activity).

**Interpretation.** Every ganglion carries the **same regular ~0.36 Hz oscillation,
in phase, for the whole recording.** That dominant shared rhythm is the headline
feature of this dataset.

![traces raster](figures/08_traces_raster.png)

**What it is.** Top: the same traces. Bottom: a **raster** (ganglion × time
heatmap; colour = ΔF/F). **How to read it.** Vertical stripes spanning all rows =
all ganglia bright at the same instants = synchrony. That's exactly what appears.

---

## 4. Pixel-level correlation (seed maps)

![seed correlation maps](figures/06_seed_maps.png)

**What it is.** For each ganglion ("seed"), every pixel is coloured by how strongly
its time course correlates with that ganglion's mean trace (red = positive).

**How to read it.** Red shows the spatial extent of pixels that move *with* the seed.

**Interpretation.** Because of the global rhythm, each seed lights up much of the
whole cord — another view of the same pervasive synchrony.

---

## 5. Ganglion × ganglion correlation — RAW

![raw correlation heatmap](figures/09_corr_raw.png)

**What it is.** Pearson correlation between every pair of ganglion ΔF/F traces.
**How to read it.** Diagonal = 1 (self). Off-diagonal: dark red = +1 (move
together), white = 0, blue = −1 (opposite). Each cell is labelled.

**Interpretation.** **Everything correlates at 0.91–1.00.** This is *not* evidence
of specific coupling — it is the single global rhythm making every region look
identical. Mean off-diagonal r = **0.96**; PCA shows one component explains
**97.5%** of the variance. **This plot is a trap if read naively.**

---

## 6. Propagation / lag — RAW

![lag matrix](figures/10_lag_matrix.png)

**What it is.** For each pair, the time lag (s) of the best cross-correlation —
does one ganglion lead another? **How to read it.** Cell [row, col] = lag of the
column ganglion relative to the row; sign = who leads.

**Interpretation.** All lags are ~0 — the global rhythm is **synchronous, no
coarse propagation**. (Caveat: at 2.88 Hz the finest resolvable lag is ~0.35 s.)

---

## 7. Removing the global signal → the REAL structure

![residual correlation heatmap](figures/11_corr_residual.png)

**What it is.** The ganglion × ganglion correlation **after regressing out the
shared common-mode signal** from every trace (`--remove-global`).

**How to read it.** Same colour scale as §5, but now red/blue mean genuinely
co-active / anti-phase *after* the dominant rhythm is removed.

**Interpretation.** A real, structured pattern emerges:
- **Adjacent ganglia co-vary** — G1–G2 ≈ +0.62, G9–G10 ≈ +0.60.
- **Anterior vs posterior anti-correlate** — e.g. G4↔G9 ≈ −0.59, G2↔G7 ≈ −0.51.

**This is reproducible:** the residual matrices from the `small` (65%) and `full`
(100%) ROI sets agree at **r = 0.82**, so it is not an artifact of ROI size. The
residual is only ~3% of total variance — a small, slow modulation riding on the
big rhythm — but it is consistent. Note: common-mode regression *can* induce some
anti-correlation, so treat the anti-phase blocks as suggestive, confirmed mainly
by their cross-ROI reproducibility.

---

## 8. What is the global rhythm? — power spectrum

![global spectrum](figures/12_spectrum.png)

**What it is.** Power spectrum of the global-mean trace (whole-frame brightness
over time). **How to read it.** X = frequency (Hz); the dotted line at 1.44 Hz is
the Nyquist limit (fastest measurable = fs/2). Y = power, **log scale** (each
gridline ×10). A tall narrow peak above the flat "grass" floor = a real rhythm.
The big rise near 0 Hz is slow drift / photobleaching, not a rhythm.

**Interpretation.** One clean peak at **0.362 Hz (2.76 s period)**, ~100× above the
floor. Faster than leech heartbeat (~0.1 Hz), slower than swim (~1–2 Hz).

---

## 9. Where & when the rhythm peaks — phase/amplitude maps

![phase and amplitude maps](figures/13_phase_amp_maps.png)

**What it is.** Per-pixel Fourier analysis at 0.362 Hz. *Left:* mean image.
*Middle:* **amplitude** (how strongly each pixel oscillates at 0.362 Hz).
*Right:* **phase** (the *timing* of each pixel within the cycle; cyclic colormap,
opacity ∝ amplitude).

**How to read it.** Amplitude bright = strong rhythm there. Phase colour: same
colour = same timing; a smooth colour sweep = a travelling wave.

**Interpretation.** Amplitude confirms the rhythm lives **on the cord/ganglia**.
The phase colour is **nearly uniform** → the rhythm fires near-synchronously
everywhere; no obvious wave.

---

## 10. Anterior–posterior propagation test

![ganglion phase gradient](figures/14_ganglion_phase.png)

**What it is.** *Left:* each ganglion plotted at its position, coloured by its phase
at 0.362 Hz. *Right:* phase vs anterior→posterior position. **Left y-axis = phase
(rad); right y-axis = the same as a time lag (ms).**

**How to read it.** A straight sloped line = constant lag per ganglion = a
**travelling wave** (slope = direction/speed). A flat line = **synchronous**.

**Interpretation.** The line is **nearly flat and non-monotonic** — the whole cord
spans only ~**0–70 ms** (out of a 2760 ms cycle). So **no significant A–P
propagation**; the rhythm is essentially synchronous. This *corrects* an earlier
"travelling wave" guess: the reproducible A–P anti-phase in §7 is **not** a phase
gradient of this carrier — it is a separate, slower amplitude modulation.

---

## 11. Does it replicate? — batch across 15 recordings

![batch phase montage](figures/15_batch_montage.png)

**What it is.** suite2p + phase analysis run on 15 LeechNo2 recordings. Top row =
amplitude maps; bottom = phase maps; titles give the peak frequency and a
phase-spread (synchrony) metric. Full table: `…/runs/_batch_phase_summary.csv`.

**How to read it.** Compare peak frequencies and whether the cord lights up
coherently across recordings.

**Interpretation.** The fast rhythm is **intermittent and variable**:
- Clear fast peak only in a subset — **−13 (0.458 Hz), −14 (0.547 Hz), −17
  (0.362 Hz)**.
- Most long recordings (−05, −06, −08, −09, −11, −12, −15) are dominated by slow
  drift (peak 0.02–0.04 Hz) with **no** clean fast rhythm.
- Short clips (30–70 frames: −02, −03, −07, −16) are too brief to trust.

Because the frequency **differs between recordings** rather than being fixed, this
argues **against a constant optical/illumination artifact** (which would repeat at
one fixed frequency) and is **more consistent with an intermittent, state-dependent
biological rhythm**. Not proof — but a meaningful constraint.

---

## 12. Key findings

1. The recording is **dominated by one near-synchronous ~0.36 Hz oscillation**
   across the whole nerve cord (97.5% of variance). Raw correlations (~0.96
   everywhere) reflect *only* this and are otherwise uninformative.
2. **No anterior–posterior propagation** of that rhythm (<~70 ms end-to-end;
   essentially synchronous).
3. After removing the global signal, a **small but reproducible** structure
   remains: **adjacent ganglia co-vary, anterior vs posterior anti-correlate**
   (cross-ROI reproducibility r = 0.82). This is a slow amplitude modulation, not a
   phase gradient.
4. Across 15 recordings the fast rhythm is **intermittent and variable in
   frequency** (0.36–0.55 Hz in 3 recordings, absent in the rest) — more consistent
   with a state-dependent biological rhythm than a fixed artifact.

## 13. Caveats

- **Widefield, not 2-photon:** out-of-focus background and neuropil mixing.
- **2.88 Hz sampling:** cannot resolve lags < ~0.35 s; a very fast wave can't be
  excluded.
- **Common-mode regression** can induce some spurious anti-correlation; the §7
  result is trusted mainly because it reproduces across ROI sizes.
- **Single prep.** The §7 structure needs replication in other recordings (the
  batch registered them; annotating + residual-analysing them is the next step).
- The rhythm's identity (heartbeat-linked? bursting? optical?) is **not yet
  established**.

## 14. How to reproduce

```bash
# 1. register a movie (auto-reads fs; writes data/calcium/runs/<stem>/)
python scripts/run_suite2p.py data/calcium/2026-06-04/helobdella_LeechNo2_elavgcamp6m-17.tif

# 2. annotate ganglia in napari (draw polygons, set 'set name', Save)
python -m analysis.calcium.annotate_ganglia data/calcium/runs/<stem> --set full

# 3. per-ganglion analysis (raw, then common-mode removed)
python -m analysis.calcium.ganglion_activity data/calcium/runs/<stem> <regions>/small
python -m analysis.calcium.ganglion_activity data/calcium/runs/<stem> <regions>/small --remove-global

# 4. global-rhythm phase analysis
python -m analysis.calcium.phase_analysis data/calcium/runs/<stem> <regions>/small

# 5. labelled overlay video
python -m analysis.calcium.render_overlay data/calcium/runs/<stem> <regions>

# 6. batch phase analysis across all runs
python scripts/batch_phase.py
```

## 15. File map

- `analysis/calcium/signals.py` — load run, local-correlation, seed maps, ΔF/F
- `analysis/calcium/annotate_ganglia.py` — napari ganglion annotation (ROI sets)
- `analysis/calcium/ganglion_activity.py` — correlation/raster/lag/events + `--remove-global`
- `analysis/calcium/phase_analysis.py` — spectrum + per-pixel phase map
- `analysis/calcium/render_overlay.py` — labelled overlay MP4
- `scripts/run_suite2p.py`, `scripts/batch_phase.py` — single + batch drivers
- `tests/test_calcium_analysis.py` — unit tests for the pure math
- Results per run: `analysis/calcium/regions/<stem>/{full,small}/{results,results_residual,phase}/`
- Batch montage + table: `data/calcium/runs/_batch_phase_montage.png`, `_batch_phase_summary.csv`
