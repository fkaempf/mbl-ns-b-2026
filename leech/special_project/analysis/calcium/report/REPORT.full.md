# Leech Ganglion Calcium Imaging — Analysis Report

**Recording:** `helobdella_LeechNo2_elavgcamp6m-17.tif`
**Indicator:** pan-neuronal `elav`-GCaMP6m, *Helobdella* ventral nerve cord
**Acquisition:**
- Frames: 692 (~**240 s** total)
- FOV: 700×700 px
- Resolution: 1.3 µm/px
- Plane / channel: single plane · single channel
- Frame interval: 0.347 s (≈ 0.35 s; used as the lag / temporal-resolution floor below)
- Sampling rate: **fs = 1/0.347 ≈ 2.88 Hz**; Nyquist = 1.44 Hz
**Tools:** suite2p 1.1.0 (registration), custom `analysis/calcium/` module (everything else)

---

## Pipeline overview (what I did)

1. **Installed suite2p 1.1.0** (from `Mouseland/suite2p`) into the project's `uv` venv.
2. **Registered** the movie with suite2p (motion correction). Result: the prep is
   essentially motionless — measured drift was ~0 px (rigid correction) and <0.5 px
   (non-rigid correction); all traces below come from the **registered** binary
   (`data.bin`).

   Importantly, residual sub-pixel motion could still produce exactly the kind of
   whole-field correlated brightness signal we see here, so the registration numbers
   alone cannot rule out a motion or illumination artifact. The main available
   constraint against that is the cross-recording frequency variability (§11), though
   that too is only suggestive (3/15 recordings) and does not exclude per-recording
   motion, illumination, or non-neural sources (§11, §13).

   A direct check — correlating the 0.362 Hz brightness signal against the
   frame-by-frame registration offsets — was not performed; it would more directly
   bound residual-motion contamination (listed as an outstanding caveat in §13).
3. **Annotated 11 ganglia** by hand in a napari tool I built
   (`annotate_ganglia.py`), drawing one polygon per ganglion on the
   local-correlation background. Saved as two ROI **sets**:
   - **`full`** (100% polygons) and **`small`** (each polygon scaled to **65%** of
     its linear extent about its centroid, pulling the ROI inward away from bright
     edges where out-of-focus background contaminates the signal).
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
with its 4 nearest neighbours; a standard way to highlight pixels whose fluctuations
are shared locally (i.e. likely the same source) while suppressing independent noise.

**How to read it.** A pixel is **bright** when it fluctuates in sync with its
neighbours over time. This coherence can come from coordinated neural activity OR
from whole-field modulation (residual motion, illumination drift); uncorrelated
shot noise stays dark. So bright = coherent signal, not necessarily neural.

**Interpretation.** The mean image is murky, but the correlation image cleanly
reveals a coherent **diagonal structure that aligns with the anatomical chain of
segmental ganglia** in the mean image. This map was used only to **localize the
cord for annotation**, not to establish that the signal is neural — the
artifact-vs-biology question is adjudicated later (§11, §13).

![local correlation full-res](figures/05_local_correlation.png)

*(Same map, full resolution, as produced by the analysis pipeline.)*

---

## 2. The annotated ganglia

![ROIs on mean image](figures/02_rois_on_mean.png)

**What it is.** The 11 hand-drawn ganglion ROIs (the `full` set) over the mean image.

**How to read it.** Each coloured patch is one ganglion, numbered along the cord.

![full vs small](figures/03_full_vs_small.png)

**What it is.** **Red = full** (100%) outlines; **green = small** (65%, shrunk
toward each ROI's centre).

**How to read it.** The green outline nests inside the red, pulled off the bright
edges.

**Interpretation.** Widefield ROIs catch out-of-focus background at their borders;
shrinking tests whether results depend on that edge.

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

**Interpretation.** Every ganglion carries a **regular, near-synchronous shared
fluctuation**. Its oscillatory character and exact frequency (~0.36 Hz) are
established by the spectrum in §8, not by these traces. That dominant shared signal
is the headline feature of this dataset (the degree of synchrony is quantified
in §10).

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
of specific coupling — it is a single dominant shared (common-mode) signal making
every region look identical. Mean off-diagonal r = **0.96**; PCA shows one component
explains **97.5%** of the variance across the 11 ganglion-mean traces (not of the raw
pixel data). (That this common-mode signal is *oscillatory* — a rhythm rather than,
say, shared drift — is not shown here but in §8, where the spectrum resolves it.)

The 97.5% and the ~0.96 mean r are two closely related summaries of the same
dominance (the traces are spatial averages of a common-mode-dominated field), not
independent confirmations. They are not numerically identical because one is a mean
of pairwise linear correlations and the other a fraction of total variance on the
leading eigenvector. Note also that PC1 here aggregates the oscillation *and* any
shared slow drift (§8): the 97.5% therefore reflects total shared
low-frequency-plus-rhythmic variance, not the rhythm alone (this matters for §12
finding 1).

**This plot is a trap if read naively.**

---

## 6. Propagation / lag — RAW

![lag matrix](figures/10_lag_matrix.png)

**What it is.** For each pair, the time lag (s) of the best cross-correlation —
does one ganglion lead another? **How to read it.** Cell [row, col] = lag of the
column ganglion relative to the row; sign = who leads.

**Interpretation.** All resolved lags are ~0, so there is **no coarse propagation
detectable above the frame interval**; this is consistent with synchrony but,
because lags are quantized to ~0.35 s, cannot distinguish true synchrony from a
sub-frame lead. (Caveat: lags are quantized to the frame interval (~0.35 s), so
sub-frame leads cannot be resolved.)

---

## 7. Removing the global signal → the REAL structure

![residual correlation heatmap](figures/11_corr_residual.png)

**What it is.** The ganglion × ganglion correlation **after regressing out the
shared common-mode signal** from every trace (`--remove-global`).

**How to read it.** Same colour scale as §5, but now red/blue mean genuinely
co-active / anti-phase *after* the dominant rhythm is removed.

**Interpretation.** A structured but unvalidated pattern emerges (see the two
cautions below before trusting it; the named pairs are *representative* of the
pattern, not standalone quantitative findings):
- **Adjacent ganglia co-vary** — representative pairs include G1–G2 ≈ +0.62,
  G9–G10 ≈ +0.60.
- **Anterior vs posterior anti-correlate** (but see below: a regression-induced
  negative-sum constraint can mimic exactly this A–P anti-phase) — representative
  pairs include G4↔G9 ≈ −0.59, G2↔G7 ≈ −0.51.

**This is reproducible:** the residual matrices from the `small` (65%) and `full`
(100%) ROI sets agree at **r = 0.82**. That cross-ROI agreement validates the
*spatial pattern* (adjacent-covary / anterior–posterior-anti), not the exact
per-pair coefficients, and shows it is not an artifact of ROI **size or
lateral boundary/edge effects** (axial out-of-focus contamination is shared by both
sets and is *not* addressed by shrinking — see the §13 widefield caveat). It does *not*, however, rule out a shared low-frequency
**drift** component: both ROI sets see the same slow drift, so reproducibility
across them cannot subtract it (see the §13 photobleaching caveat — the relevant
unaddressed confound here). The residual is only ~3% of total variance — a small,
slow modulation riding on the big rhythm. Treat it as a **small candidate
structure** pending that drift control, not a settled result.

Two further cautions on the anti-phase blocks specifically. (1) Regressing out the
mean of 11 traces mechanically forces the residuals to sum to ~zero, which imposes
a **negative-sum constraint** on the off-diagonal correlations — so *some* net
anti-correlation is expected **by construction**, regardless of biology, and the
anterior–posterior anti-phase pattern is exactly what this artifact most easily
mimics. The magnitude matters, though: a pure single-mean negative-sum constraint
induces only a weak *average* anti-correlation of order 1/(N−1) ≈ −0.1 across 11
traces, which is far smaller than the representative pair values near −0.5/−0.6
above. So the constraint plausibly **biases** the pattern but cannot by itself
manufacture anti-phase that strong — which is the motivation for the phase-shuffled
null below, not a reason to dismiss the result outright. (2) Cross-ROI
reproducibility validates the *stability* of the pattern but
**cannot distinguish induced from genuine** anti-phase, because the same regression
is applied to both ROI sets. A proper control (future work, not claimed here) would
compare the observed anti-correlation against the level expected from regression
alone — e.g. a phase-shuffled or shared-component null.

One further statistical caveat on every per-pair r in this section: the residual is
slow and strongly autocorrelated, so the effective number of independent samples
behind each coefficient is far smaller than 692 time points. Even the cross-ROI
r = 0.82 should be read as a stability check, not a powered estimate.

---

## 8. What is the global rhythm? — power spectrum

![global spectrum](figures/12_spectrum.png)

**What it is.** Power spectrum of the global-mean trace (whole-frame brightness
over time). **How to read it.** X = frequency (Hz); the dotted line at 1.44 Hz is
the Nyquist limit (fastest measurable = fs/2). Y = power, **log scale** (each
gridline ×10). A tall narrow peak above the flat "grass" floor = a real rhythm.
The big rise near 0 Hz is slow drift / photobleaching, not a rhythm.

**Interpretation.** A single narrow peak at **0.362 Hz (2.76 s period)** standing
roughly two log-decades above the surrounding noise floor (read off the plot, not
a formal SNR). The 240 s record contains ~87 cycles of this rhythm, so the peak is
well-sampled here — unlike the 30–70-frame short clips in §11, which are too brief
to resolve it. Faster than leech heartbeat (~0.1 Hz), slower than swim (~1–2 Hz);
these comparison bands are for orientation only — the species/prep-specific values,
and whether GCaMP can even follow them, are not established here. These bands are
generic-leech literature values and may not apply to *Helobdella*, which differs in
size and behavioural repertoire. Because they are not validated for *Helobdella*,
the fact that 0.362 Hz falls between them carries **no inferential weight** about
behavioural identity — it is given only to orient readers familiar with other leech
preparations. Note too that
GCaMP6m's slow rise/decay kinetics (~hundreds of ms) act as a temporal low-pass
filter on the underlying spiking, so **0.362 Hz is the dominant frequency of the
indicator signal, not necessarily of the underlying firing** (see §13).

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
Within the high-amplitude (on-cord) pixels the phase colour is **nearly uniform**
(quantified as a 0–70 ms spread in §10) → the rhythm modulates near-synchronously
everywhere, peaking at nearly the same phase across the cord; no obvious wave.
Caveat: the amplitude-opacity weighting can visually suppress genuine phase variation
in moderately-active pixels, so read this map together with the per-ganglion phase
plot in §10. (Off-cord, low-amplitude pixels have noise-dominated, essentially random
phase; the opacity-weighting by amplitude is what makes the map readable.)

---

## 10. Anterior–posterior propagation test

![ganglion phase gradient](figures/14_ganglion_phase.png)

**What it is.** *Left:* each ganglion plotted at its position, coloured by its phase
at 0.362 Hz. *Right:* phase vs anterior→posterior position. **Left y-axis = phase
(rad); right y-axis = the same as a time lag (ms).**

**How to read it.** A straight sloped line = constant lag per ganglion = a
**travelling wave** (slope = direction/speed). A flat line = **synchronous**.

**Interpretation.** The line is **nearly flat and non-monotonic** — the whole cord
spans only ~**0–70 ms** (out of a 2760 ms cycle). So **no detectable A–P
propagation at this sampling resolution**; the rhythm is **consistent with
synchrony to within the ~350 ms resolution floor**.
Crucially, that ~0–70 ms spread is well below the ~350 ms frame interval (the lag
caveat in §6/§13), so it sits at or below the resolution floor: read it as an
**upper bound consistent with synchrony**, not as a measured small lag. This
*corrects* an earlier
"travelling wave" guess: the reproducible A–P anti-phase in §7 is therefore **not**
a phase gradient of this carrier; if real, it would have to be a separate, slower
amplitude modulation rather than a phase gradient of the carrier — that is the only
interpretation left open by the flat phase here, not an independently measured slow
modulation.

---

## 11. Does it replicate? — batch across 15 recordings

![batch phase montage](figures/15_batch_montage.png)

**What it is.** suite2p + phase analysis run on 15 LeechNo2 recordings. Top row =
amplitude maps; bottom = phase maps; titles give the peak frequency and a
phase-spread (synchrony) metric. Full table: `…/runs/_batch_phase_summary.csv`.

**How to read it.** Compare peak frequencies and whether the cord lights up
coherently across recordings.

**Interpretation.** The fast rhythm is **intermittent and variable**, and the
replication is **partial — it was identifiable in only 3 of 15 recordings**:
- Clear fast peak only in a subset — **−13 (0.458 Hz), −14 (0.547 Hz), −17
  (0.362 Hz)**.
- Most long recordings (−05, −06, −08, −09, −11, −12, −15) are dominated by slow
  drift (peak 0.02–0.04 Hz) with **no** clean fast rhythm.
- Short clips (30–70 frames: −02, −03, −07, −16) are too brief to trust.

So this is **weak/partial replication** (3/15 usable, 7 drift-dominated, 4
unusably short): it establishes the rhythm's *existence* and *frequency
variability*, not its prevalence.

Because the frequency **differs between recordings** rather than being fixed, this
argues **against a single fixed-frequency optical/illumination artifact** (which
would repeat at one frequency) and is **more consistent with an intermittent,
state-dependent biological rhythm**. With only 3 recordings showing a fast peak,
the spread of frequencies is itself thinly sampled, so this constraint is
suggestive rather than strong. It does *not*, however, exclude
**per-recording differences in motion, illumination, or focus**, nor a **non-neural
physiological rhythm (heartbeat, perfusion, vascular pulsation)** — all of which
could also vary in frequency between preps and produce different *apparent*
frequencies. One weak additional constraint: the amplitude map (§9) localizes the
rhythm **on the cord/ganglia** rather than across the whole field, which argues
somewhat against a spatially diffuse perfusion or illumination source — though it
does not exclude vasculature running along the cord. A further mundane possibility:
the three recordings differ in length, so their frequency resolution and peak-picking
differ, which can shift the apparent peak independent of any change in the true
rhythm. But differing record lengths perturb the apparent peak only at the ~1/T
scale (~0.004 Hz for a 240 s record), which is small relative to the observed
0.36–0.55 Hz spread, so resolution is a minor contributor rather than a full
alternative explanation. So "more consistent with a biological rhythm" should not be
read as "more consistent with a *neural* rhythm." Not proof — but a meaningful
constraint.

---

## 12. Key findings

1. The recording is **dominated by one near-synchronous shared low-frequency signal
   whose dominant rhythmic component is ~0.36 Hz** across the whole nerve cord
   (PC1 = 97.5% of the variance across the 11 ganglion-mean traces — not of the raw
   pixel data; this PC1 also contains shared slow drift, see §8). Raw correlations
   (~0.96 everywhere) confirm this global dominance but are uninformative about
   specific inter-ganglion coupling (they reflect only the shared signal).
2. **No detectable anterior–posterior propagation** of that rhythm: the ≤~70 ms
   end-to-end phase spread is below the ~350 ms sampling floor and is an **upper
   bound consistent with synchrony, not a measured lag**.
3. After removing the global signal, a **small but reproducible** structure
   remains: **adjacent ganglia co-vary, anterior vs posterior anti-correlate**
   (cross-ROI reproducibility r = 0.82) — pending a drift/regression-null control
   (see Caveats); it may be partly induced by common-mode regression and slow
   drift. This is a slow amplitude modulation, not a phase gradient.
4. Across 15 recordings the fast rhythm is **intermittent and variable in
   frequency** (0.36–0.55 Hz in 3 recordings, absent in the rest), which argues
   against a single fixed-frequency optical artifact but does not establish a
   biological — let alone neural — origin (3/15 recordings; identity unestablished —
   see Caveats).

## 13. Caveats

- **Widefield, not 2-photon:** out-of-focus background and neuropil mixing.
- **2.88 Hz sampling:** lags are quantized to the frame interval (~0.35 s), so
  sub-frame leads can't be resolved and a very fast wave can't be excluded.
- **Common-mode regression** imposes a negative-sum constraint: removing the mean
  of 11 traces forces the residuals to sum to ~zero, so *some* anti-correlation is
  expected **by construction**, not merely possible. The §7 anti-phase result is
  trusted only as far as it reproduces across ROI sizes — which validates pattern
  stability but cannot separate induced from genuine anti-phase.
- **GCaMP6m kinetics** (~hundreds-of-ms rise/decay) temporally low-pass the signal,
  so the measured peak frequency (§8) reflects indicator-filtered, not raw, firing,
  and the "no fast wave" limit (§10) is likewise blurred by indicator response time.
  That said, at 0.36 Hz (2.76 s period) the carrier is several-fold slower than the
  indicator's response time, so GCaMP6m can track *this* rhythm with only modest
  attenuation; the low-pass concern bites hardest for any faster structure (e.g. the
  unresolved sub-frame leads of §6/§10), not for the 0.36 Hz peak itself.
- **Photobleaching / slow drift** (visible as the low-frequency rise in §8 and the
  slow-dominated recordings in §11) contaminates low-frequency components, and so
  could shape the slow ~3% residual modulation that §7 treats as the real finding.
- **Residual-motion control not yet run:** the 0.362 Hz brightness signal was never
  correlated against the frame-by-frame registration offsets — the single most direct
  test that the rhythm is not sub-pixel motion (see Pipeline overview).
- **Temporal autocorrelation:** the slow residual and the rhythm are strongly
  autocorrelated, so effective d.o.f. ≪ 692 frames; every reported r-value is
  descriptive, not powered.
- **No formal significance testing** was performed anywhere; "reproducible" rests on
  the single cross-ROI **r = 0.82** comparison, not on a hypothesis test or
  across-animal replication.
- **Single prep / single recording.** The headline §7 residual structure rests on
  one recording (−17) and its ROI annotation alone. The 14 other already-registered
  LeechNo2 recordings have not yet been annotated or residual-analysed, so even
  *within*-animal replication of the residual pattern is currently absent — not just
  across-animal. Closing that gap (annotating + residual-analysing the registered
  batch) is the next step.
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
