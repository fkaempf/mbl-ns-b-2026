# Calcium imaging analysis: helobdella_LeechNo2_elavgcamp6m-17

Pan-neuronal GCaMP6m (elav promoter) in a juvenile *Helobdella* leech ganglion.
Single channel, single plane, 692 frames at 700x700 px, fs = 2.882 Hz
(0.347 s/frame), ~240 s total. Annotated "10x juvenile after dopamin pharyngals",
movement flag "no".

**Framing.** This is a low-magnification (10x), preliminary transgenic line. The
signal is interpretable at the GANGLION / regional level, not as individual neurons.
All findings below are deliberately high level; no single-neuron claims are made.

Pipeline (5 stages, in `scripts/`): s1 QC + motion, s2 suite2p segmentation,
s2b correlation-blob segmentation, s3 temporal rhythm, s4 spatial dynamics.

## Headline result

**The ganglion shows one strong, clock-like, ganglion-wide calcium oscillation at
0.37 Hz (2.7 s period) that is sustained undamped for the full ~240 s and is
spatially synchronous across essentially the whole imaged ganglion.**

## Motion correction: NOT needed (verified, not assumed)

- Rigid drift = **0.00 px** across all 692 frames (phase correlation, cross-checked
  at full resolution, upsample factor 10). The prep is stationary; the "movement: no"
  annotation is correct.
- Non-rigid jitter = 0.00 px (3x3 block-wise check). No local warping.
- Photobleaching = **0.5 to 0.8%** over 240 s (negligible; dF/F absorbs it).
- suite2p registration was run anyway as insurance and confirmed xoff/yoff flat at 0.
- No saturation (global max 526 vs an 11-bit ceiling of 2047), no dropped frames.

Conclusion: segmentation and analysis run directly on the raw movie with no
interpolation artifacts. Frame-to-frame correlation (~0.55) reflects real calcium
dynamics, not motion, and must not be mistaken for a motion-correction requirement.

## Segmentation: two methods, suite2p alone was insufficient

| Method | n regions/patches | Note |
|--------|------------------:|------|
| suite2p (defaults, tau 1.0, diam 15, thr 0.8) | 24 | Severe UNDER-segmentation of this dense confocal band |
| correlation-blob + watershed (primary) | 180 | Captures the visible active patches across the ganglion |

The correlation/std projections show dozens to ~100+ active somata along the
ganglionic band; suite2p accepted only 24, so it was kept only as a high-confidence
cross-check. The **primary** signal set is the 180 correlation-blob "active patches"
(`derived/traces_*.npy`), with suite2p preserved as `derived/traces_*_s2p.npy`.
At 10x these patches are treated as regional signals, not single cells. Both methods,
and the whole-ganglion mask mean, independently recover the same 0.37 Hz rhythm, so
the result does not depend on the segmentation choice.

## Temporal structure (Stage 3)

- **Dominant frequency 0.372 Hz, period 2.69 s.** Spectral prominence ~230x the
  broadband median. A clear harmonic at ~0.73 Hz confirms a real, non-sinusoidal
  periodic waveform.
- **Extremely regular:** inter-peak interval CV = 0.077 (88 cycles, median 2.78 s);
  autocorrelation first side-peak height 0.84 at 2.78 s.
- **Sustained and frequency-stable:** the spectrogram shows a single fixed band at
  ~0.37 Hz for the whole recording (per-window frequency std 0.010 Hz); no drift.
- **Well below Nyquist** (0.37 Hz vs 1.44 Hz, ~26% of Nyquist): the fundamental is
  not aliased. (Anything above ~1.4 Hz would be; ~8 samples/cycle here.)
- Modest amplitude (ganglion mean dF/F peak ~0.08), expected for 10x bulk imaging.
- A slow downward baseline drift exists, separate from and far slower than the rhythm.

## Spatial structure (Stage 4)

- **Rhythm is ganglion-wide:** 93% of tissue pixels carry significant power at
  0.37 Hz; it is not confined to a sub-region.
- **In-phase, no traveling wave:** phase regression along the ganglion long axis is
  flat (slope ~0, total phase change ~1 deg across the band); regional cross-
  correlation lags are sub-frame (~70 ms). The ganglion oscillates synchronously.
- **One coherent rhythm with variable SNR:** 6-region correlation averages 0.21
  (range -0.50 to 0.85), best read as a single ganglion-wide oscillator with
  spatially varying signal quality rather than distinct functional territories.
- Peak-vs-trough raw frames confirm the modulation is real fluorescence brightening
  of the ganglion body (motion already excluded).

## Interpretation (cautious)

A single, highly regular, ganglion-wide 0.37 Hz oscillation that is synchronous
across the whole ganglion is most consistent with a shared network drive / coordinated
rhythmic activity rather than independent local pacemakers. The recording is annotated
"after dopamine pharyngals" (dopamine is a known trigger of feeding and rhythmic motor
programs in leech), but **there is no pre-dopamine baseline in this dataset, so no
causal or pharmacological claim can be made** about whether dopamine produced or shaped
this rhythm. The frequency (~0.37 Hz) is a candidate motor/visceral rhythm; identifying
it (feeding/pharyngeal vs heartbeat vs other) requires a labeled baseline and ideally
simultaneous behavioral or electrophysiological readout.

## Figures (`plots/`)

Stage 1 QC/motion: `s1_projections.png`, `s1_correlation_image.png`,
`s1_brightness_bleach.png`, `s1_motion.png`.
Stage 2/2b segmentation: `s2_rois_on_mean.png`, `s2_example_traces.png`,
`s2_registration_check.png`, `s2b_rois_on_corr.png`, `s2b_method_comparison.png`,
`s2b_example_traces.png`.
Stage 3 temporal: `s3_ganglion_signal.png`, `s3_spectrum.png`, `s3_spectrogram.png`,
`s3_autocorr_regularity.png`, `s3_regional_comparison.png`.
Stage 4 spatial: `s4_activity_maps.png`, `s4_pixelwise_rhythm_power.png`,
`s4_phase_map.png`, `s4_regional_correlation.png`, `s4_peak_trough_frames.png`.

Metrics JSONs in `metrics/`; derived arrays (traces, masks, correlation image,
ganglion signal, rhythm power/phase maps) in `derived/`.

## Intricate figure set (detailed visualization pass)

A second pass produced 25 detailed multi-panel figures (scripts `fig_*.py`), grouped
by theme. New quantitative nuances from this pass:

- **The rhythm is carried by a strong subset of patches.** ~93% of the 180 patches
  have their own dominant frequency within 0.05 Hz of 0.37 Hz, and ~40% follow the
  ganglion mean with r > 0.5; the remaining patches are noisy (median per-patch SNR
  ~4.1, low-photon 10x data). Pooling the rhythmic subset is what yields the very clean
  ganglion signal. Mean pairwise patch correlation across ALL 180 is low (~0.01) for
  this reason, while inter-REGION coherence at 0.37 Hz is high (~0.88).
- **Spatially synchronous, confirmed three ways:** the kymograph along the ganglion
  long axis shows vertical (not tilted) stripes; the cycle-averaged phase montage shows
  the whole ganglion brightening and dimming together; ROI phase vs long-axis position
  is flat (slope ~0.001 rad/px). No traveling wave.
- **Stable across the recording:** per-pixel oscillation amplitude maps over the 4
  recording quarters are similar; the wavelet scalogram and spectrogram show a single
  fixed 0.37 Hz band; instantaneous frequency (Hilbert) hovers at 0.37 Hz; interval CV
  0.077 with a tight Poincare cluster.

Figure groups (in `plots/`):
- **Population/temporal** `fig_pop_*`: clustered population raster, cycle-average
  waveform, Hilbert instantaneous freq/amplitude, inter-peak-interval + Poincare
  dynamics, 36-patch small-multiples.
- **Spectral** `fig_spec_*`: detailed PSD with harmonics, Morlet wavelet scalogram,
  high-res spectrogram, inter-region coherence, per-patch PSD heatmap.
- **Spatial** `fig_spatial_*`: power+phase 2x2, 4-way ROI property maps, phase-gradient
  / traveling-wave test, seed-correlation maps, correlation-vs-distance.
- **Cycle-resolved spatiotemporal** `fig_cycle_*`: 8-bin phase montage, long-axis
  kymograph, 5-cycle peak/trough/difference gallery, per-quarter amplitude maps, and a
  looped cycle-averaged movie `fig_cycle_movie.mp4`.
- **Dashboards/QC** `fig_dash_*`: one-page master dashboard, segmentation comparison
  (suite2p vs blob, with matching), 40-patch ROI gallery, QC panel, trace-quality panel.

## Recommendations for the dataset

1. **Record a pre-dopamine baseline** of the same prep so the "after dopamine" rhythm
   can be compared (frequency/amplitude change). This is the single biggest gap.
2. **Use the +palmmcherry dual-channel recordings** (a static structural channel) as a
   motion reference for the recordings flagged "leech moves" / "david moves" in
   Book1.xlsx; those are the ones that genuinely need motion correction. This -17 file
   does not.
3. **Higher magnification (20x+)** if single-neuron resolution is the goal; at 10x this
   line supports ganglion/regional analysis only.
4. The faster sub-rhythm question (anything >1.4 Hz) needs a higher frame rate; current
   sampling (~8 samples/cycle) resolves the 0.37 Hz fundamental well but little above it.
