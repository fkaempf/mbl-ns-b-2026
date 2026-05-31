"# mbl-ns-b-2026"

## epsp_auc_timeline.m

Opens multiple ABF files, orders them by their true recording time, and for
every sweep computes the area under the curve (AUC) of the compound EPSP on the
postsynaptic cell. All EPSPs are placed on one continuous experiment timeline
and plotted as **AUC vs time**, to track synaptic strength across a long-term
recording.

### Usage
1. In MATLAB, `cd` to this folder (so `loadephys` etc. are on the path) and run:
   ```
   epsp_auc_timeline
   ```
2. In the file picker, multi-select the ABF files (e.g. the four `long term *.abf`).
3. Outputs are written to the data folder's `figures/` subfolder (or the data
   folder): a CSV table, the main AUC-vs-time figure (`.fig` + `.png`), and a QC
   figure.

### How it works
- **Order** — recording start time is read straight from each ABF2 header
  (bytes 16/20), so files sort by true acquisition time.
- **Per sweep** — the presynaptic spike train is auto-detected on `Red_Vm`
  (col 3); the first spike marks the EPSP onset. The compound EPSP on `Grn_Vm`
  (col 1) is baseline-subtracted and integrated (`trapz`) over a fixed window
  after onset. Sweeps with no detectable spike fall back to the median onset.
- **Timeline** — each EPSP's time = (file start − first file start) +
  sweep×`sweepIntervalSec` + onset.

### Tuning (parameters at the top of the script)
- `epspCh` / `spikeCh` — channel indices (default 1 = Grn_Vm, 3 = Red_Vm).
- `winDur` — EPSP integration window after onset (s).
- `baselineWin` — pre-train baseline window (s).
- `sweepIntervalSec` — sweep start-to-start interval (6 s for this protocol).
- `spikeDerivThresh` / `refractorySec` — presynaptic spike detection.
- `baselineSweepsForNorm` — set >0 to normalize all AUCs to the first N sweeps.

Use the **QC figure** (one representative sweep per file, with the detected
onset and shaded integration window) to verify detection and adjust the above.
