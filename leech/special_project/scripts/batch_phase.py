"""Run the global-rhythm phase analysis on every completed suite2p run and
assemble a montage, to test whether the dominant oscillation replicates.

  python scripts/batch_phase.py [glob]   # default: LeechNo2 elavgcamp6m runs
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.calcium import phase_analysis, signals  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "calcium" / "runs"


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "helobdella_LeechNo2_elavgcamp6m-*"
    runs = sorted(d for d in RUNS.glob(pat)
                  if (d / "suite2p" / "plane0" / "ops.npy").exists())
    print(f"{len(runs)} runs match {pat!r}")

    rows = []
    for d in runs:
        try:
            r = signals.load_run(d)
            gm = r.movie.reshape(r.movie.shape[0], -1).mean(axis=1)
            fpk, _ = phase_analysis.dominant_frequency(gm, r.fs)
            amp, phase = phase_analysis.pixel_phase_amplitude(r.movie, fpk, r.fs)
            # phase spread among the strongly-oscillating pixels (synchrony)
            strong = amp > np.percentile(amp, 99)
            z = np.exp(1j * phase[strong]).mean()
            spread = float(np.sqrt(-2 * np.log(np.abs(z))))  # circular SD (rad)
            rows.append((d.name, r.fs, r.movie.shape[0], fpk, spread, amp, phase, r.mean_img))
            print(f"  {d.name:42} fs {r.fs:.2f}  {r.movie.shape[0]:4}fr  "
                  f"peak {fpk:.3f}Hz  phase-spread {spread:.2f}rad")
        except Exception as e:
            print(f"  {d.name}: SKIP ({e})")

    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6), squeeze=False)
    for j, (name, fs, T, fpk, spread, amp, phase, mean) in enumerate(rows):
        a = np.clip(amp / (np.percentile(amp, 99.5) + 1e-9), 0, 1)
        axes[0][j].imshow(amp, cmap="inferno", vmax=np.percentile(amp, 99.5))
        axes[0][j].set_title(f"{name.split('-')[-1]}\n{fpk:.3f}Hz", fontsize=9)
        axes[0][j].axis("off")
        axes[1][j].imshow(mean, cmap="gray")
        axes[1][j].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, alpha=a)
        axes[1][j].set_title(f"spread {spread:.2f}rad", fontsize=9)
        axes[1][j].axis("off")
    axes[0][0].set_ylabel("amplitude")
    fig.suptitle("global rhythm across recordings — top: amplitude, bottom: phase")
    fig.tight_layout()
    out = RUNS / "_batch_phase_montage.png"
    fig.savefig(out, dpi=110)
    print("montage ->", out)

    # frequency summary table
    csv = RUNS / "_batch_phase_summary.csv"
    with open(csv, "w") as f:
        f.write("run,fs_hz,frames,peak_hz,period_s,phase_spread_rad\n")
        for name, fs, T, fpk, spread, *_ in rows:
            f.write(f"{name},{fs:.4f},{T},{fpk:.4f},{1/fpk:.3f},{spread:.4f}\n")
    print("summary ->", csv)


if __name__ == "__main__":
    main()
