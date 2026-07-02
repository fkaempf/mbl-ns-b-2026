"""Item 1: characterise the FC2 bump before vs after the first aversive-laser hit.

For each fly the session splits at its first shock into a naive window ``[t0, first_shock)``
and an equal-duration experienced window right after. Per window we take the mean
16-column profile and fit ``A*cos(theta-phi)+C`` to it (item 1's sinusoid-with-a-shelf),
and we summarise the per-frame fit magnitude ``A``, baseline shelf ``C``, bump sharpness
``r2``, the heading-minus-phase offset spread, and the bump mobility ``|dphi/dt|``.

Two figures in plots/imaging/analysis:
  * fc2_first_shock_profiles - per-fly mean column profile pre vs post, fits overlaid;
  * fc2_first_shock_params   - the metrics as paired pre->post points across the 4 flies
    (n = flies is the honest unit), with a paired Wilcoxon in each panel title.

Run:  python fc2_first_shock.py
"""

import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

import fc2_analysis as fa
from imaging_unify import norm_matrix, NCOL
from cxstyle import PINK, WHITE
from utils import save_fig

PRE_COLOR, POST_COLOR = WHITE, PINK
ANG = np.arange(NCOL) * 2 * np.pi / NCOL + np.pi / NCOL      # column angles (fit convention)


def windows(df):
    """Naive vs experienced index masks, split at the first shock, matched duration."""
    t = df["time"].to_numpy()
    fs = fa.first_shock_time(df)
    if fs is None:
        return None
    dur = fs - t[0]                                          # length of the naive period
    pre = (t >= t[0]) & (t < fs)
    post = (t >= fs) & (t < fs + dur)
    return pre, post


def profile_and_fit(Fn, mask):
    """Mean 16-column profile over ``mask`` and its single sinusoid fit (A, phi, C)."""
    prof = np.nanmean(Fn[mask], axis=0)
    A, phi, C, r2 = fa.fit_bump(prof[None, :])
    return prof, float(A[0]), float(phi[0]), float(C[0])


def fitted_curve(A, phi, C, th):
    return A * np.cos(th - np.radians(phi)) + C


def collect(fly):
    """Per-fly pre/post metrics (median over frames) and the mean profiles + fits."""
    df = fa.add_fit(fa.load(fly))
    Fn = norm_matrix(df)
    w = windows(df)
    if w is None:
        return None
    pre, post = w
    mov = df["speed"].to_numpy() >= fa.MOVE_CUTOFF
    out = dict(fly=fly, dur=float(fa.first_shock_time(df) - df["time"].iloc[0]))
    for name, m in (("pre", pre), ("post", post)):
        prof, A, phi, C = profile_and_fit(Fn, m)
        out[name] = dict(
            prof=prof, A=A, phi=phi, C=C,
            med_A=float(np.nanmedian(df["fit_A"].to_numpy()[m])),
            med_C=float(np.nanmedian(df["fit_C"].to_numpy()[m])),
            med_r2=float(np.nanmedian(df["fit_r2"].to_numpy()[m])),
            off_spread=float(st.circstd(df["offset"].to_numpy()[m & mov],
                                        high=180, low=-180, nan_policy="omit")),
            med_av=float(np.nanmedian(fa.bump_angular_velocity(
                df["fit_phase"].to_numpy(), df["time"].to_numpy())[m])),
            med_speed=float(np.nanmedian(df["speed"].to_numpy()[m])),
        )
    return out


def fig_profiles(rows):
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 3.4),
                             squeeze=False, constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 200)
    cols = np.arange(1, NCOL + 1)
    for ax, r in zip(axes[0], rows):
        for name, color in (("pre", PRE_COLOR), ("post", POST_COLOR)):
            d = r[name]
            ax.plot(cols, d["prof"], "o", color=color, ms=4, alpha=0.9)
            ax.plot(1 + th / (2 * np.pi) * NCOL, fitted_curve(d["A"], d["phi"], d["C"], th),
                    "--", color=color, lw=1.4,
                    label=f"{name}: A={d['A']:.2f} C={d['C']:.2f}")
        ax.set_title(f"{r['fly']}  (±{r['dur']:.0f} s)", fontsize=9)
        ax.set_xlabel("FSB column"); ax.legend(fontsize=6, loc="upper right")
    axes[0, 0].set_ylabel("maxmin activity")
    save_fig(fig, "fc2_first_shock_profiles.png",
             title="FC2 mean column profile: naive (white) vs after first shock (pink)",
             subdir="imaging/analysis")
    plt.close(fig)


def fig_params(rows):
    metrics = [("med_A", "bump magnitude A"), ("med_C", "baseline shelf C"),
               ("off_spread", "heading-phase spread (deg)"), ("med_av", "|dphi/dt| (deg/s)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.6),
                             squeeze=False, constrained_layout=True)
    for ax, (key, label) in zip(axes[0], metrics):
        pre = np.array([r["pre"][key] for r in rows])
        post = np.array([r["post"][key] for r in rows])
        for i, r in enumerate(rows):
            ax.plot([0, 1], [pre[i], post[i]], "-o", color=PINK, alpha=0.8, ms=5)
            ax.annotate(r["fly"].replace("_00", "_"), (1, post[i]), fontsize=6,
                        color=WHITE, xytext=(4, 0), textcoords="offset points", va="center")
        try:
            p = st.wilcoxon(pre, post).pvalue
        except ValueError:
            p = np.nan
        up = int(np.sum(post > pre))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["naive", "post"])
        ax.set_xlim(-0.3, 1.5); ax.set_title(f"{label}\n{up}/{len(rows)} up, p={p:.2f}", fontsize=8)
    save_fig(fig, "fc2_first_shock_params.png",
             title="FC2 fit metrics pre vs post first shock (each line = one fly, n = 4)",
             subdir="imaging/analysis")
    plt.close(fig)


def main():
    rows = [r for r in (collect(f) for f in fa.ANALYSIS_FLIES) if r is not None]
    fig_profiles(rows)
    fig_params(rows)
    for r in rows:
        print(f"{r['fly']:10s} A {r['pre']['med_A']:.3f}->{r['post']['med_A']:.3f}  "
              f"C {r['pre']['med_C']:.3f}->{r['post']['med_C']:.3f}  "
              f"offspread {r['pre']['off_spread']:.0f}->{r['post']['off_spread']:.0f}  "
              f"|dphi/dt| {r['pre']['med_av']:.0f}->{r['post']['med_av']:.0f}")
    print("saved plots/imaging/analysis/fc2_first_shock_{profiles,params}.png")


if __name__ == "__main__":
    main()
