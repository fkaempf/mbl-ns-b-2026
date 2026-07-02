"""Item 4: how the FC2 bump magnitude and phase change around task events.

Event-triggered averages, the neural analogue of shock_response.py. Rows are the signals
(bump magnitude A; bump angular velocity |dphi/dt|, which spikes if the bump jumps or
resets; walking speed for behavioural context), columns are the events (aversive-laser
onset, wall on, wall off). The bold white curve is the mean +/- SEM pooled over all events;
thin coloured curves are the per-fly means, so one fly cannot drive the pooled trace.

One figure in plots/imaging/analysis: fc2_events.

Run:  python fc2_events.py
"""

import numpy as np
import matplotlib.pyplot as plt

import fc2_analysis as fa
from cxstyle import PINK, YELLOW, WHITE
from utils import save_fig

FLY_COLORS = [PINK, YELLOW, "#37e0d0", WHITE]
SIGNALS = [("fit_A", "bump magnitude A"),
           ("av", "bump |dphi/dt| (deg/s)"),
           ("speed", "walking speed (mm/s)")]
EVENTS = [("laser onset", fa.shock_onset_times, 5.0, 5.0),
          ("wall on", fa.wall_on_times, 5.0, 15.0),
          ("wall off", fa.wall_off_times, 5.0, 15.0)]


def per_fly_signals(df):
    return {"fit_A": df["fit_A"].to_numpy(),
            "av": fa.bump_angular_velocity(df["fit_phase"].to_numpy(), df["time"].to_numpy()),
            "speed": df["speed"].to_numpy()}


def main():
    data = []
    for f in fa.ANALYSIS_FLIES:
        df = fa.add_fit(fa.load(f))
        data.append((f, df, per_fly_signals(df)))

    nrow, ncol = len(SIGNALS), len(EVENTS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow),
                             squeeze=False, constrained_layout=True, sharex="col")
    for c, (ename, efn, pre, post) in enumerate(EVENTS):
        n_events_total = 0
        for r, (skey, slabel) in enumerate(SIGNALS):
            ax = axes[r][c]
            pool = []
            for i, (fly, df, sig) in enumerate(data):
                t = df["time"].to_numpy()
                ev = efn(df)
                grid, M = fa.event_aligned(t, sig[skey], ev, pre, post)
                if M.shape[0]:
                    ax.plot(grid, np.nanmean(M, axis=0), color=FLY_COLORS[i], lw=0.9, alpha=0.5)
                    pool.append(M)
            if pool:
                allM = np.vstack(pool)
                mean, sem = fa.mean_sem(allM)
                ax.plot(grid, mean, color=WHITE, lw=2.2)
                ax.fill_between(grid, mean - sem, mean + sem, color=WHITE, alpha=0.2)
                n_events_total = allM.shape[0]
            ax.axvline(0, color=WHITE, lw=0.8, ls="--", alpha=0.6)
            if skey == "av":
                ax.set_ylim(0, 450)                 # focus on the pooled bump-motion signal
            if r == 0:
                ax.set_title(f"{ename}  (n={n_events_total} events, {len(data)} flies)", fontsize=9)
            if c == 0:
                ax.set_ylabel(slabel, fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel(f"time from {ename} (s)", fontsize=9)
    fig.suptitle("FC2 bump around task events (white = pooled mean +/- SEM; thin = per fly)",
                 fontsize=12)
    save_fig(fig, "fc2_events.png", subdir="imaging/analysis")
    plt.close(fig)

    # quick numeric read: pre vs post magnitude and bump motion at the laser
    for fly, df, sig in data:
        ev = fa.shock_onset_times(df)
        g, MA = fa.event_aligned(df["time"].to_numpy(), sig["fit_A"], ev, 5, 5)
        _, MV = fa.event_aligned(df["time"].to_numpy(), sig["av"], ev, 5, 5)
        pre = (g < -0.5); post = (g > 0) & (g < 1.5)
        dA = np.nanmean(MA[:, post]) - np.nanmean(MA[:, pre])
        vpk = np.nanmax(np.nanmean(MV, axis=0))
        print(f"{fly:10s} laser: dA(post-pre)={dA:+.3f}  peak |dphi/dt|={vpk:.0f} deg/s  n={MA.shape[0]}")
    print("saved plots/imaging/analysis/fc2_events.png")


if __name__ == "__main__":
    main()
