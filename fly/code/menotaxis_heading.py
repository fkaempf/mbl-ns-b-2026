"""Menotaxis heading before vs after confinement (Part C).

In ``rig1_experiment_06`` the fly does menotaxis (0-900 s), is confined in a
virtual enclosure (900-1800 s), then does menotaxis again (1800 s-end). This
script asks whether the *preferred heading* is the same before and after the
confinement.

The fly turns more or less continuously in these windows (very few sustained
straight bouts), so "fixation" is weak. We therefore summarise the *preferred
walking direction*: the heading distribution weighted by speed (i.e. by distance
travelled in each direction), with a duration/speed-weighted circular mean and
resultant length R, and compare before vs after the confinement. The count of
sustained straight bouts is reported too, as a measure of how menotaxis-like
(fixating) each window is.

Run from this directory:  python menotaxis_heading.py
"""

import numpy as np

import scalloping as sc
from utils import experiment_id, experiment_label, load_trajectory, save_panel

EXPERIMENT = "20260624/rig1_experiment_06"

WINDOWS = {"before": (0, 900), "after": (1800, np.inf)}

# ---------------------------------------------------------------------------
data, x, y, time_s = load_trajectory(EXPERIMENT)
label, eid = experiment_label(EXPERIMENT), experiment_id(EXPERIMENT)
fs = 1.0 / np.median(np.diff(time_s))
heading_all = data.integrated_heading_lab.to_numpy()


def wrap(a):
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


MIN_SPEED = 5.0  # mm/s — only count heading while the fly is actually walking


def analyse_window(name):
    lo, hi = WINDOWS[name]
    m = (time_s >= lo) & (time_s < hi)
    t = time_s[m] - time_s[m][0]
    h = wrap(heading_all[m])
    _, _, speed, _ = sc.velocity(x[m], y[m], fs)
    omega = sc.angular_velocity(heading_all[m], fs)

    moving = speed > MIN_SPEED
    # preferred walking direction: heading weighted by speed (distance per direction)
    mu, R = sc.circ_mean_R(h[moving], weights=speed[moving])

    bouts = sc.straight_bouts(omega, speed, fs)
    fix_frac = sum(b - a for a, b in bouts) / len(t) if len(bouts) else 0.0
    return dict(name=name, t=t, h=h, speed=speed, omega=omega, moving=moving,
                bouts=bouts, mu=mu, R=R, fix_frac=fix_frac)


def heading_for_plot(h):
    """Insert NaNs at wrap discontinuities so the line isn't bridged."""
    hp = h.copy()
    hp[np.abs(np.diff(h, prepend=h[0])) > np.pi] = np.nan
    return hp


def main():
    res = {n: analyse_window(n) for n in WINDOWS}

    print(f"\n=== Part C: menotaxis heading before vs after ({eid}) ===")
    for n in WINDOWS:
        d = res[n]
        print(f"{n:7s}: preferred walking direction {np.degrees(d['mu']):.0f} deg "
              f"(R={d['R']:.2f})  |  {len(d['bouts'])} sustained straight bouts "
              f"({100*d['fix_frac']:.0f}% of window)")
    shift = wrap(res["after"]["mu"] - res["before"]["mu"])
    print(f"before->after preferred-direction shift: {np.degrees(shift):.0f} deg")

    sub = "menotaxis"
    ttl = f"{label} · menotaxis heading before vs after confinement"
    colors = {"before": "#4477aa", "after": "#228833"}

    def timeseries(d):
        def draw(ax):
            ax.plot(d["t"], np.degrees(heading_for_plot(d["h"])), lw=0.4,
                    color=colors[d["name"]])
            ax.axhline(np.degrees(d["mu"]), color="#cc3311", lw=1.5,
                       label=f"pref {np.degrees(d['mu']):.0f}°")
            ax.set_ylim(-180, 180); ax.set_yticks([-180, -90, 0, 90, 180])
            ax.set_xlabel("time in window (s)"); ax.set_ylabel("heading (deg)")
            ax.set_title(f"{d['name']}: heading over time")
            ax.legend(fontsize=8, loc="upper right")
        return draw

    def polar(d):
        def draw(ax):
            ax.hist(d["h"][d["moving"]], bins=np.linspace(-np.pi, np.pi, 37),
                    weights=d["speed"][d["moving"]], color=colors[d["name"]])
            ax.plot([d["mu"], d["mu"]], [0, ax.get_ylim()[1]], color="#cc3311", lw=2)
            ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
            ax.set_title(f"{d['name']}: heading (speed-weighted, radial)\n"
                         f"pref {np.degrees(d['mu']):.0f}° R={d['R']:.2f}", fontsize=9)
        return draw

    def polar_overlay(ax):
        for n in WINDOWS:
            d = res[n]
            ax.hist(d["h"][d["moving"]], bins=np.linspace(-np.pi, np.pi, 37),
                    weights=d["speed"][d["moving"]], histtype="step", lw=2,
                    color=colors[n], label=f"{n} (R={d['R']:.2f})")
            ax.plot([d["mu"], d["mu"]], [0, ax.get_ylim()[1]], color=colors[n], lw=2)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(f"heading before vs after (shift {np.degrees(shift):.0f}°)", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.1, -0.1))

    for n in WINDOWS:
        d = res[n]
        save_panel(sub, f"{n}_01_heading_timeseries.png", timeseries(d),
                   figsize=(9, 3.5), title=ttl)
        save_panel(sub, f"{n}_02_heading_polar.png", polar(d),
                   figsize=(5.5, 5.5), title=ttl, subplot_kw={"projection": "polar"})
    save_panel(sub, "before_after_heading_polar.png", polar_overlay,
               figsize=(6, 6), title=ttl, subplot_kw={"projection": "polar"})
    return res


if __name__ == "__main__":
    main()
