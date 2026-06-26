"""Characterise scalloping while a fly is confined in a circular virtual enclosure.

Experiment ``rig1_experiment_06`` runs menotaxis (0-900 s) -> confinement in a
100 mm-diameter virtual enclosure (900-1800 s) -> menotaxis (1800 s-end). During
confinement the walked path fills a disc whose rim is *scalloped*: repeated
out-and-back excursions to the wall.

Part A characterises the scalloping in the confinement window; Part B compares
rhythm/turning metrics across all three windows as a within-fly control.

Run from this directory:  python confinement_scalloping.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks

import scalloping as sc
from utils import experiment_id, experiment_label, load_trajectory, save_panel

EXPERIMENT = "260624/2026_06_24/rig1_experiment_06"
BOUNCE_HALFWIN_S = 1.2            # window each side of contact for the aligned average

WINDOWS = {                       # name -> (t_start_s, t_end_s)
    "menotaxis_before": (0, 900),
    "confinement": (900, 1800),
    "menotaxis_after": (1800, np.inf),
}

# ---------------------------------------------------------------------------
data, x, y, time_s = load_trajectory(EXPERIMENT)
label, eid = experiment_label(EXPERIMENT), experiment_id(EXPERIMENT)
fs = 1.0 / np.median(np.diff(time_s))


def window_mask(name):
    lo, hi = WINDOWS[name]
    return (time_s >= lo) & (time_s < hi)


# =========================================================================
# Part A: scalloping in the confinement window
# =========================================================================

def analyse_confinement():
    m = window_mask("confinement")
    xc, yc, tc = x[m], y[m], time_s[m] - time_s[m][0]

    cx, cy, radius_fit = sc.arena_geometry(xc, yc)
    r = sc.radial(xc, yc, cx, cy)
    _, _, speed, direction = sc.velocity(xc, yc, fs)
    heading = data.integrated_heading_lab.to_numpy()[m]
    omega = sc.angular_velocity(heading, fs)

    contacts = sc.detect_contacts(r, radius_fit, fs)
    g = sc.scallop_metrics(xc, yc, r, cx, cy, radius_fit, contacts,
                           direction, speed, omega, fs)

    iei = np.diff(tc[contacts])
    f_r, p_r = sc.psd(r, fs)
    f_w, p_w = sc.psd(omega, fs)
    fpk, _ = sc.spectral_peak(f_r, p_r)
    rad_c, spd_c = sc.speed_vs_radius(r, speed, radius_fit)
    circ = sc.net_circling(g["azimuth"])
    frac_left = np.mean(g["turn_sign"] > 0)

    # ---- report ----
    print(f"\n=== Part A: confinement scalloping ({eid}) ===")
    print(f"fitted enclosure: centre=({cx:.1f},{cy:.1f})  radius={radius_fit:.1f} mm "
          f"(nominal {sc.ENCLOSURE_RADIUS_MM:.0f})")
    print(f"contacts: {len(contacts)}  ->  {len(contacts)/(tc[-1]/60):.1f}/min")
    print(f"inter-contact interval: median {np.median(iei):.2f}s  "
          f"IQR {np.percentile(iei,25):.2f}-{np.percentile(iei,75):.2f}s")
    print(f"retreat depth: median {np.nanmedian(g['retreat_depth']):.1f} mm")
    print(f"turn angle |.|: median {np.degrees(np.nanmedian(np.abs(g['turn_angle']))):.0f} deg")
    print(f"peak angular velocity: median {np.degrees(np.nanmedian(np.abs(g['peak_omega']))):.0f} deg/s")
    print(f"incidence/reflection: {np.degrees(np.nanmedian(g['incidence'])):.0f} / "
          f"{np.degrees(np.nanmedian(g['reflection'])):.0f} deg")
    print(f"handedness: {100*frac_left:.0f}% left / {100*(1-frac_left):.0f}% right")
    print(f"net circling of contact point: {circ:.1f} turns")
    print(f"spectral peak of r(t): {fpk:.2f} Hz  (period {1/fpk:.2f}s)")

    # ---- bounce-aligned average (the stereotyped wall-bounce) ----
    hw = BOUNCE_HALFWIN_S
    tau, U, V = sc.bounce_aligned(xc, yc, cx, cy, contacts, g["turn_sign"], fs, half_win_s=hw)
    mU, sU = sc.mean_sem(U)
    mV, sV = sc.mean_sem(V)
    tau_r, Rstack = sc.bounce_aligned_signal(r, contacts, fs, half_win_s=hw)
    mR, sR = sc.mean_sem(Rstack)
    nb = V.shape[0]

    # ---- full-cycle alternative: previous -> this -> next contact ----
    ph, Rcyc = sc.bounce_cycle_aligned(r, contacts)
    mRc, sRc = sc.mean_sem(Rcyc)
    ph_xy, Ucyc, Vcyc = sc.bounce_cycle_aligned_xy(xc, yc, cx, cy, contacts, g["turn_sign"])
    mUc, sUc = sc.mean_sem(Ucyc)
    mVc, sVc = sc.mean_sem(Vcyc)
    n_half = (len(ph_xy) - 1) // 2
    ncyc = Ucyc.shape[0]

    # ---- standalone panels, each saved on its own ----
    sub = f"{eid}_scalloping"
    ttl = f"{label} · confinement scalloping (900-1800s)"

    def p_trajectory(ax):
        ax.scatter(xc, yc, c=tc, cmap="viridis", s=1, linewidths=0)
        ax.add_patch(plt.Circle((cx, cy), radius_fit, fill=False, color="k", lw=1.2))
        ax.scatter(xc[contacts], yc[contacts], c="red", s=12, zorder=3, label="contacts")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("trajectory (colour=time) + contacts")
        ax.legend(loc="upper right", fontsize=7)

    def p_radial(ax):
        ax.plot(tc, r, lw=0.4)
        ax.axhline(radius_fit, color="k", ls="--", lw=0.8, label="wall")
        ax.scatter(tc[contacts], r[contacts], c="red", s=8, zorder=3)
        ax.set_xlabel("time in window (s)"); ax.set_ylabel("radial r (mm)")
        ax.set_title("radial distance with contacts"); ax.legend(fontsize=7)

    def p_timing(ax):
        ax.hist(iei, bins=40, color="#4477aa")
        ax.set_xlabel("inter-contact interval (s)"); ax.set_ylabel("count")
        ax.set_title(f"contact timing (median {np.median(iei):.1f}s)")

    def p_retreat(ax):
        ax.hist(g["retreat_depth"], bins=30, color="#228833")
        ax.set_xlabel("retreat depth (mm)"); ax.set_ylabel("count")
        ax.set_title(f"retreat depth (median {np.nanmedian(g['retreat_depth']):.0f} mm)")

    def p_turn(ax):
        ax.hist(np.degrees(g["turn_angle"]), bins=40, color="#aa6644")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("turn angle at wall (deg, +=left)"); ax.set_ylabel("count")
        ax.set_title(f"turning / handedness ({100*frac_left:.0f}% left)")

    def p_incidence(ax):
        ax.hist(np.degrees(g["incidence"]), bins=24, alpha=0.6, label="incidence")
        ax.hist(np.degrees(g["reflection"]), bins=24, alpha=0.6, label="reflection")
        ax.set_xlabel("angle to inward wall normal (deg)"); ax.set_ylabel("count")
        ax.set_title("incidence vs reflection"); ax.legend(fontsize=7)

    def p_speed(ax):
        ax.plot(rad_c, spd_c, "-o", ms=3)
        ax.axvline(radius_fit, color="k", ls="--", lw=0.8)
        ax.set_xlabel("radial position (mm)"); ax.set_ylabel("mean speed (mm/s)")
        ax.set_title("speed vs radius")

    def p_psd(ax):
        ax.loglog(f_r, p_r, label="r(t)")
        ax.loglog(f_w, p_w, alpha=0.6, label="angular vel")
        if np.isfinite(fpk):
            ax.axvline(fpk, color="red", ls="--", lw=0.8, label=f"peak {fpk:.2f} Hz")
        ax.set_xlim(0.05, 10)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power")
        ax.set_title("spectral power (detrended)"); ax.legend(fontsize=7)

    def p_bounce_traj(ax):
        pre, post = tau <= 0, tau >= 0
        # individual bounces: approach light blue, retreat light red
        for i in range(nb):
            ax.plot(V[i][pre], U[i][pre], color="#1f77b4", lw=0.5, alpha=0.12, zorder=1)
            ax.plot(V[i][post], U[i][post], color="#d62728", lw=0.5, alpha=0.12, zorder=1)
        # SEM envelope: offset the mean path by +/- SEM perpendicular to itself
        P = np.column_stack([mV, mU])
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        N = np.column_stack([-T[:, 1], T[:, 0]])
        sig = np.sqrt((N[:, 0] * sV) ** 2 + (N[:, 1] * sU) ** 2)
        upper, lower = P + sig[:, None] * N, P - sig[:, None] * N

        def band(mask, color):
            poly = np.vstack([upper[mask], lower[mask][::-1]])
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.25, lw=0, zorder=2)

        band(pre, "#1f77b4")
        band(post, "#d62728")
        ax.plot(mV[pre], mU[pre], color="#1f77b4", lw=2.5, zorder=3, label="approach")
        ax.plot(mV[post], mU[post], color="#d62728", lw=2.5, zorder=3, label="retreat")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.scatter([0], [0], c="k", s=30, zorder=4, label="contact")
        ax.set_aspect("equal")
        ax.set_xlabel("tangential (mm)")
        ax.set_ylabel("radial offset from wall (mm)")
        ax.set_title(f"bounce-aligned mean trajectory ±SEM (±{hw:g} s, n={nb})")
        ax.legend(fontsize=7)

    def p_bounce_radial(ax):
        ax.plot(tau_r, mR, color="k", lw=2)
        ax.fill_between(tau_r, mR - sR, mR + sR, alpha=0.3, color="#cc3311",
                        label="±SEM")
        ax.axvline(0, color="0.5", ls="--", lw=0.8)
        ax.axhline(radius_fit, color="b", ls="--", lw=0.8, label="wall")
        ax.set_xlabel("time from contact (s)"); ax.set_ylabel("radial r (mm)")
        ax.set_title(f"bounce-aligned radial distance ±SEM (±{hw:g} s, n={Rstack.shape[0]})")
        ax.legend(fontsize=7)

    def p_cycle_traj(ax):
        for i in range(ncyc):                     # faint individual cycles
            ax.plot(Vcyc[i], Ucyc[i], color="0.88", lw=0.3, zorder=1)
        pts = np.column_stack([mVc, mUc]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="twilight", lw=2.5, zorder=3)
        lc.set_array(ph_xy[:-1]); ax.add_collection(lc)
        for idx, lab, col in [(0, "prev bounce", "#1f77b4"),
                              (n_half, "this bounce", "k"),
                              (2 * n_half, "next bounce", "#d62728")]:
            ax.scatter(mVc[idx], mUc[idx], color=col, s=45, zorder=4, label=lab)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_aspect("equal"); ax.autoscale_view()
        ax.set_xlabel("tangential (mm)"); ax.set_ylabel("radial offset from wall (mm)")
        ax.set_title(f"bounce-cycle mean trajectory: prev→this→next (n={ncyc})")
        cb = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("cycle phase (−1 prev · 0 this · +1 next)")
        ax.legend(fontsize=7)

    def p_cycle_radial(ax):
        ax.plot(ph, mRc, color="k", lw=2)
        ax.fill_between(ph, mRc - sRc, mRc + sRc, alpha=0.3, color="#cc3311", label="±SEM")
        ax.axhline(radius_fit, color="b", ls="--", lw=0.8, label="wall")
        for xv in (-1, 0, 1):
            ax.axvline(xv, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("cycle phase (−1 prev bounce · 0 this · +1 next)")
        ax.set_ylabel("radial r (mm)")
        ax.set_title(f"bounce-cycle radial ±SEM (n={Rcyc.shape[0]} cycles)")
        ax.legend(fontsize=7)

    panels = [
        ("01_trajectory_contacts.png", p_trajectory, (7, 6)),
        ("02_radial_distance.png", p_radial, (9, 4)),
        ("03_contact_timing.png", p_timing, (6, 4.5)),
        ("04_retreat_depth.png", p_retreat, (6, 4.5)),
        ("05_turn_angle.png", p_turn, (6, 4.5)),
        ("06_incidence_reflection.png", p_incidence, (6, 4.5)),
        ("07_speed_vs_radius.png", p_speed, (6, 4.5)),
        ("08_spectral_power.png", p_psd, (6, 4.5)),
        ("09_bounce_average_trajectory.png", p_bounce_traj, (6.5, 6)),
        ("10_bounce_average_radial.png", p_bounce_radial, (6.5, 4.5)),
        ("11_bounce_cycle_trajectory.png", p_cycle_traj, (7.5, 6)),
        ("12_bounce_cycle_radial.png", p_cycle_radial, (6.5, 4.5)),
    ]
    for fname, draw, figsize in panels:
        save_panel(sub, fname, draw, figsize=figsize, title=ttl)

    return dict(cx=cx, cy=cy, radius=radius_fit, contacts=contacts, g=g,
                f_w=f_w, p_w=p_w, fpk=fpk)


# =========================================================================
# Part B: within-fly controls — the same rhythm/turning metrics in all three
# windows. The wall-specific geometry only applies under confinement; what
# transfers is how confined the path is, how much/how rhythmically the fly turns,
# and its speed. Confinement should stand out as bounded + rhythmically turning.
# =========================================================================

def window_metrics(name):
    m = window_mask(name)
    xw, yw = x[m], y[m]
    cx, cy = np.median(xw), np.median(yw)
    r = sc.radial(xw, yw, cx, cy)
    _, _, speed, _ = sc.velocity(xw, yw, fs)
    omega = sc.angular_velocity(data.integrated_heading_lab.to_numpy()[m], fs)
    dur_min = (time_s[m][-1] - time_s[m][0]) / 60
    # turn events = |omega| excursions above 200 deg/s, >=0.3 s apart
    tpk, _ = find_peaks(np.abs(omega), height=np.radians(200), distance=int(0.3 * fs))
    f_w, p_w = sc.psd(omega, fs)
    fpk, _ = sc.spectral_peak(f_w, p_w, band=(0.05, 2.0))
    return dict(name=name, r=r, speed=speed, omega=omega, f_w=f_w, p_w=p_w,
                fpk=fpk, r95=np.percentile(r, 95), turn_rate=len(tpk) / dur_min,
                med_speed=np.median(speed), med_absomega=np.degrees(np.median(np.abs(omega))))


def analyse_controls():
    mets = [window_metrics(n) for n in WINDOWS]

    print(f"\n=== Part B: within-fly controls ({eid}) ===")
    print(f"{'window':18s} {'radial p95':>11s} {'turn rate/min':>14s} "
          f"{'med speed':>10s} {'med|omega|':>11s} {'spec peak':>10s}")
    for d in mets:
        print(f"{d['name']:18s} {d['r95']:10.1f}  {d['turn_rate']:13.1f}  "
              f"{d['med_speed']:9.1f}  {d['med_absomega']:9.0f}   {d['fpk']:8.2f}")

    sub = f"{eid}_controls"
    ttl = f"{label} · confinement vs menotaxis controls"
    colors = {"menotaxis_before": "#4477aa", "confinement": "#cc3311",
              "menotaxis_after": "#228833"}

    def p_radial(ax):
        for d in mets:
            ax.hist(d["r"], bins=40, histtype="step", density=True,
                    color=colors[d["name"]], label=d["name"])
        ax.set_xlabel("radial dist. from window centroid (mm)"); ax.set_ylabel("density")
        ax.set_title("confinement = bounded radius"); ax.legend(fontsize=7)

    def p_turning(ax):
        for d in mets:
            ax.hist(np.degrees(np.abs(d["omega"])), bins=np.linspace(0, 600, 61),
                    histtype="step", density=True, color=colors[d["name"]],
                    label=d["name"])
        ax.set_yscale("log")
        ax.set_xlabel("|angular velocity| (deg/s)"); ax.set_ylabel("density")
        ax.set_title("turning"); ax.legend(fontsize=7)

    def p_speed(ax):
        for d in mets:
            ax.hist(d["speed"], bins=np.linspace(0, 40, 61), histtype="step",
                    density=True, color=colors[d["name"]], label=d["name"])
        ax.set_xlabel("speed (mm/s)"); ax.set_ylabel("density")
        ax.set_title("speed"); ax.legend(fontsize=7)

    def p_psd(ax):
        for d in mets:
            ax.loglog(d["f_w"], d["p_w"], color=colors[d["name"]], label=d["name"])
        ax.set_xlim(0.03, 5)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("angular-vel power")
        ax.set_title("turning rhythm (PSD)"); ax.legend(fontsize=7)

    for fname, draw in [("01_radial_boundedness.png", p_radial),
                        ("02_turning.png", p_turning),
                        ("03_speed.png", p_speed),
                        ("04_turning_rhythm_psd.png", p_psd)]:
        save_panel(sub, fname, draw, figsize=(6.5, 4.5), title=ttl)
    return mets


if __name__ == "__main__":
    analyse_confinement()
    analyse_controls()
