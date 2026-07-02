"""Animate one wall presentation: raw imaging + FC2 compass + unwrapped bump + the wall.

For a chosen fly and wall trial this renders a movie of the whole presentation - PRE
seconds before the wall turns on through POST seconds after it turns off - driven
frame-by-frame on the imaging volume clock (~10.6 Hz, so it plays back in real time).
Four synchronised panels:

  * RAW      - the registered two-photon frame (sum over the 5 fast-Z planes) of the
               fan-shaped body: the actual FC2 activity you see moving,
  * COMPASS  - the population bump as a needle (PVA, pink) against the fly's heading
               (yellow), both in the fly's heading frame (offset-aligned), so you watch
               the bump track / slip from heading,
  * UNWRAPPED- the 16 FSB columns unrolled into a raster over the whole window with a
               sweeping time cursor (the bump as a moving stripe), extracted phase on top,
  * WALL     - the planar VR trajectory with the barrier (grey wall in its red laser zone)
               that spawns at wall-on; a yellow dot + facing arrow is the fly, shocks ring red.

The raw frames are read straight from the lab volume's *_registered.tif (only the window's
volumes), mapped to the unified clock via the imaging *_mean_pixel_values.pkl unix_time.

Run from this directory:
  python imaging_video.py                # Fly5_002, its strongest-bump trial
  python imaging_video.py Fly6_002 3     # a specific fly + 1-based trial
  python imaging_video.py Fly5_002 all   # every wall trial of that fly
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy import stats as st
import tifffile

from utils import DATA_DIR, PLOTS_DIR, add_scalebar
from cxstyle import PINK, YELLOW, WHITE, PINK_CMAP, PREVIEW_BG
from imaging_unify import (IMG_DIR, NCOL, ROI_COLS, TRAJ_CMAP, aligned_bump,
                           csmooth, norm_matrix, LASER_MX, LASER_MY)

PRE, POST = 10.0, 10.0          # seconds shown before wall-on / after wall-off
SMOOTH_N = 25                   # samples (~0.2 s) circular low-pass on the raster traces
VEC_SMOOTH_N = 121              # samples (~1 s) circular low-pass on the heading + FC2 vectors
SPEEDUP = 2.5                   # playback speed-up over real time
R_MIN = 0.15                    # only draw the FC2 bump where the PVA amplitude is at least this
VOL_VOL = "imaging0630processed"  # processed-data folder name on the lab volume
VOL_ROOT = f"/Volumes/nsb/NSB_2026/Fly/Heat_Project/{VOL_VOL}"
NPLANE = 5
DEFAULT_FLY = "Fly5_002"


def vol_folder(fly):
    """The fly's processed folder on the lab volume (holds the registered tif + pkl)."""
    return os.path.join(VOL_ROOT, fly)


def load_imaging(fly):
    """Registered movie path, per-volume unix times (s). Files are found recursively because
    some flies (e.g. Fly1) keep them in a sub-folder and use non-standard names."""
    src = vol_folder(fly)
    tifs = [t for t in glob.glob(os.path.join(src, "**", "*_registered.tif"), recursive=True)
            if "fbmask" not in os.path.basename(t)]
    tif = sorted(tifs)[0]
    # Trusted first-party data: our own imaging pipeline's per-volume ROI means (a pandas
    # DataFrame) copied from the lab volume, used here only for its unix_time column.
    mp = pd.read_pickle(sorted(glob.glob(os.path.join(src, "**", "*mean_pixel_values.pkl"),
                                         recursive=True))[0])
    tv = mp["unix_time"].to_numpy(float) / 1e9        # imaging volume times (s)
    return tif, tv


def read_window_frames(tif, vols):
    """Sum-over-z registered frames for the given volume indices -> (n, H, W) float32.

    The registered tif is a (nvol, nplane, H, W) fast-Z stack stored as nvol*nplane flat pages,
    so volume v's planes are pages v*nplane .. v*nplane+nplane-1; we read only the window's pages
    and sum the planes. Plane count and frame size are read from the file (they vary by fly:
    e.g. 64x128 for most, 84x128 for Fly6)."""
    with tifffile.TiffFile(tif) as tf:
        _, nplane, H, W = tf.series[0].shape
    pages = [v * nplane + p for v in vols for p in range(nplane)]
    arr = tifffile.imread(tif, key=pages).reshape(len(vols), nplane, H, W)
    return arr.astype(np.float32).sum(1)


def wall_patches(ax, cx, cy, rot, w, th):
    """The barrier as a grey wall inside its red laser zone, drawn at -rot (vrcmd
    rotation_z is y-flipped vs the vr_x/vr_y frame). Returns the patches, hidden, so the
    animation can reveal them at wall-on."""
    ang = -rot
    hw, hth = w + 2 * LASER_MX, th + 2 * LASER_MY
    heat = Rectangle((cx - hw / 2, cy - hth / 2), hw, hth, angle=ang, rotation_point="center",
                     facecolor="red", alpha=0.3, edgecolor="red", lw=0.6, zorder=2, visible=False)
    wall = Rectangle((cx - w / 2, cy - th / 2), w, th, angle=ang, rotation_point="center",
                     facecolor="0.5", edgecolor="white", lw=0.8, zorder=3, visible=False)
    ax.add_patch(heat); ax.add_patch(wall)
    return heat, wall


def make_video(fly, k, tif, tv, walls, uni, map_modes=("static", "bump")):
    """Render trial ``k`` (1-based) of ``fly`` to plots/imaging/videos/.

    One video per entry in ``map_modes`` (frames are read from the volume only once). On the map
    the heading arrow is always a FIXED length (direction only) - speed magnitude lives only in
    the compass, so nothing on the map wobbles with speed:
      * "static"  - just the (smoothed) heading arrow,
      * "bump"    - adds the FC2 bump vector on the map as a pink arrow (length scaled by bump
                    amplitude) alongside the heading arrow, saved with a _mapbumpvec suffix.

    The heading and FC2 bump vectors are temporally smoothed (VEC_SMOOTH_N); the raw imaging
    panel is left unsmoothed."""
    ton, toff, cx, cy, rot, w, th = walls[k - 1]
    tu = uni["time"].to_numpy()
    t0 = tu[0]
    x, y, h = uni["vr_x"].to_numpy(), uni["vr_y"].to_numpy(), uni["vr_heading"].to_numpy()
    las = uni["laser_on"].to_numpy()

    # --- behaviour window (dense, ~119 Hz): PRE before wall-on .. POST after wall-off ---
    i0 = int(np.searchsorted(tu, ton - PRE))
    i1 = int(min(np.searchsorted(tu, toff + POST), len(tu) - 1))
    sb = slice(i0, i1)
    tb = tu[sb] - ton                                   # 0 at wall-on
    xs, ys = x[sb], y[sb]
    r = uni["bump_amp"].to_numpy()
    colphase = csmooth(uni["bump_phase"].to_numpy(), SMOOTH_N) / 360 * NCOL + 0.5
    Wn = norm_matrix(uni)                               # (T x 16) maxmin columns
    shock_t = tb[np.where(np.diff(las[sb].astype(int)) == 1)[0]]   # shock onsets in window

    # --- allocentric compass: heading + bump in the SAME VR-world frame as the trajectory ---
    # vr_heading is the allocentric VR-world heading (0 = +y / north, clockwise), the frame the
    # wall and path live in. The bump is placed in that frame by a single constant gauge offset
    # (circular mean of heading - column phase over this trial), so it is free to drift from
    # heading rather than being glued to it.
    head_allo = csmooth(uni["vr_heading"].to_numpy(), VEC_SMOOTH_N) % 360
    phase_s = csmooth(uni["bump_phase"].to_numpy(), VEC_SMOOTH_N)
    off = st.circmean(((uni["vr_heading"].to_numpy()[sb] - uni["bump_phase"].to_numpy()[sb]) % 360),
                      low=0, high=360, nan_policy="omit")
    bump_allo = (phase_s + off) % 360

    # --- compass magnitudes, temporally smoothed too (not just the angles) so the needle
    # lengths don't jitter: speed for the heading needle, bump amplitude for the bump needle ---
    speed_s = pd.Series(uni["speed"].to_numpy()).rolling(VEC_SMOOTH_N, center=True,
                                                         min_periods=1).mean().to_numpy()
    r_s = pd.Series(r).rolling(VEC_SMOOTH_N, center=True, min_periods=1).mean().to_numpy()
    sp = speed_s[sb]
    sp_ref = np.nanpercentile(sp, 98)
    sp_norm = np.clip(sp / (sp_ref if sp_ref > 0 else 1.0), 0, 1)   # heading-needle length (smoothed)

    # --- imaging window (sparse, ~10.6 Hz): the frames that drive the movie ---
    vols = np.where((tv >= ton - PRE) & (tv <= toff + POST))[0]
    frames = read_window_frames(tif, vols)
    vmin, vmax = np.percentile(frames, 1), np.percentile(frames, 99.5)
    tvw = tv[vols] - ton                                # volume times, 0 at wall-on
    row_of = np.searchsorted(tu, tv[vols]).clip(0, len(tu) - 1)   # nearest behaviour row
    curb = np.searchsorted(tb, tvw).clip(0, len(tb) - 1)          # index within the window path

    eid = uni.attrs.get("eid", "")
    window_s = (toff - ton) + PRE + POST
    fps = max(1, int(round(len(vols) / window_s * SPEEDUP)))   # SPEEDUP x real-time playback
    outdir = os.path.join(PLOTS_DIR, "imaging", "videos")
    os.makedirs(outdir, exist_ok=True)

    def render(map_mode):
        # ---------------- figure ----------------
        fig = plt.figure(figsize=(13.5, 7.2))
        fig.patch.set_facecolor(PREVIEW_BG)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 1],
                              hspace=0.28, wspace=0.18, left=0.03, right=0.97, top=0.9, bottom=0.08)
        axRaw = fig.add_subplot(gs[0, 0])
        axComp = fig.add_subplot(gs[1, 0], projection="polar")
        axRas = fig.add_subplot(gs[0, 1])
        axTraj = fig.add_subplot(gs[1, 1])
        for a in (axRaw, axRas, axTraj):
            a.set_facecolor(PREVIEW_BG)
        axComp.set_facecolor(PREVIEW_BG)

        # RAW imaging plane
        im = axRaw.imshow(frames[0], cmap=PINK_CMAP, vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="equal")
        axRaw.axis("off")
        axRaw.set_title("raw imaging - FSB (sum of 5 z-planes)", color=WHITE, fontsize=10)

        # COMPASS: bump (pink) vs heading (yellow), allocentric, in the SAME cartesian frame as
        # the map (default polar: 0 rad = +x = right, CCW, up = north = vr_heading 0). Needles
        # are drawn at 90 - angle (deg) so a needle here points the exact same way as the fly's
        # arrow in the trajectory panel.
        axComp.set_rmax(1.05); axComp.set_rticks([]); axComp.set_xticks(np.radians([0, 90, 180, 270]))
        axComp.set_xticklabels(["E", "N", "W", "S"], color=WHITE, fontsize=8)
        axComp.tick_params(colors=WHITE)
        axComp.spines["polar"].set_color(WHITE)
        head_line, = axComp.plot([], [], color=YELLOW, lw=3, solid_capstyle="round", zorder=5)
        bump_line, = axComp.plot([], [], color=PINK, lw=3, solid_capstyle="round", zorder=6)
        head_tip, = axComp.plot([], [], "o", color=YELLOW, ms=6, zorder=7)
        bump_tip, = axComp.plot([], [], "o", color=PINK, ms=6, zorder=8)
        axComp.set_title("FC2 bump vs heading (allocentric)", color=WHITE, fontsize=10, pad=12)
        axComp.legend([head_line, bump_line], ["heading (∝ speed)", "FC2 bump"],
                      loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                      frameon=False, labelcolor=WHITE, fontsize=8)

        # UNWRAPPED raster + sweeping cursor
        axRas.imshow(Wn[sb].T, aspect="auto", origin="lower", extent=[tb[0], tb[-1], 0.5, 16.5],
                     cmap=PINK_CMAP, vmin=0, vmax=0.9, interpolation="nearest")
        axRas.plot(tb, np.where(r[sb] > R_MIN, colphase[sb], np.nan), color="#37e0d0", lw=1.0, alpha=0.8)
        axRas.axvline(0, color=WHITE, lw=0.8, alpha=0.6)                        # wall on
        axRas.axvline(toff - ton, color=WHITE, lw=0.8, ls=":", alpha=0.6)       # wall off
        if len(shock_t):
            axRas.plot(shock_t, np.full(len(shock_t), 0.9), "^", color="red", ms=6, clip_on=False)
        cursor = axRas.axvline(tvw[0], color=WHITE, lw=1.4)
        axRas.set_xlim(tb[0], tb[-1]); axRas.set_ylim(0.5, 16.5)
        axRas.set_xlabel("time from wall on (s)", color=WHITE)
        axRas.set_ylabel("FSB column (1-16)", color=WHITE)
        axRas.set_title("unwrapped bump (16 columns)", color=WHITE, fontsize=10)
        axRas.tick_params(colors=WHITE)
        for spine in axRas.spines.values():
            spine.set_color(WHITE)

        # WALL panel: static faint full path, moving fly, wall revealed at wall-on
        axTraj.plot(xs, ys, color=WHITE, lw=0.7, alpha=0.25)
        heat, wall = wall_patches(axTraj, cx, cy, rot, w, th)
        trail, = axTraj.plot([], [], color=WHITE, lw=1.5, alpha=0.9)           # path so far (white, so the pink FC2 arrow stands out)
        fly_dot, = axTraj.plot([], [], "o", color=YELLOW, ms=8, zorder=7)
        face_arrow = axTraj.annotate("", xytext=(xs[0], ys[0]), xy=(xs[0], ys[0]),
                                     arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2), zorder=7)
        bump_arrow = axTraj.annotate("", xytext=(xs[0], ys[0]), xy=(xs[0], ys[0]),
                                     arrowprops=dict(arrowstyle="-|>", color=PINK, lw=2), zorder=8)
        shock_sc = axTraj.scatter([], [], facecolor="none", edgecolor="red", s=55, lw=1.4, zorder=6)
        # frame the view to include the whole wall (laser zone), not just the path, so the wall
        # is never clipped - matching the trials/ figures (which autoscale over path + wall)
        ang = np.radians(-rot)
        cc, ss = np.cos(ang), np.sin(ang)
        hwx, hwy = (w + 2 * LASER_MX) / 2, (th + 2 * LASER_MY) / 2
        wcx = [cx + dx * cc - dy * ss for dx in (-hwx, hwx) for dy in (-hwy, hwy)]
        wcy = [cy + dx * ss + dy * cc for dx in (-hwx, hwx) for dy in (-hwy, hwy)]
        allx = np.r_[xs, wcx]; ally = np.r_[ys, wcy]
        pad = 6
        axTraj.set_xlim(allx.min() - pad, allx.max() + pad)
        axTraj.set_ylim(ally.min() - pad, ally.max() + pad)
        axTraj.set_aspect("equal")
        axTraj.axis("off")
        add_scalebar(axTraj)
        atitle = {"static": "arrow: heading",
                  "heading": "arrow: heading ∝ speed",
                  "bump": "arrows: heading (yellow) + FC2 bump (pink)"}[map_mode]
        axTraj.set_title(f"trajectory + wall  ({atitle})", color=WHITE, fontsize=10)

        sup = fig.suptitle("", color=WHITE, fontsize=13)
        arrow_len = 0.06 * ((allx.max() - allx.min()) + (ally.max() - ally.min()))   # scale to the view

        def update(fi):
            row = row_of[fi]
            cb = curb[fi]
            tt = tvw[fi]
            im.set_data(frames[fi])
            # allocentric compass (same cartesian frame as the map): heading needle = velocity
            # vector (direction = facing, length = speed normalised over the trial); bump needle
            # drawn only when the PVA is strong. 90 - angle maps vr-heading convention to +x=east.
            hd = np.radians(90 - head_allo[row])
            hlen = sp_norm[cb]                                 # smoothed speed
            head_line.set_data([hd, hd], [0, hlen]); head_tip.set_data([hd], [hlen])
            if r_s[row] > R_MIN:
                bd = np.radians(90 - bump_allo[row]); amp = min(0.4 + r_s[row], 1.0)   # smoothed amp
                bump_line.set_data([bd, bd], [0, amp]); bump_tip.set_data([bd], [amp])
            else:
                bump_line.set_data([], []); bump_tip.set_data([], [])
            cursor.set_xdata([tt, tt])
            # trajectory: path so far, the fly, its heading arrow, and shocks that have happened
            trail.set_data(xs[:cb + 1], ys[:cb + 1])
            fly_dot.set_data([xs[cb]], [ys[cb]])
            # heading arrow (yellow) from the smoothed heading: static length, or ∝ speed in the
            # "heading" version. 90 - angle maps vr-heading convention to the +x=east map frame.
            Lh = arrow_len * (sp_norm[cb] if map_mode == "heading" else 1.0)
            if Lh > 1e-6:
                hdir = np.radians(90 - head_allo[row])
                face_arrow.set_visible(True)
                face_arrow.set_position((xs[cb], ys[cb]))
                face_arrow.xy = (xs[cb] + Lh * np.cos(hdir), ys[cb] + Lh * np.sin(hdir))
            else:
                face_arrow.set_visible(False)
            # FC2 bump arrow (pink), FIXED length (direction only, like the heading arrow) in the
            # "bump" version - magnitude stays in the compass, so nothing on the map changes length
            if map_mode == "bump":
                bdir = np.radians(90 - bump_allo[row])
                bump_arrow.set_visible(True)
                bump_arrow.set_position((xs[cb], ys[cb]))
                bump_arrow.xy = (xs[cb] + arrow_len * np.cos(bdir), ys[cb] + arrow_len * np.sin(bdir))
            else:
                bump_arrow.set_visible(False)
            past = shock_t[shock_t <= tt]
            if len(past):
                si = np.searchsorted(tb, past).clip(0, len(tb) - 1)
                shock_sc.set_offsets(np.column_stack([xs[si], ys[si]]))
            else:
                shock_sc.set_offsets(np.empty((0, 2)))
            on = tt >= 0 and tt <= (toff - ton)
            heat.set_visible(on); wall.set_visible(on)
            state = "WALL ON" if on else ("before" if tt < 0 else "after")
            sup.set_text(f"FC2 imaging + wall - {fly} ({eid}) - trial {k}   "
                         f"t = {tt:+.1f} s   [{state}]")
            return [im, head_line, bump_line, head_tip, bump_tip, cursor, trail, fly_dot,
                    face_arrow, bump_arrow, shock_sc]

        ani = animation.FuncAnimation(fig, update, frames=len(vols), interval=1000 / fps, blit=False)
        suffix = {"static": "", "heading": "_mapheadvec", "bump": "_mapbumpvec"}[map_mode]
        out = os.path.join(outdir, f"fc2_{fly}_trial_{k:03d}{suffix}.mp4")
        try:
            ani.save(out, fps=fps, dpi=140, savefig_kwargs=dict(facecolor=PREVIEW_BG))
        except Exception as e:
            out = out[:-4] + ".gif"
            ani.save(out, fps=fps, writer="pillow", savefig_kwargs=dict(facecolor=PREVIEW_BG))
            print(f"  (ffmpeg failed: {type(e).__name__}) -> GIF")
        plt.close(fig)
        print(f"  saved {os.path.relpath(out, PLOTS_DIR)}  ({len(vols)} frames @ {fps} fps)")

    for m in map_modes:
        render(m)


def best_trial(uni, walls):
    """1-based index of the wall trial with the strongest mean bump amplitude in-window."""
    tu = uni["time"].to_numpy(); amp = uni["bump_amp"].to_numpy()
    scores = []
    for k, (ton, toff, cx, cy, rot, w, th) in enumerate(walls, 1):
        m = (tu >= ton - PRE) & (tu <= ton + POST)
        scores.append(np.nanmean(amp[m]) if m.any() else np.nan)
    return int(np.nanargmax(scores)) + 1


def main():
    import bounces as bo
    fly = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FLY
    # Trusted first-party data: this _unified.pkl was written by imaging_unify.py.
    uni = pd.read_pickle(os.path.join(DATA_DIR, IMG_DIR, f"{fly}_unified.pkl"))
    walls = bo.wall_trials(f"{IMG_DIR}/{fly}")
    tif, tv = load_imaging(fly)
    if len(sys.argv) > 2 and sys.argv[2] == "all":
        trials = range(1, len(walls) + 1)
    elif len(sys.argv) > 2:
        trials = [int(sys.argv[2])]
    else:
        trials = [best_trial(uni, walls)]
    print(f"{fly}: {len(walls)} walls, rendering trial(s) {list(trials)}")
    for k in trials:
        make_video(fly, k, tif, tv, walls, uni)


if __name__ == "__main__":
    main()
