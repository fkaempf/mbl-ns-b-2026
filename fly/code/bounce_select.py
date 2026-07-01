"""Quick napari tool to curate bounces by eye.

Renders every bounce (same wall-frame view as the gallery: 30 s before / 60 s after,
true finite wall, coloured by time) into a thumbnail, stacks them in napari, and lets
you tag each as *good*. The selection is saved to
``plots/bounce_gallery/good_bounces.csv`` and reloaded next time, so you can curate
across sessions.

Controls (focus the canvas):
  slider / mouse-wheel   move between bounces
  f / b                  next / previous bounce
  g                      toggle the current bounce good/not-good
  s   (or the button)    save the selection to CSV

Run:  python bounce_select.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection

import bounce_clusters as bc
from utils import PLOTS_DIR

THUMB_IN, THUMB_DPI = 7.0, 100          # 700 x 700 px thumbnails (all identical size)
CSV = os.path.join(PLOTS_DIR, "bounce_gallery", "good_bounces.csv")


def render(b):
    """Render one bounce to a fixed-size RGB array (wall + time-coloured trajectory)."""
    fig = plt.figure(figsize=(THUMB_IN, THUMB_IN), dpi=THUMB_DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    bc.draw_wall(ax, b)
    U, V, tau = b["U"], b["V"], b["tau"]
    pts = np.column_stack([U, V]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, cmap="viridis", array=tau[:-1], lw=1.4, zorder=3))
    i0 = int(np.argmin(np.abs(tau)))
    ax.scatter(U[i0], V[i0], color="black", s=45, zorder=5)
    ax.text(0.02, 0.98, f"{b['eid']}\nbounce {b['i']}/{b['n']}  t={b['t_min']:.1f} min",
            transform=ax.transAxes, va="top", ha="left", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off"); ax.autoscale()
    canvas = FigureCanvasAgg(fig); canvas.draw()
    w, h = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def load_prior(allb):
    good = np.zeros(len(allb), bool)
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        sel = {(r.eid, int(r.bounce)) for _, r in prev[prev.good].iterrows()}
        for k, b in enumerate(allb):
            good[k] = (b["eid"], b["i"]) in sel
        print(f"loaded {int(good.sum())} previously good bounces from {CSV}")
    return good


def save_csv(allb, good, verbose=True):
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    pd.DataFrame({
        "eid": [b["eid"] for b in allb],
        "bounce": [b["i"] for b in allb],
        "t_min": [round(b["t_min"], 2) for b in allb],
        "good": good,
    }).to_csv(CSV, index=False)
    if verbose:
        print(f"saved {int(good.sum())}/{len(good)} good bounces -> {CSV}")


def main():
    import napari
    from magicgui import magicgui

    allb = []
    for f in bc.EXPERIMENTS:
        allb += bc.extract(f)
    print(f"rendering {len(allb)} bounces ...")
    imgs = np.stack([render(b) for b in allb])
    good = load_prior(allb)

    viewer = napari.Viewer(title="bounce selector")
    viewer.add_image(imgs, name="bounces", rgb=True)
    viewer.text_overlay.visible = True

    def refresh(*_):
        i = int(viewer.dims.current_step[0])
        b = allb[i]
        viewer.text_overlay.text = (
            f"{i + 1}/{len(allb)}   {b['eid']} bounce {b['i']}/{b['n']}   "
            f"t={b['t_min']:.1f} min   {'GOOD ✓' if good[i] else '— not selected'}   "
            f"[g]ood (autosaves)  [f/b]nav  ({int(good.sum())} good)")
        viewer.text_overlay.color = "lime" if good[i] else "white"

    @viewer.bind_key("g", overwrite=True)
    def toggle(v):
        i = int(v.dims.current_step[0]); good[i] = not good[i]
        save_csv(allb, good, verbose=False)         # autosave on every tag
        refresh()

    @viewer.bind_key("f", overwrite=True)
    def nxt(v):
        v.dims.set_current_step(0, (int(v.dims.current_step[0]) + 1) % len(allb))

    @viewer.bind_key("b", overwrite=True)
    def prv(v):
        v.dims.set_current_step(0, (int(v.dims.current_step[0]) - 1) % len(allb))

    @viewer.bind_key("s", overwrite=True)
    def _save(v):
        save_csv(allb, good)

    @magicgui(call_button="Save good bounces -> CSV")
    def save_button():
        save_csv(allb, good)

    viewer.window.add_dock_widget(save_button, area="right")
    viewer.dims.events.current_step.connect(refresh)
    refresh()
    napari.run()


if __name__ == "__main__":
    main()
