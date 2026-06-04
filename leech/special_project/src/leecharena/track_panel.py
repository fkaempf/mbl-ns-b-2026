"""Track tab: automatic leech-position tracking with tuneable params + correction.

Workflow: load a clip, click the leeches on the current frame (seed points), tune
params, hit "Verify 10 s" to preview tracking on a short window (scrub to review), then
"Predict full video" (runs in a background worker, writes tracks_<video>.csv). The
`tracks` points layer always shows the current frame's positions; drag/add/delete a
point to correct that frame (saved with corrected=1). Existing tracks load on open.

Thin GUI over the tested engine (tracking.py) and store (tracks_store.py); napari /
magicgui / qtpy imported lazily so this module imports without Qt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tracking import TrackParams, track_clip
from .tracks_store import load_tracks, save_tracks, tracks_path_for, upsert_track_frame


def build_track_panel(ctx) -> "object":
    from magicgui.widgets import Container, FloatSpinBox, Label, PushButton, SpinBox

    viewer = ctx.viewer
    state = {"df": None, "frame": 0, "loading": False, "path": None}

    seed_layer = viewer.add_points(
        np.empty((0, 2)), name="track seeds", face_color="yellow", size=18, symbol="cross",
    )
    track_lines = viewer.add_shapes(
        name="track bodies", edge_color="cyan", edge_width=2, face_color="transparent",
    )
    track_layer = viewer.add_points(
        np.empty((0, 2)), name="tracks",
        features={"track": np.array([], dtype=int), "node": np.array([], dtype=int)},
        face_color="red", size=13,
        text={"string": "{track}.{node}", "color": "white", "size": 9},
    )

    # --- param widgets (defaults = validated-good values) --------------------
    n_spin = SpinBox(label="n leeches", value=2, min=1, max=8)
    dead = FloatSpinBox(label="food deadzone (·r)", value=0.32, min=0.0, max=0.9, step=0.02)
    motion = SpinBox(label="motion thresh", value=14, min=1, max=120)
    dark = SpinBox(label="dark thresh", value=22, min=1, max=120)
    amin = SpinBox(label="area min (px)", value=30, min=1, max=4000)
    amax = SpinBox(label="area max (px)", value=6000, min=100, max=200000)
    jump = SpinBox(label="max jump (px)", value=90, min=5, max=600)
    hold = SpinBox(label="max hold (frames)", value=90, min=1, max=4000)
    verify_btn = PushButton(text="Verify 10 s (from current frame)")
    predict_btn = PushButton(text="Predict full video (background)")
    clear_seeds_btn = PushButton(text="Clear seeds")
    play_btn = PushButton(text="▶ Play")
    play_fps = SpinBox(label="play fps", value=15, min=1, max=60)
    progress = Label(value="Click each leech on this frame to seed, then Verify.")

    def _video():
        return ctx.state.get("video")

    def _seeds():
        d = seed_layer.data
        if len(d) == 0:
            return None
        return [(float(p[1]), float(p[0])) for p in d]   # (row, col) -> (x, y)

    def _params():
        seeds = _seeds()
        n = len(seeds) if seeds else int(n_spin.value)
        return TrackParams(
            n_tracks=n, deadzone_frac=float(dead.value),
            motion_thresh=float(motion.value), dark_thresh=float(dark.value),
            area_min=int(amin.value), area_max=int(amax.value),
            max_jump=float(jump.value), max_hold=int(hold.value),
        )

    def _fps():
        reader = ctx.state.get("reader")
        return (reader.fps if reader and reader.fps else 30.0)

    def show_tracks_for(frame):
        """Sync the tracks layer (2 nodes/track) + body lines to this frame."""
        state["loading"] = True
        try:
            df = state["df"]
            sub = None if df is None else df[df["frame"] == int(frame)]
            if sub is None or sub.empty:
                track_layer.data = np.empty((0, 2))
                track_lines.data = []
                return
            pts = np.array([[float(r.y), float(r.x)] for r in sub.itertuples()], dtype=float)
            track_layer.data = pts
            track_layer.features = pd.DataFrame(
                {"track": [int(t) for t in sub["track"]], "node": [int(n) for n in sub["node"]]})
            # a line per track connecting its two nodes
            track_lines.data = []
            for tid, g in sub.groupby("track"):
                if len(g) >= 2:
                    g = g.sort_values("node")
                    track_lines.add(
                        np.array([[float(g.iloc[0]["y"]), float(g.iloc[0]["x"])],
                                  [float(g.iloc[1]["y"]), float(g.iloc[1]["x"])]]),
                        shape_type="line")
        finally:
            state["loading"] = False

    def on_frame(idx):
        state["frame"] = int(idx)
        show_tracks_for(idx)

    ctx.on_frame_changed(on_frame)

    def on_track_edit(*_):
        """User dragged/added/deleted a track point -> save this frame (corrected=1)."""
        if state["loading"] or state["path"] is None or _video() is None:
            return
        fr = state["frame"]
        d = track_layer.data
        feats = track_layer.features
        t_s = round(fr / _fps(), 3)
        rows = []
        for i in range(len(d)):
            tid = int(feats["track"].iloc[i]) if "track" in feats and i < len(feats) else 0
            nd = int(feats["node"].iloc[i]) if "node" in feats and i < len(feats) else i
            rows.append({"frame": fr, "time_s": t_s, "track": tid, "node": nd,
                         "x": round(float(d[i][1]), 1), "y": round(float(d[i][0]), 1), "held": 0})
        try:
            upsert_track_frame(state["path"], fr, rows)
            state["df"] = load_tracks(state["path"])
            show_tracks_for(fr)        # refresh the body line (guarded against re-entry)
            progress.value = f"corrected frame {fr} ({len(rows)} pt)"
        except Exception as exc:  # noqa: BLE001
            ctx.status(f"correction save failed: {exc}")

    track_layer.events.data.connect(on_track_edit)

    def clear_seeds(*_):
        seed_layer.data = np.empty((0, 2))
        progress.value = "Seeds cleared."

    def verify(*_):
        video = _video()
        if video is None:
            progress.value = "Load a video first."
            return
        start = state["frame"]
        stop = start + int(10 * _fps())
        progress.value = f"tracking frames {start}..{stop}…"
        try:
            rows = track_clip(video, _params(), frame_range=(start, stop), seeds=_seeds())
        except Exception as exc:  # noqa: BLE001
            progress.value = f"verify failed: {exc}"
            return
        state["df"] = pd.DataFrame(rows) if rows else None
        state["path"] = str(tracks_path_for(ctx.config.annotations_path, video.name))
        state["play_start"], state["play_stop"] = start, stop   # Play loops this window
        held = 0 if not rows else 100 * np.mean([r["held"] > 0 for r in rows])
        progress.value = (f"10 s preview: {len(rows)} points, {held:.0f}% held. "
                          "Scrub or ▶ Play to review (not saved).")
        ctx.set_frame(start)
        show_tracks_for(state["frame"])

    def predict(*_):
        video = _video()
        if video is None:
            progress.value = "Load a video first."
            return
        seeds, p = _seeds(), _params()
        path = str(tracks_path_for(ctx.config.annotations_path, video.name))

        def finish(rows):
            save_tracks(path, rows)
            state["df"] = load_tracks(path)
            state["path"] = path
            progress.value = f"done: {len(rows)} rows -> {Path(path).name}"
            show_tracks_for(state["frame"])

        try:
            from napari.qt.threading import thread_worker

            @thread_worker
            def run():
                return track_clip(video, p, seeds=seeds)

            w = run()
            w.returned.connect(finish)
            w.errored.connect(lambda e: setattr(progress, "value", f"predict failed: {e}"))
            progress.value = "predicting full video in background…"
            w.start()
        except Exception:  # no Qt worker available -> run inline (blocks)
            progress.value = "predicting full video (this may take minutes)…"
            finish(track_clip(video, p, seeds=seeds))

    # --- playback: a QTimer advances the session frame so the overlay animates ------
    from qtpy.QtCore import QTimer

    timer = QTimer()
    state["timer"] = timer        # keep a reference so it isn't garbage-collected

    def _tick():
        reader = ctx.state.get("reader")
        if reader is None:
            return
        lo = state.get("play_start") or 0
        hi = state.get("play_stop")
        hi = (hi - 1) if hi else (reader.n_frames - 1)
        nxt = state["frame"] + 1
        if nxt > hi:
            nxt = lo                # loop the window/clip
        ctx.set_frame(nxt)

    timer.timeout.connect(_tick)

    def toggle_play(*_):
        if timer.isActive():
            timer.stop()
            play_btn.text = "▶ Play"
        else:
            if ctx.state.get("reader") is None:
                progress.value = "Load a video first."
                return
            timer.start(int(1000 / max(1, int(play_fps.value))))
            play_btn.text = "⏸ Pause"

    verify_btn.changed.connect(verify)
    predict_btn.changed.connect(predict)
    clear_seeds_btn.changed.connect(clear_seeds)
    play_btn.changed.connect(toggle_play)

    # Arrow keys step the frame (Right = next, Left = previous).
    try:
        @viewer.bind_key("Right", overwrite=True)
        def _next_frame(_v):
            ctx.set_frame(state["frame"] + 1)

        @viewer.bind_key("Left", overwrite=True)
        def _prev_frame(_v):
            ctx.set_frame(state["frame"] - 1)
    except Exception as exc:  # noqa: BLE001 — e.g. a stub viewer in tests
        progress.value = f"could not bind arrow keys: {exc}"

    def on_video(path):
        tp = tracks_path_for(ctx.config.annotations_path, Path(path).name)
        state["path"] = str(tp)
        state["frame"] = 0
        seed_layer.data = np.empty((0, 2))
        df = load_tracks(tp)
        state["df"] = df if len(df) else None
        show_tracks_for(0)
        if state["df"] is not None:
            progress.value = f"loaded existing {tp.name} ({len(df)} rows). Scrub to review."

    ctx.on_video_loaded(on_video)

    return Container(widgets=[
        Label(value="— seed (select 'track seeds' layer, click each leech) —"),
        n_spin, clear_seeds_btn,
        Label(value="— params —"), dead, motion, dark, amin, amax, jump, hold,
        Label(value="— run —"), verify_btn, predict_btn,
        Label(value="— playback —"), play_fps, play_btn,
        Label(value="— correct: scrub frame slider, drag a 'tracks' point —"),
        progress,
    ])
