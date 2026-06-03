"""napari app to annotate leech orientation on randomized frames of a single-dish clip.

Per frame: confirm/adjust the arena circle, mark food (or absent), and click each
leech's anterior/posterior/middle. "Save & next" appends rows and advances to the next
unannotated sampled frame. Resumable: frames already in the store are skipped.

This is a thin GUI layer over the tested modules (video, arena, sampling, store).
napari/magicgui are imported lazily so the package imports without Qt.

CLI: leech-arena-annotate --video clips/raw_dish0.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .arena import circle_to_corners, corners_to_circle, detect_circle
from .config import Config
from .sampling import planned_frames, unannotated
from .store import annotated_frames, append_rows
from .video import VideoReader

ROLES = ["anterior", "posterior", "middle"]


def advance_role(role, leech_idx, roles=ROLES):
    """Next (role, leech_idx) after placing a point on `role`.

    Steps through `roles` in order; after the last role it wraps to the first and
    bumps to the next leech. An unrecognized role restarts at the first role.
    """
    try:
        i = roles.index(role)
    except ValueError:
        return roles[0], int(leech_idx)
    if i + 1 < len(roles):
        return roles[i + 1], int(leech_idx)
    return roles[0], int(leech_idx) + 1


def _xy_from_rowcol(points_rowcol: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points_rowcol, dtype=float))
    return pts[:, [1, 0]]


def _build_rows(video_id, frame_idx, time_s, arena, food, leeches) -> list[dict]:
    """Assemble store rows for one frame.

    arena: (cx, cy, r) or None. food: (x, y) or None.
    leeches: dict[int, dict[role, (x, y)]].
    """
    base = {
        "video": video_id, "frame": int(frame_idx), "time_s": float(time_s),
        "arena_cx": np.nan, "arena_cy": np.nan, "arena_r": np.nan,
        "food_x": np.nan, "food_y": np.nan,
    }
    if arena is not None:
        base["arena_cx"], base["arena_cy"], base["arena_r"] = arena
    if food is not None:
        base["food_x"], base["food_y"] = food

    def leech_row(idx, roles):
        row = dict(base)
        row["leech_idx"] = int(idx)
        for role, key in (("anterior", "ant"), ("posterior", "post"), ("middle", "mid")):
            xy = roles.get(role)
            row[f"{key}_x"] = xy[0] if xy is not None else np.nan
            row[f"{key}_y"] = xy[1] if xy is not None else np.nan
        return row

    if not leeches:
        empty = dict(base)
        empty["leech_idx"] = -1
        for key in ("ant", "post", "mid"):
            empty[f"{key}_x"] = np.nan
            empty[f"{key}_y"] = np.nan
        return [empty]
    return [leech_row(idx, roles) for idx, roles in sorted(leeches.items())]


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def annotation_from_rows(rows):
    """Inverse of _build_rows: parse one frame's stored rows back into
    (arena, food, leeches) for repopulating the GUI when revisiting a frame.

    rows: list of dicts keyed by the store COLUMNS (e.g. df slice .to_dict('records')).
    Returns (arena | None, food | None, leeches dict[int, dict[role, (x, y)]]).
    """
    rows = list(rows)
    if not rows:
        return None, None, {}
    base = rows[0]
    cx, cy, r = base.get("arena_cx"), base.get("arena_cy"), base.get("arena_r")
    arena = None if _is_missing(cx) or _is_missing(cy) or _is_missing(r) \
        else (float(cx), float(cy), float(r))
    fx, fy = base.get("food_x"), base.get("food_y")
    food = None if _is_missing(fx) or _is_missing(fy) else (float(fx), float(fy))

    leeches: dict = {}
    for row in rows:
        idx = int(row["leech_idx"])
        if idx < 0:
            continue
        for role, key in (("anterior", "ant"), ("posterior", "post"), ("middle", "mid")):
            x, y = row.get(f"{key}_x"), row.get(f"{key}_y")
            if not _is_missing(x) and not _is_missing(y):
                leeches.setdefault(idx, {})[role] = (float(x), float(y))
    return arena, food, leeches


def launch(video_path: str | Path, config: Config) -> None:
    import napari
    from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, SpinBox

    video_path = Path(video_path)
    video_id = video_path.name
    reader = VideoReader(video_path)

    planned = planned_frames(reader.n_frames, config.n_sample, config.seed)
    done = annotated_frames(config.annotations_path, video_id)
    queue = unannotated(planned, done)
    state = {"pos": 0}  # index into queue

    viewer = napari.Viewer(title=f"leech arena — {video_id}")
    image_layer = viewer.add_image(np.zeros((reader.height, reader.width, 3), np.uint8),
                                   name="frame")
    arena_layer = viewer.add_shapes(name="arena", edge_color="yellow", face_color="transparent",
                                    edge_width=3)
    food_layer = viewer.add_points(np.empty((0, 2)), name="food", face_color="lime",
                                   size=14, symbol="star")
    leech_layer = viewer.add_points(
        np.empty((0, 2)), name="leeches",
        features={"role": np.array([], dtype=object), "leech_idx": np.array([], dtype=int)},
        face_color="red", size=10,
        text={"string": "{leech_idx}:{role}", "color": "white", "size": 8},
    )
    leech_layer.feature_defaults["role"] = ROLES[0]
    leech_layer.feature_defaults["leech_idx"] = 0

    # --- dock widgets --------------------------------------------------------
    progress = Label(value="")
    role_combo = ComboBox(label="leech role", choices=ROLES, value=ROLES[0])
    leech_spin = SpinBox(label="leech #", value=0, min=0, max=99)
    food_present = CheckBox(label="food present", value=True)
    detect_btn = PushButton(text="Auto-detect arena")
    save_btn = PushButton(text="Save & next")
    skip_btn = PushButton(text="Skip frame (no save)")
    status = Label(value="")

    def set_arena(corners):
        arena_layer.data = []
        if corners is not None:
            arena_layer.add(corners, shape_type="ellipse")

    def current_arena():
        if len(arena_layer.data) == 0:
            return None
        return corners_to_circle(arena_layer.data[-1])

    def push_defaults(*_):
        leech_layer.feature_defaults["role"] = role_combo.value
        leech_layer.feature_defaults["leech_idx"] = int(leech_spin.value)

    role_combo.changed.connect(push_defaults)
    leech_spin.changed.connect(push_defaults)

    def auto_detect(*_):
        gray = image_layer.data
        circle = detect_circle(
            gray, dp=config.hough.dp, min_dist=config.hough.min_dist,
            param1=config.hough.param1, param2=config.hough.param2,
            min_radius=config.hough.min_radius, max_radius=config.hough.max_radius,
        )
        if circle is None:
            status.value = "No arena detected — draw/adjust the ellipse manually."
            set_arena(None)
        else:
            set_arena(circle_to_corners(*circle))
            status.value = f"Arena: cx={circle[0]:.0f} cy={circle[1]:.0f} r={circle[2]:.0f}"

    def load_current():
        if state["pos"] >= len(queue):
            progress.value = "All sampled frames annotated."
            status.value = "Done. Close the window."
            image_layer.data = np.zeros((reader.height, reader.width, 3), np.uint8)
            return
        frame_idx = queue[state["pos"]]
        image_layer.data = reader.frame(frame_idx)
        food_layer.data = np.empty((0, 2))
        leech_layer.data = np.empty((0, 2))
        leech_spin.value = 0
        role_combo.value = ROLES[0]
        push_defaults()
        auto_detect()
        progress.value = f"frame {frame_idx}  ({state['pos'] + 1} / {len(queue)} this session)"

    def collect_leeches():
        leeches: dict[int, dict] = {}
        if len(leech_layer.data) == 0:
            return leeches
        feats = leech_layer.features
        xy = _xy_from_rowcol(leech_layer.data)
        for i in range(len(xy)):
            idx = int(feats["leech_idx"].iloc[i])
            role = str(feats["role"].iloc[i])
            leeches.setdefault(idx, {})[role] = (float(xy[i, 0]), float(xy[i, 1]))
        return leeches

    def save_next(*_):
        try:
            if state["pos"] >= len(queue):
                return
            frame_idx = queue[state["pos"]]
            arena = current_arena()
            food = None
            if food_present.value and len(food_layer.data) > 0:
                fxy = _xy_from_rowcol(food_layer.data)[0]
                food = (float(fxy[0]), float(fxy[1]))
            rows = _build_rows(video_id, frame_idx, reader.time_s(frame_idx),
                               arena, food, collect_leeches())
            append_rows(config.annotations_path, rows)
            status.value = f"Saved frame {frame_idx} ({len(rows)} row(s))."
            state["pos"] += 1
            load_current()
        except Exception as exc:
            status.value = f"Save failed: {exc}"

    def skip(*_):
        state["pos"] += 1
        load_current()

    detect_btn.changed.connect(auto_detect)
    save_btn.changed.connect(save_next)
    skip_btn.changed.connect(skip)

    widget = Container(widgets=[
        progress,
        Label(value="— arena —"), detect_btn,
        Label(value="— leech (set role/# then click) —"), role_combo, leech_spin,
        Label(value="— food —"), food_present,
        save_btn, skip_btn, status,
    ])
    viewer.window.add_dock_widget(widget, name="annotate", area="right")

    load_current()
    napari.run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Annotate leech orientation on a single-dish clip.")
    parser.add_argument("--config", default="arena_config.yaml")
    parser.add_argument("--video", required=True, help="single-dish MP4 clip")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    launch(args.video, config)


if __name__ == "__main__":
    main()
