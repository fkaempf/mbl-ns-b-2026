"""Annotate tab: orientation annotation panel for the unified napari app.

Port of leecharena.annotate.launch() into a panel factory. Reuses the tested
modules (sampling, store, arena) and the module-level helpers from annotate
(_build_rows, _xy_from_rowcol). napari/magicgui are imported lazily so this
module imports cleanly without Qt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annotate import (
    ROLES_2,
    ROLES_3,
    _build_rows,
    _xy_from_rowcol,
    advance_role,
    annotation_from_rows,
)
from .arena import circle_to_corners, corners_to_circle, detect_circle
from .sampling import even_frames, more_frames, planned_frames
from .store import annotated_frames, upsert_frame
from .store import load as load_store


def build_annotate_panel(ctx) -> object:
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        Label,
        PushButton,
        SpinBox,
    )

    viewer = ctx.viewer
    config = ctx.config

    # --- napari layers (mirror annotate.launch) ------------------------------
    arena_layer = viewer.add_shapes(
        name="arena", edge_color="yellow", face_color="transparent", edge_width=3,
    )
    food_layer = viewer.add_points(
        np.empty((0, 2)), name="food", face_color="lime", size=14, symbol="star",
    )
    leech_layer = viewer.add_points(
        np.empty((0, 2)), name="leeches",
        features={"role": np.array([], dtype=object), "leech_idx": np.array([], dtype=int)},
        face_color="red", size=10,
        text={"string": "{leech_idx}:{role}", "color": "white", "size": 8},
    )
    leech_layer.feature_defaults["role"] = ROLES_3[0]
    leech_layer.feature_defaults["leech_idx"] = 0

    # --- closure state -------------------------------------------------------
    # loading=True suppresses the auto-advance handler while we repopulate layers.
    state = {"pos": 0, "queue": [], "leech_count": 0, "loading": False, "shown_frame": None}

    def active_roles():
        return ROLES_3 if node_mode.value.startswith("ant/post/mid") else ROLES_2

    # --- widgets -------------------------------------------------------------
    progress = Label(value="")
    node_mode = ComboBox(label="nodes", choices=["ant/post/mid", "ant/post"],
                         value="ant/post/mid")
    sample_mode = ComboBox(label="sampling", choices=["random (n_sample)", "even interval"],
                           value="random (n_sample)")
    interval_spin = SpinBox(label="even interval (s)", value=10, min=1, max=600)
    start_btn = PushButton(text="Start / reload queue")
    target = Label(value="")
    role_combo = ComboBox(label="leech role", choices=ROLES_3, value=ROLES_3[0])
    leech_spin = SpinBox(label="leech #", value=0, min=0, max=99)
    food_present = CheckBox(label="food present", value=True)
    auto_arena = CheckBox(label="auto-detect arena each frame", value=True)
    detect_btn = PushButton(text="Auto-detect arena")
    save_btn = PushButton(text="Save & next  [Space]")
    back_btn = PushButton(text="◀ Back (previous frame)")
    skip_btn = PushButton(text="Next ▶")
    add_spin = SpinBox(label="add N frames", value=10, min=1, max=500)
    add_btn = PushButton(text="➕ Add more frames to pool")

    def _video_id():
        video = ctx.state.get("video")
        return video.name if video is not None else None

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

    def update_target(*_):
        target.value = f"Next click → {role_combo.value} (leech {int(leech_spin.value)})"

    def on_node_mode(*_):
        roles = active_roles()
        role_combo.choices = roles
        role_combo.value = roles[0]
        update_target()

    role_combo.changed.connect(push_defaults)
    leech_spin.changed.connect(push_defaults)
    role_combo.changed.connect(update_target)
    leech_spin.changed.connect(update_target)
    node_mode.changed.connect(on_node_mode)

    def on_leech_data(*_):
        """Stamp each newly placed point with the current target role/leech, then
        auto-advance the target (anterior→posterior→middle→next leech). Removals
        don't advance.

        We write the point's features DIRECTLY rather than relying on the layer's
        feature_defaults: napari reverts feature_defaults changes made during an add
        when it finalizes that add, so the next point would otherwise keep the old
        role. Setting the features data sticks.
        """
        n = len(leech_layer.data)
        if state["loading"]:           # repopulating from CSV — don't stamp/advance
            state["leech_count"] = n
            return
        if n > state["leech_count"]:
            feats = leech_layer.features.copy()
            for i in range(state["leech_count"], n):
                feats.loc[i, "role"] = role_combo.value
                feats.loc[i, "leech_idx"] = int(leech_spin.value)
                nr, nl = advance_role(role_combo.value, int(leech_spin.value), roles=active_roles())
                role_combo.value = nr
                leech_spin.value = nl
            leech_layer.features = feats
            update_target()
        state["leech_count"] = n

    leech_layer.events.data.connect(on_leech_data)

    def auto_detect(*_):
        gray = ctx.image_layer.data
        circle = detect_circle(
            gray, dp=config.hough.dp, min_dist=config.hough.min_dist,
            param1=config.hough.param1, param2=config.hough.param2,
            min_radius=config.hough.min_radius, max_radius=config.hough.max_radius,
        )
        if circle is None:
            ctx.status("No arena detected — draw/adjust the ellipse manually.")
            set_arena(None)
        else:
            set_arena(circle_to_corners(*circle))
            ctx.status(f"Arena: cx={circle[0]:.0f} cy={circle[1]:.0f} r={circle[2]:.0f}")

    def build_queue(*_):
        """Build (or rebuild) the queue for the current clip and resume at the first
        not-yet-annotated frame. Also fired automatically when a video is loaded."""
        reader = ctx.state.get("reader")
        video_id = _video_id()
        if reader is None or video_id is None:
            ctx.status("Load a video first.")
            return
        if sample_mode.value.startswith("even"):
            step = max(1, round(int(interval_spin.value) * (reader.fps or 30.0)))
            planned = even_frames(reader.n_frames, step)   # 0, step, 2*step … to end
        else:
            planned = planned_frames(reader.n_frames, config.n_sample, config.seed)
        done = annotated_frames(config.annotations_path, video_id)
        state["queue"] = planned  # keep all planned frames so Back can revisit any
        state["pos"] = next((i for i, f in enumerate(planned) if f not in done), 0)
        set_arena(None)  # fresh detect for this clip; then carried across its frames
        ctx.status(
            f"{video_id}: {len(done)}/{len(planned)} annotated"
            f"{' — loaded existing' if done else ' — new file'}; "
            f"resuming at {state['pos'] + 1}/{len(planned)}."
        )
        load_current()

    def load_existing(frame_idx) -> bool:
        """Repopulate arena/food/leeches from any saved rows for this frame.
        Returns True if the frame already had annotations."""
        df = load_store(config.annotations_path)
        if df.empty:
            return False
        sub = df[(df["video"] == _video_id()) & (df["frame"] == int(frame_idx))]
        if sub.empty:
            return False
        arena, food, leeches = annotation_from_rows(sub.to_dict("records"))
        state["loading"] = True
        try:
            if arena is not None:
                set_arena(circle_to_corners(*arena))
            # reflect the saved food state exactly (override any carried-over point)
            if food is not None:
                food_layer.data = np.array([[food[1], food[0]]])  # (row, col) = (y, x)
                food_present.value = True
            else:
                food_layer.data = np.empty((0, 2))
                food_present.value = False
            pts, roles, idxs = [], [], []
            for idx in sorted(leeches):
                for role, (x, y) in leeches[idx].items():
                    pts.append([y, x]); roles.append(role); idxs.append(idx)
            if pts:
                leech_layer.data = np.array(pts, dtype=float)
                leech_layer.features = pd.DataFrame({"role": roles, "leech_idx": idxs})
            state["leech_count"] = len(pts)
        finally:
            state["loading"] = False
        # On return to an annotated frame, the next leech starts one above the highest
        # existing id so a newly added animal doesn't collide with an existing one.
        if idxs:
            leech_spin.value = max(idxs) + 1
            role_combo.value = active_roles()[0]
            push_defaults()
            update_target()
        return True

    def load_current():
        queue = state["queue"]
        n_q = len(queue)
        # Auto-save the frame we're leaving (its annotation is still on the layers),
        # so any frame change persists work without needing the Save button.
        prev = state.get("shown_frame")
        target = queue[state["pos"]] if state["pos"] < n_q else None
        if prev is not None and prev != target and _has_leeches():
            try:
                _persist_frame(prev)
            except Exception as exc:
                ctx.status(f"Auto-save failed: {exc}")
        if n_q == 0:
            progress.value = "Load a clip and Start the queue."
            state["shown_frame"] = None
            return
        if state["pos"] >= n_q:
            progress.value = f"All {n_q} sampled frames done — use ◀ Back to revisit."
            ctx.status("Queue complete.")
            state["shown_frame"] = None
            return
        frame_idx = queue[state["pos"]]
        ctx.show_frame(frame_idx)
        # reset per-frame leech points (guarded so clearing doesn't trip auto-advance).
        # Food is carried over like the arena (it's ~static); load_existing overrides it
        # for a revisited annotated frame.
        state["loading"] = True
        leech_layer.data = np.empty((0, 2))
        state["loading"] = False
        state["leech_count"] = 0
        leech_spin.value = 0
        role_combo.value = active_roles()[0]
        push_defaults()
        update_target()
        had = load_existing(frame_idx)
        # A revisited annotated frame restores its saved arena. Otherwise: if
        # "auto-detect arena each frame" is on, re-detect every frame (handles a
        # drifting dish); if off, carry the previous ellipse over, detecting only
        # when there isn't one yet.
        if not had and (auto_arena.value or len(arena_layer.data) == 0):
            auto_detect()
        state["shown_frame"] = frame_idx
        mark = "  ✓ annotated" if had else ""
        progress.value = f"frame {frame_idx}  ({state['pos'] + 1} / {n_q}){mark}"

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

    def _persist_frame(frame_idx) -> int | None:
        """Write the current layers' annotation for frame_idx (upsert). Returns row
        count, or None if no video is loaded."""
        reader = ctx.state.get("reader")
        video_id = _video_id()
        if reader is None or video_id is None:
            return None
        arena = current_arena()
        food = None
        if food_present.value and len(food_layer.data) > 0:
            fxy = _xy_from_rowcol(food_layer.data)[0]
            food = (float(fxy[0]), float(fxy[1]))
        rows = _build_rows(video_id, frame_idx, reader.time_s(frame_idx),
                           arena, food, collect_leeches())
        # upsert (not append) so re-saving a revisited frame replaces its rows
        upsert_frame(config.annotations_path, video_id, frame_idx, rows)
        return len(rows)

    def _has_leeches() -> bool:
        return len(leech_layer.data) > 0

    def save_next(*_):
        try:
            if state["pos"] >= len(state["queue"]):
                ctx.status("Queue empty — nothing to save.")
                return
            frame_idx = state["queue"][state["pos"]]
            n = _persist_frame(frame_idx)
            if n is None:
                ctx.status("Load a video first.")
                return
            ctx.status(f"Saved frame {frame_idx} ({n} row(s)).")
            state["pos"] += 1
            state["shown_frame"] = None      # already saved; skip the auto-save in load_current
            load_current()
        except Exception as exc:
            ctx.status(f"Save failed: {exc}")

    def skip(*_):
        state["pos"] += 1
        load_current()

    def back(*_):
        q = len(state["queue"])
        if q == 0:
            return
        state["pos"] = q - 1 if state["pos"] >= q else max(0, state["pos"] - 1)
        load_current()

    def add_frames(*_):
        reader = ctx.state.get("reader")
        if reader is None:
            ctx.status("Load a video first.")
            return
        was_done = state["pos"] >= len(state["queue"])
        extra = more_frames(
            reader.n_frames, exclude=set(state["queue"]),
            k=int(add_spin.value), seed=config.seed + len(state["queue"]),
        )
        if not extra:
            ctx.status("No more frames available to add.")
            return
        state["queue"].extend(extra)
        ctx.status(f"Added {len(extra)} frame(s); pool now {len(state['queue'])}.")
        if was_done:           # we'd run out — jump into the newly added frames
            load_current()
        else:
            progress.value = f"frame {state['queue'][state['pos']]}  " \
                             f"({state['pos'] + 1} / {len(state['queue'])})"

    start_btn.changed.connect(build_queue)
    detect_btn.changed.connect(auto_detect)
    save_btn.changed.connect(save_next)
    skip_btn.changed.connect(skip)
    back_btn.changed.connect(back)
    add_btn.changed.connect(add_frames)

    # Keyboard: Space = Save & next. An application-level event filter catches the
    # key before napari's canvas (whose hold-to-pan otherwise swallows Space) and
    # regardless of which dock widget has focus — the canvas-only bind_key proved
    # unreliable. We ignore Space while a text/number field is being edited.
    try:
        from qtpy.QtCore import QEvent, QObject, Qt
        from qtpy.QtWidgets import QApplication, QLineEdit

        class _SpaceFilter(QObject):
            def eventFilter(self, obj, event):
                if (event.type() == QEvent.KeyPress
                        and event.key() == Qt.Key_Space
                        and not event.isAutoRepeat()):
                    fw = QApplication.focusWidget()
                    # Only ignore Space while actively TYPING in a text field. A
                    # spinbox that merely holds focus (e.g. after auto-advance bumps
                    # the leech #) reports the spinbox itself, NOT its inner QLineEdit,
                    # so Space still advances — fixing "Space worked after anterior but
                    # not after posterior".
                    if not isinstance(fw, QLineEdit):
                        save_next()
                        return True
                return False

        filt = _SpaceFilter()
        QApplication.instance().installEventFilter(filt)
        state["space_filter"] = filt          # keep a reference so it isn't GC'd
    except Exception as exc:  # e.g. no QApplication in a stub test
        progress.value = f"could not bind Space: {exc}"

    # Auto-build the queue (and load any existing annotations) whenever a clip opens.
    ctx.on_video_loaded(build_queue)

    update_target()

    return Container(widgets=[
        node_mode, sample_mode, interval_spin, start_btn,
        progress,
        Label(value="— arena —"), auto_arena, detect_btn,
        Label(value="— leech (click; role auto-advances; → next leech) —"),
        target, role_combo, leech_spin,
        Label(value="— food (select 'food' layer, then click) —"), food_present,
        save_btn,
        back_btn, skip_btn,
        Label(value="— pool —"), add_spin, add_btn,
    ])
