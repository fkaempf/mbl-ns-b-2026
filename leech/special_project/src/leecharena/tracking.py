"""Classical detect-then-track for leech POSITION (no ML, no labels).

Per frame, foreground = motion (|frame - median background|) OR dark-appearance (a
still leech is darker than the dish), restricted to the dish ANNULUS — inside the rim
and OUTSIDE a central deadzone where the food sits. Fragmented bodies are merged by a
morphological close. A fixed number of tracks is then maintained by greedy
nearest-neighbour association with hold-last (a momentarily-faint leech keeps its last
position instead of being dropped) and re-acquisition of stale tracks.

The geometry/association helpers are pure (numpy only); cv2 is imported lazily inside
the functions that need it, so this module imports without OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackParams:
    n_tracks: int = 2
    deadzone_frac: float = 0.32      # central food deadzone radius, as fraction of dish r
    motion_thresh: float = 14.0      # |frame - bg| gray difference for motion
    dark_thresh: float = 22.0        # how much darker than bg a still leech must be
    area_min: int = 30               # blob area band (px) — reject specks ...
    area_max: int = 6000             # ... and hands/transitions
    max_jump: float = 90.0           # max px a track may move between processed frames
    max_hold: int = 90               # frames a track may coast before it can re-acquire
    rim_frac: float = 0.90           # outer mask radius, as fraction of dish r
    redetect_every: int = 10         # re-Hough the dish every N frames (0 = static dish)
    recenter_smooth: float = 0.3     # EMA factor applied to a re-detected dish center
    max_center_jump: float = 40.0    # reject a re-detection that jumps further than this


def annulus_mask(shape, cx: float, cy: float, r: float, deadzone_frac: float,
                 rim_frac: float = 0.90) -> np.ndarray:
    """Boolean mask of the dish annulus: inside rim_frac*r, outside deadzone_frac*r."""
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    d2 = (X - cx) ** 2 + (Y - cy) ** 2
    return (d2 <= (rim_frac * r) ** 2) & (d2 >= (deadzone_frac * r) ** 2)


def detect_blobs(gray, bg_gray, mask, params: TrackParams):
    """Foreground blobs in the masked annulus, largest first, each as
    (area, cx, cy, end0, end1) where the two ends are the blob's body extremes."""
    import cv2

    gray = np.asarray(gray, dtype=np.float32)
    bg_gray = np.asarray(bg_gray, dtype=np.float32)
    motion = np.abs(gray - bg_gray) > params.motion_thresh
    dark = (bg_gray - gray) > params.dark_thresh
    fg = ((motion | dark) & mask).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (params.area_min <= area <= params.area_max):
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w0, h0 = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ys, xs = np.where(lab[y0:y0 + h0, x0:x0 + w0] == i)
        pix = np.column_stack([xs + x0, ys + y0])      # (x, y) pixels of this component
        e0, e1 = blob_endpoints(pix)
        blobs.append((area, float(cent[i][0]), float(cent[i][1]), e0, e1))
    blobs.sort(key=lambda b: b[0], reverse=True)
    return blobs


def associate(tracks, held, dets, params: TrackParams):
    """Update tracks (list of [x,y] or None) and held counters from detections.

    Greedy nearest-neighbour: freshest tracks pick first; a match within max_jump
    resets held to 0; an unmatched track increments held (holds its last position).
    Leftover detections re-acquire an empty or stale (held > max_hold) track.
    """
    tracks = [None if t is None else [float(t[0]), float(t[1])] for t in tracks]
    held = list(held)
    pts = [(float(d[1]), float(d[2])) for d in dets]   # (cx, cy) of each detection
    used = set()

    for ti in sorted(range(len(tracks)), key=lambda t: held[t]):
        if tracks[ti] is None:
            continue
        tx, ty = tracks[ti]
        best, bd = None, float("inf")
        for pi, (px, py) in enumerate(pts):
            if pi in used:
                continue
            d = np.hypot(px - tx, py - ty)
            if d < bd:
                bd, best = d, pi
        if best is not None and bd <= params.max_jump:
            tracks[ti] = [pts[best][0], pts[best][1]]
            used.add(best)
            held[ti] = 0
        else:
            held[ti] += 1

    for pi, p in enumerate(pts):
        if pi in used:
            continue
        free = [t for t in range(len(tracks)) if tracks[t] is None or held[t] > params.max_hold]
        if free:
            ti = free[0]
            tracks[ti] = [p[0], p[1]]
            held[ti] = 0
            used.add(pi)
    return tracks, held


def init_tracks(seeds, params: TrackParams):
    """Initial (tracks, held) for a clip.

    seeds: list of (x, y) starting positions clicked by the user on the first frame,
    or None to auto-acquire. A seeded track starts live (held=0); an unseeded slot
    starts stale (held > max_hold) so the first detections can acquire it.
    """
    tracks = [None] * params.n_tracks
    held = [params.max_hold + 1] * params.n_tracks
    if seeds:
        for i, s in enumerate(seeds[:params.n_tracks]):
            tracks[i] = [float(s[0]), float(s[1])]
            held[i] = 0
    return tracks, held


def blob_endpoints(pixels):
    """Two body-end points of a blob as ((x,y),(x,y)): the extreme pixels along the
    blob's principal (long) axis via PCA. `pixels` is an (N,2) array of (x,y) coords."""
    pts = np.asarray(pixels, dtype=float)
    if len(pts) < 2:
        p = pts[0] if len(pts) else (0.0, 0.0)
        return (float(p[0]), float(p[1])), (float(p[0]), float(p[1]))
    c = pts.mean(axis=0)
    centered = pts - c
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]      # principal (long) axis
    proj = centered @ axis
    a, b = pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]
    return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))


def assign_nodes(prev, ends):
    """Order the two `ends` ((x,y),(x,y)) to stay consistent with `prev` ((x,y),(x,y)).

    Returns ends as-is if prev is None; otherwise returns them in whichever order keeps
    node0 near prev node0 and node1 near prev node1 (so the two ends don't swap labels
    frame-to-frame). Correctness of which is anterior/posterior is irrelevant — only
    temporal consistency matters.
    """
    a, b = ends
    if prev is None:
        return a, b
    p0, p1 = prev
    keep = np.hypot(a[0] - p0[0], a[1] - p0[1]) + np.hypot(b[0] - p1[0], b[1] - p1[1])
    swap = np.hypot(b[0] - p0[0], b[1] - p0[1]) + np.hypot(a[0] - p1[0], a[1] - p1[1])
    return (a, b) if keep <= swap else (b, a)


def update_center(cur, detected, smooth: float, max_jump: float):
    """EMA-update the dish center toward a fresh Hough detection.

    Holds `cur` if `detected` is None or jumps further than max_jump (a mis-detect);
    otherwise moves a `smooth` fraction toward it. Keeps the annulus following a
    drifting dish without jerking on bad detections.
    """
    if detected is None:
        return (float(cur[0]), float(cur[1]))
    if np.hypot(detected[0] - cur[0], detected[1] - cur[1]) > max_jump:
        return (float(cur[0]), float(cur[1]))
    return (float(cur[0] + smooth * (detected[0] - cur[0])),
            float(cur[1] + smooth * (detected[1] - cur[1])))


def _contaminated(gray, bg_gray, mask) -> bool:
    """Frame whose masked median brightness is far from the background (hand/transition)."""
    return abs(float(np.median(gray[mask])) - float(np.median(bg_gray[mask]))) > 45.0


def track_clip(video, params: TrackParams | None = None, frame_range=None,
               seeds=None, progress=None):
    """Track a clip; return rows {frame, time_s, track, x, y, held}.

    frame_range: (start, stop) frame indices, or None for the whole clip.
    seeds: optional list of (x, y) start positions (clicked on the first frame) used to
           initialise the tracks; None auto-acquires from the first detections.
    progress: optional callback(done, total).
    """
    import cv2

    from .arena import detect_circle
    from .compositing import composite

    params = params or TrackParams()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg = composite(video, method="median", sample=80)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY).astype(np.float32)
    circ = detect_circle(bg_gray, min_dist=1000.0, param2=40.0,
                         min_radius=int(0.30 * min(h, w)), max_radius=int(0.50 * min(h, w)))
    cx0, cy0, r = circ if circ else (w / 2.0, h / 2.0, 0.42 * min(h, w))
    minR, maxR = int(0.30 * min(h, w)), int(0.50 * min(h, w))
    cx, cy = cx0, cy0       # tracked (drifting) dish center

    def detect_center(gray_u8):
        c = detect_circle(gray_u8, min_dist=1000.0, param2=40.0, min_radius=minR, max_radius=maxR)
        return None if c is None else (c[0], c[1])

    start, stop = (frame_range or (0, n_frames))
    start = max(0, int(start)); stop = min(n_frames, int(stop))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    tracks, held = init_tracks(seeds, params)
    track_ends = [None] * params.n_tracks      # last (end0, end1) per track, for consistency
    rows = []
    for fi in range(start, stop):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # follow a drifting dish: periodically re-detect its center, smooth, recenter
        if params.redetect_every and (fi - start) % params.redetect_every == 0:
            cx, cy = update_center((cx, cy), detect_center(gray.astype(np.uint8)),
                                   params.recenter_smooth, params.max_center_jump)
        mask = annulus_mask((h, w), cx, cy, r, params.deadzone_frac, params.rim_frac)
        # shift the static background to align its dish with this frame's dish
        dx, dy = int(round(cx - cx0)), int(round(cy - cy0))
        bg_al = np.roll(np.roll(bg_gray, dy, axis=0), dx, axis=1) if (dx or dy) else bg_gray
        dets = [] if _contaminated(gray, bg_al, mask) else detect_blobs(gray, bg_al, mask, params)
        ends_by_centroid = {(round(d[1], 3), round(d[2], 3)): (d[3], d[4]) for d in dets}
        tracks, held = associate(tracks, held, dets, params)
        t_s = fi / fps
        for ti in range(params.n_tracks):
            if tracks[ti] is None or held[ti] > params.max_hold:
                continue
            if held[ti] == 0:        # matched a blob this frame -> use its fresh endpoints
                fresh = ends_by_centroid.get((round(tracks[ti][0], 3), round(tracks[ti][1], 3)))
                if fresh is not None:
                    track_ends[ti] = assign_nodes(track_ends[ti], fresh)
            ends = track_ends[ti]
            if ends is None:         # no body yet -> degenerate pair at the centroid
                ends = ((tracks[ti][0], tracks[ti][1]), (tracks[ti][0], tracks[ti][1]))
            for node, (ex, ey) in enumerate(ends):
                rows.append({"frame": fi, "time_s": round(t_s, 3), "track": ti, "node": node,
                             "x": round(float(ex), 1), "y": round(float(ey), 1), "held": held[ti]})
        if progress is not None and (fi - start) % 200 == 0:
            progress(fi - start, stop - start)
    cap.release()
    if progress is not None:
        progress(stop - start, stop - start)
    return rows
