import numpy as np
import pytest

from leecharena.tracking import (
    TrackParams,
    annulus_mask,
    associate,
    detect_blobs,
    init_tracks,
    update_center,
)


def test_update_center_smooths_toward_detection():
    # EMA 0.5 -> halfway from (100,100) toward (110,120)
    new = update_center((100.0, 100.0), (110.0, 120.0), smooth=0.5, max_jump=50)
    assert new == pytest.approx((105.0, 110.0))


def test_update_center_rejects_bad_jump():
    # detection 200px away (> max_jump) is a mis-detect -> keep current center
    new = update_center((100.0, 100.0), (400.0, 400.0), smooth=0.5, max_jump=50)
    assert new == pytest.approx((100.0, 100.0))


def test_update_center_none_detection_holds():
    new = update_center((100.0, 100.0), None, smooth=0.5, max_jump=50)
    assert new == pytest.approx((100.0, 100.0))


def test_blob_endpoints_finds_major_axis_extremes():
    from leecharena.tracking import blob_endpoints
    # a horizontal bar of pixels (x 10..90, y=50) -> ends near (10,50) and (90,50)
    xs = np.arange(10, 91)
    pts = np.column_stack([xs, np.full_like(xs, 50)])
    e0, e1 = blob_endpoints(pts)
    got = sorted([e0, e1])
    assert got[0][0] == pytest.approx(10, abs=1)
    assert got[1][0] == pytest.approx(90, abs=1)
    assert got[0][1] == pytest.approx(50, abs=1)


def test_assign_nodes_keeps_consistent_order():
    from leecharena.tracking import assign_nodes
    prev = ((0.0, 0.0), (100.0, 0.0))
    # new ends arrive swapped; assign_nodes should restore prev's ordering
    a, b = assign_nodes(prev, ((98.0, 1.0), (2.0, 1.0)))
    assert a == pytest.approx((2.0, 1.0))     # node0 stays near prev node0 (origin)
    assert b == pytest.approx((98.0, 1.0))


def test_assign_nodes_no_prev_returns_as_is():
    from leecharena.tracking import assign_nodes
    a, b = assign_nodes(None, ((5.0, 5.0), (9.0, 9.0)))
    assert (a, b) == ((5.0, 5.0), (9.0, 9.0))


def test_init_tracks_from_seeds_sets_positions_and_live():
    p = TrackParams(n_tracks=2, max_hold=90)
    tracks, held = init_tracks([(120.0, 130.0), (300.0, 310.0)], p)
    assert tracks == [[120.0, 130.0], [300.0, 310.0]]
    assert held == [0, 0]                      # seeded tracks are immediately live


def test_init_tracks_without_seeds_is_empty_and_stale():
    p = TrackParams(n_tracks=2, max_hold=90)
    tracks, held = init_tracks(None, p)
    assert tracks == [None, None]
    assert all(h > p.max_hold for h in held)   # nothing to track yet -> free to acquire


def test_annulus_mask_excludes_rim_and_center():
    m = annulus_mask((200, 200), cx=100, cy=100, r=100, deadzone_frac=0.3)
    assert m[100, 100] == False          # center (food) excluded
    assert m[100, 160] == True           # mid annulus (60px out, within 0.9r, beyond 0.3r)
    assert m[100, 198] == False          # near rim (98px > 0.9r) excluded
    assert m[100, 120] == False          # 20px out < 0.3r deadzone -> excluded


def test_detect_blobs_finds_body_skips_food_and_specks():
    p = TrackParams(deadzone_frac=0.3, motion_thresh=14, dark_thresh=22,
                    area_min=30, area_max=6000)
    bg = np.full((200, 200), 200, np.float32)        # bright background
    gray = bg.copy()
    gray[100:115, 150:165] = 60                      # a leech in the annulus (dark, ~225px)
    gray[95:105, 95:105] = 60                        # "food" in the center deadzone
    gray[100:102, 40:42] = 60                        # tiny speck (~4px, below area_min)
    mask = annulus_mask(gray.shape, 100, 100, 100, p.deadzone_frac)

    blobs = detect_blobs(gray, bg, mask, p)
    xs = [round(b[1]) for b in blobs]            # b = (area, cx, cy, end0, end1)
    # the annulus leech (~x157) is found; food (~x100) and speck (~x40) are not
    assert any(150 <= x <= 165 for x in xs)
    assert not any(90 <= x <= 110 for x in xs)
    assert not any(35 <= x <= 45 for x in xs)


def test_associate_matches_nearest_within_maxjump():
    p = TrackParams(n_tracks=2, max_jump=50, max_hold=90)
    tracks = [[100.0, 100.0], [200.0, 200.0]]
    held = [0, 0]
    dets = [(100, 205.0, 205.0), (100, 105.0, 102.0)]   # near track1 / track0
    tracks, held = associate(tracks, held, dets, p)
    assert tracks[0] == pytest.approx([105.0, 102.0])
    assert tracks[1] == pytest.approx([205.0, 205.0])
    assert held == [0, 0]


def test_associate_holds_when_no_detection():
    p = TrackParams(n_tracks=2, max_jump=50, max_hold=90)
    tracks = [[100.0, 100.0], [200.0, 200.0]]
    held = [0, 0]
    tracks, held = associate(tracks, held, [], p)         # nothing detected
    assert tracks == [[100.0, 100.0], [200.0, 200.0]]     # positions held
    assert held == [1, 1]


def test_associate_rejects_jump_beyond_maxjump():
    p = TrackParams(n_tracks=1, max_jump=50, max_hold=90)
    tracks = [[100.0, 100.0]]
    held = [0]
    tracks, held = associate(tracks, held, [(100, 400.0, 400.0)], p)  # too far
    assert tracks == [[100.0, 100.0]]                     # not moved
    assert held == [1]


def test_associate_reacquires_stale_track():
    p = TrackParams(n_tracks=1, max_jump=50, max_hold=5)
    tracks = [[100.0, 100.0]]
    held = [6]                                            # stale (> max_hold)
    tracks, held = associate(tracks, held, [(100, 400.0, 400.0)], p)
    assert tracks[0] == pytest.approx([400.0, 400.0])     # re-acquired far detection
    assert held[0] == 0
