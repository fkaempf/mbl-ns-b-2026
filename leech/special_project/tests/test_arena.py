import numpy as np
import pytest

from leecharena.arena import circle_to_corners, corners_to_circle, detect_circle


def test_circle_bbox_round_trip():
    cx, cy, r = 120.0, 80.0, 50.0
    corners = circle_to_corners(cx, cy, r)
    assert corners.shape == (4, 2)
    back_cx, back_cy, back_r = corners_to_circle(corners)
    assert back_cx == pytest.approx(cx)
    assert back_cy == pytest.approx(cy)
    assert back_r == pytest.approx(r)


def test_corners_to_circle_averages_oval():
    # A slight oval bbox: width 100, height 120 -> r = (100 + 120) / 4 = 55.
    corners = np.array([[0.0, 0.0], [0.0, 100.0], [120.0, 100.0], [120.0, 0.0]])
    cx, cy, r = corners_to_circle(corners)
    assert (cx, cy) == pytest.approx((50.0, 60.0))
    assert r == pytest.approx(55.0)


def test_detect_circle_on_synthetic_disk():
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (100, 90), 60, 255, thickness=-1)  # filled disk
    result = detect_circle(img, min_dist=200, param2=30, min_radius=40, max_radius=80)
    assert result is not None
    cx, cy, r = result
    assert cx == pytest.approx(100, abs=6)
    assert cy == pytest.approx(90, abs=6)
    assert r == pytest.approx(60, abs=8)


def test_detect_circle_returns_none_on_blank():
    pytest.importorskip("cv2")
    img = np.zeros((100, 100), np.uint8)
    assert detect_circle(img, min_radius=10, max_radius=40) is None
