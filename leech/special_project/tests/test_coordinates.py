import numpy as np
import pytest

from leechtemplate.coordinates import (
    Calibration,
    canonical_to_pixel,
    infer_side,
    pixel_to_canonical,
    rowcol_from_xy,
    xy_from_rowcol,
)


def make_calib():
    # Anterior at (100, 100), posterior straight down at (100, 300) -> AP length 200.
    # Right reference to the screen-right of the midline.
    return Calibration(
        anterior_midline=(100.0, 100.0),
        posterior_midline=(100.0, 300.0),
        right_ref=(200.0, 200.0),
    )


def test_landmarks_map_to_expected_canonical():
    calib = make_calib()
    pts = np.array([[100.0, 100.0], [100.0, 300.0]])  # anterior, posterior
    canon = pixel_to_canonical(pts, calib)
    np.testing.assert_allclose(canon[0], [0.0, 0.0], atol=1e-9)  # origin
    np.testing.assert_allclose(canon[1], [0.0, 1.0], atol=1e-9)  # posterior y==1


def test_right_reference_is_positive_x():
    calib = make_calib()
    canon = pixel_to_canonical(np.array([[200.0, 200.0]]), calib)
    assert canon[0, 0] > 0  # right side -> +x


def test_left_point_is_negative_x():
    calib = make_calib()
    canon = pixel_to_canonical(np.array([[0.0, 200.0]]), calib)
    assert canon[0, 0] < 0


def test_isotropic_scale():
    # A point one AP-length to the right of the origin should have x == 1.
    calib = make_calib()
    canon = pixel_to_canonical(np.array([[300.0, 100.0]]), calib)  # 200 px right of origin
    np.testing.assert_allclose(canon[0], [1.0, 0.0], atol=1e-9)


def test_round_trip():
    calib = make_calib()
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 400, size=(50, 2))
    back = canonical_to_pixel(pixel_to_canonical(pts, calib), calib)
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_round_trip_rotated_axis():
    # Oblique A-P axis: round-trip must still hold.
    calib = Calibration(
        anterior_midline=(50.0, 60.0),
        posterior_midline=(250.0, 180.0),
        right_ref=(300.0, 0.0),
    )
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 300, size=(30, 2))
    back = canonical_to_pixel(pixel_to_canonical(pts, calib), calib)
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_infer_side():
    assert infer_side(0.5, 0.02) == "R"
    assert infer_side(-0.5, 0.02) == "L"
    assert infer_side(0.01, 0.02) == "M"
    assert infer_side(-0.01, 0.02) == "M"


def test_napari_order_helpers_invert():
    pts_rowcol = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(
        rowcol_from_xy(xy_from_rowcol(pts_rowcol)), pts_rowcol
    )


def test_coincident_ap_landmarks_raise():
    calib = Calibration((1.0, 1.0), (1.0, 1.0), (2.0, 2.0))
    with pytest.raises(ValueError):
        pixel_to_canonical(np.array([[0.0, 0.0]]), calib)


def test_right_ref_on_midline_raises():
    calib = Calibration((100.0, 100.0), (100.0, 300.0), (100.0, 200.0))
    with pytest.raises(ValueError):
        calib.basis()
