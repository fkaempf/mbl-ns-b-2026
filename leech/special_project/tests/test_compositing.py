import os
import tempfile

import numpy as np
import pytest


def _make_clip(path, n=60):
    """Synthetic clip: a static white block in every frame plus a white block that
    sweeps across the top, touching any given pixel for only a few frames."""
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 320))
    for i in range(n):
        frame = np.zeros((320, 320, 3), np.uint8)
        frame[150:170, 150:170] = 255          # static: present in all frames
        c = 10 + i * 4                          # moving block, sweeps left -> right
        frame[20:40, c:c + 10] = 255
        vw.write(frame)
    vw.release()


def test_composite_mean_keeps_static_and_fades_movement():
    pytest.importorskip("cv2")
    from leecharena.compositing import composite

    d = tempfile.mkdtemp()
    clip = os.path.join(d, "clip.mp4")
    _make_clip(clip)

    img = composite(clip, method="mean")

    assert img.shape == (320, 320, 3)
    assert img.dtype == np.uint8
    # Static block survives the average.
    assert img[160, 160].min() > 200
    # A pixel the moving block only grazes for a few frames fades toward background.
    assert img[30, 100].max() < 60


def test_composite_median_removes_movement_entirely():
    pytest.importorskip("cv2")
    from leecharena.compositing import composite

    d = tempfile.mkdtemp()
    clip = os.path.join(d, "clip.mp4")
    _make_clip(clip)

    img = composite(clip, method="median")

    assert img[160, 160].min() > 200          # static block kept
    assert img[30, 100].max() < 10            # transient block gone


def test_composite_sample_subsamples_frames():
    pytest.importorskip("cv2")
    from leecharena.compositing import composite

    d = tempfile.mkdtemp()
    clip = os.path.join(d, "clip.mp4")
    _make_clip(clip)

    img = composite(clip, method="mean", sample=10)
    assert img.shape == (320, 320, 3)
    assert img.dtype == np.uint8
    assert img[160, 160].min() > 200          # static block still present


def test_composite_rejects_unknown_method():
    pytest.importorskip("cv2")
    from leecharena.compositing import composite

    d = tempfile.mkdtemp()
    clip = os.path.join(d, "clip.mp4")
    _make_clip(clip)

    with pytest.raises(ValueError):
        composite(clip, method="bogus")
