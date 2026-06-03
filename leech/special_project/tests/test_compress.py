import os
import tempfile

import numpy as np
import pytest

from leecharena.compress import build_ffmpeg_cmd, scale_filter


def test_scale_filter_keeps_even_dimensions():
    f = scale_filter(0.5)
    assert f.startswith("scale=")
    assert "trunc(iw*0.5/2)*2" in f
    assert "trunc(ih*0.5/2)*2" in f


def test_build_cmd_basic():
    cmd = build_ffmpeg_cmd("ffmpeg", "in.mp4", "out.mp4", crf=28, preset="fast")
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and "in.mp4" in cmd
    assert cmd[-1] == "out.mp4"
    assert "libx264" in cmd
    assert cmd[cmd.index("-crf") + 1] == "28"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert "-an" in cmd            # audio dropped
    assert "-vf" not in cmd        # no scaling requested


def test_build_cmd_with_scale():
    cmd = build_ffmpeg_cmd("ffmpeg", "in.mp4", "out.mp4", scale=0.5)
    assert "-vf" in cmd
    assert "scale=" in cmd[cmd.index("-vf") + 1]


def test_build_cmd_scale_one_is_noop():
    cmd = build_ffmpeg_cmd("ffmpeg", "in.mp4", "out.mp4", scale=1.0)
    assert "-vf" not in cmd


def test_compress_video_end_to_end():
    """Real ffmpeg run if cv2 + imageio-ffmpeg are available; skipped otherwise."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    from leecharena.compress import compress_video

    d = tempfile.mkdtemp()
    raw = os.path.join(d, "raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 320))
    for i in range(60):
        frame = np.zeros((320, 320, 3), np.uint8)
        cv2.circle(frame, (160, 160), 120, (255, 255, 255), 3)
        cv2.circle(frame, (160 + i, 80), 6, (0, 0, 255), -1)
        vw.write(frame)
    vw.release()

    out = os.path.join(d, "small.mp4")
    compress_video(raw, out, crf=30)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    # H.264 at crf 30 should be smaller than the mp4v source for this simple content.
    assert os.path.getsize(out) < os.path.getsize(raw)
