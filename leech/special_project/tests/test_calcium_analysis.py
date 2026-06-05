"""Unit tests for the pure calcium-analysis math (no Qt, no suite2p run)."""
import numpy as np
import pytest

from analysis.calcium import signals, ganglion_activity as ga, contraction as ct


def _make_movie(T=200, Ly=8, Lx=8):
    rng = np.random.default_rng(0)
    return rng.normal(100, 1, size=(T, Ly, Lx)).astype("float32")


def test_region_traces_is_mask_mean():
    movie = _make_movie()
    mask = np.zeros((8, 8), bool); mask[2:4, 2:4] = True
    masks = mask[None]
    tr = signals.region_traces(movie, masks)
    expected = movie[:, mask].mean(axis=1)
    assert tr.shape == (1, movie.shape[0])
    np.testing.assert_allclose(tr[0], expected, rtol=1e-5)


def test_dff_is_zero_for_flat_signal():
    F = np.full((1, 300), 50.0, "float32")
    d = signals.dff(F, fs=3.0)
    assert np.allclose(d, 0.0, atol=1e-4)


def test_dff_recovers_known_transient():
    fs = 3.0
    F = np.full(600, 100.0)
    F[300:330] += 50.0                      # +50% over baseline of 100
    d = signals.dff(F[None], fs=fs, win_s=30.0)[0]
    assert d[315] == pytest.approx(0.5, abs=0.05)
    assert abs(d[50]) < 0.05


def test_correlation_matrix_identity_and_anticorr():
    t = np.linspace(0, 10, 500)
    a = np.sin(t); b = -np.sin(t); c = np.cos(t)
    corr = ga.correlation_matrix(np.stack([a, b, c]))
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-6)
    assert abs(corr[0, 2]) < 0.1            # sin vs cos ~ orthogonal


def test_crosscorr_recovers_known_lag():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 1000)
    a = base.copy()
    b = np.roll(base, 7)                     # b lags a by 7 samples
    lag, r = ga.crosscorr_lag(a, b, max_lag=20)
    assert lag == 7
    assert r > 0.9


def test_lag_matrix_antisymmetric():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 800)
    traces = np.stack([base, np.roll(base, 5)])
    lags = ga.lag_matrix(traces, fs=10.0, max_lag_s=2.0)
    assert lags[0, 1] == pytest.approx(-lags[1, 0])
    assert lags[0, 1] != 0


def test_detect_events_counts_transients():
    fs = 5.0
    trace = np.zeros(1000)
    for onset in (100, 400, 700):
        trace[onset:onset + 10] = 5.0       # 3 clear events
    events = ga.detect_events(trace[None], k=3.0, fs=fs)
    assert len(events[0]) == 3
    np.testing.assert_array_equal(events[0], [100, 400, 700])


def test_regress_out_global_removes_common_mode():
    rng = np.random.default_rng(4)
    T = 2000
    glob = np.sin(np.linspace(0, 40, T))            # shared global signal
    traces = np.stack([5 * glob + 0.1 * rng.normal(0, 1, T) for _ in range(6)])
    # raw: everything ~ perfectly correlated
    raw = ga.correlation_matrix(traces)
    assert raw[~np.eye(6, dtype=bool)].mean() > 0.95
    resid, g = ga.regress_out_global(traces)
    res_corr = ga.correlation_matrix(resid)
    # after removing the global signal, residuals are ~uncorrelated
    assert abs(res_corr[~np.eye(6, dtype=bool)].mean()) < 0.2
    assert np.corrcoef(g, glob)[0, 1] > 0.99


def test_regress_out_global_keeps_extra_shared_pair():
    rng = np.random.default_rng(5)
    T, n = 3000, 12
    glob = np.sin(np.linspace(0, 50, T))
    extra = np.sin(np.linspace(0, 13, T) + 1.0)     # shared by only G0,G1
    base = [5 * glob + 1.0 * rng.normal(0, 1, T) for _ in range(n)]
    base[0] = base[0] + 3 * extra
    base[1] = base[1] + 3 * extra
    resid, _ = ga.regress_out_global(np.stack(base))
    rc = ga.correlation_matrix(resid)
    # With the extra signal shared by only 2 of 12 regions, it barely leaks
    # into the global mean, so genuine G0-G1 coupling survives and stands out
    # well above coupling to a non-participating region.
    assert rc[0, 1] > 0.6
    assert rc[0, 1] - abs(rc[0, 3]) > 0.3


def test_seed_correlation_map_finds_correlated_pixels():
    T = 300
    rng = np.random.default_rng(3)
    sig = rng.normal(0, 1, T).astype("float32")
    movie = rng.normal(0, 0.1, (T, 6, 6)).astype("float32")
    movie[:, 1, 1] += sig                    # this pixel tracks the seed
    cmap = signals.seed_correlation_map(movie, sig)
    assert cmap[1, 1] > 0.9
    assert abs(cmap[4, 4]) < 0.3


# ---- body-wall contraction (kymograph edge tracking + peak delay) ----------
def test_kymograph_samples_along_line():
    movie = np.zeros((3, 20, 20), "float32")
    movie[:, 10, :] = 5.0                       # bright row at r=10
    kymo = ct.kymograph(movie, (0, 5), (19, 5))  # vertical line down col 5
    assert kymo.shape == (3, 20)
    assert kymo[0].argmax() == 10               # picks up the bright row


def test_track_edge_follows_moving_step():
    T, n = 60, 50
    kymo = np.zeros((T, n), "float32")
    true = (15 + 10 * np.sin(np.linspace(0, 4 * np.pi, T))).astype(int)
    for t in range(T):
        kymo[t, true[t]:] = 1.0                 # step edge that moves over time
    edge = ct.track_edge(kymo)
    assert np.corrcoef(edge, true)[0, 1] > 0.98
    assert np.abs(edge - true).mean() < 1.5


def test_peak_intervals_recovers_period():
    fs = 10.0
    t = np.arange(0, 30, 1 / fs)
    x = np.sin(2 * np.pi * 0.5 * t)             # 0.5 Hz -> 2 s period
    peaks, period = ct.peak_intervals(x, fs)
    assert period == pytest.approx(2.0, abs=0.1)
    assert len(peaks) >= 13


def test_peak_delays_sign_and_value():
    fs = 10.0
    ref = np.array([100, 200, 300])
    tgt = ref + 5                                # tgt 5 samples = 0.5 s after ref
    d = ct.peak_delays(ref, tgt, fs)
    assert np.allclose(d, 0.5)


def test_kymograph_perp_tracks_wall_moving_perpendicular():
    # a bright horizontal wall that moves up/down (perpendicular) over time;
    # the line is drawn ALONG the wall (horizontal). Edge tracking should
    # recover the perpendicular displacement.
    T, Y, X = 80, 60, 40
    movie = np.zeros((T, Y, X), "float32")
    pos = (30 + 6 * np.sin(np.linspace(0, 4 * np.pi, T))).astype(int)
    for t in range(T):
        movie[t, pos[t]:pos[t] + 2, :] = 5.0          # bright wall at row pos[t]
    kymo = ct.kymograph_perp(movie, (30, 5), (30, 34), half_width=15)
    edge = ct.track_edge(kymo)
    # |corr| ~ 1: edge follows the wall (sign depends on perp-axis orientation)
    assert abs(np.corrcoef(edge, pos)[0, 1]) > 0.95


def test_phase_at_recovers_phase():
    fs = 20.0
    t = np.arange(0, 20, 1 / fs)
    a = np.cos(2 * np.pi * 0.5 * t)                    # phase 0 at f=0.5
    b = np.cos(2 * np.pi * 0.5 * t - np.pi / 2)        # lags a by 90 deg
    dphi = np.angle(np.exp(1j * (ct.phase_at(b, 0.5, fs) - ct.phase_at(a, 0.5, fs))))
    assert dphi == pytest.approx(-np.pi / 2, abs=0.1)


def test_ncc_shift_ignores_brightness_but_catches_movement():
    n, T = 41, 60
    base = np.exp(-((np.arange(n) - 20) ** 2) / 8.0)   # a bump profile
    # (a) pure brightness oscillation, NO movement -> NCC shift ~ 0
    bright = np.stack([base * (1 + 0.8 * np.sin(2 * np.pi * t / 12)) for t in range(T)])
    s_bright = ct.track_shift_ncc(bright)
    assert np.std(s_bright) < 0.5
    # (b) real movement (bump translates) -> NCC shift recovers it
    moved = np.stack([np.roll(base, int(round(4 * np.sin(2 * np.pi * t / 12))))
                      for t in range(T)])
    s_moved = ct.track_shift_ncc(moved)
    assert np.ptp(s_moved) > 5          # tracks the ~8 px peak-to-peak motion
