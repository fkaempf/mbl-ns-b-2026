from leecharena.sampling import planned_frames, unannotated


def test_deterministic_given_seed():
    a = planned_frames(1000, 30, seed=7)
    b = planned_frames(1000, 30, seed=7)
    assert a == b


def test_different_seed_differs():
    a = planned_frames(1000, 30, seed=1)
    b = planned_frames(1000, 30, seed=2)
    assert a != b


def test_count_and_uniqueness():
    frames = planned_frames(100, 30, seed=0)
    assert len(frames) == 30
    assert len(set(frames)) == 30
    assert all(0 <= f < 100 for f in frames)


def test_sample_capped_at_total():
    frames = planned_frames(10, 50, seed=0)
    assert len(frames) == 10
    assert sorted(frames) == list(range(10))


def test_empty_video():
    assert planned_frames(0, 30, seed=0) == []


def test_unannotated_preserves_order_and_skips_done():
    planned = [5, 2, 9, 1, 7]
    remaining = unannotated(planned, annotated={2, 1})
    assert remaining == [5, 9, 7]
