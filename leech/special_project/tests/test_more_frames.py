from leecharena.sampling import more_frames


def test_more_frames_excludes_pool_and_is_deterministic():
    a = more_frames(100, exclude={0, 1, 2}, k=5, seed=0)
    assert len(a) == 5
    assert all(x not in {0, 1, 2} for x in a)
    assert all(0 <= x < 100 for x in a)
    assert len(set(a)) == 5                       # unique
    assert a == more_frames(100, exclude={0, 1, 2}, k=5, seed=0)  # deterministic


def test_more_frames_caps_at_available():
    a = more_frames(5, exclude={0, 1, 2}, k=10, seed=0)
    assert sorted(a) == [3, 4]                     # only 3,4 left


def test_more_frames_empty_when_all_excluded():
    assert more_frames(3, exclude={0, 1, 2}, k=5, seed=0) == []
