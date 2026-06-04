from leecharena.sampling import even_frames, more_frames


def test_even_frames_steps_to_end():
    f = even_frames(100, step=20)
    assert f == [0, 20, 40, 60, 80]            # runs until the video end


def test_even_frames_step_one_is_all_frames():
    assert even_frames(5, step=1) == [0, 1, 2, 3, 4]


def test_even_frames_clamps_bad_step():
    assert even_frames(3, step=0) == [0, 1, 2]   # step <1 clamps to 1 (all frames), no crash
    assert even_frames(0, step=5) == []


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
