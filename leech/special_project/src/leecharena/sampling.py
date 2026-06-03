"""Seeded, resumable random frame sampling.

`planned_frames` is deterministic given (n_total, n_sample, seed): the same clip and
config always proposes the same frames in the same order. `unannotated` filters out
frames already in the store so annotation can stop and resume.
"""

from __future__ import annotations

import numpy as np


def planned_frames(n_total: int, n_sample: int, seed: int) -> list[int]:
    """Return up to n_sample unique frame indices in a deterministic shuffled order.

    Uniform sample without replacement from [0, n_total), then shuffled so the user
    annotates in a random order (not sorted, which would bias toward early frames if
    they stop partway).
    """
    if n_total <= 0:
        return []
    rng = np.random.default_rng(seed)
    k = min(n_sample, n_total)
    chosen = rng.choice(n_total, size=k, replace=False)
    rng.shuffle(chosen)
    return [int(i) for i in chosen]


def unannotated(planned: list[int], annotated: set[int]) -> list[int]:
    """Planned frames not yet in the store, preserving planned order."""
    return [f for f in planned if f not in annotated]
