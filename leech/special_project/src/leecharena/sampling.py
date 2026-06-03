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


def more_frames(n_total: int, exclude, k: int, seed: int) -> list[int]:
    """Up to k additional frame indices from [0, n_total), none of them in `exclude`.

    Used by the "add more frames" button to grow the annotation pool with fresh,
    not-yet-queued frames. Deterministic given (n_total, exclude, k, seed)."""
    exclude = set(exclude)
    available = [i for i in range(n_total) if i not in exclude]
    if not available:
        return []
    rng = np.random.default_rng(seed)
    k = min(k, len(available))
    chosen = rng.choice(len(available), size=k, replace=False)
    return [available[int(i)] for i in chosen]
