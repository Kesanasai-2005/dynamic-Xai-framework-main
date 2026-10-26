"""Shared top-K nonzero feature helpers."""

import numpy as np


def top_nonzero(vec, top_k: int) -> list[int]:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    nz = np.flatnonzero(np.abs(vec) > 1e-12)
    return sorted(nz.tolist(), key=lambda i: abs(vec[i]), reverse=True)[:top_k]
