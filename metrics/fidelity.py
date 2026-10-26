"""Fidelity F: bounded LIME local surrogate R² (exp.score)."""

import numpy as np


def compute(lime_score: float) -> float:
    """clip(LIME local weighted R², 0, 1)."""
    try:
        return float(np.clip(float(lime_score), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0
