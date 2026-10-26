"""Consistency C: sign agreement on top-K nonzero SHAP features vs LIME."""

import numpy as np

from metrics._topk import top_nonzero


def compute(shap_vec, lime_vec, top_k: int = 10) -> float:
    """Fraction of top-K nonzero SHAP features with matching LIME sign."""
    shap_vec = np.asarray(shap_vec, dtype=float).reshape(-1)
    lime_vec = np.asarray(lime_vec, dtype=float).reshape(-1)
    shap_top = top_nonzero(shap_vec, top_k)
    if not shap_top:
        return 0.0
    matches = sum(
        1
        for i in shap_top
        if abs(lime_vec[i]) > 1e-12 and np.sign(shap_vec[i]) == np.sign(lime_vec[i])
    )
    return float(matches / len(shap_top))
