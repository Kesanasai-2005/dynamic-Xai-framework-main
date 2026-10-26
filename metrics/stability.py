"""Stability S: cosine similarity of SHAP after small numeric perturbations."""

import numpy as np


def compute(shap_vec, perturbed_shap_vecs) -> float:
    """
    Mean cosine similarity between baseline SHAP and perturbed SHAP vectors.
    Clipped to [0, 1].
    """
    a = np.asarray(shap_vec, dtype=float).reshape(-1)
    sims = []
    for p in perturbed_shap_vecs:
        b = np.asarray(p, dtype=float).reshape(-1)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-12:
            sims.append(float(np.dot(a, b) / denom))
    if not sims:
        return 0.0
    return float(np.clip(np.mean(sims), 0.0, 1.0))
