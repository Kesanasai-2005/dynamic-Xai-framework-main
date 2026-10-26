"""Hybrid Interpretability Index (HII)."""


def compute(
    consistency: float,
    stability: float,
    fidelity: float,
    feature_agreement: float,
    weights=None,
) -> float:
    """
    Weighted average of explanation-quality metrics.
    Default: equal weights (0.25 each).
    """
    w = weights or {
        "consistency": 0.25,
        "stability": 0.25,
        "fidelity": 0.25,
        "feature_agreement": 0.25,
    }
    total_w = sum(w.values()) or 1.0
    score = (
        w.get("consistency", 0) * consistency
        + w.get("stability", 0) * stability
        + w.get("fidelity", 0) * fidelity
        + w.get("feature_agreement", 0) * feature_agreement
    ) / total_w
    return round(float(score), 4)
