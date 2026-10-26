"""Feature agreement A: Jaccard overlap of nonzero top-K SHAP vs LIME sets."""

from metrics._topk import top_nonzero


def compute(shap_vec, lime_vec, top_k: int = 10) -> float:
    shap_set = set(top_nonzero(shap_vec, top_k))
    lime_set = set(top_nonzero(lime_vec, top_k))
    union = shap_set | lime_set
    if not union:
        return 0.0
    return float(len(shap_set & lime_set) / len(union))
