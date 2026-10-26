"""SHAP-based explanations (model-agnostic KernelExplainer fallback + TreeExplainer)."""

import numpy as np
import shap


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "tree", "boost", "lgbm", "catboost"))


def explain(model, X_background, X_explain, feature_names=None, max_samples: int = 50):
    """
    Return mean |SHAP| importance per feature and per-sample SHAP matrix.
    Uses TreeExplainer when possible; otherwise KernelExplainer on a small background.
    """
    X_explain = np.asarray(X_explain)
    X_background = np.asarray(X_background)
    n = min(max_samples, len(X_explain))
    X_e = X_explain[:n]

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_e)
    else:
        bg = shap.sample(X_background, min(50, len(X_background)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_e, nsamples=100)

    # Binary / multi-class: reduce to positive / mean abs
    if isinstance(shap_values, list):
        # list of arrays per class → mean abs across classes
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        abs_vals = stacked.mean(axis=0)
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            # (n, features, classes)
            abs_vals = np.abs(sv).mean(axis=2)
        else:
            abs_vals = np.abs(sv)

    mean_importance = abs_vals.mean(axis=0)
    names = feature_names or [f"f{i}" for i in range(len(mean_importance))]
    importance = dict(zip(names, mean_importance.tolist()))

    return {
        "importance": importance,
        "shap_values": abs_vals,
        "feature_names": names,
        "n_explained": n,
    }
