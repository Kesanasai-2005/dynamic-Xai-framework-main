"""Single-instance SHAP / LIME figures for Streamlit (matplotlib, no plt.show)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "tree", "boost", "lgbm", "catboost"))


def _class_names(model, class_names=None) -> list[str]:
    if class_names is not None:
        return [str(c) for c in class_names]
    return [str(c) for c in model.classes_]


def shap_waterfall_fig(
    model,
    X_background,
    x_row,
    feature_names,
    class_idx: int,
    max_display: int = 20,
):
    """Return a matplotlib Figure with a SHAP waterfall for one row / class."""
    X_background = np.asarray(X_background)
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    names = list(feature_names)

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(pd.DataFrame(x, columns=names))
        # Explanation: (n, features) or (n, features, classes)
        if getattr(explanation.values, "ndim", 0) == 3:
            row_exp = explanation[0, :, class_idx]
        else:
            row_exp = explanation[0]
    else:
        bg = shap.sample(X_background, min(50, len(X_background)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(x, nsamples=100)
        if isinstance(sv, list):
            values = np.asarray(sv[class_idx][0])
            base = float(np.asarray(explainer.expected_value)[class_idx])
        else:
            sv = np.asarray(sv)
            if sv.ndim == 3:
                values = sv[0, :, class_idx]
                base = float(np.asarray(explainer.expected_value)[class_idx])
            else:
                values = sv[0]
                base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        row_exp = shap.Explanation(
            values=values,
            base_values=base,
            data=x[0],
            feature_names=names,
        )

    fig = plt.figure(figsize=(8, max(4, min(max_display, len(names)) * 0.28)))
    shap.plots.waterfall(row_exp, max_display=max_display, show=False)
    fig.tight_layout()
    return fig


def lime_fig(
    model,
    X_train,
    x_row,
    feature_names,
    class_names=None,
    class_idx: int = 1,
    num_features: int = 10,
):
    """Return a matplotlib Figure with LIME weights for one row / class."""
    X_train = np.asarray(X_train)
    x = np.asarray(x_row, dtype=float).reshape(-1)
    names = list(feature_names)
    labels = _class_names(model, class_names)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=names,
        class_names=labels,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    exp = explainer.explain_instance(
        x,
        model.predict_proba,
        num_features=min(num_features, len(names)),
        labels=(class_idx,),
    )
    fig = exp.as_pyplot_figure(label=class_idx)
    fig.tight_layout()
    return fig, exp.as_list(label=class_idx)


def prediction_summary(model, x_row, class_names=None) -> dict:
    """Predicted label, probabilities, and class index for display."""
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    labels = _class_names(model, class_names)
    probs = model.predict_proba(x)[0]
    pred_i = int(np.argmax(probs))
    return {
        "predicted": labels[pred_i],
        "predicted_idx": pred_i,
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "class_names": labels,
    }
