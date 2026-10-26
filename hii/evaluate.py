"""
Per-application HII evaluation (notebook recipe).

C = sign agreement on top-K nonzero SHAP vs LIME
S = cosine similarity after small num__* perturbations
F = clip(LIME.score, 0, 1)
A = Jaccard of nonzero top-K sets
HII = weighted average (default equal 0.25)
Aggregated as the mean over up to n_apps test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from hii.hybrid_index import compute as compute_hii
from metrics import consistency, feature_agreement, fidelity, stability


def resolve_class_idx(model, preferred: str | None = None) -> int:
    names = [str(c) for c in model.classes_]
    if preferred is not None and preferred in names:
        return names.index(preferred)
    if "Approved" in names:
        return names.index("Approved")
    return 1 if len(names) > 1 else 0


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "tree", "boost", "lgbm", "catboost"))


def _make_shap_explain(model, background, class_idx: int, shap_evals: int):
    """Return callable X -> (n, features) signed SHAP for the target class."""
    background = np.asarray(background)

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)

        def explain(X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            exp = explainer(X)
            vals = np.asarray(exp.values)
            if vals.ndim == 3:
                return vals[:, :, class_idx]
            return vals

        return explain

    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(
        lambda z: model.predict_proba(np.asarray(z))[:, class_idx],
        masker,
        algorithm="permutation",
    )

    def explain(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # PermutationExplainer requires >= 2 * n_features + 1
        n_features = X.shape[1]
        max_evals = max(int(shap_evals), 2 * n_features + 1)
        return np.asarray(explainer(X, max_evals=max_evals).values)

    return explain


def _lime_vector(exp, class_idx: int, n_features: int) -> np.ndarray:
    weights = dict(exp.local_exp[class_idx])
    return np.array([weights.get(i, 0.0) for i in range(n_features)], dtype=float)


def _mean_abs_importance(matrix: np.ndarray, feature_names: list[str]) -> dict:
    mean_abs = np.abs(np.asarray(matrix)).mean(axis=0)
    return dict(zip(feature_names, mean_abs.tolist()))


def _mean_signed_importance(matrix: np.ndarray, feature_names: list[str]) -> dict:
    mean = np.asarray(matrix, dtype=float).mean(axis=0)
    return dict(zip(feature_names, mean.tolist()))


def evaluate_model(
    model,
    X_train,
    X_test,
    feature_names,
    *,
    y_test=None,
    n_apps: int = 20,
    top_k: int = 10,
    n_perturbations: int = 2,
    shap_evals: int = 100,
    lime_num_samples: int = 1000,
    background_size: int = 100,
    perturb_scale: float = 0.05,
    positive_class: str | None = None,
    weights=None,
    random_state: int = 42,
) -> dict:
    """
    Compute mean C/S/F/A/HII over a random subset of test rows.
    Returns interpretability summary, importance dicts, and per-row table.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    feature_names = list(feature_names)
    n_features = len(feature_names)
    class_idx = resolve_class_idx(model, positive_class)
    class_names = [str(c) for c in model.classes_]

    rng = np.random.default_rng(random_state)
    n_apps = min(int(n_apps), len(X_test))
    if n_apps < 1:
        empty = {
            "consistency": 0.0,
            "stability": 0.0,
            "fidelity": 0.0,
            "feature_agreement": 0.0,
            "hii": 0.0,
        }
        return {
            "interpretability": empty,
            "shap_importance": {},
            "lime_importance": {},
            "signed_shap": {},
            "per_application": pd.DataFrame(),
            "class_idx": class_idx,
            "n_apps": 0,
        }

    positions = np.sort(rng.choice(len(X_test), size=n_apps, replace=False))
    X_eval = X_test[positions]
    y_eval = None if y_test is None else np.asarray(y_test)[positions]

    bg_n = min(background_size, len(X_train))
    background = X_train[:bg_n]
    explain_shap = _make_shap_explain(model, background, class_idx, shap_evals)

    numeric_mask = np.array([str(f).startswith("num__") for f in feature_names])
    if not numeric_mask.any():
        # no scaled numeric cols — perturb all features lightly
        numeric_mask = np.ones(n_features, dtype=bool)

    shap_matrix = explain_shap(X_eval)

    perturbed_rows = []
    for row in X_eval:
        for _ in range(n_perturbations):
            p = row.copy()
            p[numeric_mask] = p[numeric_mask] + rng.normal(
                0.0, perturb_scale, size=int(numeric_mask.sum())
            )
            perturbed_rows.append(p)
    perturbed_shap = explain_shap(np.asarray(perturbed_rows))

    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    rows = []
    shap_store = np.zeros((n_apps, n_features))
    lime_store = np.zeros((n_apps, n_features))

    for j in range(n_apps):
        shap_vec = np.asarray(shap_matrix[j], dtype=float).reshape(-1)
        lime_exp = lime_explainer.explain_instance(
            X_eval[j],
            model.predict_proba,
            labels=(class_idx,),
            num_features=n_features,
            num_samples=lime_num_samples,
        )
        lime_vec = _lime_vector(lime_exp, class_idx, n_features)
        shap_store[j] = shap_vec
        lime_store[j] = lime_vec

        pert_vecs = [
            perturbed_shap[j * n_perturbations + p] for p in range(n_perturbations)
        ]
        C = consistency.compute(shap_vec, lime_vec, top_k=top_k)
        S = stability.compute(shap_vec, pert_vecs)
        F = fidelity.compute(lime_exp.score)
        A = feature_agreement.compute(shap_vec, lime_vec, top_k=top_k)
        HII = compute_hii(C, S, F, A, weights=weights)

        pred = model.predict(X_eval[j : j + 1])[0]
        proba = float(model.predict_proba(X_eval[j : j + 1])[0, class_idx])
        row = {
            "Application_Index": int(positions[j]),
            "Predicted": pred,
            "Target_Class_Probability": proba,
            "Consistency_C": C,
            "Stability_S": S,
            "Fidelity_F": F,
            "Feature_Agreement_A": A,
            "HII": HII,
            "LIME_R2_raw": float(lime_exp.score),
        }
        if y_eval is not None:
            row["Actual"] = y_eval[j]
        rows.append(row)

    per_app = pd.DataFrame(rows)
    means = {
        "consistency": float(per_app["Consistency_C"].mean()),
        "stability": float(per_app["Stability_S"].mean()),
        "fidelity": float(per_app["Fidelity_F"].mean()),
        "feature_agreement": float(per_app["Feature_Agreement_A"].mean()),
    }
    means["hii"] = compute_hii(
        means["consistency"],
        means["stability"],
        means["fidelity"],
        means["feature_agreement"],
        weights=weights,
    )
    for k in ("consistency", "stability", "fidelity", "feature_agreement"):
        means[k] = round(means[k], 4)

    return {
        "interpretability": means,
        "shap_importance": _mean_abs_importance(shap_store, feature_names),
        "lime_importance": _mean_abs_importance(lime_store, feature_names),
        "signed_shap": _mean_signed_importance(shap_store, feature_names),
        "per_application": per_app,
        "class_idx": class_idx,
        "class_name": class_names[class_idx],
        "n_apps": n_apps,
    }
