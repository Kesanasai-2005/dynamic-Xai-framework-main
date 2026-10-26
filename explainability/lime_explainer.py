"""LIME tabular explanations."""

import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def explain(
    model,
    X_train,
    X_explain,
    feature_names=None,
    class_names=None,
    max_samples: int = 50,
    num_features: int = 10,
):
    """
    Return mean |LIME| importance per feature across explained instances.
    """
    X_train = np.asarray(X_train)
    X_explain = np.asarray(X_explain)
    n_features = X_train.shape[1]
    names = feature_names or [f"f{i}" for i in range(n_features)]

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=names,
        class_names=class_names or ["0", "1"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    n = min(max_samples, len(X_explain))
    scores = np.zeros(n_features)
    counts = np.zeros(n_features)
    lime_matrix = np.zeros((n, n_features))

    for i in range(n):
        exp = explainer.explain_instance(
            X_explain[i],
            model.predict_proba,
            num_features=min(num_features, n_features),
        )
        # map local feature weights back to indices
        local = dict(exp.as_map()[exp.available_labels()[0]])
        for feat_idx, weight in local.items():
            scores[feat_idx] += abs(weight)
            counts[feat_idx] += 1
            lime_matrix[i, feat_idx] = abs(weight)

    mean_importance = np.divide(
        scores, np.maximum(counts, 1), out=np.zeros_like(scores), where=counts > 0
    )
    importance = dict(zip(names, mean_importance.tolist()))

    return {
        "importance": importance,
        "lime_values": lime_matrix,
        "feature_names": names,
        "n_explained": n,
    }
