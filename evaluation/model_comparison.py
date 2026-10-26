"""Aggregate prediction metrics + HII into a comparison table."""

import pandas as pd


def compare(results: dict) -> pd.DataFrame:
    """
    results: {
      model_name: {
        "performance": {...},
        "interpretability": {consistency, stability, fidelity, feature_agreement, hii}
      }
    }
    """
    rows = []
    for name, payload in results.items():
        perf = payload.get("performance", {})
        interp = payload.get("interpretability", {})
        rows.append(
            {
                "Model": name,
                "Accuracy": perf.get("accuracy"),
                "Precision": perf.get("precision"),
                "Recall": perf.get("recall"),
                "F1": perf.get("f1"),
                "Consistency": interp.get("consistency"),
                "Stability": interp.get("stability"),
                "Fidelity": interp.get("fidelity"),
                "Feature Agreement": interp.get("feature_agreement"),
                "HII": interp.get("hii"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "HII" in df.columns:
        df = df.sort_values("HII", ascending=False).reset_index(drop=True)
    return df


_COMPONENTS = {
    "consistency": "Consistency (C)",
    "stability": "Stability (S)",
    "fidelity": "Fidelity (F)",
    "feature_agreement": "Feature agreement (A)",
}

_COMPONENT_SHORT = {
    "consistency": "C",
    "stability": "S",
    "fidelity": "F",
    "feature_agreement": "A",
}


def _level(x: float) -> str:
    return "high" if x >= 0.75 else "moderate" if x >= 0.5 else "low"


def _top_features(importance: dict, k: int = 3) -> list[str]:
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [name.split("__", 1)[-1] for name, _ in ranked]


def summarize(report: dict) -> list[str]:
    """Short, scannable takeaways — one clear point per line."""
    info = report["dataset_info"]
    df = report["comparison"]
    results = report["results"]
    if df.empty:
        return ["No models were run."]

    by_acc = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_acc, best_hii = by_acc.iloc[0], df.iloc[0]
    has_acc = pd.notna(best_acc["Accuracy"])
    test_src = "uploaded test set" if info.get("test_source") == "uploaded" else "held-out split"
    unlabeled = " (no labels — accuracy skipped)" if not info.get("test_has_target", True) else ""

    lines = [
        f"Data — {info['train_rows']} train / {info['test_rows']} test rows ({test_src}){unlabeled}, "
        f"target `{info['target']}`, {info['target_classes']} classes.",
    ]

    if not has_acc:
        if len(df) == 1:
            row = df.iloc[0]
            lines.append(f"Only model — {row['Model']}, HII {row['HII']:.2f} (no test labels).")
        else:
            lines.append(
                f"Best interpretability — {best_hii['Model']} (HII {best_hii['HII']:.2f}). "
                "Accuracy unavailable (test has no target)."
            )
    elif len(df) == 1:
        row = df.iloc[0]
        lines.append(
            f"Only model — {row['Model']} at {row['Accuracy']:.0%} accuracy, HII {row['HII']:.2f}."
        )
    else:
        if best_acc["Accuracy"] == by_acc.iloc[-1]["Accuracy"]:
            lines.append(f"Accuracy — all models tie at {best_acc['Accuracy']:.0%}.")
        else:
            lines.append(
                f"Best accuracy — {best_acc['Model']} ({best_acc['Accuracy']:.0%}). "
                f"Lowest — {by_acc.iloc[-1]['Model']} ({by_acc.iloc[-1]['Accuracy']:.0%})."
            )
        lines.append(
            f"Best interpretability — {best_hii['Model']} (HII {best_hii['HII']:.2f})."
        )
        if best_acc["Model"] == best_hii["Model"]:
            lines.append(f"Trade-off — none; {best_acc['Model']} leads on both.")
        else:
            lines.append(
                f"Trade-off — pick {best_acc['Model']} for accuracy, "
                f"{best_hii['Model']} when explanations matter more."
            )

    for _, row in df.iterrows():
        interp = results[row["Model"]]["interpretability"]
        weakest = min(_COMPONENTS, key=lambda c: interp[c])
        top = _top_features(results[row["Model"]]["shap_importance"])
        acc = "—" if pd.isna(row["Accuracy"]) else f"{row['Accuracy']:.0%}"
        lines.append(
            f"{row['Model']} — Acc {acc} · HII {row['HII']:.2f} ({_level(row['HII'])}) · "
            f"weakest {_COMPONENT_SHORT[weakest]}={interp[weakest]:.2f} · "
            f"top features: {', '.join(top)}."
        )

    shared = set.intersection(*(set(_top_features(r["shap_importance"])) for r in results.values()))
    if len(results) > 1 and shared:
        lines.append(f"Shared drivers — {', '.join(sorted(shared))} rank high in every model.")

    if info["test_rows"] < 20:
        lines.append(
            f"Caution — only {info['test_rows']} test rows; treat scores as noisy."
        )

    if not has_acc:
        lines.append(f"Recommendation — prefer {best_hii['Model']} for explanations on this unlabeled test set.")
    elif best_acc["Model"] != best_hii["Model"]:
        lines.append(
            f"Recommendation — use {best_hii['Model']} to explain decisions; "
            f"{best_acc['Model']} when raw accuracy wins."
        )
    else:
        lines.append(f"Recommendation — prefer {best_hii['Model']} overall for this dataset.")

    return lines
