"""
Dynamic Explainable AI Framework — CLI entry point.

Pipeline:
  load → analyze → preprocess → train → per-app HII (SHAP/LIME) → report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evaluation.model_comparison import compare, summarize
from hii.evaluate import evaluate_model
from models import decision_tree, logistic_regression, random_forest
from preprocessing.preprocess import analyze_dataset, load_dataset, preprocess
from visualization.plots import save_comparison_figure

MODEL_REGISTRY = {
    "Logistic Regression": logistic_regression,
    "Random Forest": random_forest,
    "Decision Tree": decision_tree,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def selected_models(cfg: dict) -> list[str]:
    flags = cfg.get("models", {})
    mapping = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
    }
    return [name for key, name in mapping.items() if flags.get(key, True)]


def run_pipeline(
    dataset_path: str,
    target: str,
    model_names: list[str] | None = None,
    config: dict | None = None,
    test_path: str | None = None,
) -> dict:
    cfg = config or {}
    prep_cfg = cfg.get("preprocessing", {})
    exp_cfg = cfg.get("explainability", {})
    hii_weights = cfg.get("hii", {}).get("weights")

    print("Loading dataset...")
    df = load_dataset(dataset_path)
    print("Dataset analysis")
    info = analyze_dataset(df, target)
    print(
        f"  rows={info['rows']} cols={info['columns']} "
        f"missing={info['missing_values']} classes={info['target_classes']}"
    )

    test_df = load_dataset(test_path) if test_path else None

    print("Preprocessing...")
    data = preprocess(
        df,
        target=target,
        test_size=prep_cfg.get("test_size", 0.2),
        random_state=prep_cfg.get("random_state", 42),
        handle_missing=prep_cfg.get("handle_missing", "median"),
        test_df=test_df,
    )
    info["test_source"] = "uploaded" if test_df is not None else "split"
    info["train_rows"] = len(data["y_train"])
    info["test_rows"] = len(data["X_test"])
    info["test_has_target"] = data["y_test"] is not None

    names = model_names or selected_models(cfg)
    # n_samples kept as alias for older configs
    n_apps = exp_cfg.get("n_apps", exp_cfg.get("n_samples", 20))
    top_k = exp_cfg.get("top_k", 10)
    eval_kwargs = {
        "n_apps": n_apps,
        "top_k": top_k,
        "n_perturbations": exp_cfg.get("n_perturbations", 2),
        "shap_evals": exp_cfg.get("shap_evals", 100),
        "lime_num_samples": exp_cfg.get("lime_num_samples", 1000),
        "background_size": exp_cfg.get("background_size", 100),
        "perturb_scale": exp_cfg.get("perturb_scale", 0.05),
        "positive_class": exp_cfg.get("positive_class"),
        "weights": hii_weights,
        "random_state": prep_cfg.get("random_state", 42),
    }

    results = {}
    print("Training models and evaluating HII...")
    for name in names:
        module = MODEL_REGISTRY[name]
        print(f"  {name}")
        model = module.train(data["X_train"], data["y_train"])
        if data["y_test"] is None:
            performance = {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
            }
        else:
            performance = module.evaluate(model, data["X_test"], data["y_test"])

        print(f"  Per-application HII (up to {n_apps} apps, top_k={top_k})...")
        hii_out = evaluate_model(
            model,
            data["X_train"],
            data["X_test"],
            data["feature_names"],
            y_test=data["y_test"],
            **eval_kwargs,
        )
        interp = hii_out["interpretability"]
        print(
            f"  class={hii_out.get('class_name')} n={hii_out['n_apps']} "
            f"C={interp['consistency']:.3f} S={interp['stability']:.3f} "
            f"F={interp['fidelity']:.3f} A={interp['feature_agreement']:.3f} "
            f"HII={interp['hii']:.3f}"
        )

        results[name] = {
            "model": model,
            "performance": performance,
            "interpretability": interp,
            "shap_importance": hii_out["shap_importance"],
            "lime_importance": hii_out["lime_importance"],
            "signed_shap": hii_out.get("signed_shap", {}),
            "hii_per_application": hii_out["per_application"],
            "hii_class": hii_out.get("class_name"),
        }

    print("Generating report...")
    comparison = compare(results)
    le = data.get("label_encoder")
    class_names = (
        [str(c) for c in le.classes_]
        if le is not None
        else [str(c) for c in sorted(set(map(str, data["y_train"])))]
    )
    return {
        "dataset_info": info,
        "results": results,
        "comparison": comparison,
        "feature_names": data["feature_names"],
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "X_test_raw": data["X_test_raw"],
        "class_names": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic Explainable AI Framework")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--test", default=None, help="Separate test CSV (same columns as dataset)")
    parser.add_argument("--output", default="data/processed/report.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or cfg["dataset"]["path"]
    target = args.target or cfg["dataset"]["target"]

    report = run_pipeline(dataset, target, config=cfg, test_path=args.test)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["comparison"].to_csv(out, index=False)
    fig_path = out.with_suffix(".png")
    save_comparison_figure(report["comparison"], str(fig_path))

    per_app_frames = []
    for name, payload in report["results"].items():
        df = payload.get("hii_per_application")
        if df is not None and not df.empty:
            part = df.copy()
            part.insert(0, "Model", name)
            per_app_frames.append(part)
    if per_app_frames:
        per_path = out.with_name(out.stem + "_per_application.csv")
        pd.concat(per_app_frames, ignore_index=True).to_csv(per_path, index=False)
        print(f"Per-app HII: {per_path}")

    print("\n" + "=" * 50)
    print("Dynamic Explainable AI Framework — Report")
    print("=" * 50)
    print(report["comparison"].to_string(index=False))
    print("\nSummary")
    for line in summarize(report):
        print(f"- {line}")
    print(f"\nSaved: {out}")
    print(f"Plot:  {fig_path}")


if __name__ == "__main__":
    main()
