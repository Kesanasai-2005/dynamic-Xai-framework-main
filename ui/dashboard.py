"""Summary dashboard: HII metrics + signed SHAP feature push."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from evaluation.model_comparison import summarize
from ui.theme import kpi_row, section, takeaways_panel
from visualization.explanation_plots import lime_fig, prediction_summary, shap_waterfall_fig

POS_COLOR = "#2a9d8f"
NEG_COLOR = "#e76f51"


def _pretty_feature(name: str) -> str:
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _metric_table(comparison: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Model",
        "Accuracy",
        "Consistency",
        "Stability",
        "Fidelity",
        "Feature Agreement",
        "HII",
    ]
    show = comparison[[c for c in cols if c in comparison.columns]].copy()
    rename = {
        "Consistency": "C",
        "Stability": "S",
        "Fidelity": "F",
        "Feature Agreement": "A",
    }
    show = show.rename(columns=rename)
    for c in show.columns:
        if c == "Model":
            continue
        if c == "Accuracy":
            show[c] = show[c].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        else:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    return show


def _csfa_bar_fig(comparison: pd.DataFrame):
    metrics = ["Consistency", "Stability", "Fidelity", "Feature Agreement", "HII"]
    models = comparison["Model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, model in enumerate(models):
        row = comparison.loc[comparison["Model"] == model].iloc[0]
        vals = [float(row[m]) for m in metrics]
        ax.bar(x + i * width, vals, width, label=model, color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["C", "S", "F", "A", "HII"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _feature_push_fig(signed_shap: dict, positive_label: str, negative_label: str, top_n: int = 12):
    if not signed_shap:
        return None
    items = sorted(signed_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    items = sorted(items, key=lambda kv: kv[1])
    names = [_pretty_feature(n) for n, _ in items]
    vals = [v for _, v in items]
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.32 * len(names) + 1)))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Mean SHAP")
    ax.set_title(f"→ {positive_label} (teal)   ← {negative_label} (coral)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _actual_label(report, row_idx: int) -> str:
    y_test = report.get("y_test")
    if y_test is None:
        return "—"
    y = y_test[row_idx]
    names = report["class_names"]
    try:
        return names[int(y)]
    except (ValueError, IndexError, TypeError):
        return str(y)


def _fmt_pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.0%}"


def _other_class(class_names: list[str], positive: str) -> str:
    others = [c for c in class_names if str(c) != str(positive)]
    return others[0] if others else "other"


def render_summary(report: dict, cfg: dict) -> None:
    comparison = report["comparison"]
    info = report["dataset_info"]
    model_names = list(report["results"].keys())

    bar = st.columns([5, 1])
    with bar[1]:
        if st.button("New analysis", width="stretch"):
            st.session_state.pop("report", None)
            st.switch_page("app.py")

    takeaways_panel(summarize(report), title="Key takeaways")

    section(
        "1 · Scoreboard",
        "Accuracy plus interpretability components (equal 25% weights → HII).",
    )
    st.dataframe(_metric_table(comparison), hide_index=True, width="stretch")
    fig = _csfa_bar_fig(comparison)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    section(
        "2 · Model detail",
        "C = SHAP↔LIME sign agreement · S = stability under noise · "
        "F = LIME local R² · A = top-feature overlap.",
    )
    selected = st.selectbox("Model", model_names, key="dash_model")
    payload = report["results"][selected]
    interp = payload["interpretability"]
    pos = str(payload.get("hii_class", report["class_names"][-1] if report["class_names"] else "1"))
    neg = _other_class([str(c) for c in report["class_names"]], pos)
    perf = payload["performance"]

    kpi_row(
        [
            ("Accuracy", _fmt_pct(perf.get("accuracy"))),
            ("HII", f"{interp['hii']:.3f}"),
            ("C Consistency", f"{interp['consistency']:.3f}"),
            ("S Stability", f"{interp['stability']:.3f}"),
            ("F Fidelity", f"{interp['fidelity']:.3f}"),
            ("A Agreement", f"{interp['feature_agreement']:.3f}"),
        ]
    )

    section(
        f"3 · What drives class “{pos}”",
        f"Teal bars push toward {pos}. Coral bars push toward {neg}.",
    )
    push_fig = _feature_push_fig(payload.get("signed_shap") or {}, pos, neg)
    if push_fig is not None:
        st.pyplot(push_fig, clear_figure=True)
        plt.close(push_fig)
    else:
        st.info("No signed SHAP available for this model.")

    with st.expander("Per-application HII rows"):
        per = payload.get("hii_per_application")
        if per is not None and not per.empty:
            st.dataframe(per, hide_index=True, width="stretch")
        else:
            st.write("No per-application rows.")

    section("4 · Local explanation", "One test row — SHAP waterfall and LIME.")
    _render_local(report, cfg, selected, pos)

    st.caption(
        f"Dataset · {info.get('train_rows', '?')} train · {info.get('test_rows', '?')} test · "
        f"target `{info.get('target')}`"
    )


def _render_local(report: dict, cfg: dict, default_model: str, default_class: str) -> None:
    n_test = len(report["X_test"])
    model_names = list(report["results"].keys())
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox(
            "Model",
            model_names,
            index=model_names.index(default_model) if default_model in model_names else 0,
            key="local_model",
        )
    with c2:
        row_idx = st.number_input(
            "Test row",
            min_value=0,
            max_value=max(n_test - 1, 0),
            value=0,
            step=1,
            key="local_row",
        )
    model = report["results"][model_name]["model"]
    model_class_names = [str(c) for c in model.classes_]
    summary = prediction_summary(model, report["X_test"][int(row_idx)], model_class_names)
    with c3:
        default_i = (
            summary["class_names"].index(default_class)
            if default_class in summary["class_names"]
            else summary["predicted_idx"]
        )
        explain_for = st.selectbox(
            "Explain class",
            summary["class_names"],
            index=default_i,
            key="local_class",
        )
    class_idx = summary["class_names"].index(explain_for)

    st.write(
        f"Predicted **{summary['predicted']}** · Actual **{_actual_label(report, int(row_idx))}** · "
        + " · ".join(f"{c} {p:.0%}" for c, p in summary["probabilities"].items())
    )
    st.dataframe(
        report["X_test_raw"].iloc[[int(row_idx)]].reset_index(drop=True),
        hide_index=True,
        width="stretch",
    )

    if st.button("Generate SHAP & LIME", type="primary", key="local_run"):
        feature_names = report["feature_names"]
        x_row = report["X_test"][int(row_idx)]
        lime_k = cfg.get("explainability", {}).get("lime_num_features", 10)
        with st.spinner("Computing explanations…"):
            shap_figure = shap_waterfall_fig(
                model,
                report["X_train"],
                x_row,
                feature_names,
                class_idx=class_idx,
                max_display=20,
            )
            lime_figure, lime_list = lime_fig(
                model,
                report["X_train"],
                x_row,
                feature_names,
                class_names=model_class_names,
                class_idx=class_idx,
                num_features=lime_k,
            )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**SHAP** ({explain_for})")
            st.pyplot(shap_figure, clear_figure=True)
        with col_b:
            st.markdown(f"**LIME** ({explain_for})")
            st.pyplot(lime_figure, clear_figure=True)
        st.dataframe(
            pd.DataFrame(lime_list, columns=["rule", "weight"]),
            hide_index=True,
            width="stretch",
        )
        plt.close("all")
