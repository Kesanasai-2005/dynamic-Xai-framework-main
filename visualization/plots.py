"""Simple plotting helpers for performance and HII comparison."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_performance(comparison_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    df = comparison_df.copy()
    ax.barh(df["Model"], df["Accuracy"], color="#2a6f97")
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Performance")
    ax.set_xlim(0, 1)
    return ax


def plot_hii(comparison_df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    df = comparison_df.copy()
    ax.barh(df["Model"], df["HII"], color="#e09f3e")
    ax.set_xlabel("Hybrid Interpretability Index")
    ax.set_title("Interpretability (HII)")
    ax.set_xlim(0, 1)
    return ax


def plot_metric_breakdown(interp: dict, model_name: str = "", ax=None):
    ax = ax or plt.gca()
    keys = ["consistency", "stability", "fidelity", "feature_agreement"]
    labels = ["Consistency", "Stability", "Fidelity", "Feature Agreement"]
    vals = [interp.get(k, 0) for k in keys]
    ax.bar(labels, vals, color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Explanation Metrics — {model_name}".strip(" —"))
    ax.tick_params(axis="x", rotation=20)
    return ax


def save_comparison_figure(comparison_df: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_performance(comparison_df, axes[0])
    plot_hii(comparison_df, axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
