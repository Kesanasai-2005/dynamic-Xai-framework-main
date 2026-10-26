"""
Upload landing page.
After the pipeline finishes, navigates to /summary.
Run: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from main import run_pipeline
from preprocessing.preprocess import validate_test
from ui.theme import apply, hero

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"

st.set_page_config(
    page_title="Dynamic XAI — Upload",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)
apply()


def load_config():
    with open(DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def _save_and_read(upload, name: str):
    path = ROOT / "data" / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getvalue())
    try:
        return str(path), pd.read_csv(path), None
    except Exception as exc:
        return str(path), None, f"Could not read {upload.name}: {exc}"


# Already have results → go straight to summary
if "report" in st.session_state and st.query_params.get("stay") != "1":
    st.switch_page("pages/summary.py")

cfg = load_config()
hero(
    "Dynamic Explainable AI",
    "Upload train and test CSVs to score models and compute Hybrid Interpretability Index.",
    chips=["SHAP", "LIME", "HII · C S F A"],
)

uploaded = st.file_uploader("Train dataset (CSV)", type=["csv"], key="train_csv")
uploaded_test = st.file_uploader(
    "Test dataset (CSV)",
    type=["csv"],
    key="test_csv",
    help="Must have the same feature columns as train. Target column is optional.",
)

errors = []
dataset_path = test_path = preview = test_df = None
if uploaded is not None:
    dataset_path, preview, err = _save_and_read(uploaded, "_upload.csv")
    errors += [err] if err else []
if uploaded_test is not None:
    test_path, test_df, err = _save_and_read(uploaded_test, "_upload_test.csv")
    errors += [err] if err else []

cols = list(preview.columns) if preview is not None else []
default_target = cfg["dataset"]["target"]


def _target_index(columns: list[str], preferred: str) -> int | None:
    if not columns:
        return None
    if preferred in columns:
        return columns.index(preferred)
    for name in ("target", "label", "y", "class", "loan_status"):
        if name in columns:
            return columns.index(name)
    return len(columns) - 1


# New key when columns change so a stale selection (e.g. old default) can't stick.
_tgt_key = f"target_col_{hash(tuple(cols))}" if cols else "target_col_empty"
target = st.selectbox(
    "Target column",
    cols,
    index=_target_index(cols, default_target),
    key=_tgt_key,
    disabled=not cols,
    placeholder="Upload train data first",
)

st.markdown("**Models to evaluate**")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    use_lr = st.checkbox("Logistic Regression", value=True)
with mc2:
    use_rf = st.checkbox("Random Forest", value=True)
with mc3:
    use_dt = st.checkbox("Decision Tree", value=True)

if preview is not None and test_df is not None and target is not None:
    errors += validate_test(preview, test_df, target)
    if not errors:
        st.success(
            f"Ready — {preview.shape[1]} columns · "
            f"train {preview.shape[0]} rows · test {test_df.shape[0]} rows."
        )
for err in errors:
    st.error(err)

selected = []
if use_lr:
    selected.append("Logistic Regression")
if use_rf:
    selected.append("Random Forest")
if use_dt:
    selected.append("Decision Tree")

ready = (
    uploaded is not None
    and uploaded_test is not None
    and target is not None
    and not errors
    and bool(selected)
)
run = st.button("Run analysis", type="primary", width="stretch", disabled=not ready)

if run:
    with st.spinner("Training models and computing per-application HII…"):
        report = run_pipeline(
            dataset_path,
            target,
            model_names=selected,
            config=cfg,
            test_path=test_path,
        )
    st.session_state["report"] = report
    st.switch_page("pages/summary.py")
