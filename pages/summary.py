"""
Results dashboard at /summary.
Requires st.session_state['report'] from the upload page.
"""

from pathlib import Path

import streamlit as st
import yaml

from ui.dashboard import render_summary
from ui.theme import apply, hero

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"

st.set_page_config(
    page_title="Dynamic XAI — Summary",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply()


def load_config():
    with open(DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


if "report" not in st.session_state:
    st.warning("No analysis yet. Upload data to get started.")
    if st.button("Go to upload", type="primary"):
        st.switch_page("app.py")
    st.stop()

report = st.session_state["report"]
info = report["dataset_info"]
n_models = len(report["results"])
best = report["comparison"].iloc[0] if not report["comparison"].empty else None

hero(
    "Results",
    "Clear takeaways first, then scores, drivers, and local explanations.",
    chips=[
        f"{n_models} models",
        f"{info.get('test_rows', '?')} test rows",
        f"Best HII · {best['Model']} ({best['HII']:.2f})" if best is not None else "HII",
    ],
)

render_summary(report, load_config())
