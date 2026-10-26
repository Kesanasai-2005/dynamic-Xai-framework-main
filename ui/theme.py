"""Shared Streamlit chrome for upload + summary pages."""

import html

import streamlit as st

CSS = """
<style>
  [data-testid="stSidebarNav"] { display: none; }
  .block-container { padding-top: 1.25rem; max-width: 1100px; }

  /* Custom KPI cards — own colors, immune to Streamlit dark-mode text */
  .dxai-kpi-row {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0.35rem 0 1rem 0;
  }
  @media (max-width: 900px) {
    .dxai-kpi-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  }
  .dxai-kpi {
    background: #f4f7f8;
    border: 1px solid #d7e0e4;
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
  }
  .dxai-kpi .dxai-kpi-label {
    color: #5a6a72 !important;
    -webkit-text-fill-color: #5a6a72 !important;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
  }
  .dxai-kpi .dxai-kpi-value {
    color: #0f3d4c !important;
    -webkit-text-fill-color: #0f3d4c !important;
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0;
  }

  .dxai-hero {
    background: linear-gradient(135deg, #0f3d4c 0%, #1a6b7c 55%, #2a9d8f 100%);
    color: #f7fafb;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
  }
  .dxai-hero h1 { color: #fff; margin: 0 0 0.35rem 0; font-size: 1.6rem; }
  .dxai-hero p { margin: 0; opacity: 0.9; }
  .dxai-chip {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    margin: 0.35rem 0.35rem 0 0;
    font-size: 0.85rem;
  }

  .dxai-panel {
    background: #ffffff;
    border: 1px solid #d7e0e4;
    border-radius: 14px;
    padding: 1.1rem 1.25rem 1rem;
    margin: 0.5rem 0 1.25rem 0;
    box-shadow: 0 1px 2px rgba(15, 61, 76, 0.04);
  }
  .dxai-panel h3 {
    margin: 0 0 0.85rem 0;
    font-size: 1.05rem;
    color: #0f3d4c;
    letter-spacing: 0.01em;
  }
  .dxai-panel ol {
    margin: 0;
    padding: 0;
    list-style: none;
    counter-reset: takeaway;
  }
  .dxai-panel li {
    counter-increment: takeaway;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.7rem 0;
    border-top: 1px solid #e8eef0;
    color: #1c2b33;
    line-height: 1.45;
    font-size: 0.98rem;
  }
  .dxai-panel li:first-child { border-top: none; padding-top: 0.15rem; }
  .dxai-panel li::before {
    content: counter(takeaway);
    flex: 0 0 1.6rem;
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    background: #0f3d4c;
    color: #fff;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.1rem;
  }
  .dxai-panel li > .dxai-body {
    flex: 1 1 auto;
    min-width: 0;
  }
  .dxai-panel li.dxai-rec {
    background: #e8f5f3;
    border: 1px solid #b7dfd8;
    border-radius: 10px;
    padding: 0.85rem 0.9rem;
    margin-top: 0.55rem;
  }
  .dxai-panel li.dxai-rec::before { background: #2a9d8f; }
  .dxai-panel .dxai-label {
    display: inline-block;
    font-weight: 700;
    color: #0f3d4c;
    min-width: 7.5rem;
  }
  .dxai-section {
    margin: 1.5rem 0 0.6rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #e8eef0;
    color: #0f3d4c;
    font-size: 1.1rem;
    font-weight: 700;
  }
  .dxai-hint { color: #5a6a72; font-size: 0.9rem; margin: 0 0 0.75rem 0; }

  @media (prefers-color-scheme: dark) {
    .dxai-panel {
      background: #1a2429;
      border-color: #2f3d45;
    }
    .dxai-panel h3, .dxai-panel .dxai-label, .dxai-section { color: #e8f0f2; }
    .dxai-panel li { color: #d5dee2; border-top-color: #2f3d45; }
    .dxai-panel li.dxai-rec {
      background: #14353a;
      border-color: #2a9d8f55;
    }
    .dxai-hint { color: #9aabaf; }
  }
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = "".join(
        f'<span class="dxai-chip">{html.escape(c)}</span>' for c in (chips or [])
    )
    st.markdown(
        f'<div class="dxai-hero"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p>"
        f'<div style="margin-top:0.55rem">{chips_html}</div></div>',
        unsafe_allow_html=True,
    )


def takeaways_panel(lines: list[str], title: str = "Summary") -> None:
    """Numbered executive summary; last Recommendation line is highlighted."""
    items = []
    for line in lines:
        safe = html.escape(line)
        # Bold the "Label —" lead-in when present
        if " — " in safe:
            label, rest = safe.split(" — ", 1)
            safe = f'<span class="dxai-label">{label}</span> {rest}'
        cls = "dxai-rec" if line.startswith("Recommendation") else ""
        items.append(f'<li class="{cls}"><span class="dxai-body">{safe}</span></li>')
    st.markdown(
        f'<div class="dxai-panel"><h3>{html.escape(title)}</h3>'
        f"<ol>{''.join(items)}</ol></div>",
        unsafe_allow_html=True,
    )


def section(title: str, hint: str | None = None) -> None:
    st.markdown(f'<div class="dxai-section">{html.escape(title)}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<p class="dxai-hint">{html.escape(hint)}</p>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str]]) -> None:
    """Render label/value KPI cards with fixed readable colors."""
    cells = "".join(
        f'<div class="dxai-kpi">'
        f'<p class="dxai-kpi-label">{html.escape(label)}</p>'
        f'<p class="dxai-kpi-value">{html.escape(value)}</p>'
        f"</div>"
        for label, value in items
    )
    st.markdown(f'<div class="dxai-kpi-row">{cells}</div>', unsafe_allow_html=True)
