"""Shared UI helpers to keep page headers consistent."""
from __future__ import annotations

from typing import Optional

import streamlit as st

BASE_STYLE = """
<style>
:root {
  --border-subtle: #e5e7eb;
  --text-muted: #5f6368;
  --text-body: #2d2f31;
}

div.block-container {
  padding-top: 1.25rem;
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.page-title {
  margin: 0;
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: -0.01rem;
}

.page-subtitle {
  margin: 0.2rem 0 0.45rem 0;
  color: var(--text-muted);
  font-size: 1rem;
}

.page-description {
  margin: 0;
  color: var(--text-body);
  font-size: 0.95rem;
  line-height: 1.55;
}

.compact-divider {
  border: 0;
  border-top: 1px solid var(--border-subtle);
  margin: 0 0 1.15rem 0;
}

.section-label {
  margin: 0 0 0.5rem 0;
  font-size: 1.05rem;
  font-weight: 700;
}
</style>
"""


def inject_base_style() -> None:
    """Apply a light styling layer once per session."""
    if st.session_state.get("_base_style_applied"):
        return
    st.markdown(BASE_STYLE, unsafe_allow_html=True)
    st.session_state["_base_style_applied"] = True


def render_page_header(
    title: str,
    subtitle: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    """Render a consistent page header with an optional subtitle and description."""
    inject_base_style()
    subtitle_html = f"<p class='page-subtitle'>{subtitle}</p>" if subtitle else ""
    description_html = f"<div class='page-description'>{description}</div>" if description else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <h1 class="page-title">{title}</h1>
                {subtitle_html}
                {description_html}
            </div>
        </div>
        <hr class="compact-divider">
        """,
        unsafe_allow_html=True,
    )


def section_title(label: str, description: Optional[str] = None) -> None:
    """Standardized section label used within body content."""
    inject_base_style()
    desc_html = f"<div class='page-description'>{description}</div>" if description else ""
    st.markdown(
        f"""
        <div class="section-label">{label}</div>
        {desc_html}
        """,
        unsafe_allow_html=True,
    )
