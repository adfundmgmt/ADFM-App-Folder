"""Shared Streamlit presentation primitives for ADFM analytics pages."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Optional

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class PageHeader:
    """Content used by the shared ADFM page header."""

    title: str
    description: str
    eyebrow: str = "ADFM Analytics"
    as_of: Optional[str] = None
    source_note: Optional[str] = None


def inject_base_style(max_width_px: int = 1500) -> None:
    """Apply a compact, page-safe visual baseline without changing Streamlit theme."""
    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: {int(max_width_px)}px; padding-top: 1.25rem; padding-bottom: 1.5rem; }}
        .adfm-eyebrow {{ font-size: .71rem; letter-spacing: .11em; text-transform: uppercase; color: #64748b; font-weight: 750; }}
        .adfm-page-title {{ margin: .12rem 0 .34rem; font-size: 1.72rem; line-height: 1.18; color: #111827; font-weight: 780; }}
        .adfm-page-description {{ max-width: 1120px; color: #64748b; line-height: 1.52; font-size: .9rem; margin-bottom: .55rem; }}
        .adfm-status {{ color: #64748b; font-size: .78rem; margin: .1rem 0 1rem; }}
        @media (prefers-color-scheme: dark) {{ .adfm-page-title {{ color: #f8fafc; }} .adfm-page-description, .adfm-status {{ color: #cbd5e1; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(header: PageHeader) -> None:
    """Render a consistent page identity and transparent as-of/source status."""
    status = " · ".join(item for item in (header.as_of, header.source_note) if item)
    st.markdown(
        "<div class='adfm-eyebrow'>" + escape(header.eyebrow) + "</div>"
        "<div class='adfm-page-title'>" + escape(header.title) + "</div>"
        "<div class='adfm-page-description'>" + escape(header.description) + "</div>"
        + ("<div class='adfm-status'>" + escape(status) + "</div>" if status else ""),
        unsafe_allow_html=True,
    )


def dataframe_download(label: str, frame: pd.DataFrame, filename: str) -> None:
    """Offer a CSV export of underlying numeric values, never display strings."""
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_footer(text: str = "© 2026 AD Fund Management LP") -> None:
    """Render the standard discreet ADFM footer."""
    st.caption(text)
