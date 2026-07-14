"""Shared Streamlit presentation primitives for ADFM analytics pages."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import streamlit as st

from .catalog import tool_for_page


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


def render_status_line(**items: object) -> None:
    """Render compact, escaped status metadata while omitting unavailable values."""
    parts = [
        f"{escape(label.replace('_', ' ').title())}: {escape(str(value))}"
        for label, value in items.items()
        if value not in (None, "")
    ]
    if parts:
        st.markdown("<div class='adfm-status'>" + " · ".join(parts) + "</div>", unsafe_allow_html=True)


def metric_table(frame: pd.DataFrame, column_config: Optional[Mapping[str, object]] = None) -> None:
    """Render a consistent full-width metric table without transforming values."""
    st.dataframe(frame, use_container_width=True, hide_index=True, column_config=dict(column_config or {}))


def dataframe_download(label: str, frame: pd.DataFrame, filename: str) -> None:
    """Offer a CSV export of underlying numeric values, never display strings."""
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_footer(text: str = "© 2026 AD Fund Management LP", data_note: Optional[str] = None) -> None:
    """Render a standard source/data-policy disclosure and discreet ADFM footer."""
    if data_note is None:
        caller = inspect.currentframe().f_back
        caller_file = Path(str(caller.f_globals.get("__file__", ""))).name if caller is not None else ""
        tool = tool_for_page(caller_file)
        if tool is not None:
            data_note = (
                f"Primary inputs: {tool.primary_inputs}. Data dates and benchmarks are shown above when applicable. "
                "Missing observations remain unavailable rather than being fabricated."
            )
    if data_note:
        st.caption(data_note)
    st.caption(text)
