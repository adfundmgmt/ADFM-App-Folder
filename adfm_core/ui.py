"""Shared Streamlit presentation primitives for ADFM analytics pages."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping, Optional, Sequence

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


def inject_explorer_style(max_width_px: int = 1560) -> None:
    """Apply the shared visual language used by the analytical explorer pages."""
    inject_base_style(max_width_px=max_width_px)
    st.markdown(
        """
        <style>
        html, body, .stApp, main, [data-testid="stAppViewContainer"] { background: #ffffff; }
        header[data-testid="stHeader"] { background: rgba(255,255,255,.96); border-bottom: 1px solid #f1f5f9; }
        section[data-testid="stSidebar"] { background: #fafaf9; border-right: 1px solid #e7e5e4; }
        .adfm-kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: .7rem; margin: .45rem 0 1rem; }
        .adfm-kpi-card { background: #fafaf8; border: 1px solid #e7e5e4; border-radius: 12px; padding: 12px 14px; min-height: 96px; }
        .adfm-kpi-label { color: #78716c; font-size: .68rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .42rem; }
        .adfm-kpi-value { color: #171717; font-size: 1.17rem; font-weight: 780; line-height: 1.14; }
        .adfm-kpi-note { color: #78716c; font-size: .73rem; line-height: 1.32; margin-top: .38rem; }
        .adfm-selection-note { background: #fcfcfb; border: 1px solid #e7e5e4; border-left: 4px solid #315f95; border-radius: 12px; padding: 13px 15px; margin: .35rem 0 1rem; color: #292524; line-height: 1.5; }
        .adfm-selection-label { color: #78716c; font-size: .69rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .3rem; }
        .adfm-section-title { color: #171717; font-size: 1rem; font-weight: 800; letter-spacing: -.01em; margin: 1rem 0 .24rem; }
        .adfm-section-subtitle { color: #78716c; font-size: .81rem; line-height: 1.42; margin-bottom: .62rem; }
        div[data-testid="stDataFrame"] { border: 1px solid #e7e5e4; border-radius: 11px; overflow: hidden; }
        div[data-testid="stPlotlyChart"] { background: #ffffff; border-radius: 12px; }
        .stTabs [data-baseweb="tab-list"] { gap: .45rem; }
        .stTabs [data-baseweb="tab"] { height: 2.35rem; padding: 0 .85rem; }
        @media (max-width: 1250px) { .adfm-kpi-grid { grid-template-columns: repeat(3, minmax(0,1fr)); } }
        @media (max-width: 760px) { .adfm-kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: Sequence[tuple[str, str, str]]) -> None:
    """Render a responsive strip of compact decision-oriented KPI cards."""
    body = []
    for label, value, note in cards:
        body.append(
            "<div class='adfm-kpi-card'>"
            f"<div class='adfm-kpi-label'>{escape(str(label))}</div>"
            f"<div class='adfm-kpi-value'>{escape(str(value))}</div>"
            f"<div class='adfm-kpi-note'>{escape(str(note))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='adfm-kpi-grid'>" + "".join(body) + "</div>", unsafe_allow_html=True
    )


def render_selection_note(label: str, text: str) -> None:
    """Render the active read or selection as the page's primary narrative cue."""
    st.markdown(
        "<div class='adfm-selection-note'>"
        f"<div class='adfm-selection-label'>{escape(str(label))}</div>"
        f"<div>{escape(str(text))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str) -> None:
    """Render a compact chart-section heading with enough metric context."""
    st.markdown(
        f"<div class='adfm-section-title'>{escape(str(title))}</div>"
        f"<div class='adfm-section-subtitle'>{escape(str(subtitle))}</div>",
        unsafe_allow_html=True,
    )


def render_page_header(header: PageHeader) -> None:
    """Render a consistent page identity and transparent as-of/source status."""
    status = " · ".join(item for item in (header.as_of, header.source_note) if item)
    st.markdown(
        "<div class='adfm-eyebrow'>" + escape(header.eyebrow) + "</div>"
        "<div class='adfm-page-title'>" + escape(header.title) + "</div>"
        "<div class='adfm-page-description'>"
        + escape(header.description)
        + "</div>"
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
        st.markdown(
            "<div class='adfm-status'>" + " · ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


def metric_table(
    frame: pd.DataFrame, column_config: Optional[Mapping[str, object]] = None
) -> None:
    """Render a consistent full-width metric table without transforming values."""
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config=dict(column_config or {}),
    )


def dataframe_download(label: str, frame: pd.DataFrame, filename: str) -> None:
    """Offer a CSV export of underlying numeric values, never display strings."""
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_footer(
    text: str = "© 2026 AD Fund Management LP", data_note: Optional[str] = None
) -> None:
    """Render a standard source/data-policy disclosure and discreet ADFM footer."""
    if data_note is None:
        caller = inspect.currentframe().f_back
        caller_file = (
            Path(str(caller.f_globals.get("__file__", ""))).name
            if caller is not None
            else ""
        )
        tool = tool_for_page(caller_file)
        if tool is not None:
            data_note = (
                f"Primary inputs: {tool.primary_inputs}. Data dates and benchmarks are shown above when applicable. "
                "Missing observations remain unavailable rather than being fabricated."
            )
    if data_note:
        st.caption(data_note)
    st.caption(text)
