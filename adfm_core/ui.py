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


def inject_institutional_theme(max_width_px: int = 1560) -> None:
    """Apply the black-and-white ADFM research-book theme to a Streamlit page."""
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: light;
            --adfm-black: #000000;
            --adfm-ink: #171717;
            --adfm-muted: #555555;
            --adfm-rule: #c9c9c9;
            --adfm-soft: #f5f5f3;
            --adfm-white: #ffffff;
        }}

        html,
        body,
        .stApp,
        main,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background: var(--adfm-white) !important;
            color: var(--adfm-black) !important;
        }}

        [data-testid="stDecoration"] {{ display: none !important; }}

        header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, .98) !important;
            border-bottom: 1px solid #e5e5e5 !important;
        }}

        .block-container {{
            max-width: {int(max_width_px)}px !important;
            padding-top: 1.7rem !important;
            padding-bottom: 2rem !important;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--adfm-white) !important;
            border-right: 1px solid var(--adfm-black) !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            background: var(--adfm-white) !important;
            padding-top: 1.25rem !important;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2 {{
            border-bottom: 2px solid var(--adfm-black) !important;
            margin: 1.05rem 0 .75rem !important;
            padding: 0 0 .48rem !important;
            color: var(--adfm-black) !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: .72rem !important;
            font-weight: 800 !important;
            letter-spacing: .13em !important;
            line-height: 1.25 !important;
            text-transform: uppercase !important;
        }}

        section[data-testid="stSidebar"] h3 {{
            border-bottom: 1px solid var(--adfm-rule) !important;
            margin: 1rem 0 .6rem !important;
            padding-bottom: .38rem !important;
            color: var(--adfm-black) !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: .68rem !important;
            font-weight: 800 !important;
            letter-spacing: .1em !important;
            text-transform: uppercase !important;
        }}

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: #303030 !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: .78rem !important;
            line-height: 1.5 !important;
        }}

        section[data-testid="stSidebar"] strong {{ color: var(--adfm-black) !important; }}
        section[data-testid="stSidebar"] hr {{ border-color: var(--adfm-black) !important; }}

        main h1,
        [data-testid="stMain"] h1 {{
            border-top: 3px solid var(--adfm-black) !important;
            border-bottom: 1px solid var(--adfm-black) !important;
            margin: 0 0 .45rem !important;
            padding: .85rem 0 .75rem !important;
            color: var(--adfm-black) !important;
            font-family: Georgia, "Times New Roman", serif !important;
            font-size: clamp(2rem, 3.2vw, 2.65rem) !important;
            font-weight: 400 !important;
            letter-spacing: -.035em !important;
            line-height: 1.05 !important;
        }}

        main h2,
        [data-testid="stMain"] h2 {{
            border-bottom: 1px solid var(--adfm-black) !important;
            margin: 1.65rem 0 .75rem !important;
            padding-bottom: .42rem !important;
            color: var(--adfm-black) !important;
            font-family: Georgia, "Times New Roman", serif !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            letter-spacing: -.018em !important;
            line-height: 1.2 !important;
        }}

        main h3,
        [data-testid="stMain"] h3 {{
            margin: 1.25rem 0 .55rem !important;
            color: var(--adfm-black) !important;
            font-family: Georgia, "Times New Roman", serif !important;
            font-size: 1.08rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
        }}

        main p,
        main li,
        main label,
        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label {{
            color: var(--adfm-ink) !important;
        }}

        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: var(--adfm-muted) !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: .76rem !important;
            line-height: 1.48 !important;
        }}

        a {{ color: var(--adfm-black) !important; }}

        button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stDownloadButton"] button {{
            border: 1px solid var(--adfm-black) !important;
            border-radius: 0 !important;
            background: var(--adfm-white) !important;
            color: var(--adfm-black) !important;
            box-shadow: none !important;
            font-weight: 700 !important;
        }}

        button:hover,
        button:focus-visible,
        [data-testid="stDownloadButton"] button:hover {{
            background: var(--adfm-black) !important;
            color: var(--adfm-white) !important;
        }}

        [data-testid="stBaseButton-primary"] {{
            border: 1px solid var(--adfm-black) !important;
            border-radius: 0 !important;
            background: var(--adfm-black) !important;
            color: var(--adfm-white) !important;
            box-shadow: none !important;
        }}

        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input {{
            border-color: #8a8a8a !important;
            border-radius: 0 !important;
            background: var(--adfm-white) !important;
            color: var(--adfm-black) !important;
            box-shadow: none !important;
        }}

        [data-baseweb="tag"] {{
            border-radius: 0 !important;
            background: var(--adfm-black) !important;
            color: var(--adfm-white) !important;
        }}

        [data-testid="stExpander"] details {{
            border: 1px solid var(--adfm-rule) !important;
            border-radius: 0 !important;
            background: var(--adfm-white) !important;
            box-shadow: none !important;
        }}

        [data-testid="stExpander"] summary {{
            color: var(--adfm-black) !important;
            font-weight: 700 !important;
        }}

        [data-testid="stAlert"] {{
            border: 1px solid var(--adfm-black) !important;
            border-left-width: 4px !important;
            border-radius: 0 !important;
            background: var(--adfm-soft) !important;
            color: var(--adfm-black) !important;
        }}

        [data-testid="stMetric"] {{
            border-top: 1px solid var(--adfm-rule) !important;
            border-bottom: 1px solid var(--adfm-rule) !important;
            padding: .65rem 0 !important;
        }}

        [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"] {{ color: var(--adfm-muted) !important; }}
        [data-testid="stMetricValue"] {{ color: var(--adfm-black) !important; }}

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            border: 1px solid var(--adfm-rule) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }}

        div[data-testid="stPlotlyChart"],
        div[data-testid="stPyplot"] {{
            border-radius: 0 !important;
            background: var(--adfm-white) !important;
            box-shadow: none !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0 !important;
            border-bottom: 1px solid var(--adfm-black) !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 2.35rem !important;
            border-radius: 0 !important;
            padding: 0 .9rem !important;
            color: var(--adfm-muted) !important;
        }}

        .stTabs [aria-selected="true"] {{ color: var(--adfm-black) !important; }}

        [class*="card"],
        [class*="Card"],
        [class*="panel"],
        [class*="Panel"],
        [class*="banner"],
        [class*="Banner"],
        [class*="callout"],
        [class*="Callout"] {{
            border-radius: 0 !important;
            box-shadow: none !important;
        }}

        .adfm-page-header {{
            border-top: 3px solid var(--adfm-black);
            border-bottom: 1px solid var(--adfm-black);
            margin-bottom: 1rem;
            padding: .8rem 0 .75rem;
        }}

        .adfm-eyebrow {{
            margin-bottom: .32rem;
            color: var(--adfm-black) !important;
            font-family: Arial, Helvetica, sans-serif;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .15em;
            line-height: 1.2;
            text-transform: uppercase;
        }}

        .adfm-page-title {{
            margin: 0;
            color: var(--adfm-black) !important;
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2rem, 3.2vw, 2.65rem);
            font-weight: 400;
            letter-spacing: -.035em;
            line-height: 1.05;
        }}

        .adfm-page-description {{
            max-width: 1120px;
            margin: .52rem 0 0;
            color: #3f3f3f !important;
            font-family: Arial, Helvetica, sans-serif;
            font-size: .84rem;
            line-height: 1.5;
        }}

        .adfm-status {{
            margin: .45rem 0 0;
            color: var(--adfm-muted) !important;
            font-family: Arial, Helvetica, sans-serif;
            font-size: .72rem;
            line-height: 1.4;
        }}

        .adfm-footer {{
            display: flex;
            justify-content: space-between;
            gap: 2rem;
            border-top: 1px solid var(--adfm-black);
            margin-top: 2rem;
            padding-top: .72rem;
            color: #333333;
            font-family: Arial, Helvetica, sans-serif;
            font-size: .65rem;
            letter-spacing: .03em;
            line-height: 1.45;
            text-transform: uppercase;
        }}

        .adfm-footer-note {{ max-width: 1080px; }}
        .adfm-footer-firm {{ white-space: nowrap; }}

        @media (max-width: 760px) {{
            .block-container {{ padding: 1.25rem 1rem 1.75rem !important; }}
            main h1, [data-testid="stMain"] h1, .adfm-page-title {{ font-size: 1.9rem !important; }}
            .adfm-footer {{ display: block; }}
            .adfm-footer-firm {{ display: block; margin-top: .4rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_base_style(max_width_px: int = 1500) -> None:
    """Apply the shared institutional visual baseline."""
    inject_institutional_theme(max_width_px=max_width_px)


def inject_explorer_style(max_width_px: int = 1560) -> None:
    """Apply the shared visual language used by the analytical explorer pages."""
    inject_base_style(max_width_px=max_width_px)
    st.markdown(
        """
        <style>
        html, body, .stApp, main, [data-testid="stAppViewContainer"] { background: #ffffff !important; }
        header[data-testid="stHeader"] { background: rgba(255,255,255,.98) !important; border-bottom: 1px solid #e5e5e5 !important; }
        section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #000000 !important; }
        .adfm-kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: .7rem; margin: .45rem 0 1rem; }
        .adfm-kpi-card { background: #ffffff; border: 1px solid #bdbdbd; border-radius: 0; padding: 12px 14px; min-height: 96px; box-shadow: none; }
        .adfm-kpi-label { color: #555555; font-size: .68rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .42rem; }
        .adfm-kpi-value { color: #000000; font-size: 1.17rem; font-weight: 780; line-height: 1.14; }
        .adfm-kpi-note { color: #555555; font-size: .73rem; line-height: 1.32; margin-top: .38rem; }
        .adfm-selection-note { background: #ffffff; border: 1px solid #bdbdbd; border-left: 4px solid #000000; border-radius: 0; padding: 13px 15px; margin: .35rem 0 1rem; color: #171717; line-height: 1.5; box-shadow: none; }
        .adfm-selection-label { color: #555555; font-size: .69rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .3rem; }
        .adfm-section-title { color: #000000; font-family: Georgia, "Times New Roman", serif; font-size: 1rem; font-weight: 800; letter-spacing: -.01em; margin: 1rem 0 .24rem; }
        .adfm-section-subtitle { color: #555555; font-size: .81rem; line-height: 1.42; margin-bottom: .62rem; }
        div[data-testid="stDataFrame"] { border: 1px solid #bdbdbd; border-radius: 0; overflow: hidden; }
        div[data-testid="stPlotlyChart"] { background: #ffffff; border-radius: 0; }
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
        "<header class='adfm-page-header'>"
        "<div class='adfm-eyebrow'>" + escape(header.eyebrow) + "</div>"
        "<div class='adfm-page-title'>" + escape(header.title) + "</div>"
        "<div class='adfm-page-description'>"
        + escape(header.description)
        + "</div>"
        + ("<div class='adfm-status'>" + escape(status) + "</div>" if status else "")
        + "</header>",
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
    st.markdown(
        "<footer class='adfm-footer'>"
        + (
            "<span class='adfm-footer-note'>" + escape(data_note) + "</span>"
            if data_note
            else "<span></span>"
        )
        + "<span class='adfm-footer-firm'>"
        + escape(text)
        + "</span></footer>",
        unsafe_allow_html=True,
    )
