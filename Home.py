from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image

from adfm_core.catalog import GROUP_ORDER, tool_definitions

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "ADFM_Logo_Naked.png"
PAGE_ICON = Image.open(LOGO_PATH).convert("RGBA")

st.set_page_config(
    page_title="ADFM Analytics",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


TOOLS = tool_definitions()
TOOLS_BY_GROUP = {
    group: [tool for tool in TOOLS if tool.group == group] for group in GROUP_ORDER
}


def logo_data_uri() -> str:
    """Return the approved ADFM shield as an embeddable image."""

    encoded = b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_tool(tool) -> None:
    """Render one catalog entry with Streamlit-native multipage navigation."""

    st.page_link(
        f"pages/{tool.page_filename}",
        label=tool.title,
        width="content",
    )
    st.markdown(
        f"<div class='tool-description'>{escape(tool.description)}</div>"
        "<div class='entry-rule'></div>",
        unsafe_allow_html=True,
    )


def render_group(group: str) -> None:
    """Render one full-width group in the sidebar's catalog order."""

    st.markdown(
        f"<div class='directory-group-title'>{escape(group)}</div>",
        unsafe_allow_html=True,
    )
    group_tools = TOOLS_BY_GROUP[group]
    for row_start in range(0, len(group_tools), 2):
        row_columns = st.columns(2, gap="large")
        for column_index, tool in enumerate(group_tools[row_start : row_start + 2]):
            with row_columns[column_index]:
                render_tool(tool)

st.markdown(
    """
    <style>
        :root {
            color-scheme: light;
        }

        html,
        body,
        .stApp,
        main,
        [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
            color: #000000;
        }

        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.98);
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #000000 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: #ffffff !important;
            padding-top: 1.25rem !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label {
            color: #303030 !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: .78rem !important;
            line-height: 1.5 !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2 {
            border-bottom: 2px solid #000000 !important;
            padding-bottom: .48rem !important;
            color: #000000 !important;
            font-size: .72rem !important;
            font-weight: 800 !important;
            letter-spacing: .13em !important;
            text-transform: uppercase !important;
        }

        section[data-testid="stSidebar"] button {
            border: 1px solid #000000 !important;
            border-radius: 0 !important;
            background: #ffffff !important;
            color: #000000 !important;
            box-shadow: none !important;
        }

        .block-container {
            max-width: 1240px;
            padding: 2.75rem 2.5rem 3rem;
        }

        .adfm-masthead {
            display: grid;
            grid-template-columns: 82px minmax(0, 1fr) auto;
            align-items: center;
            column-gap: 1.25rem;
            border-top: 3px solid #000000;
            border-bottom: 1px solid #000000;
            padding: 1.35rem 0 1.25rem;
        }

        .adfm-mark {
            display: block;
            width: 70px;
            height: 70px;
            object-fit: contain;
        }

        .firm-name {
            margin: 0 0 0.3rem;
            color: #000000;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            line-height: 1.2;
            text-transform: uppercase;
        }

        .adfm-title {
            margin: 0;
            color: #000000;
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2.05rem, 4vw, 3rem);
            font-weight: 400;
            letter-spacing: -0.035em;
            line-height: 1;
        }

        .adfm-subtitle {
            margin: 0.48rem 0 0;
            color: #3f3f3f;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.84rem;
            line-height: 1.4;
        }

        .research-label {
            align-self: start;
            margin-top: 0.15rem;
            color: #000000;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-align: right;
            text-transform: uppercase;
        }

        .directory-introduction {
            display: grid;
            grid-template-columns: minmax(240px, 0.8fr) minmax(360px, 1.2fr);
            gap: 3rem;
            align-items: end;
            padding: 2.4rem 0 1.65rem;
        }

        .directory-title {
            margin: 0;
            color: #000000;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.55rem;
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .directory-copy {
            margin: 0;
            color: #4a4a4a;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.82rem;
            line-height: 1.55;
            text-align: right;
        }

        .directory-group-title {
            border-bottom: 2px solid #000000;
            margin: 1.65rem 0 1rem;
            padding: 0 0 0.55rem;
            color: #000000;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            line-height: 1.2;
            text-transform: uppercase;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] {
            margin: 0;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] a {
            display: inline-flex !important;
            width: auto !important;
            min-height: 0 !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            padding: 0 !important;
            box-shadow: none !important;
            text-decoration: none !important;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] a p {
            margin: 0 !important;
            color: #000000 !important;
            font-family: Georgia, "Times New Roman", serif !important;
            font-size: 1.18rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.012em !important;
            line-height: 1.3 !important;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] a:hover p,
        [data-testid="stMain"] [data-testid="stPageLink"] a:focus-visible p {
            text-decoration: underline !important;
            text-decoration-thickness: 1px !important;
            text-underline-offset: 0.18em !important;
        }

        [data-testid="stMain"] [data-testid="stPageLink"] a:focus-visible {
            outline: 1px solid #000000 !important;
            outline-offset: 4px !important;
        }

        .tool-description {
            max-width: 34rem;
            margin-top: 0.38rem;
            color: #505050;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.79rem;
            line-height: 1.5;
        }

        .entry-rule {
            height: 1px;
            margin: 1rem 0 1.2rem;
            background: #d7d7d7;
        }

        .adfm-footer {
            display: flex;
            justify-content: space-between;
            gap: 2rem;
            border-top: 1px solid #000000;
            margin-top: 2.25rem;
            padding-top: 0.8rem;
            color: #333333;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.66rem;
            letter-spacing: 0.04em;
            line-height: 1.4;
            text-transform: uppercase;
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 1.5rem 1.15rem 2.25rem;
            }

            .adfm-masthead {
                grid-template-columns: 58px minmax(0, 1fr);
                column-gap: 0.9rem;
                padding: 1rem 0;
            }

            .adfm-mark {
                width: 52px;
                height: 52px;
            }

            .adfm-title {
                font-size: 2rem;
            }

            .research-label {
                display: none;
            }

            .directory-introduction {
                display: block;
                padding: 1.75rem 0 1.35rem;
            }

            .directory-copy {
                margin-top: 0.55rem;
                text-align: left;
            }

            .adfm-footer {
                display: block;
            }

            .adfm-footer span:last-child {
                display: block;
                margin-top: 0.3rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <header class="adfm-masthead">
        <img class="adfm-mark" src="{logo_data_uri()}" alt="AD Fund Management shield">
        <div>
            <div class="firm-name">AD Fund Management LP</div>
            <h1 class="adfm-title">ADFM Analytics</h1>
            <p class="adfm-subtitle">Proprietary market research and analytical tools.</p>
        </div>
        <div class="research-label">Internal Research</div>
    </header>
    <section class="directory-introduction">
        <h2 class="directory-title">Research Directory</h2>
        <p class="directory-copy">
            Choose a tool below. Each is built to examine a specific question
            across equities, market structure, macro, rates, liquidity,
            positioning, and risk.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


for group in GROUP_ORDER:
    render_group(group)
