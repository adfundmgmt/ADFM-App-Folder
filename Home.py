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


def current_page_path(page_filename: str) -> str:
    """Resolve a catalog page after a numeric-prefix reorder or hot deploy."""

    _, separator, stable_name = page_filename.partition("_")
    if separator:
        matches = sorted((ROOT / "pages").glob(f"*_{stable_name}"))
        if len(matches) == 1:
            return matches[0].relative_to(ROOT).as_posix()
    return f"pages/{page_filename}"


def render_tool(tool) -> None:
    """Render one fully clickable catalog entry with native page routing."""

    with st.container(key=f"directory_entry_{tool.number}"):
        st.page_link(
            current_page_path(tool.page_filename),
            label=f"**{tool.title}**",
            width="content",
        )
        st.markdown(
            f"<div class='tool-description'>{escape(tool.description)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='entry-rule'></div>", unsafe_allow_html=True)


def render_group(group: str, *, first: bool = False) -> None:
    """Render a naturally sized directory group in catalog order."""

    title_class = "directory-group-title directory-group-title--first" if first else "directory-group-title"
    st.markdown(
        f"<div class='{title_class}'>{escape(group)}</div>",
        unsafe_allow_html=True,
    )

    group_tools = TOOLS_BY_GROUP[group]
    if len(group_tools) == 1:
        render_tool(group_tools[0])
        return

    for row_start in range(0, len(group_tools), 2):
        row_columns = st.columns(2, gap="large")
        for column_index, tool in enumerate(group_tools[row_start : row_start + 2]):
            with row_columns[column_index]:
                render_tool(tool)


st.html(
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
            height: 3rem !important;
            background: rgba(255, 255, 255, 0.98);
        }

        [data-testid="stToolbar"] {
            height: 100% !important;
        }

        [data-testid="stToolbarActions"],
        [data-testid="stMainMenu"],
        [data-testid="stDecoration"] {
            display: none !important;
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
            max-width: 1180px;
            padding: calc(3.25rem + env(safe-area-inset-top, 0px)) 2rem 3rem;
        }

        .block-container > [data-testid="stVerticalBlock"],
        .block-container [data-testid="stColumn"] [data-testid="stVerticalBlock"],
        [class*="st-key-directory_entry_"] > [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }

        .adfm-masthead {
            display: grid;
            grid-template-columns: 56px minmax(0, 1fr);
            align-items: center;
            column-gap: 0.9rem;
            border-top: 3px solid #000000;
            border-bottom: 1px solid #000000;
            padding: 0.7rem 0 0.75rem;
            overflow: visible;
        }

        .adfm-mark {
            display: block;
            width: 48px;
            height: 48px;
            object-fit: contain;
        }

        .firm-name {
            margin: 0 0 0.24rem;
            color: #000000;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            line-height: 1.2;
            text-transform: uppercase;
            white-space: normal;
            overflow: visible;
        }

        .adfm-title {
            margin: 0 !important;
            padding: 0 !important;
            color: #000000;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: clamp(2rem, 3vw, 2.2rem) !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em;
            line-height: 0.98 !important;
            white-space: normal !important;
            overflow: visible !important;
        }

        .adfm-subtitle {
            margin: 0.3rem 0 0;
            color: #414141;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.88rem;
            line-height: 1.35;
        }

        .directory-group-title {
            border-bottom: 2px solid #000000;
            margin: 2rem 0 0.9rem;
            padding: 0 0 0.55rem;
            color: #000000;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            line-height: 1.2;
            text-transform: uppercase;
        }

        .directory-group-title--first {
            margin-top: 2.25rem;
        }

        [class*="st-key-directory_entry_"] {
            position: relative;
            height: 100%;
            padding: 0.9rem 0 1.15rem;
            cursor: pointer;
        }

        [class*="st-key-directory_entry_"] > [data-testid="stVerticalBlock"] {
            height: 100%;
        }

        div[data-testid="stPageLink"] {
            margin: 0;
            padding-right: 2rem;
        }

        div[data-testid="stPageLink"]::after {
            content: "→";
            position: absolute;
            top: 0.14rem;
            right: 0;
            color: #777777;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.95rem;
            line-height: 1;
            transition: color 120ms ease, transform 120ms ease;
        }

        div[data-testid="stPageLink"] a {
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

        div[data-testid="stPageLink"] a::after {
            content: "";
            position: absolute;
            inset: 0;
            z-index: 1;
        }

        div[data-testid="stPageLink"] p,
        div[data-testid="stPageLink"] p strong {
            margin: 0 !important;
            color: #000000 !important;
            font-family: Georgia, "Times New Roman", serif !important;
            font-size: 1.28rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.014em !important;
            line-height: 1.25 !important;
        }

        [class*="st-key-directory_entry_"]:hover div[data-testid="stPageLink"] p,
        [class*="st-key-directory_entry_"]:hover div[data-testid="stPageLink"] p strong,
        div[data-testid="stPageLink"] a:focus-visible p,
        div[data-testid="stPageLink"] a:focus-visible p strong {
            text-decoration: underline !important;
            text-decoration-thickness: 1px !important;
            text-underline-offset: 0.18em !important;
        }

        [class*="st-key-directory_entry_"]:hover div[data-testid="stPageLink"]::after {
            color: #000000;
            transform: translateX(2px);
        }

        div[data-testid="stPageLink"] a:focus-visible::after {
            outline: 1px solid #000000;
            outline-offset: 4px;
        }

        .tool-description {
            max-width: 42rem;
            margin-top: 0.5rem;
            color: #505050;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        [class*="st-key-directory_entry_"] [data-testid="stElementContainer"]:has(.entry-rule) {
            margin-top: auto !important;
        }

        .entry-rule {
            height: 1px;
            margin-top: 1rem;
            background: #d7d7d7;
        }

        @media (max-width: 760px) {
            header[data-testid="stHeader"] {
                height: 2.6rem !important;
            }

            .block-container {
                max-width: none;
                padding: calc(3.2rem + env(safe-area-inset-top, 0px)) 1rem 2.25rem;
            }

            .adfm-masthead {
                grid-template-columns: 46px minmax(0, 1fr);
                column-gap: 0.72rem;
                margin-top: 0.3rem;
                padding: 0.55rem 0 0.65rem;
            }

            .adfm-mark {
                width: 43px;
                height: 43px;
            }

            .firm-name {
                margin-bottom: 0.18rem;
                font-size: 0.61rem;
                letter-spacing: 0.15em;
            }

            .adfm-title {
                font-size: clamp(1.68rem, 7.3vw, 1.9rem) !important;
                line-height: 1 !important;
            }

            .adfm-subtitle {
                margin-top: 0.22rem;
                font-size: 0.81rem;
                line-height: 1.35;
            }

            .directory-group-title {
                margin: 1.7rem 0 0.8rem;
                padding-bottom: 0.5rem;
                font-size: 0.68rem;
            }

            .directory-group-title--first {
                margin-top: 2.2rem;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0 !important;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:not(:has([class*="st-key-directory_entry_"])) {
                display: none !important;
            }

            [class*="st-key-directory_entry_"] {
                padding: 0.78rem 0 1.1rem;
            }

            div[data-testid="stPageLink"] p,
            div[data-testid="stPageLink"] p strong {
                font-size: 1.22rem !important;
                line-height: 1.28 !important;
            }

            .tool-description {
                max-width: none;
                margin-top: 0.42rem;
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .entry-rule {
                margin-top: 0.9rem;
            }
        }
    </style>
    """,
)


st.markdown(
    f"""
    <header class="adfm-masthead" data-home-revision="2026-08-22-responsive-directory-v1">
        <img class="adfm-mark" src="{logo_data_uri()}" alt="AD Fund Management shield">
        <div>
            <div class="firm-name">AD Fund Management LP</div>
            <h1 class="adfm-title">ADFM Analytics</h1>
            <p class="adfm-subtitle">Market research and analytical tools.</p>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)


for group_index, group in enumerate(GROUP_ORDER):
    render_group(group, first=group_index == 0)
