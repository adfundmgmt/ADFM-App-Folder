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
    initial_sidebar_state="expanded",
)

TOOLS = tool_definitions()
TOOLS_BY_GROUP = {
    group: [tool for tool in TOOLS if tool.group == group] for group in GROUP_ORDER
}


def legacy_url_path(page_filename: str) -> str:
    stem = Path(page_filename).stem
    _, separator, stable_name = stem.partition("_")
    return stable_name if separator else stem


def logo_data_uri() -> str:
    encoded = b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_tool(tool) -> None:
    page = NAV_PAGE_BY_FILENAME[tool.page_filename]
    with st.container(key=f"directory_entry_{tool.number}"):
        st.page_link(page, label=f"**{tool.title}**", width="content")
        st.markdown(
            f"<div class='tool-description'>{escape(tool.description)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='entry-rule'></div>", unsafe_allow_html=True)


def render_group(group: str, *, first: bool = False) -> None:
    title_class = (
        "directory-group-title directory-group-title--first"
        if first
        else "directory-group-title"
    )
    st.markdown(
        f"<div class='{title_class}'>{escape(group)}</div>", unsafe_allow_html=True
    )
    group_tools = TOOLS_BY_GROUP[group]
    if len(group_tools) == 1:
        render_tool(group_tools[0])
        return
    for row_start in range(0, len(group_tools), 2):
        columns = st.columns(2, gap="large")
        for column_index, tool in enumerate(group_tools[row_start : row_start + 2]):
            with columns[column_index]:
                render_tool(tool)


def render_home() -> None:
    st.html(
        """
        <style>
        :root { color-scheme: light; }
        html, body, .stApp, main, [data-testid="stAppViewContainer"] {
            background: #ffffff !important; color: #000000;
        }
        header[data-testid="stHeader"] { background: rgba(255,255,255,.98); }
        [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDecoration"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff !important; border-right: 1px solid #000000 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: #ffffff !important; padding-top: 1.25rem !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label {
            color: #303030 !important; font-family: Arial, Helvetica, sans-serif !important;
            font-size: .78rem !important; line-height: 1.5 !important;
        }
        .block-container {
            max-width: 1180px; padding: calc(3.25rem + env(safe-area-inset-top, 0px)) 2rem 3rem;
        }
        .block-container > [data-testid="stVerticalBlock"],
        .block-container [data-testid="stColumn"] [data-testid="stVerticalBlock"],
        [class*="st-key-directory_entry_"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
        .adfm-masthead {
            display: grid; grid-template-columns: 56px minmax(0,1fr); align-items: center;
            column-gap: .9rem; border-top: 3px solid #000000; border-bottom: 1px solid #000000;
            padding: .7rem 0 .75rem; overflow: visible !important;
        }
        .adfm-mark { display:block; width:48px; height:48px; object-fit:contain; }
        .firm-name {
            margin:0 0 .24rem; font-family:Arial,Helvetica,sans-serif; font-size:.66rem;
            font-weight:700; letter-spacing:.18em; text-transform:uppercase;
            white-space: normal !important; overflow: visible !important;
        }
        .adfm-title {
            margin:0 !important; padding:0 !important; font-family:Arial,Helvetica,sans-serif !important;
            font-size:clamp(2rem,3vw,2.2rem) !important; font-weight:800 !important;
            letter-spacing:-.04em; line-height:.98 !important;
            white-space: normal !important; overflow: visible !important;
        }
        .adfm-subtitle {
            margin:.3rem 0 0; color:#414141; font-family:Arial,Helvetica,sans-serif;
            font-size:.88rem; line-height:1.35;
        }
        .directory-group-title {
            border-bottom:2px solid #000000; margin:2rem 0 .9rem; padding:0 0 .55rem;
            font-family:Arial,Helvetica,sans-serif; font-size:.7rem; font-weight:800;
            letter-spacing:.14em; line-height:1.2; text-transform:uppercase;
        }
        .directory-group-title--first { margin-top:2.25rem; }
        [class*="st-key-directory_entry_"] {
            position:relative; height:100%; padding:.9rem 0 1.15rem; cursor:pointer;
        }
        div[data-testid="stPageLink"] { margin:0; padding-right:2rem; }
        div[data-testid="stPageLink"] a {
            display:inline-flex !important; width:auto !important; min-height:0 !important;
            border:0 !important; border-radius:0 !important; background:transparent !important;
            padding:0 !important; box-shadow:none !important; text-decoration:none !important;
        }
        div[data-testid="stPageLink"] p,
        div[data-testid="stPageLink"] p strong {
            margin:0 !important; color:#000000 !important; font-family:Georgia,"Times New Roman",serif !important;
            font-size:1.28rem !important; font-weight:800 !important; letter-spacing:-.014em !important;
            line-height:1.25 !important;
        }
        [class*="st-key-directory_entry_"]:hover div[data-testid="stPageLink"] p,
        [class*="st-key-directory_entry_"]:hover div[data-testid="stPageLink"] p strong {
            text-decoration:underline !important; text-decoration-thickness:1px !important;
            text-underline-offset:.18em !important;
        }
        .tool-description {
            max-width:42rem; margin-top:.5rem; color:#505050; font-family:Arial,Helvetica,sans-serif;
            font-size:.84rem; line-height:1.5;
        }
        .entry-rule { height:1px; margin-top:1rem; background:#d7d7d7; }
        @media (max-width:760px) {
            .block-container { max-width:none; padding: calc(3.2rem + env(safe-area-inset-top, 0px)) 1rem 2.25rem; }
            .adfm-masthead { grid-template-columns:46px minmax(0,1fr); column-gap:.72rem; padding:.55rem 0 .65rem; }
            .adfm-mark { width:43px; height:43px; }
            .firm-name { font-size:.61rem; letter-spacing:.15em; }
            .adfm-title { font-size:clamp(1.68rem,7.3vw,1.9rem) !important; }
            .adfm-subtitle { font-size:.81rem; }
            div[data-testid="stHorizontalBlock"] { flex-direction:column !important; gap:0 !important; }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                width:100% !important; min-width:100% !important; flex:1 1 100% !important;
            }
            div[data-testid="stPageLink"] p,
            div[data-testid="stPageLink"] p strong { font-size:1.22rem !important; }
            .tool-description { max-width:none; font-size:.88rem; }
        }
        </style>
        """
    )
    st.markdown(
        f"""
        <header class="adfm-masthead">
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


HOME_PAGE = st.Page(render_home, title="Home", default=True)
TOOL_PAGES = [
    st.Page(
        f"pages/{tool.page_filename}",
        title=tool.title,
        url_path=legacy_url_path(tool.page_filename),
    )
    for tool in TOOLS
]
NAV_PAGE_BY_FILENAME = {
    tool.page_filename: page for tool, page in zip(TOOLS, TOOL_PAGES, strict=True)
}

NAVIGATION = st.navigation([HOME_PAGE, *TOOL_PAGES], position="sidebar")
NAVIGATION.run()
