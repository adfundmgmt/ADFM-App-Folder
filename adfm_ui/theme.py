from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

TERMINAL_COLORWAY = [
    "#4CC9F0", "#F72585", "#B8F35D", "#FF9F1C", "#9B5DE5",
    "#2EC4B6", "#FFD166", "#FF595E", "#7BDFF2", "#A0C4FF",
    "#C77DFF", "#80ED99",
]
_DASH_SEQ = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
_MARKER_SEQ = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star"]


TERMINAL_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#060B16",
        "plot_bgcolor": "#0C1423",
        "font": {"family": "JetBrains Mono, IBM Plex Mono, Menlo, monospace", "color": "#E5ECF6", "size": 12},
        "title": {"font": {"color": "#F8FAFC", "size": 18}},
        "legend": {
            "bgcolor": "rgba(6,11,22,0.7)",
            "bordercolor": "#1E2A3D",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "xaxis": {
            "gridcolor": "#1B2638",
            "linecolor": "#2A3A55",
            "zerolinecolor": "#2A3A55",
            "tickfont": {"color": "#D8E2F0"},
            "titlefont": {"color": "#D8E2F0"},
        },
        "yaxis": {
            "gridcolor": "#1B2638",
            "linecolor": "#2A3A55",
            "zerolinecolor": "#2A3A55",
            "tickfont": {"color": "#D8E2F0"},
            "titlefont": {"color": "#D8E2F0"},
        },
        "colorway": TERMINAL_COLORWAY,
        "hoverlabel": {
            "bgcolor": "#0F1A2E",
            "bordercolor": "#38506D",
            "font": {"color": "#F8FAFC", "family": "JetBrains Mono, monospace"},
        },
    }
}


def _inject_terminal_css() -> None:
    if st.session_state.get("_terminal_css_applied"):
        return

    css = """
    <style>
        :root { --terminal-bg:#060B16; --terminal-panel:#0C1423; --terminal-border:#1E2A3D; --terminal-text:#E5ECF6; --terminal-muted:#9FB3C8; }
        html, body, [class*="stApp"] { background: var(--terminal-bg); color: var(--terminal-text); font-family: "JetBrains Mono","IBM Plex Mono",Menlo,monospace; }
        .stApp, .main, section[data-testid="stSidebar"] { background: var(--terminal-bg) !important; }
        #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
        .block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 98%; }
        [data-testid="stSidebar"] > div:first-child { background: #081120; border-right: 1px solid var(--terminal-border); }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) { background: var(--terminal-panel); border:1px solid var(--terminal-border); border-radius:6px; padding:0.4rem 0.6rem; }
        .stTextInput > div > div > input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input, .stTextArea textarea {
            background-color:#0A1322 !important; color:var(--terminal-text) !important; border:1px solid #2A3A55 !important; min-height: 2rem !important; font-size: 0.86rem !important;
        }
        .stButton button, .stDownloadButton button { background:#101B2D; color:#DCE7F4; border:1px solid #2F4460; border-radius:4px; padding:0.2rem 0.65rem; }
        .terminal-panel { background: var(--terminal-panel); border:1px solid var(--terminal-border); border-radius:6px; padding:0.6rem 0.75rem; margin-bottom:0.5rem; }
        .terminal-status { background:#091424; border:1px solid #2A3A55; border-radius:6px; padding:0.35rem 0.6rem; margin-bottom:0.6rem; font-size:0.78rem; letter-spacing:0.02em; color:#C9DAEC; }
        .terminal-status .tag { color:#8AE9FF; margin-right:0.8rem; }
        .stDataFrame, .stTable { border:1px solid var(--terminal-border); border-radius:6px; overflow:hidden; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.session_state["_terminal_css_applied"] = True


def render_terminal_status_bar(app_name: str = "ADFM TERMINAL", mode: str = "LIVE", status: str = "OK", tags: Iterable[str] | None = None) -> None:
    tags = list(tags or ["YF", "FRED"])
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag_html = "".join(f'<span class="tag">[{t}]</span>' for t in tags)
    st.markdown(
        f"<div class='terminal-status'><span class='tag'>{app_name}</span><span class='tag'>MODE:{mode}</span><span class='tag'>STATUS:{status}</span><span class='tag'>LOCAL:{now_local}</span>{tag_html}</div>",
        unsafe_allow_html=True,
    )


@contextmanager
def terminal_panel(title: str | None = None):
    st.markdown("<div class='terminal-panel'>", unsafe_allow_html=True)
    if title:
        st.markdown(f"**{title}**")
    container = st.container()
    try:
        with container:
            yield container
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def _style_scatter_redundancy(fig):
    scatter_traces = [t for t in fig.data if getattr(t, "type", "") in {"scatter", "scattergl"}]
    if len(scatter_traces) <= 8:
        return
    for i, trace in enumerate(scatter_traces):
        dash = _DASH_SEQ[(i // len(TERMINAL_COLORWAY)) % len(_DASH_SEQ)]
        marker = _MARKER_SEQ[i % len(_MARKER_SEQ)]
        trace.update(line={"dash": dash, "width": max(getattr(getattr(trace, "line", None), "width", 0) or 0, 2)})
        mode = getattr(trace, "mode", "lines") or "lines"
        if "markers" not in mode:
            mode = f"{mode}+markers"
        trace.update(mode=mode, marker={"symbol": marker, "size": 6})


def apply_plotly_terminal(fig):
    if fig is None:
        return fig
    fig.update_layout(template=TERMINAL_TEMPLATE)
    _style_scatter_redundancy(fig)
    fig.update_xaxes(showgrid=True, gridcolor="#1B2638", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1B2638", zeroline=False)
    return fig


def apply_matplotlib_terminal() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#060B16",
            "axes.facecolor": "#0C1423",
            "savefig.facecolor": "#060B16",
            "axes.edgecolor": "#2A3A55",
            "axes.labelcolor": "#DCE7F4",
            "xtick.color": "#C7D7EA",
            "ytick.color": "#C7D7EA",
            "text.color": "#E5ECF6",
            "grid.color": "#22324A",
            "axes.grid": True,
            "grid.alpha": 0.6,
            "font.family": ["JetBrains Mono", "IBM Plex Mono", "monospace"],
            "axes.prop_cycle": plt.cycler(color=TERMINAL_COLORWAY)
            + plt.cycler(linestyle=["-", "--", ":", "-.", "-", "--", ":", "-.", "-", "--", ":", "-."])
            + plt.cycler(marker=["o", "s", "^", "D", "v", "P", "X", "*", "o", "s", "^", "D"]),
        }
    )


def format_table_terminal(df: pd.DataFrame):
    return df.style.set_table_styles(
        [
            {"selector": "", "props": [("background-color", "#0C1423"), ("color", "#E5ECF6"), ("border", "1px solid #1E2A3D")]},
            {"selector": "th", "props": [("background-color", "#101B2D"), ("color", "#A9D6FF"), ("border", "1px solid #24364E")]},
            {"selector": "td", "props": [("border", "1px solid #1E2A3D")]},
        ]
    )


def _patch_streamlit_plotly() -> None:
    if st.session_state.get("_terminal_plotly_patched"):
        return
    original = st.plotly_chart

    def _wrapped_plotly_chart(fig, *args, **kwargs):
        return original(apply_plotly_terminal(fig), *args, **kwargs)

    st.plotly_chart = _wrapped_plotly_chart
    st.session_state["_terminal_plotly_patched"] = True


def apply_terminal_theme(page_title: str | None = None, layout: str = "wide") -> None:
    if page_title and not st.session_state.get("_terminal_page_configured"):
        st.set_page_config(page_title=page_title, layout=layout)
        st.session_state["_terminal_page_configured"] = True

    _inject_terminal_css()
    apply_matplotlib_terminal()
    _patch_streamlit_plotly()
    if not st.session_state.get("_terminal_status_rendered"):
        render_terminal_status_bar(app_name="ADFM ANALYTICS", mode="TERMINAL", status="READY")
        st.session_state["_terminal_status_rendered"] = True
