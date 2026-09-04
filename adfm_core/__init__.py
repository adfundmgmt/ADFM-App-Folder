"""Shared ADFM application primitives.

Pages may adopt these modules incrementally. The package keeps shared exports
small while applying narrowly scoped compatibility behavior for legacy pages.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Optional

from .catalog import TOOL_CATALOG, ToolDefinition
from .data_integrity import DataIntegrityPolicy, DataQualityReport
from .market_data import MarketDataConfig

try:
    from .ui import PageHeader, inject_institutional_theme
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise

    @dataclass(frozen=True)
    class PageHeader:
        """Framework-neutral fallback used by non-Streamlit runtimes."""

        title: str
        description: str
        eyebrow: str = "ADFM Analytics"
        as_of: Optional[str] = None
        source_note: Optional[str] = None

    def inject_institutional_theme(max_width_px: int = 1560) -> None:
        """No-op when the legacy Streamlit presentation package is absent."""
        return None


def _called_from_analytics_page() -> bool:
    """Return True while a cataloged Streamlit page is configuring itself."""

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = frame.f_code.co_filename.replace("\\", "/")
            if (
                filename.startswith("pages/") or "/pages/" in filename
            ) and filename.endswith(".py"):
                return True
            frame = frame.f_back
    finally:
        del frame
    return False


def _called_from_position_sizing() -> bool:
    """Return True only while the Position Sizing Lab page is executing."""

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = frame.f_code.co_filename.replace("\\", "/")
            if filename.endswith("pages/22_Position_Sizing_Lab.py"):
                return True
            frame = frame.f_back
    finally:
        del frame
    return False


def _inject_position_sizing_contrast() -> None:
    """Strengthen simulator marks and chart contrast without changing other pages."""

    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        """
        <style>
        main:has(.sim-head) .sim-grid {
            gap: 7px !important;
        }

        main:has(.sim-head) .sim-dot {
            border-width: 2px !important;
            border-color: #7f7f7f !important;
            background: #ffffff !important;
            color: #262626 !important;
            font-size: .72rem !important;
            font-weight: 900 !important;
        }

        main:has(.sim-head) .sim-dot.win {
            border-color: #548235 !important;
            background: #c6e0b4 !important;
            color: #1f4e21 !important;
        }

        main:has(.sim-head) .sim-dot.loss {
            border-color: #c00000 !important;
            background: #f4b183 !important;
            color: #9c0006 !important;
        }

        main:has(.sim-head) .sim-dot.flat {
            border-color: #4472c4 !important;
            background: #d9e1f2 !important;
            color: #203864 !important;
        }

        main:has(.sim-head) .sim-dot.empty {
            border-color: #a6a6a6 !important;
            background: #f2f2f2 !important;
            color: transparent !important;
        }

        main:has(.sim-head) .sim-balance {
            border-top-width: 2px !important;
            border-bottom-width: 2px !important;
        }

        main:has(.sim-head) .sim-balance span:first-child {
            color: #262626 !important;
        }

        main:has(.sim-head) [data-testid="stAlert"] {
            border-width: 2px !important;
            border-color: #595959 !important;
            background: #f7f7f7 !important;
        }

        main:has(.sim-head) path.js-line[style*="112, 173, 71"] {
            stroke: #2e7d32 !important;
            stroke-width: 3px !important;
        }

        main:has(.sim-head) path.js-line[style*="237, 125, 49"] {
            stroke: #c65911 !important;
            stroke-width: 2.8px !important;
        }

        main:has(.sim-head) path.js-fill[style*="112, 173, 71"] {
            fill: #70ad47 !important;
            fill-opacity: .22 !important;
        }

        main:has(.sim-head) path.js-fill[style*="237, 125, 49"] {
            fill: #ed7d31 !important;
            fill-opacity: .18 !important;
        }

        main:has(.sim-head) .xtick text,
        main:has(.sim-head) .ytick text,
        main:has(.sim-head) .g-xtitle text,
        main:has(.sim-head) .g-ytitle text {
            fill: #404040 !important;
        }

        main:has(.sim-head) path.xgrid,
        main:has(.sim-head) path.ygrid {
            stroke: #d0d0d0 !important;
        }

        @media (max-width: 760px) {
            main:has(.sim-head) .sim-grid {
                gap: 5px !important;
            }

            main:has(.sim-head) .sim-dot {
                font-size: .64rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _install_institutional_theme() -> None:
    """Apply the shared theme immediately after each tool calls page config."""

    try:
        import streamlit as st
        from PIL import Image
        from streamlit.delta_generator import DeltaGenerator
    except ImportError:
        return

    if getattr(st, "_adfm_institutional_theme_installed", False):
        return

    original_set_page_config = st.set_page_config
    original_button = DeltaGenerator.button
    logo_path = Path(__file__).resolve().parents[1] / "assets" / "ADFM_Logo_Naked.png"

    @wraps(original_set_page_config)
    def set_page_config(*args, **kwargs):
        is_analytics_page = _called_from_analytics_page()
        is_position_sizing = _called_from_position_sizing()
        if is_analytics_page:
            kwargs = dict(kwargs)
            kwargs.setdefault("page_icon", Image.open(logo_path).convert("RGBA"))
        result = original_set_page_config(*args, **kwargs)
        if is_analytics_page:
            inject_institutional_theme()
        if is_position_sizing:
            _inject_position_sizing_contrast()
        return result

    @wraps(original_button)
    def button(self, label, *args, **kwargs):
        if _called_from_position_sizing() and str(label).strip() == "Start simulation":
            kwargs = dict(kwargs)
            kwargs["type"] = "secondary"
        return original_button(self, label, *args, **kwargs)

    st.set_page_config = set_page_config
    DeltaGenerator.button = button
    st._adfm_institutional_theme_installed = True


_install_institutional_theme()


def _called_from_monthly_seasonality() -> bool:
    """Return True only while the Monthly Seasonality Explorer page is executing."""

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = frame.f_code.co_filename.replace("\\", "/")
            if filename.endswith("pages/24_Monthly_Seasonality_Explorer.py"):
                return True
            frame = frame.f_back
    finally:
        del frame
    return False


def _suppress_legacy_terrain_output() -> None:
    """Hide the retired 3D terrain block on the legacy seasonality page.

    The Monthly Returns matrix is rendered earlier on that page. This shim is
    intentionally limited to the old terrain heading and its immediately
    following Plotly chart and caption, leaving every other page unchanged.
    """

    try:
        import streamlit as st
    except ImportError:
        return

    if getattr(st, "_adfm_terrain_suppression_installed", False):
        return

    original_subheader = st.subheader
    original_plotly_chart = st.plotly_chart
    original_caption = st.caption
    state = {"plot": False, "caption": False}

    @wraps(original_subheader)
    def subheader(body, *args, **kwargs):
        if _called_from_monthly_seasonality() and str(body).strip() == "3D Seasonal Waterfall Terrain":
            state["plot"] = True
            state["caption"] = True
            return None
        return original_subheader(body, *args, **kwargs)

    @wraps(original_plotly_chart)
    def plotly_chart(*args, **kwargs):
        if _called_from_monthly_seasonality() and state["plot"]:
            state["plot"] = False
            return None
        return original_plotly_chart(*args, **kwargs)

    @wraps(original_caption)
    def caption(body, *args, **kwargs):
        if _called_from_monthly_seasonality() and state["caption"]:
            state["caption"] = False
            return None
        return original_caption(body, *args, **kwargs)

    st.subheader = subheader
    st.plotly_chart = plotly_chart
    st.caption = caption
    st._adfm_terrain_suppression_installed = True


_suppress_legacy_terrain_output()


__all__ = [
    "DataIntegrityPolicy",
    "DataQualityReport",
    "MarketDataConfig",
    "PageHeader",
    "TOOL_CATALOG",
    "ToolDefinition",
    "inject_institutional_theme",
]
