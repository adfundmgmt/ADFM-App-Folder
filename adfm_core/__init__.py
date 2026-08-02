"""Shared ADFM application primitives.

Pages may adopt these modules incrementally. The package keeps shared exports
small while applying narrowly scoped compatibility behavior for legacy pages.
"""

from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path

from .catalog import TOOL_CATALOG, ToolDefinition
from .data_integrity import DataIntegrityPolicy, DataQualityReport
from .market_data import MarketDataConfig
from .ui import PageHeader, inject_institutional_theme


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


def _install_institutional_theme() -> None:
    """Apply the shared theme immediately after each tool calls page config."""

    try:
        import streamlit as st
        from PIL import Image
    except ImportError:
        return

    if getattr(st, "_adfm_institutional_theme_installed", False):
        return

    original_set_page_config = st.set_page_config
    logo_path = Path(__file__).resolve().parents[1] / "assets" / "ADFM_Logo_Naked.png"

    @wraps(original_set_page_config)
    def set_page_config(*args, **kwargs):
        is_analytics_page = _called_from_analytics_page()
        if is_analytics_page:
            kwargs = dict(kwargs)
            kwargs.setdefault("page_icon", Image.open(logo_path).convert("RGBA"))
        result = original_set_page_config(*args, **kwargs)
        if is_analytics_page:
            inject_institutional_theme()
        return result

    st.set_page_config = set_page_config
    st._adfm_institutional_theme_installed = True


_install_institutional_theme()


def _called_from_monthly_seasonality() -> bool:
    """Return True only while the Monthly Seasonality Explorer page is executing."""

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = frame.f_code.co_filename.replace("\\", "/")
            if filename.endswith("pages/8_Monthly_Seasonality_Explorer.py"):
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
