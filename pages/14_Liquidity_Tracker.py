"""Presentation patch for the ADFM Liquidity Conditions Monitor.

The analytical engine remains in ``adfm_core/_liquidity_tracker_base.py``. This
wrapper removes two redundant charts, replaces the raw multi-series panel with
an interpretable driver view, and keeps the base implementation isolated so the
calculation logic remains auditable.
"""

from __future__ import annotations

import inspect
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.io.formats.style import Styler

import adfm_core.ui as ui


BASE_PAGE = Path(__file__).resolve().parents[1] / "adfm_core" / "_liquidity_tracker_base.py"

_original_checkbox = st.checkbox
_original_plotly_chart = st.plotly_chart
_original_render_section_header = ui.render_section_header
_state = {"skip_plots": 0}


# pandas 3 removed Styler.applymap. The base page is retained unchanged for
# auditability, so provide the compatibility alias before it executes.
_added_applymap_alias = False
if not hasattr(Styler, "applymap") and hasattr(Styler, "map"):
    Styler.applymap = Styler.map  # type: ignore[attr-defined]
    _added_applymap_alias = True


def _checkbox(label: str, *args: Any, **kwargs: Any) -> Any:
    if label == "Show level-versus-impulse map":
        return False
    if label == "Show raw primary-source panels":
        return _original_checkbox("Show primary liquidity drivers", *args, **kwargs)
    return _original_checkbox(label, *args, **kwargs)


def _plotly_chart(*args: Any, **kwargs: Any) -> Any:
    if _state["skip_plots"] > 0:
        _state["skip_plots"] -= 1
        return None
    return _original_plotly_chart(*args, **kwargs)


def _latest(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def _render_primary_drivers(namespace: dict[str, Any]) -> None:
    _original_render_section_header(
        "Primary Liquidity Drivers",
        "The first panel shows which primary sleeve is changing. The second ranks the individual components driving the latest reading. Positive scores ease liquidity; negative scores tighten it.",
    )

    sleeve_frame = namespace.get("display_sleeve_impulses", pd.DataFrame())
    primary_impulses = namespace.get("primary_impulses", pd.DataFrame())
    plot_layout = namespace.get("plot_layout")
    gray = namespace.get("GRAY", "#6B7280")
    blue = namespace.get("BLUE", "#4472C4")
    orange = namespace.get("ORANGE", "#ED7D31")
    purple = namespace.get("PURPLE", "#7030A0")
    green = namespace.get("GREEN", "#70AD47")
    red = namespace.get("RED", "#C00000")

    if callable(plot_layout) and isinstance(sleeve_frame, pd.DataFrame):
        sleeves = [
            sleeve
            for sleeve in ("Balance Sheet", "Funding", "Transmission")
            if sleeve in sleeve_frame.columns
        ]
        if sleeves:
            colors = {"Balance Sheet": blue, "Funding": orange, "Transmission": purple}
            fig = go.Figure()
            for sleeve in sleeves:
                fig.add_trace(
                    go.Scatter(
                        x=sleeve_frame.index,
                        y=sleeve_frame[sleeve],
                        name=sleeve,
                        mode="lines",
                        line=dict(color=colors[sleeve], width=2.4),
                    )
                )
            fig.add_hrect(y0=-0.35, y1=0.35, fillcolor="rgba(107,114,128,.07)", line_width=0)
            fig.add_hline(y=0, line_dash="dot", line_color=gray)
            plot_layout(fig, 420, margin=dict(l=52, r=28, t=64, b=44))
            fig.update_yaxes(title_text="Sleeve impulse")
            _original_plotly_chart(fig, width="stretch")

    if callable(plot_layout) and isinstance(primary_impulses, pd.DataFrame):
        latest_components = pd.Series(
            {column: _latest(primary_impulses[column]) for column in primary_impulses.columns},
            dtype=float,
        ).dropna().sort_values()
        if not latest_components.empty:
            bar_colors = [
                red if value < -0.35 else green if value > 0.35 else gray
                for value in latest_components
            ]
            fig = go.Figure()
            fig.add_vline(x=0, line_dash="dot", line_color=gray)
            fig.add_trace(
                go.Bar(
                    x=latest_components.values,
                    y=latest_components.index,
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{value:+.2f}" for value in latest_components.values],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}<br>Impulse: %{x:+.2f}<extra></extra>",
                )
            )
            plot_layout(
                fig,
                max(390, 36 * len(latest_components) + 90),
                margin=dict(l=190, r=58, t=30, b=42),
                showlegend=False,
                hovermode="closest",
            )
            fig.update_xaxes(title_text="Latest component impulse")
            fig.update_yaxes(showgrid=False)
            _original_plotly_chart(fig, width="stretch")


def _render_section_header(title: str, subtitle: str) -> None:
    # Reset any unused skip count when the base page advances to a new section.
    _state["skip_plots"] = 0

    if title == "Current Sleeve Attribution":
        _state["skip_plots"] = 1
        return

    if title == "Level × Impulse Map":
        _state["skip_plots"] = 1
        return

    if title == "Primary-Source Plumbing and Transmission":
        caller = inspect.currentframe().f_back
        namespace = caller.f_globals if caller is not None else {}
        _render_primary_drivers(namespace)
        # The base implementation will attempt to render three raw charts next.
        _state["skip_plots"] = 3
        return

    _original_render_section_header(title, subtitle)


st.checkbox = _checkbox
st.plotly_chart = _plotly_chart
ui.render_section_header = _render_section_header

try:
    runpy.run_path(str(BASE_PAGE), run_name="__main__")
finally:
    st.checkbox = _original_checkbox
    st.plotly_chart = _original_plotly_chart
    ui.render_section_header = _original_render_section_header
    if _added_applymap_alias:
        delattr(Styler, "applymap")
