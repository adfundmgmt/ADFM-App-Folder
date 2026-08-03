"""Presentation layer for the ADFM Liquidity Conditions Monitor.

The analytical engine remains in ``adfm_core/_liquidity_tracker_base.py``.
This wrapper simplifies the page without changing the underlying calculations.
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
from plotly.subplots import make_subplots

import adfm_core.ui as ui


BASE_PAGE = Path(__file__).resolve().parents[1] / "adfm_core" / "_liquidity_tracker_base.py"

_original_checkbox = st.checkbox
_original_selectbox = st.selectbox
_original_plotly_chart = st.plotly_chart
_original_info = st.info
_original_render_section_header = ui.render_section_header
_original_render_kpi_cards = ui.render_kpi_cards
_original_render_selection_note = ui.render_selection_note

_state = {"skip_plots": 0, "skip_info": 0}


_added_applymap_alias = False
if not hasattr(Styler, "applymap") and hasattr(Styler, "map"):
    Styler.applymap = Styler.map  # type: ignore[attr-defined]
    _added_applymap_alias = True


def _checkbox(label: str, *args: Any, **kwargs: Any) -> Any:
    if label == "Show benchmark overlay":
        return False
    if label == "Show level-versus-impulse map":
        return False
    if label == "Show raw primary-source panels":
        return _original_checkbox("Show primary liquidity drivers", *args, **kwargs)
    if label == "Show Fed FCI-G overlay":
        return _original_checkbox("Show Fed financial conditions", *args, **kwargs)
    return _original_checkbox(label, *args, **kwargs)


def _selectbox(label: str, *args: Any, **kwargs: Any) -> Any:
    if label == "Benchmark overlay":
        return "SPY"
    return _original_selectbox(label, *args, **kwargs)


def _plotly_chart(*args: Any, **kwargs: Any) -> Any:
    if _state["skip_plots"] > 0:
        _state["skip_plots"] -= 1
        return None
    return _original_plotly_chart(*args, **kwargs)


def _info(*args: Any, **kwargs: Any) -> Any:
    if _state["skip_info"] > 0:
        _state["skip_info"] -= 1
        return None
    return _original_info(*args, **kwargs)


def _render_kpi_cards(*args: Any, **kwargs: Any) -> None:
    return None


def _render_selection_note(*args: Any, **kwargs: Any) -> None:
    return None


def _latest(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def _bucket(value: float, positive: str, negative: str) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value >= 0.35:
        return positive
    if value <= -0.35:
        return negative
    return "Mixed"


def _render_fci_g(namespace: dict[str, Any]) -> None:
    if not bool(namespace.get("show_fcig_overlay", True)):
        return

    load_fcig = namespace.get("load_fcig")
    filter_lookback = namespace.get("filter_lookback")
    plot_layout = namespace.get("plot_layout")
    lookback = namespace.get("lookback", "2y")
    gray = namespace.get("GRAY", "#6B7280")
    blue = namespace.get("BLUE", "#4472C4")
    orange = namespace.get("ORANGE", "#ED7D31")

    if not callable(load_fcig) or not callable(filter_lookback) or not callable(plot_layout):
        return

    _original_render_section_header(
        "Federal Reserve FCI-G Overlay",
        "Above zero means financial conditions are a growth headwind. Below zero means they are a growth tailwind. This is a transmission measure, so it remains separate from the liquidity score.",
    )

    fcig, _ = load_fcig()
    if fcig.empty:
        _original_info(
            "Federal Reserve FCI-G is temporarily unavailable. The primary liquidity composite above is unaffected."
        )
        return

    fcig_display = filter_lookback(fcig, lookback)
    fig = go.Figure()
    colors = {"FCI-G Baseline": blue, "FCI-G 1Y Lookback": orange}
    for column in fcig_display.columns:
        fig.add_trace(
            go.Scatter(
                x=fcig_display.index,
                y=fcig_display[column],
                name=column,
                mode="lines",
                line=dict(color=colors.get(column), width=2.4),
            )
        )

    y_values = pd.to_numeric(fcig_display.stack(), errors="coerce").dropna()
    if not y_values.empty:
        y_min = min(float(y_values.min()), -0.25)
        y_max = max(float(y_values.max()), 0.25)
        fig.add_hrect(
            y0=0,
            y1=y_max,
            fillcolor="rgba(192,0,0,.055)",
            line_width=0,
            annotation_text="Growth headwind",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=y_min,
            y1=0,
            fillcolor="rgba(112,173,71,.055)",
            line_width=0,
            annotation_text="Growth tailwind",
            annotation_position="bottom left",
        )

    fig.add_hline(y=0, line_dash="dot", line_color=gray)
    plot_layout(fig, 430, margin=dict(l=52, r=28, t=68, b=44))
    fig.update_yaxes(title_text="FCI-G")
    _original_plotly_chart(fig, width="stretch")


def _render_liquidity_level_and_impulse(namespace: dict[str, Any]) -> None:
    display_level = namespace.get("display_level", pd.Series(dtype=float))
    display_impulse = namespace.get("display_impulse", pd.Series(dtype=float))
    plot_layout = namespace.get("plot_layout")
    gray = namespace.get("GRAY", "#6B7280")
    blue = namespace.get("BLUE", "#4472C4")
    black = namespace.get("BLACK", "#111827")

    if not callable(plot_layout):
        return

    current_level = _latest(display_level)
    current_impulse = _latest(display_impulse)
    level_read = _bucket(current_level, "Easy", "Tight")
    impulse_read = _bucket(current_impulse, "Improving", "Deteriorating")

    if pd.notna(current_level) and pd.notna(current_impulse):
        subtitle = (
            f"Current level: {current_level:+.2f} ({level_read}). "
            f"Current impulse: {current_impulse:+.2f} ({impulse_read}). "
            "The top panel shows how easy or restrictive conditions are. "
            "The bottom panel shows whether they are improving or deteriorating."
        )
    else:
        subtitle = (
            "The top panel shows how easy or restrictive conditions are. "
            "The bottom panel shows whether they are improving or deteriorating."
        )

    _original_render_section_header("Liquidity Level and Marginal Impulse", subtitle)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.5, 0.5],
        subplot_titles=("Liquidity level", "Marginal impulse"),
    )

    for row in (1, 2):
        fig.add_hrect(
            y0=-0.35,
            y1=0.35,
            fillcolor="rgba(107,114,128,.07)",
            line_width=0,
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color=gray, row=row, col=1)

    fig.add_trace(
        go.Scatter(
            x=display_level.index,
            y=display_level,
            name="Liquidity Level",
            mode="lines",
            line=dict(color=blue, width=2.8),
            showlegend=False,
            hovertemplate="%{x|%b %d, %Y}<br>Level: %{y:+.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=display_impulse.index,
            y=display_impulse,
            name="Liquidity Impulse",
            mode="lines",
            line=dict(color=black, width=2.8),
            showlegend=False,
            hovertemplate="%{x|%b %d, %Y}<br>Impulse: %{y:+.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    plot_layout(fig, 590, margin=dict(l=54, r=28, t=76, b=44), showlegend=False)
    fig.update_yaxes(title_text="Easy ↕ Tight", row=1, col=1)
    fig.update_yaxes(title_text="Improving ↕ Deteriorating", row=2, col=1)
    _original_plotly_chart(fig, width="stretch")

    _render_fci_g(namespace)


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
            fig.add_hrect(
                y0=-0.35,
                y1=0.35,
                fillcolor="rgba(107,114,128,.07)",
                line_width=0,
            )
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
    _state["skip_plots"] = 0
    _state["skip_info"] = 0

    if title == "Liquidity Regime Snapshot":
        return

    if title == "Liquidity Level and Marginal Impulse":
        caller = inspect.currentframe().f_back
        namespace = caller.f_globals if caller is not None else {}
        _render_liquidity_level_and_impulse(namespace)
        _state["skip_plots"] = 1
        return

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
        _state["skip_plots"] = 3
        return

    if title == "Federal Reserve FCI-G Overlay":
        _state["skip_plots"] = 1
        _state["skip_info"] = 1
        return

    _original_render_section_header(title, subtitle)


st.checkbox = _checkbox
st.selectbox = _selectbox
st.plotly_chart = _plotly_chart
st.info = _info
ui.render_section_header = _render_section_header
ui.render_kpi_cards = _render_kpi_cards
ui.render_selection_note = _render_selection_note

try:
    runpy.run_path(str(BASE_PAGE), run_name="__main__")
finally:
    st.checkbox = _original_checkbox
    st.selectbox = _original_selectbox
    st.plotly_chart = _original_plotly_chart
    st.info = _original_info
    ui.render_section_header = _original_render_section_header
    ui.render_kpi_cards = _original_render_kpi_cards
    ui.render_selection_note = _original_render_selection_note
    if _added_applymap_alias:
        delattr(Styler, "applymap")
