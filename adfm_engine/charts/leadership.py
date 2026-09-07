from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.palette import PASTEL
STATE_COLORS = {
    "Leading": PASTEL["sage"],
    "Improving": PASTEL["blue"],
    "Weakening": PASTEL["amber"],
    "Lagging": PASTEL["rose"],
}

def fmt_signed_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+.2%}"

def make_rotation_map(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for state in ("Leading", "Improving", "Weakening", "Lagging"):
        subset = frame.loc[frame["State"] == state]
        if subset.empty:
            continue
        custom = [
            [
                row["Relationship"],
                row["Family"],
                fmt_signed_percent(row["1W"]),
                fmt_signed_percent(row["1M"]),
                fmt_signed_percent(row["3M"]),
                fmt_signed_percent(row["6M"]),
            ]
            for _, row in subset.iterrows()
        ]
        figure.add_trace(
            go.Scatter(
                x=subset["Leadership Score"],
                y=subset["Acceleration"],
                mode="markers+text",
                name=state,
                text=subset["Pair"],
                textposition="top center",
                textfont={"size": 9, "color": "#202020"},
                marker={"size": 10, "color": STATE_COLORS[state], "line": {"color": "#ffffff", "width": 1}},
                customdata=custom,
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                    "<br>Score %{x:.1f}<br>Acceleration %{y:.1f}"
                    "<br>1W %{customdata[2]} · 1M %{customdata[3]}"
                    "<br>3M %{customdata[4]} · 6M %{customdata[5]}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line_color="#777777", line_width=1)
    figure.add_vline(x=0, line_color="#777777", line_width=1)
    figure.update_layout(
        height=540,
        margin={"l": 45, "r": 25, "t": 20, "b": 55},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 11},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        hoverlabel={"bgcolor": "white", "font": {"color": "black"}},
    )
    figure.update_xaxes(title="Leadership score", range=[-112, 112], showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    y_limit = max(55.0, float(np.nanmax(np.abs(frame["Acceleration"]))) * 1.22)
    figure.update_yaxes(title="Momentum acceleration", range=[-y_limit, y_limit], showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    return figure

def make_detail_figure(series: pd.Series, start: pd.Timestamp) -> go.Figure:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    display = clean.loc[start:].copy()
    if display.empty:
        display = clean.copy()
    base = float(display.iloc[0])
    rebased = clean / base * 100.0
    view = rebased.loc[display.index.min() :]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=view.index, y=view, mode="lines", name="Ratio", line={"color": "#000000", "width": 2.2}, hovertemplate="%{y:.2f}<extra>Ratio</extra>"))
    for window, color in ((21, PASTEL["lavender"]), (50, PASTEL["blue"]), (200, PASTEL["rose"])):
        moving_average = rebased.rolling(window, min_periods=max(10, window // 2)).mean().loc[view.index]
        figure.add_trace(go.Scatter(x=moving_average.index, y=moving_average, mode="lines", name=f"{window}D", line={"color": color, "width": 1.25}, hovertemplate=f"%{{y:.2f}}<extra>{window}D</extra>"))
    figure.add_trace(go.Scatter(x=[view.index[-1]], y=[view.iloc[-1]], mode="markers", marker={"color": "#000000", "size": 7}, showlegend=False, hovertemplate="%{y:.2f}<extra>Last</extra>"))
    figure.update_layout(
        height=390,
        margin={"l": 50, "r": 25, "t": 20, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 11},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    figure.update_yaxes(title="Rebased ratio", showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    return figure
