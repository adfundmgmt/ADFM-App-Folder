"""Exact Plotly transforms from the original yield curve monitor."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL, PASTEL_RATES_SCALE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.analytics.yields import *

def clean_plot_layout(
    fig: go.Figure, height: int, y_title: Optional[str] = None
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=18, t=24, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(color="#334155", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#d9dee7",
        tickfont=dict(color="#64748b"),
    )
    fig.update_yaxes(
        gridcolor="#edf0f4",
        zeroline=False,
        title_text=y_title,
        tickfont=dict(color="#64748b"),
    )
    return fig


def chart_display_mode() -> Dict[str, object]:
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }


def history_chart(
    rates: pd.DataFrame,
    selected_curve: str,
    available_yields: List[str],
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
    )

    yield_style = {
        "Y3M": (COLORS["slate"], 1.4),
        "Y5": (COLORS["grey"], 1.5),
        "Y10": (COLORS["blue"], 2.4),
        "Y30": (COLORS["purple"], 2.0),
    }

    for col in available_yields:
        color, width = yield_style.get(col, (COLORS["grey"], 1.5))
        current = latest(rates[col])
        fig.add_trace(
            go.Scatter(
                x=rates.index,
                y=rates[col],
                mode="lines",
                name=f"{label_for_series(col)}  {fmt_pct(current)}",
                line=dict(color=color, width=width),
                hovertemplate=(
                    f"<b>{label_for_series(col)}</b><br>"
                    "%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    curve_bp = rates[selected_curve] * 100.0
    current_curve = latest(curve_bp)
    fig.add_trace(
        go.Scatter(
            x=rates.index,
            y=curve_bp,
            mode="lines",
            name=f"{label_for_series(selected_curve)}  {fmt_bp(current_curve)}",
            line=dict(color=COLORS["amber"], width=2.1),
            hovertemplate=(
                f"<b>{label_for_series(selected_curve)}</b><br>"
                "%{x|%b %d, %Y}<br>%{y:.0f} bp<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="#94a3b8",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Yield (%)",
        row=1,
        col=1,
        gridcolor="#edf0f4",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Curve (bp)",
        row=2,
        col=1,
        gridcolor="#edf0f4",
        zeroline=False,
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)

    fig.update_layout(
        height=570,
        margin=dict(l=12, r=18, t=30, b=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(color="#334155", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    return fig


def snapshot_chart(curve_data, curve_compare, available_yields):
    latest_curve, comparison_curve = curve_comparison_values(
        curve_data,
        curve_compare,
    )

    x_vals = [
        float(
            YAHOO_YIELD_TICKERS[
                FIELD_TO_TICKER[c]
            ]["years"]
        )
        for c in available_yields
    ]
    x_labels = [YIELD_LABELS[c] for c in available_yields]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=latest_curve.values,
            mode="lines+markers",
            name="Latest",
            line=dict(color=COLORS["blue"], width=3),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=comparison_curve,
            mode="lines+markers",
            name=f"{curve_compare} ago",
            line=dict(
                color=COLORS["grey"],
                width=1.7,
                dash="dash",
            ),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        xaxis=dict(
            title="Tenor",
            tickvals=x_vals,
            ticktext=x_labels,
        ),
        yaxis=dict(title="Yield (%)"),
    )
    return clean_plot_layout(fig, 360)

def pressure_chart(matrix):
    heat_cols = list(PERIODS.keys())
    z = matrix[heat_cols].to_numpy(dtype=float)
    text = np.full(z.shape, "", dtype=object)
    finite_mask = np.isfinite(z)
    text[finite_mask] = np.vectorize(
        lambda v: f"{v:+.0f}"
    )(z[finite_mask])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heat_cols,
            y=matrix["Series"],
            colorscale=PASTEL_RATES_SCALE,
            zmid=0,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="bp", thickness=10),
            hovertemplate=(
                "<b>%{y}</b><br>%{x}: %{z:+.0f} bp<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis=dict(side="top"))
    return clean_plot_layout(fig, 360)
