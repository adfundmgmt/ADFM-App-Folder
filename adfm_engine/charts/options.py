"""Original compass, term structure, and fixed-moneyness surface."""
from __future__ import annotations
from datetime import date,datetime
from typing import Mapping
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from adfm_engine.palette import PASTEL
from adfm_engine.analytics.options_positioning import prepare_chain
GRID_COLOR = "rgba(148,163,184,0.23)"
PRIMARY_COLOR = PASTEL["blue"]
PUT_COLOR = PASTEL["coral"]
CALL_COLOR = PASTEL["periwinkle"]
SELECTED_COLOR = PASTEL["rose"]
PEER_COLOR = PASTEL["lavender"]
def compass_chart(frame: pd.DataFrame, selected: str) -> go.Figure:
    plot = frame.dropna(subset=["put_skew_percentile", "iv_richness_percentile"])
    fig = go.Figure()
    quadrant_colors = (
        (0, 50, 50, 100, "rgba(83,196,174,.20)"),
        (50, 100, 50, 100, "rgba(199,112,169,.20)"),
        (0, 50, 0, 50, "rgba(98,180,168,.12)"),
        (50, 100, 0, 50, "rgba(204,112,169,.13)"),
    )
    for x0, x1, y0, y1, color in quadrant_colors:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
    fig.add_hline(y=50, line=dict(color="#475569", width=1))
    fig.add_vline(x=50, line=dict(color="#475569", width=1))
    fig.add_trace(
        go.Scatter(
            x=plot["put_skew_percentile"],
            y=plot["iv_richness_percentile"],
            text=plot["ticker"],
            customdata=np.column_stack(
                [plot["atm_iv"] * 100.0, plot["put_skew"] * 100.0, plot["expiry"]]
            ),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=[15 if ticker == selected else 10 for ticker in plot["ticker"]],
                color=[SELECTED_COLOR if ticker == selected else PEER_COLOR for ticker in plot["ticker"]],
                line=dict(color="white", width=1.2),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>IV richness rank: %{y:.0f}<br>Put-skew rank: %{x:.0f}"
                "<br>ATM IV: %{customdata[0]:.1f}%<br>Put skew: %{customdata[1]:+.1f} vol pts"
                "<br>Expiry: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    annotations = (
        (24, 88, "Rich IV<br>Call/upside skew"),
        (76, 88, "Rich IV<br>Put/downside skew"),
        (24, 12, "Cheap IV<br>Call/upside skew"),
        (76, 12, "Cheap IV<br>Put/downside skew"),
    )
    for x, y, text in annotations:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=12, color="#64748b"))
    fig.update_xaxes(title="25-delta put-skew percentile in selected universe", range=[-4, 104], showgrid=False)
    fig.update_yaxes(title="IV-minus-realized percentile in selected universe", range=[-4, 104], showgrid=False)
    fig.update_layout(
        height=620,
        template="plotly_white",
        margin=dict(l=55, r=25, t=25, b=55),
        showlegend=False,
        hovermode="closest",
        font=dict(family="Arial, sans-serif", color="#1f2937"),
    )
    return fig

def term_structure_chart(frame: pd.DataFrame) -> go.Figure:
    plot = frame.sort_values("dte")
    fig = go.Figure()
    for column, label, color, dash in (
        ("atm_iv", "ATM IV", PRIMARY_COLOR, "solid"),
        ("put_25d_iv", "25-delta put IV", PUT_COLOR, "dash"),
        ("call_25d_iv", "25-delta call IV", CALL_COLOR, "dot"),
    ):
        fig.add_trace(
            go.Scatter(
                x=plot["dte"],
                y=plot[column] * 100.0,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate="%{x:.0f} DTE<br>%{y:.1f}%<extra></extra>",
            )
        )
    fig.update_xaxes(title="Days to expiration", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title="Implied volatility", ticksuffix="%", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_layout(
        height=445,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=45, r=25, t=30, b=45),
        legend=dict(orientation="h", y=1.04, x=0),
    )
    return fig

def iv_surface_chart(
    term_chains: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame]],
    risk_free_rate: float,
) -> go.Figure:
    grid = np.arange(80.0, 120.1, 2.5)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for snapshot, calls, puts in term_chains:
        spot = float(snapshot["spot"])
        time_years = max(float(snapshot["dte"]), 1.0) / 365.0
        call_frame = prepare_chain(
            calls,
            "call",
            spot=spot,
            time_years=time_years,
            risk_free_rate=risk_free_rate,
        )
        put_frame = prepare_chain(
            puts,
            "put",
            spot=spot,
            time_years=time_years,
            risk_free_rate=risk_free_rate,
        )
        call_frame["moneyness"] = call_frame["strike"] / spot * 100.0
        put_frame["moneyness"] = put_frame["strike"] / spot * 100.0
        otm = pd.concat(
            [
                put_frame.loc[put_frame["moneyness"].le(100.0)],
                call_frame.loc[call_frame["moneyness"].gt(100.0)],
            ],
            ignore_index=True,
        ).dropna(subset=["moneyness", "impliedVolatility"])
        otm = otm.loc[otm["impliedVolatility"].between(0.02, 5.0)].sort_values("moneyness")
        otm = otm.groupby("moneyness", as_index=False)["impliedVolatility"].median()
        if len(otm) < 2:
            continue
        values = np.interp(grid, otm["moneyness"], otm["impliedVolatility"] * 100.0, left=np.nan, right=np.nan)
        rows.append(values)
        labels.append(f"{snapshot['expiry']} · {int(float(snapshot['dte']))}D")
    fig = go.Figure(
        go.Heatmap(
            x=grid,
            y=labels,
            z=np.asarray(rows),
            colorscale="RdBu_r",
            colorbar=dict(title="IV %"),
            hovertemplate="%{y}<br>Moneyness: %{x:.1f}%<br>IV: %{z:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=100, line=dict(color="#111827", width=1.5))
    fig.update_xaxes(title="Strike / spot", ticksuffix="%")
    fig.update_yaxes(title="Expiration", autorange="reversed")
    fig.update_layout(height=max(390, 70 * len(labels)), template="plotly_white", margin=dict(l=55, r=35, t=25, b=50))
    return fig

