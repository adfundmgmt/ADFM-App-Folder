"""Original CFTC chart transformations and hover behavior."""
from __future__ import annotations
from datetime import date,timedelta
from typing import Final,Mapping
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.palette import EXCEL,PASTEL_20
from adfm_engine.analytics.cftc import add_metrics,COHORTS
PRICE_COLOR="#111111"
POSITION_COLOR=EXCEL["rose"]
GRID_COLOR="rgba(127,140,141,0.20)"

def positioning_chart(
    history: pd.DataFrame,
    price: pd.Series,
    market: str,
    cohort: str,
    price_label: str | None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not price.empty:
        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price.values,
                name=price_label or "Price",
                mode="lines",
                line=dict(color=PRICE_COLOR, width=1.4),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=history["report_date"],
            y=pd.to_numeric(history["net_pct_oi"], errors="coerce") * 100,
            name=f"{cohort} positioning",
            mode="lines",
            line=dict(color=POSITION_COLOR, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(192,80,77,0.08)",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, line=dict(color=EXCEL["slate_blue"], width=1), secondary_y=True)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text=price_label or "Price", showgrid=False, secondary_y=False)
    fig.update_yaxes(
        title_text="Net position as % of open interest",
        ticksuffix="%",
        showgrid=True,
        gridcolor=GRID_COLOR,
        secondary_y=True,
    )
    fig.update_layout(
        height=540,
        template="plotly_white",
        margin=dict(l=45, r=65, t=25, b=45),
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
        title=dict(text=market, font=dict(size=15), x=0.01),
    )
    return fig


def cohort_chart(history: pd.DataFrame, report_type: str) -> go.Figure:
    fig = go.Figure()
    for i, cohort in enumerate(COHORTS[report_type]):
        series = add_metrics(history, report_type, cohort)
        fig.add_trace(
            go.Scatter(
                x=series["report_date"],
                y=series["net_pct_oi"] * 100,
                name=cohort,
                mode="lines",
                line=dict(color=PASTEL_20[i % len(PASTEL_20)], width=1.7),
            )
        )
    fig.add_hline(y=0, line=dict(color=EXCEL["slate_blue"], width=1))
    fig.update_layout(
        height=470,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=45, r=25, t=20, b=40),
    )
    fig.update_yaxes(title="Net position as % of open interest", ticksuffix="%", gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    return fig


