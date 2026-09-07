from __future__ import annotations
from typing import Any,Mapping,Optional
import pandas as pd
from adfm_engine.analytics.sec_fundamentals import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.palette import PASTEL
from adfm_engine.analytics.underwriter import currency_prefix
def quarterly_chart(frame: pd.DataFrame, unit: str) -> go.Figure:
    indexed = frame.set_index("Period End").copy()
    revenue = pd.to_numeric(indexed.get("Revenue"), errors="coerce")
    operating_income = pd.to_numeric(indexed.get("Operating Income"), errors="coerce")
    margin = operating_income.div(revenue.replace(0, pd.NA)) * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=indexed.index,
            y=revenue / 1_000_000_000,
            name=f"Revenue ({currency_prefix(unit)}bn)",
            marker_color=PASTEL["blue"],
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>Revenue: {currency_prefix(unit)}%{{y:,.2f}}bn<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    if margin.notna().any():
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=margin,
                name="Operating Margin",
                line={"color": PASTEL["coral"], "width": 2.4},
                marker={"size": 6},
                hovertemplate="%{x|%Y-%m-%d}<br>Margin: %{y:,.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
    fig.update_yaxes(
        title_text=f"Revenue ({currency_prefix(unit)}bn)",
        tickprefix=currency_prefix(unit),
        secondary_y=False,
        gridcolor="#e5e5e5",
    )
    fig.update_yaxes(
        title_text="Operating margin", ticksuffix="%", secondary_y=True, showgrid=False
    )
    fig.update_layout(
        height=410,
        margin={"l": 25, "r": 25, "t": 30, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#171717", "family": "Arial"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
        bargap=0.28,
    )
    return fig

def price_history_chart(close: pd.Series, ticker: str, currency: str) -> go.Figure:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    visible_start = pd.Timestamp(clean.index[-1]) - pd.DateOffset(years=1)
    visible = clean.loc[pd.to_datetime(clean.index) >= visible_start]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=visible.index,
            y=visible,
            name=ticker,
            mode="lines",
            line={"color": PASTEL["blue"], "width": 2.5},
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>{currency_prefix(currency)}%{{y:,.2f}}<extra></extra>"
            ),
        )
    )
    for window, color in ((50, PASTEL["coral"]), (200, PASTEL["sage"])):
        average = (
            clean.rolling(window, min_periods=window).mean().reindex(visible.index)
        )
        if average.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=average.index,
                    y=average,
                    name=f"{window}D average",
                    mode="lines",
                    line={"color": color, "width": 1.4},
                    hovertemplate=(
                        f"%{{x|%Y-%m-%d}}<br>{currency_prefix(currency)}%{{y:,.2f}}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        height=390,
        margin={"l": 25, "r": 25, "t": 18, "b": 25},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#171717", "family": "Arial"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        tickprefix=currency_prefix(currency),
        tickformat=",.0f",
        gridcolor="#e5e5e5",
        title_text="Price",
    )
    return fig

