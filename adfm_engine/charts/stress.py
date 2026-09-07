from __future__ import annotations
from datetime import date,timedelta
from typing import Dict,List
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.analytics.stress import *

def stress_chart(target_px,target_label,risk_score,dislocation_score,onset_dates,lookback_years,today):
    cutoff = pd.Timestamp(today - timedelta(days=int(lookback_years * 365.25)))
    plot_idx = target_px.index[target_px.index >= cutoff]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
    )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=target_px.reindex(plot_idx),
            name=target_label,
            mode="lines",
            line=dict(width=2.4, color=INDEX_COLOR),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                + target_label
                + ": %{y:,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    visible_onsets = onset_dates[onset_dates >= cutoff]
    if len(visible_onsets):
        marker_risk = risk_score.reindex(visible_onsets)
        marker_dis = dislocation_score.reindex(visible_onsets)
        custom = np.column_stack(
            [
                [f"{x:+.2f}" if pd.notna(x) else "NA" for x in marker_risk],
                [f"{x:+.2f}" if pd.notna(x) else "NA" for x in marker_dis],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=visible_onsets,
                y=target_px.reindex(visible_onsets).values,
                name="Watch onset",
                mode="markers",
                marker=dict(
                    size=8,
                    symbol="diamond",
                    color=WATCH_COLOR,
                    line=dict(color=INDEX_COLOR, width=0.8),
                ),
                customdata=custom,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Watch onset"
                    "<br>Risk-Off: %{customdata[0]}"
                    "<br>Dislocation: %{customdata[1]}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=risk_score.reindex(plot_idx),
            name="Global Risk-Off",
            mode="lines",
            line=dict(width=2.0, color=RISK_COLOR),
            hovertemplate="%{x|%Y-%m-%d}<br>Risk-Off: %{y:+.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=dislocation_score.reindex(plot_idx),
            name="Global Dislocation",
            mode="lines",
            line=dict(width=1.9, dash="dot", color=DISLOCATION_COLOR),
            hovertemplate="%{x|%Y-%m-%d}<br>Dislocation: %{y:+.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    for level in [WATCH_RISK, HEDGE_RISK, FRACTURE_LEVEL]:
        fig.add_hline(
            y=level,
            row=2,
            col=1,
            line_dash="dash",
            line_width=1,
            line_color=THRESHOLD_COLOR,
            opacity=0.35,
        )

    fig.add_hline(
        y=0.0,
        row=2,
        col=1,
        line_width=1,
        line_color=THRESHOLD_COLOR,
        opacity=0.30,
    )

    fig.update_layout(
        height=700,
        margin=dict(l=25, r=25, t=45, b=25),
        hovermode="x",
        legend=dict(orientation="h", y=1.04, x=0),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    fig.update_yaxes(title_text=target_label, tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(title_text="Global stress score", tickformat=".2f", row=2, col=1)

    return fig
