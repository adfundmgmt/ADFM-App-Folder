from __future__ import annotations
from datetime import date,timedelta
from io import StringIO
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
from adfm_engine.data.registry import SeriesDefinition
from adfm_engine.palette import PASTEL,PASTEL_DIVERGING_SCALE
import plotly.graph_objects as go
from adfm_engine.analytics.calendar import TYPE_COLORS,ASSET_LABELS,RISK_COLORS
def _timeline(events: pd.DataFrame, today_: date) -> go.Figure:
    fig = go.Figure()
    plot = events.copy()
    plot["PlotDate"] = pd.to_datetime(plot["Date"])
    for event_type, group in plot.groupby("Type", sort=False):
        fig.add_trace(go.Scatter(x=group["PlotDate"], y=group["Risk Score"], mode="markers", name=str(event_type), marker=dict(size=np.clip(group["Risk Score"] / 4.7, 9, 22), color=TYPE_COLORS.get(str(event_type), RISK_COLORS["neutral"]), opacity=0.86, line=dict(width=1, color="white")), text=group["Event"], hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<br>Risk: %{y:.0f}<extra></extra>"))
    today_ts = pd.Timestamp(today_)
    fig.add_shape(type="line", x0=today_ts, x1=today_ts, y0=35, y1=103, xref="x", yref="y", line=dict(color="#0f172a", width=1, dash="dot"))
    fig.add_vrect(x0=today_ts, x1=today_ts + pd.Timedelta(days=7), fillcolor="#f1f5f9", opacity=0.55, line_width=0, layer="below")
    fig.update_layout(height=390, margin=dict(l=12, r=12, t=18, b=12), yaxis=dict(title="Risk score", range=[35, 103], gridcolor="#eef2f7"), xaxis=dict(title="", gridcolor="#f8fafc"), legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    return fig

def _heatmap(perf: pd.DataFrame) -> go.Figure:
    perf = perf[perf["Ticker"] != "^VIX"].copy()
    if perf.empty:
        return go.Figure()
    windows = ["Today", "1W", "1M", "3M", "YTD"]
    z, text = [], []
    for _, row in perf.iterrows():
        values, labels = [], []
        for w in windows:
            val = row[w]
            values.append(val * 100 if np.isfinite(val) else np.nan)
            labels.append("N/A" if not np.isfinite(val) else f"{val:+.2%}")
        z.append(values)
        text.append(labels)
    fig = go.Figure(data=go.Heatmap(z=z, x=windows, y=perf["Asset"].tolist(), text=text, texttemplate="%{text}", colorscale=PASTEL_DIVERGING_SCALE, zmid=0, colorbar=dict(title="%", len=0.85), hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=12, r=12, t=18, b=12), plot_bgcolor="white", paper_bgcolor="white", xaxis=dict(side="top"))
    return fig

