"""Preserved from the original Cross-Asset Ratio Chartbook; no UI runtime."""
from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.analytics.ratio_universe import MA_COLORS
from adfm_engine.analytics.ratios import rsi_wilder

def make_empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(
        height=280,
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def make_fig(
    ratio: pd.Series,
    title: str,
    display_start: pd.Timestamp,
    ma_settings: Dict[int, bool],
    show_rsi_flag: bool,
    rsi_len: int,
    compact: bool = False,
) -> go.Figure:
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if ratio.empty:
        return make_empty_fig("No data")

    ratio_view = ratio.loc[display_start:].copy()

    if ratio_view.empty:
        ratio_view = ratio.copy()

    x_start = ratio_view.index.min()
    x_end = ratio_view.index.max()

    rows = 2 if show_rsi_flag else 1
    row_heights = [0.78, 0.22] if show_rsi_flag else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Scatter(
            x=ratio_view.index,
            y=ratio_view.values,
            mode="lines",
            name="Ratio",
            line={"color": "black", "width": 2.0},
            hovertemplate="%{y:.2f}<extra>Ratio</extra>",
        ),
        row=1,
        col=1,
    )

    visible_y = [ratio_view]

    for ma_len, enabled in sorted(ma_settings.items()):
        if not enabled:
            continue

        min_obs = max(2, min(ma_len, int(ma_len * 0.40)))
        ma = ratio.rolling(ma_len, min_periods=min_obs).mean()
        ma_view = ma.loc[x_start:x_end].dropna()

        if ma_view.empty:
            continue

        visible_y.append(ma_view)

        fig.add_trace(
            go.Scatter(
                x=ma_view.index,
                y=ma_view.values,
                mode="lines",
                name=f"{ma_len}D",
                line={"color": MA_COLORS.get(ma_len, "#555555"), "width": 1.2},
                hovertemplate=f"%{{y:.2f}}<extra>{ma_len}D</extra>",
            ),
            row=1,
            col=1,
        )

    latest_x = ratio_view.index[-1]
    latest_y = ratio_view.iloc[-1]

    fig.add_trace(
        go.Scatter(
            x=[latest_x],
            y=[latest_y],
            mode="markers",
            name="Last",
            marker={"color": "black", "size": 6},
            showlegend=False,
            hovertemplate="%{y:.2f}<extra>Last</extra>",
        ),
        row=1,
        col=1,
    )

    y_all = pd.concat(visible_y).replace([np.inf, -np.inf], np.nan).dropna()

    if not y_all.empty:
        ymin = float(y_all.min())
        ymax = float(y_all.max())
        pad = (ymax - ymin) * 0.06 if ymax != ymin else max(abs(ymin) * 0.05, 1.0)
        fig.update_yaxes(range=[ymin - pad, ymax + pad], row=1, col=1)

    if show_rsi_flag:
        rsi = rsi_wilder(ratio, window=rsi_len)
        rsi_view = rsi.loc[x_start:x_end].dropna()

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y2",
            x0=x_start,
            x1=x_end,
            y0=30,
            y1=70,
            fillcolor="gray",
            opacity=0.08,
            line_width=0,
        )

        fig.add_trace(
            go.Scatter(
                x=rsi_view.index,
                y=rsi_view.values,
                mode="lines",
                name="RSI",
                line={"color": "black", "width": 1.1},
                showlegend=False,
                hovertemplate="%{y:.1f}<extra>RSI</extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[70, 70],
                mode="lines",
                name="RSI 70",
                line={"color": "#b22222", "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[30, 30],
                mode="lines",
                name="RSI 30",
                line={"color": "#2e8b57", "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    fig.update_layout(
        title_text="",
        height=(430 if show_rsi_flag else 315) if compact else (620 if show_rsi_flag else 465),
        margin={"l": 42, "r": 20, "t": 42, "b": 34},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
    )

    fig.update_xaxes(
        range=[x_start, x_end],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Ratio Index",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        row=1,
        col=1,
    )

    return fig


