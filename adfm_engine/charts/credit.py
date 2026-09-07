"""Original credit Plotly transforms, including sovereign source hover details."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from adfm_engine.analytics.credit_definitions import *
from adfm_engine.analytics.credit import clean_series

def chart_layout(height: int = 390, showlegend: bool = True) -> dict:
    return dict(
        height=height,
        margin=dict(l=12, r=16, t=26, b=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color=COLORS["slate"]),
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )


def apply_axis_style(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor=COLORS["grid"])
    return fig


def sovereign_bar_chart(frame: pd.DataFrame, group: str, x_limit: float) -> go.Figure:
    group_frame = (
        frame.loc[frame["Group"] == group]
        .sort_values("Move bp", ascending=True)
        .reset_index(drop=True)
    )
    median = float(group_frame["Move bp"].median()) if not group_frame.empty else 0.0
    colors = [
        SOVEREIGN_UP if value > 0 else SOVEREIGN_DOWN if value < 0 else SOVEREIGN_FLAT
        for value in group_frame["Move bp"]
    ]
    custom = (
        np.column_stack(
            [
                group_frame["Start Yield"],
                group_frame["End Yield"],
                group_frame["Start Date"].dt.strftime("%Y-%m-%d"),
                group_frame["End Date"].dt.strftime("%Y-%m-%d"),
                group_frame["Source"],
            ]
        )
        if not group_frame.empty
        else np.empty((0, 5), dtype=object)
    )

    fig = go.Figure()
    fig.add_vline(x=0, line_width=1.1, line_color=SOVEREIGN_ZERO)
    if not group_frame.empty:
        fig.add_vline(
            x=median,
            line_width=1.6,
            line_dash="dash",
            line_color=SOVEREIGN_MEDIAN,
        )
        fig.add_trace(
            go.Bar(
                x=group_frame["Move bp"],
                y=group_frame["Label"],
                orientation="h",
                marker_color=colors,
                marker_line_color="rgba(17,24,39,.10)",
                marker_line_width=.5,
                customdata=custom,
                text=[
                    f"{value:+.0f} bp   {start:.2f}% → {end:.2f}%"
                    for value, start, end in zip(
                        group_frame["Move bp"],
                        group_frame["Start Yield"],
                        group_frame["End Yield"],
                    )
                ],
                textposition="outside",
                textfont=dict(color="#111827", size=11),
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>Move: %{x:+.0f} bp"
                    "<br>%{customdata[0]:.2f}% → %{customdata[1]:.2f}%"
                    "<br>%{customdata[2]} → %{customdata[3]}"
                    "<br>Source: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=max(360, 34 * max(len(group_frame), 7) + 90),
        margin=dict(l=24, r=150, t=48, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        bargap=.18,
        title=dict(
            text=f"{group.upper()} · median {median:+.0f} bp",
            x=0,
            xanchor="left",
            font=dict(size=15, color="#111827"),
        ),
        xaxis=dict(
            title="Change in benchmark 10Y yield (basis points)",
            range=[-x_limit, x_limit],
            zeroline=False,
            gridcolor="#EEF2F6",
            tickfont=dict(color="#64748B", size=11),
            title_font=dict(color="#64748B", size=11),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#374151", size=11),
        ),
    )
    return fig


def spread_chart(fred, hy_oas, bbb_oas, ig_oas, display_start):
    spread_frame = pd.DataFrame(index=fred.index)
    if not hy_oas.empty:
        spread_frame['HY OAS'] = hy_oas * 100.0
    if not bbb_oas.empty:
        spread_frame['BBB OAS'] = bbb_oas * 100.0
    if not ig_oas.empty:
        spread_frame['IG OAS'] = ig_oas * 100.0
    spread_frame = spread_frame.loc[spread_frame.index >= display_start].dropna(how='all')
    if spread_frame.empty:
        return None
    else:
        fig = go.Figure()
        spread_colors = {'HY OAS': COLORS['red'], 'BBB OAS': COLORS['orange'], 'IG OAS': COLORS['blue']}
        for column in spread_frame.columns:
            fig.add_trace(go.Scatter(x=spread_frame.index, y=spread_frame[column], mode='lines', name=f'{column} {spread_frame[column].dropna().iloc[-1]:.0f} bp', line=dict(color=spread_colors[column], width=2.3)))
        fig.update_layout(**chart_layout(height=390, showlegend=True))
        fig.update_yaxes(title_text='Option-adjusted spread (bp)')
        apply_axis_style(fig)
    return fig

def funding_chart(fred, dgs10, dgs30, display_start):
    rate_frame = pd.DataFrame(index=fred.index)
    if not dgs10.empty:
        rate_frame['10Y'] = dgs10
    if not dgs30.empty:
        rate_frame['30Y'] = dgs30
    rate_frame = rate_frame.loc[rate_frame.index >= display_start].dropna(how='all')
    if rate_frame.empty:
        return None
    else:
        fig = go.Figure()
        for column, color in [('10Y', COLORS['blue']), ('30Y', COLORS['purple'])]:
            if column in rate_frame:
                fig.add_trace(go.Scatter(x=rate_frame.index, y=rate_frame[column], mode='lines', name=f'{column} {rate_frame[column].dropna().iloc[-1]:.2f}%', line=dict(color=color, width=2.4)))
        fig.update_layout(**chart_layout(height=390, showlegend=True))
        fig.update_yaxes(title_text='Yield (%)')
        apply_axis_style(fig)
    return fig

def appetite_chart(proxy, display_start):
    proxy_cols = [c for c in ['HYG/LQD', 'BKLN/LQD', 'SRLN/LQD', 'EMB/LQD', 'KRE/SPY', 'XLF/SPY'] if c in proxy.columns]
    if not proxy_cols:
        return None
    else:
        proxy_view = proxy.loc[proxy.index >= display_start, proxy_cols].dropna(how='all')
        fig = go.Figure()
        for i, column in enumerate(proxy_view.columns):
            s = clean_series(proxy_view[column])
            if s.empty or s.iloc[0] == 0:
                continue
            rebased = s / s.iloc[0] * 100.0
            fig.add_trace(go.Scatter(x=rebased.index, y=rebased, mode='lines', name=column, line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2.0)))
        fig.add_hline(y=100, line_width=1, line_color=COLORS['grid'])
        fig.update_layout(**chart_layout(height=390, showlegend=True))
        fig.update_yaxes(title_text='Rebased to 100')
        apply_axis_style(fig)
    return fig
