"""Original liquidity Plotly transformations, isolated from presentation."""
from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.palette import PASTEL
from adfm_engine.analytics.liquidity_definitions import *
from adfm_engine.analytics.liquidity import latest, filter_lookback
BLUE=PASTEL["blue"]
GREEN=PASTEL["sage"]
ORANGE=PASTEL["coral"]
PURPLE=PASTEL["plum"]

def plot_layout(fig: go.Figure, height: int, margin: Optional[Dict[str, int]] = None, showlegend: bool = True, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=margin or dict(l=50, r=36, t=82, b=48),
        font=dict(color="#334155", family="Arial, sans-serif"),
        hovermode=hovermode,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.025, xanchor="left", x=0.0, font=dict(size=11), bgcolor="rgba(255,255,255,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(226,232,240,.48)", showline=True, linecolor="#cbd5e1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, zeroline=False)
    return fig


def _add_regime_bands(
    fig: go.Figure,
    *,
    row: int,
    positive_label: str,
    negative_label: str,
) -> None:
    fig.add_hrect(
        y0=-0.35,
        y1=0.35,
        fillcolor="rgba(107,114,128,.07)",
        line_width=0,
        row=row,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=GRAY,
        row=row,
        col=1,
    )
    fig.add_hline(
        y=0.35,
        line_dash="dot",
        line_color="rgba(112,173,71,.55)",
        row=row,
        col=1,
    )
    fig.add_hline(
        y=-0.35,
        line_dash="dot",
        line_color="rgba(192,0,0,.45)",
        row=row,
        col=1,
    )
    fig.add_annotation(
        text=positive_label,
        xref="paper",
        x=0.995,
        yref=f"y{'' if row == 1 else row}",
        y=0.43,
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color="#548235"),
    )
    fig.add_annotation(
        text=negative_label,
        xref="paper",
        x=0.995,
        yref=f"y{'' if row == 1 else row}",
        y=-0.43,
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color="#9C0006"),
    )


def main_chart(display_level, display_impulse):
    fig_main = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.5, 0.5],
        subplot_titles=("Liquidity level", "Marginal impulse"),
    )


    _add_regime_bands(
        fig_main,
        row=1,
        positive_label="Easy",
        negative_label="Tight",
    )


    _add_regime_bands(
        fig_main,
        row=2,
        positive_label="Improving",
        negative_label="Deteriorating",
    )


    fig_main.add_trace(
        go.Scatter(
            x=display_level.index,
            y=display_level,
            name="Liquidity Level",
            mode="lines",
            line=dict(color=BLUE, width=2.8),
            showlegend=False,
            hovertemplate="%{x|%b %d, %Y}<br>Level: %{y:+.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


    fig_main.add_trace(
        go.Scatter(
            x=display_impulse.index,
            y=display_impulse,
            name="Liquidity Impulse",
            mode="lines",
            line=dict(color=BLACK, width=2.8),
            showlegend=False,
            hovertemplate="%{x|%b %d, %Y}<br>Impulse: %{y:+.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )


    plot_layout(
        fig_main,
        600,
        margin=dict(l=56, r=30, t=78, b=44),
        showlegend=False,
    )


    fig_main.update_yaxes(title_text="Level score", row=1, col=1)


    fig_main.update_yaxes(title_text="Impulse score", row=2, col=1)


    return fig_main

def financial_conditions_chart(fcig, lookback):
    fcig_display = filter_lookback(fcig, lookback)


    fig_fcig = go.Figure()


    fcig_colors = {
        "FCI-G Baseline": BLUE,
        "FCI-G 1Y Lookback": ORANGE,
    }


    for column in fcig_display.columns:
        fig_fcig.add_trace(
            go.Scatter(
                x=fcig_display.index,
                y=fcig_display[column],
                name=column,
                mode="lines",
                line=dict(
                    color=fcig_colors.get(column),
                    width=2.4,
                ),
            )
        )


    y_values = pd.to_numeric(
        fcig_display.stack(),
        errors="coerce",
    ).dropna()


    if not y_values.empty:
        y_min = min(float(y_values.min()), -0.25)
        y_max = max(float(y_values.max()), 0.25)
        fig_fcig.add_hrect(
            y0=0,
            y1=y_max,
            fillcolor="rgba(192,0,0,.055)",
            line_width=0,
            annotation_text="Growth headwind",
            annotation_position="top left",
        )
        fig_fcig.add_hrect(
            y0=y_min,
            y1=0,
            fillcolor="rgba(112,173,71,.055)",
            line_width=0,
            annotation_text="Growth tailwind",
            annotation_position="bottom left",
        )


    fig_fcig.add_hline(
        y=0,
        line_dash="dot",
        line_color=GRAY,
    )


    plot_layout(
        fig_fcig,
        430,
        margin=dict(l=52, r=28, t=68, b=44),
    )


    fig_fcig.update_yaxes(title_text="FCI-G")


    return fig_fcig

def driver_charts(display_sleeve_impulses, primary_impulses):
    fig_sleeves = fig_components = None
    primary_sleeves = [sleeve for sleeve in ('Balance Sheet', 'Funding', 'Transmission') if sleeve in display_sleeve_impulses.columns]
    if primary_sleeves:
        sleeve_colors = {'Balance Sheet': BLUE, 'Funding': ORANGE, 'Transmission': PURPLE}
        fig_sleeves = go.Figure()
        for sleeve in primary_sleeves:
            fig_sleeves.add_trace(go.Scatter(x=display_sleeve_impulses.index, y=display_sleeve_impulses[sleeve], name=sleeve, mode='lines', line=dict(color=sleeve_colors[sleeve], width=2.4)))
        fig_sleeves.add_hrect(y0=-0.35, y1=0.35, fillcolor='rgba(107,114,128,.07)', line_width=0)
        fig_sleeves.add_hline(y=0, line_dash='dot', line_color=GRAY)
        plot_layout(fig_sleeves, 420, margin=dict(l=52, r=28, t=64, b=44))
        fig_sleeves.update_yaxes(title_text='Sleeve impulse')
    latest_components = pd.Series({column: latest(primary_impulses[column]) for column in primary_impulses.columns}, dtype=float).dropna().sort_values()
    if not latest_components.empty:
        bar_colors = [RED if value < -0.35 else GREEN if value > 0.35 else GRAY for value in latest_components]
        fig_components = go.Figure()
        fig_components.add_vline(x=0, line_dash='dot', line_color=GRAY)
        fig_components.add_trace(go.Bar(x=latest_components.values, y=latest_components.index, orientation='h', marker_color=bar_colors, text=[f'{value:+.2f}' for value in latest_components.values], textposition='outside', cliponaxis=False, hovertemplate='%{y}<br>Impulse: %{x:+.2f}<extra></extra>'))
        plot_layout(fig_components, max(390, 36 * len(latest_components) + 90), margin=dict(l=190, r=58, t=30, b=42), showlegend=False, hovermode='closest')
        fig_components.update_xaxes(title_text='Latest component impulse')
        fig_components.update_yaxes(showgrid=False)
    return fig_sleeves, fig_components
