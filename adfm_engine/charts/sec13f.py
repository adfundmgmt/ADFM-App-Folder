import pandas as pd
import plotly.graph_objects as go
from adfm_engine.palette import PASTEL
from adfm_engine.analytics.sec13f_controls import SORT_OPTIONS
def exposure_chart(ranking: pd.DataFrame, sort_label: str, top_n: int) -> go.Figure:
    sort_column = SORT_OPTIONS[sort_label]
    plot = ranking.sort_values(sort_column, ascending=False).head(top_n).copy()
    plot = plot.sort_values(sort_column)
    if sort_column == "PORTFOLIO_WEIGHT_PCT":
        axis_title, suffix = "Share of disclosed 13F portfolio", "%"
    elif sort_column == "POSITION_VALUE_USD":
        axis_title, suffix = "Reported market value ($)", ""
    else:
        axis_title, suffix = "Reported shares", ""
    fig = go.Figure(
        go.Bar(
            x=plot[sort_column],
            y=plot["MANAGER"],
            orientation="h",
            marker=dict(color=PASTEL["blue"]),
            customdata=plot[
                ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "PORTFOLIO_VALUE_USD"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Portfolio weight: %{customdata[0]:.2f}%"
                "<br>Position value: $%{customdata[1]:,.0f}"
                "<br>13F portfolio: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(440, 32 * len(plot) + 115),
        margin=dict(l=10, r=25, t=30, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(title=axis_title, ticksuffix=suffix, gridcolor="#e5e5e5"),
        yaxis=dict(title=None, automargin=True),
    )
    return fig
