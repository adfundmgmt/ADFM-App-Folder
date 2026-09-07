from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from adfm_engine.palette import PASTEL
from adfm_engine.data.market import adjusted_ohlcv

TITLE = "Relative Volatility Lab"
HISTORY_OPTIONS = ("1y", "2y", "3y", "5y", "10y", "max")
RVOL_WINDOWS = (5, 10, 21, 42, 63, 126, 252)
NORMALIZATION_WINDOWS = (21, 63, 126, 252, 504, 1260)
DIAGNOSTIC_SHORT_WINDOW = 5
DIAGNOSTIC_LONG_WINDOW = 21
SOXX_TICKER = "SOXX"
NDX_TICKER = "^NDX"
EQUAL_WEIGHT_TICKER = "QEW"
CAP_WEIGHT_TICKER = "QQQ"
PRIMARY_COLOR = PASTEL["blue"]
COMPARISON_COLOR = PASTEL["coral"]
IMPLIED_COLOR = PASTEL["lavender"]
IMPLIED_COMPARISON_COLOR = PASTEL["teal"]
ALERT_COLOR = PASTEL["rose"]
GRID_COLOR = "rgba(148,163,184,0.23)"


def normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()


def display_ticker(ticker: str) -> str:
    return {"^NDX": "NDX", "^GSPC": "SPX", "^VXN": "VXN", "^VIX": "VIX"}.get(
        ticker, ticker
    )


def close_series(raw_frames: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    frame = raw_frames.get(ticker)
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=ticker)
    adjusted = adjusted_ohlcv(frame)
    if "Close" not in adjusted:
        return pd.Series(dtype=float, name=ticker)
    close = pd.to_numeric(adjusted["Close"], errors="coerce").dropna()
    close.name = ticker
    return close


def latest_value(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def fmt(value: float, suffix: str = "", digits: int = 1) -> str:
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "N/A"


def aligned_level_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    frame = pd.concat(
        [numerator.rename("numerator"), denominator.rename("denominator")], axis=1
    )
    ratio = frame["numerator"].div(frame["denominator"].replace(0, np.nan))
    return ratio.replace([np.inf, -np.inf], np.nan).rename("level_ratio")


def style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=45, r=30, t=55, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        font=dict(family="Arial, sans-serif", color="#1f2937"),
    )


def overview_chart(
    frame: pd.DataFrame,
    primary: str,
    comparison: str,
    rvol_window: int,
    primary_implied: str,
    comparison_implied: str,
) -> go.Figure:
    plot = frame.dropna(subset=["primary_rvol", "comparison_rvol", "rvol_ratio"])
    ratio_median = float(plot["rvol_ratio"].median())
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.64, 0.36],
    )
    fig.add_trace(
        go.Scatter(
            x=plot.index,
            y=plot["primary_rvol"],
            name=f"{primary} {rvol_window}D synthetic VIX",
            line=dict(color=PRIMARY_COLOR, width=2.1),
            hovertemplate="%{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot.index,
            y=plot["comparison_rvol"],
            name=f"{comparison} {rvol_window}D synthetic VIX",
            line=dict(color=COMPARISON_COLOR, width=1.8),
            hovertemplate="%{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if (
        primary_implied
        and "primary_implied_level" in plot
        and plot["primary_implied_level"].notna().any()
    ):
        fig.add_trace(
            go.Scatter(
                x=plot.index,
                y=plot["primary_implied_level"],
                name=f"{primary_implied} implied",
                line=dict(color=IMPLIED_COLOR, width=1.35, dash="dash"),
                hovertemplate="%{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if (
        comparison_implied
        and "comparison_implied_level" in plot
        and plot["comparison_implied_level"].notna().any()
    ):
        fig.add_trace(
            go.Scatter(
                x=plot.index,
                y=plot["comparison_implied_level"],
                name=f"{comparison_implied} implied",
                line=dict(color=IMPLIED_COMPARISON_COLOR, width=1.35, dash="dot"),
                hovertemplate="%{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=plot.index,
            y=plot["rvol_ratio"],
            name=f"{primary} / {comparison} RVOL",
            line=dict(color=PRIMARY_COLOR, width=2.0),
            hovertemplate="%{y:.2f}x<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if "implied_ratio" in plot and plot["implied_ratio"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot.index,
                y=plot["implied_ratio"],
                name=f"{primary_implied} / {comparison_implied} implied",
                line=dict(color=IMPLIED_COMPARISON_COLOR, width=1.55, dash="dot"),
                hovertemplate="%{y:.2f}x<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(
        y=ratio_median,
        line=dict(color=IMPLIED_COLOR, width=1.2, dash="dash"),
        annotation_text=f"median {ratio_median:.2f}x",
        annotation_position="bottom left",
        row=2,
        col=1,
    )
    latest_ratio = latest_value(plot["rvol_ratio"])
    if np.isfinite(latest_ratio):
        fig.add_trace(
            go.Scatter(
                x=[plot.index[-1]],
                y=[latest_ratio],
                name="Latest ratio",
                mode="markers",
                marker=dict(color=ALERT_COLOR, size=9),
                hovertemplate="Latest: %{y:.2f}x<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Annualized vol, %", row=1, col=1)
    fig.update_yaxes(title_text="RVOL ratio", ticksuffix="x", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(height=760)
    style_axes(fig)
    return fig


def normalized_chart(
    frame: pd.DataFrame,
    primary: str,
    comparison: str,
    primary_implied: str,
    comparison_implied: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["primary_zscore"],
            name=f"{primary} synthetic VIX z-score",
            line=dict(color=PRIMARY_COLOR, width=2.0),
            hovertemplate="%{y:.2f}σ<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["comparison_zscore"],
            name=f"{comparison} synthetic VIX z-score",
            line=dict(color=COMPARISON_COLOR, width=1.8),
            hovertemplate="%{y:.2f}σ<extra></extra>",
        )
    )
    if (
        primary_implied
        and "primary_implied_zscore" in frame
        and frame["primary_implied_zscore"].notna().any()
    ):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["primary_implied_zscore"],
                name=f"{primary_implied} z-score",
                line=dict(color=IMPLIED_COLOR, width=1.35, dash="dash"),
                hovertemplate="%{y:.2f}σ<extra></extra>",
            )
        )
    if (
        comparison_implied
        and "comparison_implied_zscore" in frame
        and frame["comparison_implied_zscore"].notna().any()
    ):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["comparison_implied_zscore"],
                name=f"{comparison_implied} z-score",
                line=dict(color=IMPLIED_COMPARISON_COLOR, width=1.35, dash="dot"),
                hovertemplate="%{y:.2f}σ<extra></extra>",
            )
        )
    fig.add_hrect(y0=2, y1=8, line_width=0, fillcolor="rgba(200,30,30,0.07)")
    fig.add_hrect(y0=-8, y1=-2, line_width=0, fillcolor="rgba(37,99,235,0.06)")
    for level, dash in ((0, "solid"), (2, "dot"), (-2, "dot")):
        fig.add_hline(
            y=level,
            line=dict(color="rgba(100,116,139,0.65)", width=1, dash=dash),
        )
    fig.update_yaxes(title_text="Standard deviations", range=[-4, 4])
    fig.update_xaxes(title_text="Date")
    fig.update_layout(height=525)
    style_axes(fig)
    return fig

