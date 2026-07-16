"""Interactive pairwise realized-volatility and synthetic-VIX comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.market_data import (
    adjusted_ohlcv,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
    unique_tickers,
)
from adfm_core.relative_volatility import (
    prior_percentile_rank,
    relative_volatility_frame,
    rolling_zscore_previous,
)
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_section_header,
    render_selection_note,
    render_status_line,
)

TITLE = "Relative Volatility Lab"
HISTORY_OPTIONS = ("1y", "2y", "3y", "5y", "10y", "max")
RVOL_WINDOWS = (5, 10, 21, 42, 63, 126, 252)
NORMALIZATION_WINDOWS = (21, 63, 126, 252, 504, 1260)
PRIMARY_COLOR = "#14213d"
COMPARISON_COLOR = "#2563eb"
IMPLIED_COLOR = "#7c8796"
ALERT_COLOR = "#c81e1e"
GRID_COLOR = "rgba(148,163,184,0.23)"


def normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()


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


def return_correlation(
    primary_close: pd.Series,
    comparison_close: pd.Series,
    sessions: int = 63,
) -> float:
    returns = pd.concat(
        [
            np.log(primary_close).diff().rename("primary"),
            np.log(comparison_close).diff().rename("comparison"),
        ],
        axis=1,
    ).dropna()
    if len(returns) < 10:
        return np.nan
    return float(returns.tail(sessions).corr().iloc[0, 1])


def ratio_read(primary: str, comparison: str, ratio: float, median: float) -> str:
    if not np.isfinite(ratio) or not np.isfinite(median):
        return "The current relative-volatility reading is unavailable for this selection."
    spread = ratio / median - 1.0 if median else np.nan
    if ratio >= median:
        direction = "above"
        implication = f"{primary} volatility is unusually elevated relative to {comparison}."
    else:
        direction = "below"
        implication = f"{primary} volatility is subdued relative to {comparison}."
    return (
        f"The {primary}/{comparison} realized-volatility ratio is {ratio:.2f}x, "
        f"{abs(spread):.0%} {direction} its loaded-history median of {median:.2f}x. "
        f"{implication}"
    )


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
    implied: str,
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
    if implied and "implied_level" in plot and plot["implied_level"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot.index,
                y=plot["implied_level"],
                name=f"{implied} implied",
                line=dict(color=IMPLIED_COLOR, width=1.35, dash="dash"),
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
    implied: str,
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
    if implied and "implied_zscore" in frame and frame["implied_zscore"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["implied_zscore"],
                name=f"{implied} z-score",
                line=dict(color=IMPLIED_COLOR, width=1.35, dash="dash"),
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


st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1560)
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; }
    div[data-testid="stForm"] { border: 0; padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Volatility setup")
    with st.form("relative_volatility_settings"):
        primary_ticker = st.text_input(
            "Primary ticker",
            value="^NDX",
            help="Any Yahoo Finance ticker, index, ETF, stock, commodity, rate, or FX symbol.",
        )
        comparison_ticker = st.text_input(
            "Comparison ticker",
            value="^GSPC",
        )
        implied_ticker = st.text_input(
            "Optional implied-vol ticker",
            value="^VIX",
            help="Leave blank when no traded implied-volatility index exists.",
        )
        history = st.selectbox("Price history", HISTORY_OPTIONS, index=3)
        rvol_window = st.selectbox(
            "Synthetic VIX window",
            RVOL_WINDOWS,
            index=2,
            format_func=lambda value: f"{value} sessions",
        )
        normalization_window = st.selectbox(
            "Z-score lookback",
            NORMALIZATION_WINDOWS,
            index=3,
            format_func=lambda value: f"{value} sessions",
        )
        st.form_submit_button("Apply", width="stretch")
    st.caption(
        "Synthetic VIX = annualized close-to-close realized volatility. "
        "The optional overlay is a market-implied index and is not required."
    )
    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Compare the volatility regime of any two liquid market instruments on a consistent basis.

        **What this page shows**
        - Annualized realized volatility for both selected tickers.
        - The primary/comparison volatility ratio and historical percentile.
        - Each instrument's causal volatility z-score.
        - An optional market-implied volatility overlay when one exists.

        **Data source**
        - Yahoo Finance adjusted daily price history.
        - Missing observations remain unavailable rather than being fabricated.
        """
    )

primary = normalize_ticker(primary_ticker)
comparison = normalize_ticker(comparison_ticker)
implied = normalize_ticker(implied_ticker)

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Compare any two assets through a selectable synthetic-VIX window, "
            "relative-volatility ratio, causal z-scores, and an optional implied-volatility index."
        ),
        eyebrow="ADFM Volatility Intelligence",
    )
)

if not primary or not comparison:
    st.error("Enter both a primary ticker and a comparison ticker.")
    render_footer()
    st.stop()

requested = unique_tickers([primary, comparison, implied])
with st.spinner("Loading volatility history..."):
    raw_frames, missing = fetch_daily_ohlcv(requested, period=history)

primary_close = close_series(raw_frames, primary)
comparison_close = close_series(raw_frames, comparison)
implied_close = close_series(raw_frames, implied) if implied else pd.Series(dtype=float)

missing_required = [
    ticker
    for ticker, close in ((primary, primary_close), (comparison, comparison_close))
    if close.empty
]
if missing_required:
    st.error("No valid daily price history was returned for: " + ", ".join(missing_required))
    if not missing.empty:
        st.dataframe(missing, hide_index=True, width="stretch")
    render_footer()
    st.stop()

if implied and implied_close.empty:
    st.warning(
        f"{implied} was unavailable. The synthetic-VIX comparison still works without the optional overlay."
    )

normalization_min_periods = max(10, min(63, normalization_window // 2))
analysis = relative_volatility_frame(
    primary_close,
    comparison_close,
    rvol_window=rvol_window,
    normalization_window=normalization_window,
    normalization_min_periods=normalization_min_periods,
)
if not implied_close.empty:
    analysis["implied_level"] = implied_close
    analysis["implied_zscore"] = rolling_zscore_previous(
        implied_close,
        normalization_window,
        normalization_min_periods,
    )

usable = analysis.dropna(subset=["primary_rvol", "comparison_rvol", "rvol_ratio"])
if usable.empty:
    st.error(
        f"There are not enough overlapping observations to calculate {rvol_window}-session volatility."
    )
    render_footer()
    st.stop()

as_of = usable.index[-1]
render_status_line(
    as_of=as_of.date().isoformat(),
    primary=primary,
    comparison=comparison,
    synthetic_vix_window=f"{rvol_window} sessions",
    normalization=f"{normalization_window} prior sessions",
)

primary_rvol = latest_value(usable["primary_rvol"])
comparison_rvol = latest_value(usable["comparison_rvol"])
current_ratio = latest_value(usable["rvol_ratio"])
ratio_median = float(usable["rvol_ratio"].median())
ratio_percentile = prior_percentile_rank(usable["rvol_ratio"])
primary_zscore = latest_value(analysis["primary_zscore"])
comparison_zscore = latest_value(analysis["comparison_zscore"])
corr_63d = return_correlation(primary_close, comparison_close)

render_selection_note(
    "Current relative-volatility read",
    ratio_read(primary, comparison, current_ratio, ratio_median),
)
render_kpi_cards(
    [
        (f"{primary} synthetic VIX", fmt(primary_rvol, "%"), f"{rvol_window}-session RVOL"),
        (f"{comparison} synthetic VIX", fmt(comparison_rvol, "%"), f"{rvol_window}-session RVOL"),
        ("RVOL ratio", fmt(current_ratio, "x", 2), f"median {fmt(ratio_median, 'x', 2)}"),
        ("Ratio percentile", fmt(ratio_percentile, "th", 0), "vs loaded prior history"),
        (f"{primary} vol z-score", fmt(primary_zscore, "σ", 2), f"vs {normalization_window} prior sessions"),
        ("63D return correlation", fmt(corr_63d, "", 2), f"{primary} vs {comparison}"),
    ]
)

overview_tab, normalized_tab, data_tab, methodology_tab = st.tabs(
    ["Volatility spread", "Normalized stress", "Data", "Methodology"]
)

with overview_tab:
    render_section_header(
        "Realized volatility and pairwise spread",
        (
            f"Annualized {rvol_window}-session close-to-close volatility. "
            "The lower panel divides the primary asset's RVOL by the comparison asset's RVOL."
        ),
    )
    st.plotly_chart(
        overview_chart(analysis, primary, comparison, rvol_window, implied),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

with normalized_tab:
    render_section_header(
        "Each asset's own volatility regime",
        (
            f"Each reading is standardized against its own prior {normalization_window} sessions. "
            "The current day is excluded from the reference mean and standard deviation."
        ),
    )
    st.plotly_chart(
        normalized_chart(analysis, primary, comparison, implied),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )
    z_table = pd.DataFrame(
        [
            {
                "Series": primary,
                "Current synthetic VIX": primary_rvol,
                "Z-score": primary_zscore,
            },
            {
                "Series": comparison,
                "Current synthetic VIX": comparison_rvol,
                "Z-score": comparison_zscore,
            },
        ]
    )
    if implied and "implied_level" in analysis:
        z_table.loc[len(z_table)] = {
            "Series": implied,
            "Current synthetic VIX": latest_value(analysis["implied_level"]),
            "Z-score": latest_value(analysis["implied_zscore"]),
        }
    st.dataframe(
        z_table.style.format(
            {"Current synthetic VIX": "{:.2f}", "Z-score": "{:+.2f}"},
            na_rep="N/A",
        ),
        hide_index=True,
        width="stretch",
    )

with data_tab:
    render_section_header(
        "Calculation history",
        "Most recent 252 observations are shown below; the download contains the full loaded history.",
    )
    export = analysis.rename(
        columns={
            "primary_rvol": f"{primary}_synthetic_vix",
            "comparison_rvol": f"{comparison}_synthetic_vix",
            "rvol_ratio": f"{primary}_{comparison}_rvol_ratio",
            "primary_zscore": f"{primary}_vol_zscore",
            "comparison_zscore": f"{comparison}_vol_zscore",
            "implied_level": f"{implied}_level" if implied else "implied_level",
            "implied_zscore": f"{implied}_zscore" if implied else "implied_zscore",
        }
    )
    export.index.name = "Date"
    display = export.dropna(how="all").tail(252).sort_index(ascending=False).reset_index()
    st.dataframe(
        display.style.format(precision=3, na_rep=""),
        hide_index=True,
        width="stretch",
        height=430,
    )
    dataframe_download(
        "Download full volatility history",
        export.reset_index(),
        f"relative_volatility_{primary}_{comparison}.csv".replace("^", ""),
    )
    if not missing.empty:
        with st.expander("Provider diagnostics"):
            st.dataframe(missing, hide_index=True, width="stretch")

with methodology_tab:
    st.markdown(
        f"""
        **Synthetic VIX calculation**

        - Daily price action is measured with log returns.
        - The selected {rvol_window}-session rolling standard deviation is annualized by multiplying by the square root of 252 and shown in percent units.
        - This is a realized-volatility estimate. It is comparable across liquid assets, but it is not an options-implied volatility index and does not contain a forward volatility risk premium.

        **Normalization and comparison**

        - Each z-score compares today's synthetic VIX with the mean and sample standard deviation of up to {normalization_window} prior observations. Excluding today keeps the calculation causal.
        - The ratio divides {primary} synthetic VIX by {comparison} synthetic VIX on overlapping dates.
        - The ratio percentile ranks the latest ratio against all earlier ratios in the loaded history; ties receive half credit.
        - The optional `{implied or 'implied-volatility'}` line is plotted as reported by Yahoo Finance and is not transformed into a realized-volatility estimate.

        Thin trading, stale observations, leverage, market-hour differences, and overnight gaps can make comparisons less representative. This dashboard is an analytical tool, not an investment recommendation.
        """
    )

render_footer()
