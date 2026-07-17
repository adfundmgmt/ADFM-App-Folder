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
    pair_volatility_diagnostics,
    realized_volatility_ratio,
    relative_volatility_frame,
    rolling_zscore_previous,
)
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
    render_status_line,
)

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
PRIMARY_COLOR = "#14213d"
COMPARISON_COLOR = "#2563eb"
IMPLIED_COLOR = "#7c8796"
IMPLIED_COMPARISON_COLOR = "#9a3412"
ALERT_COLOR = "#c81e1e"
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
        primary_implied_ticker = st.text_input(
            "Primary implied-vol ticker",
            value="^VXN",
            help="Defaults to VXN for the Nasdaq 100.",
        )
        comparison_implied_ticker = st.text_input(
            "Comparison implied-vol ticker",
            value="^VIX",
            help="Defaults to VIX for the S&P 500.",
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
        "Implied-volatility inputs are optional and are used as a paired ratio."
    )
    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Compare the volatility regime of any two liquid market instruments on a consistent basis.

        **What this page shows**
        - Annualized realized volatility for both selected tickers.
        - Numerator, denominator, and ratio percentiles plus the 5-session ratio change.
        - The paired implied-volatility ratio beside the realized ratio.
        - Fixed 5D/21D acceleration, downside, semiconductor, and breadth diagnostics.
        - Each instrument's causal volatility z-score.

        **Data source**
        - Yahoo Finance adjusted daily price history.
        - Missing observations remain unavailable rather than being fabricated.
        """
    )

primary = normalize_ticker(primary_ticker)
comparison = normalize_ticker(comparison_ticker)
primary_implied = normalize_ticker(primary_implied_ticker)
comparison_implied = normalize_ticker(comparison_implied_ticker)
primary_label = display_ticker(primary)
comparison_label = display_ticker(comparison)
primary_implied_label = display_ticker(primary_implied)
comparison_implied_label = display_ticker(comparison_implied)

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Compare any two assets through a selectable synthetic-VIX window, "
            "ratio decomposition, implied-versus-realized pricing, and fixed-window stress diagnostics."
        ),
        eyebrow="ADFM Volatility Intelligence",
    )
)

if not primary or not comparison:
    st.error("Enter both a primary ticker and a comparison ticker.")
    render_footer()
    st.stop()

requested = unique_tickers(
    [
        primary,
        comparison,
        primary_implied,
        comparison_implied,
        SOXX_TICKER,
        NDX_TICKER,
        EQUAL_WEIGHT_TICKER,
        CAP_WEIGHT_TICKER,
    ]
)
with st.spinner("Loading volatility history..."):
    raw_frames, missing = fetch_daily_ohlcv(requested, period=history)

primary_close = close_series(raw_frames, primary)
comparison_close = close_series(raw_frames, comparison)
primary_implied_close = close_series(raw_frames, primary_implied)
comparison_implied_close = close_series(raw_frames, comparison_implied)
soxx_close = close_series(raw_frames, SOXX_TICKER)
ndx_close = close_series(raw_frames, NDX_TICKER)
equal_weight_close = close_series(raw_frames, EQUAL_WEIGHT_TICKER)
cap_weight_close = close_series(raw_frames, CAP_WEIGHT_TICKER)

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

optional_missing = [
    ticker
    for ticker, close in (
        (primary_implied, primary_implied_close),
        (comparison_implied, comparison_implied_close),
        (SOXX_TICKER, soxx_close),
        (NDX_TICKER, ndx_close),
        (EQUAL_WEIGHT_TICKER, equal_weight_close),
        (CAP_WEIGHT_TICKER, cap_weight_close),
    )
    if ticker and close.empty
]
if optional_missing:
    st.warning(
        "Optional diagnostics are unavailable for: "
        + ", ".join(dict.fromkeys(optional_missing))
        + ". Core pair analysis is unaffected."
    )

normalization_min_periods = max(10, min(63, normalization_window // 2))
analysis = relative_volatility_frame(
    primary_close,
    comparison_close,
    rvol_window=rvol_window,
    normalization_window=normalization_window,
    normalization_min_periods=normalization_min_periods,
)
pair_diagnostics = pair_volatility_diagnostics(
    primary_close,
    comparison_close,
    short_window=DIAGNOSTIC_SHORT_WINDOW,
    long_window=DIAGNOSTIC_LONG_WINDOW,
)
analysis = analysis.join(pair_diagnostics, how="outer")
if not primary_implied_close.empty:
    analysis["primary_implied_level"] = primary_implied_close
    analysis["primary_implied_zscore"] = rolling_zscore_previous(
        primary_implied_close,
        normalization_window,
        normalization_min_periods,
    )
if not comparison_implied_close.empty:
    analysis["comparison_implied_level"] = comparison_implied_close
    analysis["comparison_implied_zscore"] = rolling_zscore_previous(
        comparison_implied_close,
        normalization_window,
        normalization_min_periods,
    )
analysis["implied_ratio"] = aligned_level_ratio(
    primary_implied_close,
    comparison_implied_close,
)
analysis["soxx_ndx_rvol_ratio_21d"] = realized_volatility_ratio(
    soxx_close,
    ndx_close,
    DIAGNOSTIC_LONG_WINDOW,
)
analysis["qew_qqq_rvol_ratio_21d"] = realized_volatility_ratio(
    equal_weight_close,
    cap_weight_close,
    DIAGNOSTIC_LONG_WINDOW,
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
primary_zscore = latest_value(analysis["primary_zscore"])
comparison_zscore = latest_value(analysis["comparison_zscore"])

overview_tab, normalized_tab, data_tab, methodology_tab = st.tabs(
    ["Volatility spread", "Normalized stress", "Data", "Methodology"]
)

with overview_tab:
    render_section_header(
        "Realized volatility and pairwise spread",
        (
            f"Annualized {rvol_window}-session close-to-close volatility. "
            "The lower panel places the realized pair ratio beside the paired implied-volatility ratio."
        ),
    )
    st.plotly_chart(
        overview_chart(
            analysis,
            primary_label,
            comparison_label,
            rvol_window,
            primary_implied_label,
            comparison_implied_label,
        ),
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
        normalized_chart(
            analysis,
            primary_label,
            comparison_label,
            primary_implied_label,
            comparison_implied_label,
        ),
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
    if primary_implied and "primary_implied_level" in analysis:
        z_table.loc[len(z_table)] = {
            "Series": primary_implied,
            "Current synthetic VIX": latest_value(
                analysis["primary_implied_level"]
            ),
            "Z-score": latest_value(analysis["primary_implied_zscore"]),
        }
    if comparison_implied and "comparison_implied_level" in analysis:
        z_table.loc[len(z_table)] = {
            "Series": comparison_implied,
            "Current synthetic VIX": latest_value(
                analysis["comparison_implied_level"]
            ),
            "Z-score": latest_value(analysis["comparison_implied_zscore"]),
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
            "primary_implied_level": f"{primary_implied}_level",
            "primary_implied_zscore": f"{primary_implied}_zscore",
            "comparison_implied_level": f"{comparison_implied}_level",
            "comparison_implied_zscore": f"{comparison_implied}_zscore",
            "implied_ratio": f"{primary_implied}_{comparison_implied}_ratio",
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
        - The two asset percentiles and the ratio percentile rank each latest reading against all earlier observations in the loaded history; ties receive half credit. The current observation is excluded from its own reference set.
        - The 5D ratio change is the point-to-point change from five valid ratio observations earlier.
        - `{primary_implied or 'Primary implied volatility'}` divided by `{comparison_implied or 'comparison implied volatility'}` is shown beside the realized ratio. Both implied series are plotted as reported by Yahoo Finance and are not transformed into realized-volatility estimates.

        **Fixed-window diagnostics**

        - Relative-volatility acceleration divides the 5-session {primary}/{comparison} RVOL ratio by the 21-session ratio. A reading above 1.0 means short-term relative stress is running above the recent regime.
        - SOXX/NDX and QEW/QQQ divide 21-session annualized realized volatility for those fixed benchmark pairs. QEW is used as the Nasdaq-100 equal-weight proxy and QQQ as the cap-weight proxy.
        - The downside-semivolatility ratio uses the annualized sample standard deviation of negative log-return sessions observed within each trailing 21-session window. It requires at least two negative sessions per asset; sparse windows remain unavailable.
        - No optional series is filled or fabricated. Missing implied-volatility or ETF history produces `N/A` diagnostics while the selected pair continues to render.

        Thin trading, stale observations, leverage, market-hour differences, and overnight gaps can make comparisons less representative. This dashboard is an analytical tool, not an investment recommendation.
        """
    )

render_footer()
