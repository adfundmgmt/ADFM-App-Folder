"""Compare realized volatility, recent changes, and historical context for two assets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.palette import PASTEL
from adfm_core.market_data import (
    adjusted_ohlcv,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
    unique_tickers,
)
from adfm_core.relative_volatility import (
    pair_volatility_diagnostics,
    prior_percentile_rank,
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
    render_sidebar_about,
    render_status_line,
)

TITLE = "Relative Volatility Lab"
HISTORY_OPTIONS = ("1y", "2y", "3y", "5y", "10y", "max")
RVOL_WINDOWS = (5, 10, 21, 42, 63, 126, 252)
NORMALIZATION_WINDOWS = (21, 63, 126, 252, 504, 1260)
DIAGNOSTIC_SHORT_WINDOW = 5
DIAGNOSTIC_LONG_WINDOW = 21
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
            name=f"{primary} {rvol_window}D realized volatility",
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
            name=f"{comparison} {rvol_window}D realized volatility",
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
            name=f"{primary} / {comparison} realized vol",
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
        annotation_text=f"History median {ratio_median:.2f}x",
        annotation_position="bottom left",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=1.0, line=dict(color="rgba(100,116,139,0.65)", width=1, dash="dot"),
        annotation_text="1.00x = equal volatility", annotation_position="top right",
        row=2, col=1,
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
    fig.update_yaxes(title_text=f"{primary} / {comparison}", ticksuffix="x", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(height=660)
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
            name=f"{primary} realized volatility z-score",
            line=dict(color=PRIMARY_COLOR, width=2.0),
            hovertemplate="%{y:.2f}σ<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["comparison_zscore"],
            name=f"{comparison} realized volatility z-score",
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
    fig.update_yaxes(title_text="Standard deviations")
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
    render_sidebar_about("13_Relative_Volatility_Lab.py")
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
        history = st.selectbox("Chart history", HISTORY_OPTIONS, index=3)
        rvol_window = st.selectbox(
            "Volatility window", RVOL_WINDOWS, index=2,
            format_func=lambda value: f"{value} sessions",
        )
        with st.expander("Advanced settings", expanded=True):
            show_implied = st.checkbox("Show implied-volatility overlays", value=False)
            primary_implied_ticker = st.text_input(
                "Primary implied-vol ticker", value="^VXN",
                help="Choose the options-implied index corresponding to the primary asset. VXN measures Nasdaq-100 implied volatility.",
            )
            comparison_implied_ticker = st.text_input(
                "Comparison implied-vol ticker", value="^VIX",
                help="Choose the options-implied index corresponding to the comparison asset. VIX measures S&P 500 implied volatility.",
            )
            normalization_window = st.selectbox(
                "Stress comparison history", NORMALIZATION_WINDOWS, index=3,
                format_func=lambda value: f"{value} prior sessions",
            )
        st.form_submit_button("Apply", width="stretch")

primary = normalize_ticker(primary_ticker)
comparison = normalize_ticker(comparison_ticker)
primary_implied = normalize_ticker(primary_implied_ticker) if show_implied else ""
comparison_implied = normalize_ticker(comparison_implied_ticker) if show_implied else ""
primary_label = display_ticker(primary)
comparison_label = display_ticker(comparison)
primary_implied_label = display_ticker(primary_implied)
comparison_implied_label = display_ticker(comparison_implied)

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Compare how much two assets are moving, whether that movement is unusual, "
            "and how their relative volatility is changing."
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
    ]
)
with st.spinner("Loading volatility history..."):
    raw_frames, missing = fetch_daily_ohlcv(requested, period=history)

primary_close = close_series(raw_frames, primary)
comparison_close = close_series(raw_frames, comparison)
primary_implied_close = close_series(raw_frames, primary_implied)
comparison_implied_close = close_series(raw_frames, comparison_implied)

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
    )
    if ticker and close.empty
]
if optional_missing:
    st.warning(
        "Implied-volatility overlays are unavailable for: "
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
    volatility_window=f"{rvol_window} sessions",
)

current = usable.iloc[-1]
primary_rvol = float(current["primary_rvol"])
comparison_rvol = float(current["comparison_rvol"])
primary_zscore = float(current["primary_zscore"])
comparison_zscore = float(current["comparison_zscore"])
previous = usable.iloc[-6] if len(usable) >= 6 else None
snapshot = pd.DataFrame([
    {
        "Asset": label,
        "Volatility (%)": float(current[column]),
        "5-session change (pp)": float(current[column] - previous[column]) if previous is not None else np.nan,
        "History percentile": prior_percentile_rank(usable[column]),
    }
    for label, column in [(primary_label, "primary_rvol"), (comparison_label, "comparison_rvol")]
])
st.dataframe(
    snapshot.style.format({
        "Volatility (%)": "{:.1f}%", "5-session change (pp)": "{:+.1f}",
        "History percentile": "{:.0f}",
    }, na_rep="N/A"), hide_index=True, width="stretch",
)
ratio = float(current["rvol_ratio"])
relative_gap = abs(ratio - 1.0) * 100.0
direction = "more" if ratio >= 1.0 else "less"
ratio_change = float(ratio - previous["rvol_ratio"]) if previous is not None else np.nan
st.markdown(
    f"**{primary_label} / {comparison_label}: {ratio:.2f}x**. "
    f"{primary_label} is {relative_gap:.0f}% {direction} volatile than {comparison_label}. "
    + (f"The ratio changed {ratio_change:+.2f}x over the last five paired observations." if np.isfinite(ratio_change) else "")
)
st.caption(
    "Volatility is annualized daily return variability, not a return forecast. "
    "A history percentile of 90 means volatility exceeds 90% of earlier readings in the loaded pair history. "
    "pp = percentage points; changes use five prior paired observations."
)

overview_tab = st.container()
normalized_tab = st.expander('Historical stress detail', expanded=False, on_change="rerun")
data_tab = st.expander('Data', expanded=False, on_change="rerun")
methodology_tab = st.expander('Methodology', expanded=False, on_change="rerun")

with overview_tab:
    render_section_header(
        "Volatility comparison",
        (
            f"Annualized {rvol_window}-session close-to-close volatility. "
            f"Below: {primary_label} / {comparison_label}. Above 1.00x means {primary_label} is more volatile."
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
    if normalized_tab.open:
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
                    "Volatility / index level": primary_rvol,
                    "Z-score": primary_zscore,
                },
                {
                    "Series": comparison,
                    "Volatility / index level": comparison_rvol,
                    "Z-score": comparison_zscore,
                },
            ]
        )
        if primary_implied and "primary_implied_level" in analysis:
            z_table.loc[len(z_table)] = {
                "Series": primary_implied,
                "Volatility / index level": latest_value(
                    analysis["primary_implied_level"]
                ),
                "Z-score": latest_value(analysis["primary_implied_zscore"]),
            }
        if comparison_implied and "comparison_implied_level" in analysis:
            z_table.loc[len(z_table)] = {
                "Series": comparison_implied,
                "Volatility / index level": latest_value(
                    analysis["comparison_implied_level"]
                ),
                "Z-score": latest_value(analysis["comparison_implied_zscore"]),
            }
        st.dataframe(
            z_table.style.format(
                {"Volatility / index level": "{:.2f}", "Z-score": "{:+.2f}"},
                na_rep="N/A",
            ),
            hide_index=True,
            width="stretch",
        )

with data_tab:
    if data_tab.open:
        render_section_header(
            "Calculation history",
            "Most recent 252 observations are shown below; the download contains the full loaded history.",
        )
        export = analysis.rename(
            columns={
                "primary_rvol": f"{primary}_realized_volatility",
                "comparison_rvol": f"{comparison}_realized_volatility",
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
    if methodology_tab.open:
        st.markdown(
            f"""
            **Realized volatility calculation**

            - Daily price action is measured with log returns.
            - The selected {rvol_window}-session rolling standard deviation is annualized by multiplying by the square root of 252 and shown in percent units.
            - This is a realized-volatility estimate. It is comparable across liquid assets, but it is not an options-implied volatility index and does not contain a forward volatility risk premium.

            **Normalization and comparison**

            - Each z-score compares today's realized volatility with the mean and sample standard deviation of up to {normalization_window} prior observations. Excluding today keeps the calculation causal.
            - The ratio divides {primary} realized volatility by {comparison} realized volatility on overlapping dates.
            - The two asset percentiles and the ratio percentile rank each latest reading against all earlier observations in the loaded history; ties receive half credit. The current observation is excluded from its own reference set.
            - The table and ratio change use five earlier paired observations. Changes in volatility levels are percentage points; changes in the ratio are multiples (x).
            - `{primary_implied or 'Primary implied volatility'}` divided by `{comparison_implied or 'comparison implied volatility'}` is shown beside the realized ratio. Both implied series are plotted as reported by Yahoo Finance and are not transformed into realized-volatility estimates.

            **Fixed-window diagnostics**

            - Relative-volatility acceleration divides the 5-session {primary}/{comparison} RVOL ratio by the 21-session ratio. A reading above 1.0 means short-term relative stress is running above the recent regime.
            - The downside-semivolatility ratio uses the annualized sample standard deviation of negative log-return sessions observed within each trailing 21-session window. It requires at least two negative sessions per asset; sparse windows remain unavailable.
            - No optional series is filled or fabricated. Missing implied-volatility or ETF history produces `N/A` diagnostics while the selected pair continues to render.

            Thin trading, stale observations, leverage, market-hour differences, and overnight gaps can make comparisons less representative. Realized volatility describes past movement; it does not establish whether options are cheap or expensive.
            """
        )

render_footer()
