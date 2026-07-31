"""Cross-asset correlation, beta, and diversification regime dashboard."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.correlation_regime import (
    conditional_pair_statistics,
    correlation_snapshot,
    current_beta_table,
    log_returns,
    pair_table,
    rolling_average_correlation,
    rolling_pair_metrics,
    window_correlation,
)
from adfm_core.market_data import (
    close_panel,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
)
from adfm_core.ui import render_footer

configure_yfinance_cache()

TITLE = "Cross-Asset Correlation Lab"
SUBTITLE = (
    "Rolling correlation, beta instability, market-mode concentration, and "
    "conditional diversification across major liquid asset classes."
)

ASSET_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "US Equities": ("SPY", "QQQ", "IWM"),
    "Global Equities": ("EFA", "EEM"),
    "Credit": ("HYG", "LQD"),
    "Rates": ("TLT", "IEF", "SHY"),
    "Commodities": ("GLD", "DBC", "USO"),
    "FX": ("UUP", "FXY"),
}
MATRIX_TICKERS: Final[tuple[str, ...]] = tuple(
    ticker for group in ASSET_GROUPS.values() for ticker in group
)
LOAD_TICKERS: Final[tuple[str, ...]] = MATRIX_TICKERS + ("^VIX",)

DISPLAY_NAMES: Final[dict[str, str]] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
    "HYG": "High Yield Credit",
    "LQD": "Investment Grade",
    "TLT": "Long Treasuries",
    "IEF": "Intermediate Treasuries",
    "SHY": "Short Treasuries",
    "GLD": "Gold",
    "DBC": "Broad Commodities",
    "USO": "Crude Oil",
    "UUP": "US Dollar",
    "FXY": "Japanese Yen",
    "^VIX": "VIX",
}

COLORS: Final[dict[str, str]] = {
    "navy": "#334155",
    "blue": "#526f8f",
    "red": "#a06452",
    "green": "#4f765f",
    "amber": "#9a733c",
    "purple": "#75668f",
    "slate": "#334155",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "border": "#dfe3e8",
}

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, .stApp, main, [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
        }
        .block-container {
            padding-top: 3.0rem;
            padding-bottom: 2.5rem;
            max-width: 1580px;
        }
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }
        header[data-testid="stHeader"] {
            background: rgba(255,255,255,.96) !important;
            border-bottom: 1px solid #f1f5f9;
        }
        .page-title {
            font-size: 1.72rem;
            font-weight: 850;
            color: #111827;
            letter-spacing: -0.025em;
            margin-bottom: 0.12rem;
        }
        .page-subtitle {
            font-size: 0.91rem;
            color: #64748b;
            margin-bottom: 0.38rem;
            line-height: 1.42;
        }
        .data-status {
            color: #64748b;
            font-size: 0.73rem;
            margin-bottom: 0.85rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.82rem 0.92rem;
            box-shadow: 0 1px 3px rgba(15,23,42,.035);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 750;
            letter-spacing: .035em;
            text-transform: uppercase;
            font-size: .68rem;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 780;
            font-size: 1.24rem;
        }
        .regime-banner {
            display: grid;
            grid-template-columns: minmax(0,1.5fr) minmax(250px,.8fr);
            gap: 1rem;
            align-items: center;
            border: 1px solid rgba(100,116,139,.18);
            border-left: 4px solid var(--accent, #526f8f);
            border-radius: 8px;
            background: rgba(100,116,139,.035);
            padding: 13px 16px;
            margin: .5rem 0 .85rem;
        }
        .regime-kicker {
            font-size: .63rem;
            font-weight: 850;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: var(--accent, #526f8f);
            margin-bottom: .26rem;
        }
        .regime-title {
            font-size: 1.22rem;
            font-weight: 830;
            color: #111827;
            letter-spacing: -.02em;
        }
        .regime-detail {
            color: #475569;
            font-size: .80rem;
            line-height: 1.42;
            margin-top: .2rem;
        }
        .regime-stat {
            border-left: 1px solid rgba(100,116,139,.18);
            padding-left: 1rem;
            text-align: right;
            color: #111827;
            font-size: 1.42rem;
            font-weight: 820;
        }
        .section-title {
            font-size: 1.01rem;
            font-weight: 780;
            color: #0f172a;
            margin-top: .8rem;
            margin-bottom: .16rem;
            letter-spacing: -.01em;
        }
        .section-subtitle {
            font-size: .80rem;
            color: #64748b;
            margin-bottom: .55rem;
            line-height: 1.4;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
        .stTabs [data-baseweb="tab"] { height: 2.35rem; padding: 0 .8rem; }
        @media (max-width: 780px) {
            .regime-banner { grid-template-columns: 1fr; }
            .regime-stat {
                border-left: 0;
                border-top: 1px solid rgba(100,116,139,.18);
                padding: .8rem 0 0;
                text-align: left;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='section-title'>{title}</div>"
        f"<div class='section-subtitle'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def display_label(ticker: str) -> str:
    return f"{DISPLAY_NAMES.get(ticker, ticker)} ({ticker})"


def style_figure(figure: go.Figure, *, height: int) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=45, r=30, t=55, b=40),
        font=dict(family="Arial, sans-serif", color="#1f2937", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


def matrix_figure(
    matrix: pd.DataFrame,
    title: str,
    *,
    change: bool = False,
) -> go.Figure:
    labels = list(matrix.columns)
    limit = 0.50 if change else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=labels,
            y=labels,
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="Δρ" if change else "ρ", thickness=12),
            hovertemplate="%{y} / %{x}<br>%{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title=dict(text=title, x=0.01, xanchor="left"))
    figure.update_xaxes(tickangle=-40, side="bottom")
    return style_figure(figure, height=660)


def rolling_regime_figure(
    average_correlation: pd.Series,
    vix: pd.Series,
) -> go.Figure:
    aligned_vix = pd.to_numeric(vix, errors="coerce").reindex(average_correlation.index)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=average_correlation.index,
            y=average_correlation,
            name="Average pairwise correlation",
            line=dict(color=COLORS["navy"], width=2.2),
            hovertemplate="%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=aligned_vix.index,
            y=aligned_vix,
            name="VIX",
            line=dict(color=COLORS["red"], width=1.3),
            opacity=0.72,
            hovertemplate="%{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_hline(y=0.50, line_dash="dot", line_color=COLORS["amber"])
    figure.update_yaxes(title_text="Average correlation", secondary_y=False)
    figure.update_yaxes(title_text="VIX", secondary_y=True, showgrid=False)
    figure.update_layout(
        title=dict(text="Cross-Asset Correlation Regime", x=0.01, xanchor="left")
    )
    return style_figure(figure, height=560)


def conditional_heatmap(
    conditional: pd.DataFrame,
    benchmark: str,
) -> go.Figure:
    matrix = conditional.pivot(
        index="Regime",
        columns="Ticker",
        values="Correlation to Benchmark",
    )
    matrix = matrix.reindex(
        columns=[ticker for ticker in MATRIX_TICKERS if ticker in matrix]
    )
    labels = [DISPLAY_NAMES.get(column, column) for column in matrix.columns]
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=labels,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="ρ", thickness=12),
            hovertemplate="%{y}<br>%{x}<br>ρ %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=dict(
            text=f"Conditional Correlation to {DISPLAY_NAMES.get(benchmark, benchmark)}",
            x=0.01,
            xanchor="left",
        )
    )
    return style_figure(figure, height=470)


def pair_diagnostics_figure(
    metrics: pd.DataFrame,
    asset: str,
    benchmark: str,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.38, 0.31, 0.31],
    )
    figure.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["Correlation"],
            name="Rolling correlation",
            line=dict(color=COLORS["navy"], width=2.1),
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=0, line_color=COLORS["muted"], line_dash="dot", row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["Beta"],
            name=f"{asset} beta to {benchmark}",
            line=dict(color=COLORS["blue"], width=1.9),
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=1, line_color=COLORS["muted"], line_dash="dot", row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["Relative Log Return"],
            name="Cumulative relative log return",
            line=dict(color=COLORS["green"], width=1.9),
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="Correlation", range=[-1.05, 1.05], row=1, col=1)
    figure.update_yaxes(title_text="Beta", row=2, col=1)
    figure.update_yaxes(title_text="Relative return", tickformat=".0%", row=3, col=1)
    figure.update_layout(
        title=dict(
            text=f"{display_label(asset)} vs {display_label(benchmark)}",
            x=0.01,
            xanchor="left",
        )
    )
    return style_figure(figure, height=780)


def scatter_figure(
    returns: pd.DataFrame,
    asset: str,
    benchmark: str,
    window: int,
) -> go.Figure:
    sample = returns[[asset, benchmark]].dropna().tail(window)
    x = sample[benchmark] * 100
    y = sample[asset] * 100
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Daily returns",
            marker=dict(color=COLORS["blue"], size=7, opacity=0.62),
            text=[timestamp.date().isoformat() for timestamp in sample.index],
            hovertemplate="%{text}<br>Benchmark %{x:.2f}%<br>Asset %{y:.2f}%<extra></extra>",
        )
    )
    if len(sample) >= 3 and x.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array([x.min(), x.max()])
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=intercept + slope * line_x,
                mode="lines",
                name=f"OLS beta {slope:.2f}",
                line=dict(color=COLORS["red"], width=2),
            )
        )
    figure.update_layout(
        title=dict(text=f"Daily Return Scatter · Last {window} Sessions", x=0.01),
        hovermode="closest",
    )
    figure.update_xaxes(title=f"{display_label(benchmark)} return, %")
    figure.update_yaxes(title=f"{display_label(asset)} return, %")
    return style_figure(figure, height=500)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(period: str):
    frames, missing = fetch_daily_ohlcv(LOAD_TICKERS, period=period)
    prices = close_panel(frames, LOAD_TICKERS, adjusted=True)
    return prices, missing


with st.sidebar:
    st.header("Correlation Setup")
    history = st.selectbox(
        "History",
        ("1y", "2y", "3y", "5y", "10y"),
        index=3,
    )
    window = st.selectbox(
        "Rolling window",
        (21, 42, 63, 126, 252),
        index=2,
        format_func=lambda value: f"{value} sessions",
    )
    benchmark = st.selectbox(
        "Benchmark",
        MATRIX_TICKERS,
        index=MATRIX_TICKERS.index("SPY"),
        format_func=display_label,
    )
    asset_options = tuple(ticker for ticker in MATRIX_TICKERS if ticker != benchmark)
    pair_asset = st.selectbox(
        "Pair asset",
        asset_options,
        index=asset_options.index("TLT") if "TLT" in asset_options else 0,
        format_func=display_label,
    )
    st.markdown("---")
    st.header("Universe")
    for group, tickers in ASSET_GROUPS.items():
        st.caption(f"{group}: {', '.join(tickers)}")
    st.markdown("---")
    st.header("Method")
    st.caption(
        "Daily adjusted-close log returns. Correlations and betas use observed "
        "overlapping sessions only. Missing prices are never forward-filled."
    )

with st.spinner("Loading cross-asset returns..."):
    prices, missing = load_market_data(history)

available = [ticker for ticker in MATRIX_TICKERS if ticker in prices]
returns = log_returns(prices[available]) if len(available) >= 2 else pd.DataFrame()
current_matrix = window_correlation(returns, window)
prior_matrix = window_correlation(returns, window, offset=window)

st.markdown(f"<div class='page-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='page-subtitle'>{SUBTITLE}</div>", unsafe_allow_html=True)

if returns.empty or current_matrix.empty:
    st.error(
        "Insufficient overlapping market history to calculate the correlation matrix."
    )
    if not missing.empty:
        st.dataframe(missing, width="stretch", hide_index=True)
    render_footer()
    st.stop()

if benchmark not in current_matrix or pair_asset not in returns:
    st.error(
        "The selected benchmark or pair asset did not return enough valid history. "
        "Choose another instrument or a shorter lookback."
    )
    if not missing.empty:
        st.dataframe(missing, width="stretch", hide_index=True)
    render_footer()
    st.stop()

as_of = returns.dropna(how="all").index.max().date().isoformat()
st.markdown(
    f"<div class='data-status'>Yahoo Finance adjusted close · Data through {as_of} · "
    f"{len(current_matrix)} assets · {window}-session window · No price filling</div>",
    unsafe_allow_html=True,
)

snapshot = correlation_snapshot(returns[current_matrix.columns], window)
pairs = pair_table(current_matrix, prior_matrix, DISPLAY_NAMES)
pairs_with_values = pairs.dropna(subset=["Correlation"])
average_history = rolling_average_correlation(returns[current_matrix.columns], window)
vix = prices["^VIX"] if "^VIX" in prices else pd.Series(index=prices.index, dtype=float)
spy_drawdown = (
    prices["SPY"].div(prices["SPY"].cummax()).sub(1.0)
    if "SPY" in prices
    else pd.Series(index=prices.index, dtype=float)
)
conditional = conditional_pair_statistics(
    returns[current_matrix.columns],
    benchmark,
    vix_levels=vix,
    drawdown=spy_drawdown,
)
pair_metrics = rolling_pair_metrics(returns, pair_asset, benchmark, window)
betas = current_beta_table(returns[current_matrix.columns], benchmark)
betas.insert(
    1,
    "Asset",
    betas["Ticker"].map(DISPLAY_NAMES).fillna(betas["Ticker"]),
)

change = snapshot.average_correlation - snapshot.prior_average_correlation
regime_label = (
    "Prior window unavailable"
    if not np.isfinite(change)
    else "Correlation rising"
    if change > 0.05
    else "Correlation falling"
    if change < -0.05
    else "Correlation stable"
)
concentration_label = (
    "High market-mode concentration"
    if snapshot.market_mode_share >= 0.50
    else "Moderate market-mode concentration"
    if snapshot.market_mode_share >= 0.35
    else "Low market-mode concentration"
)
accent = (
    COLORS["red"]
    if change > 0.05
    else COLORS["green"]
    if change < -0.05
    else COLORS["blue"]
)
change_display = f"{change:+.2f}" if np.isfinite(change) else "N/A"
st.markdown(
    f"""
    <div class="regime-banner" style="--accent:{accent}">
        <div>
            <div class="regime-kicker">Current diversification regime</div>
            <div class="regime-title">{regime_label}</div>
            <div class="regime-detail">{concentration_label}. Current and prior readings use separate {window}-session windows.</div>
        </div>
        <div class="regime-stat">ρ {snapshot.average_correlation:+.2f}<div class="regime-detail">Δ {change_display} vs prior window</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

highest = (
    pairs_with_values.iloc[0]
    if not pairs_with_values.empty
    else pd.Series(dtype=object)
)
lowest = (
    pairs_with_values.iloc[-1]
    if not pairs_with_values.empty
    else pd.Series(dtype=object)
)
metric_columns = st.columns(6)
metric_columns[0].metric(
    "Average correlation",
    f"{snapshot.average_correlation:+.2f}",
    f"{change_display} vs prior",
)
metric_columns[1].metric(
    "Market mode",
    f"{snapshot.market_mode_share:.0%}",
    "First eigenvalue share",
)
metric_columns[2].metric(
    "Effective bets",
    f"{snapshot.effective_bets:.1f}",
    f"Across {snapshot.assets} assets",
)
metric_columns[3].metric(
    "Highest pair",
    f"{highest.get('Correlation', np.nan):+.2f}",
    f"{highest.get('Ticker 1', 'N/A')} / {highest.get('Ticker 2', 'N/A')}",
)
metric_columns[4].metric(
    "Lowest pair",
    f"{lowest.get('Correlation', np.nan):+.2f}",
    f"{lowest.get('Ticker 1', 'N/A')} / {lowest.get('Ticker 2', 'N/A')}",
)
metric_columns[5].metric(
    "Coverage",
    f"{snapshot.assets}/{len(MATRIX_TICKERS)}",
    f"{snapshot.observations} sessions",
)

matrix_tab, regime_tab, pair_tab, data_tab = st.tabs(
    ["Correlation Matrix", "Regime Structure", "Pair Lab", "Data + Methodology"]
)

with matrix_tab:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            matrix_figure(current_matrix, f"Current {window}-Session Correlation"),
            width="stretch",
        )
    with right:
        common = current_matrix.index.intersection(prior_matrix.index)
        if len(common) >= 2:
            change_matrix = (
                current_matrix.loc[common, common] - prior_matrix.loc[common, common]
            )
            st.plotly_chart(
                matrix_figure(
                    change_matrix,
                    "Change Versus Prior Non-Overlapping Window",
                    change=True,
                ),
                width="stretch",
            )
        else:
            st.info(
                "The selected history does not contain two complete, non-overlapping windows."
            )
    section_header(
        "Largest correlation changes",
        "Pairs ranked by the absolute change from the prior non-overlapping window.",
    )
    movers = pairs.reindex(pairs["Change"].abs().sort_values(ascending=False).index)
    st.dataframe(
        movers[["Asset 1", "Asset 2", "Correlation", "Prior", "Change"]].head(20),
        width="stretch",
        hide_index=True,
        column_config={
            "Correlation": st.column_config.NumberColumn(format="%+.2f"),
            "Prior": st.column_config.NumberColumn(format="%+.2f"),
            "Change": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

with regime_tab:
    st.plotly_chart(
        rolling_regime_figure(average_history, vix),
        width="stretch",
    )
    section_header(
        "Conditional correlation structure",
        "Same return series split by benchmark direction, high-volatility sessions, and equity drawdowns.",
    )
    if conditional.empty:
        st.warning("Conditional correlation statistics are unavailable.")
    else:
        st.plotly_chart(
            conditional_heatmap(conditional, benchmark),
            width="stretch",
        )
    section_header(
        "Current beta term structure",
        "Correlation and beta to the selected benchmark across 21, 63, 126, and 252 sessions.",
    )
    st.dataframe(
        betas.drop(columns="Ticker"),
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="%+.2f")
            for column in betas.columns
            if column.startswith(("Corr", "Beta"))
        },
    )

with pair_tab:
    if pair_metrics.empty:
        st.warning("The selected pair does not have enough overlapping data.")
    else:
        st.plotly_chart(
            pair_diagnostics_figure(pair_metrics, pair_asset, benchmark),
            width="stretch",
        )
        st.plotly_chart(
            scatter_figure(returns, pair_asset, benchmark, window),
            width="stretch",
        )

with data_tab:
    section_header(
        "All pair statistics",
        "Underlying pairwise values used by the matrix and change ranking.",
    )
    st.dataframe(
        pairs,
        width="stretch",
        hide_index=True,
        column_config={
            "Correlation": st.column_config.NumberColumn(format="%+.3f"),
            "Prior": st.column_config.NumberColumn(format="%+.3f"),
            "Change": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    st.download_button(
        "Download pair statistics CSV",
        data=pairs.to_csv(index=False).encode("utf-8"),
        file_name=f"adfm_cross_asset_correlations_{as_of}.csv",
        mime="text/csv",
    )
    section_header(
        "Conditional statistics",
        "Exact observations, correlation, beta, and annualized volatility behind the conditional heatmap.",
    )
    conditional_export = conditional.copy()
    conditional_export.insert(
        2,
        "Asset",
        conditional_export["Ticker"]
        .map(DISPLAY_NAMES)
        .fillna(conditional_export["Ticker"]),
    )
    st.dataframe(
        conditional_export,
        width="stretch",
        hide_index=True,
        column_config={
            "Correlation to Benchmark": st.column_config.NumberColumn(format="%+.3f"),
            "Beta to Benchmark": st.column_config.NumberColumn(format="%+.3f"),
            "Annualized Volatility": st.column_config.NumberColumn(format="%.1%%"),
        },
    )
    st.download_button(
        "Download conditional statistics CSV",
        data=conditional_export.to_csv(index=False).encode("utf-8"),
        file_name=f"adfm_conditional_correlations_{as_of}.csv",
        mime="text/csv",
    )
    section_header(
        "Rolling regime history",
        "The complete average-correlation and VIX series shown in the regime chart.",
    )
    rolling_export = pd.concat(
        [average_history, pd.to_numeric(vix, errors="coerce").rename("VIX")],
        axis=1,
    ).dropna(how="all")
    rolling_export.index.name = "Date"
    st.dataframe(
        rolling_export.reset_index(),
        width="stretch",
        hide_index=True,
        column_config={
            "Average Correlation": st.column_config.NumberColumn(format="%+.3f"),
            "VIX": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download rolling history CSV",
        data=rolling_export.to_csv().encode("utf-8"),
        file_name=f"adfm_correlation_regime_history_{as_of}.csv",
        mime="text/csv",
    )
    section_header(
        "Selected pair history",
        "Rolling correlation, beta, and cumulative relative log return for the active pair.",
    )
    pair_export = pair_metrics.copy()
    pair_export.index.name = "Date"
    st.dataframe(
        pair_export.reset_index(),
        width="stretch",
        hide_index=True,
        column_config={
            "Correlation": st.column_config.NumberColumn(format="%+.3f"),
            "Beta": st.column_config.NumberColumn(format="%+.3f"),
            "Relative Log Return": st.column_config.NumberColumn(format="%+.1%%"),
        },
    )
    st.download_button(
        "Download selected pair history CSV",
        data=pair_export.to_csv().encode("utf-8"),
        file_name=f"adfm_pair_history_{pair_asset}_{benchmark}_{as_of}.csv",
        mime="text/csv",
    )
    with st.expander("Data coverage and calculation definitions"):
        if missing.empty:
            st.success("All requested market series returned valid OHLCV data.")
        else:
            st.dataframe(missing, width="stretch", hide_index=True)
        st.markdown(
            f"""
            - **Return input:** daily adjusted-close log returns through {as_of}.
            - **Current matrix:** trailing {window} observed sessions.
            - **Prior matrix:** the immediately preceding non-overlapping {window}-session sample.
            - **Market mode:** largest eigenvalue divided by the sum of correlation-matrix eigenvalues.
            - **Effective bets:** entropy effective rank of the current correlation matrix.
            - **High VIX:** sessions in the top quartile of VIX levels within the selected history.
            - **Drawdown regime:** sessions when SPY was at least 10% below its running peak.
            - Missing observations remain missing. Returns are paired only on overlapping dates.
            """
        )

render_footer(
    data_note=(
        "Primary inputs: Yahoo Finance adjusted-close data for liquid equity, credit, "
        "rates, commodity, FX, and volatility proxies. Correlation is sample-dependent "
        "and does not imply stable diversification."
    )
)
