"""Conviction-led, historically adjusted position-sizing decision tool."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from adfm_core.market_data import (
    adjusted_ohlcv,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
    unique_tickers,
)
from adfm_core.palette import PASTEL
from adfm_core.position_sizing import (
    HORIZON_TRADING_DAYS,
    annualized_volatility,
    bootstrap_portfolio_paths,
    calculate_sizing,
    daily_gap_proxy,
    earnings_reaction_frame,
    expected_shortfall,
    first_touch_statistics,
    historical_tail_move,
    historical_windows,
    maximum_drawdown_from_prices,
    rolling_annualized_volatility,
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

TITLE = "ADFM Conviction & Position Sizing Lab"
NY_TZ = ZoneInfo("America/New_York")
BENCHMARKS = ("SPY", "QQQ", "TLT", "UUP", "USO", "^VIX")
GRID = "rgba(148,163,184,.23)"
BLUE = PASTEL["blue"]
RED = PASTEL["rose"]
GREEN = PASTEL["sage"]
ORANGE = PASTEL["coral"]


def pct(value: float, digits: int = 1, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "N/A"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value * 100:,.{digits}f}%"


def money(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}" if np.isfinite(value) else "N/A"


def close_series(frames: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    raw = frames.get(ticker)
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=ticker)
    out = pd.to_numeric(adjusted_ohlcv(raw).get("Close"), errors="coerce").dropna()
    out.name = ticker
    return out


def adjusted_frame(frames: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    raw = frames.get(ticker)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return adjusted_ohlcv(raw).dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def earnings_dates(symbol: str) -> tuple[pd.Timestamp, ...]:
    try:
        dates = yf.Ticker(symbol).get_earnings_dates(limit=48)
        if dates is None or dates.empty:
            return ()
        index = pd.DatetimeIndex(dates.index)
        if index.tz is not None:
            index = index.tz_convert(None)
        return tuple(pd.Timestamp(value).normalize() for value in index)
    except Exception:
        return ()


def style(fig: go.Figure, height: int = 430, hovermode: str = "x unified") -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode=hovermode,
        margin=dict(l=48, r=24, t=48, b=42),
        legend=dict(orientation="h", y=1.02, x=0),
        font=dict(family="Arial, sans-serif", color="#1f2937"),
    )
    return fig


def risk_chart(close: pd.Series, rolling_vol: pd.Series) -> go.Figure:
    close = close.tail(1260)
    drawdown = close / close.cummax() - 1.0
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=.07,
        row_heights=[.52, .24, .24],
    )
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Adjusted price", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown", fill="tozeroy", line=dict(color=RED)), row=2, col=1)
    fig.add_trace(go.Scatter(x=close.index, y=rolling_vol.reindex(close.index) * 100, name="63D realized volatility", line=dict(color=ORANGE)), row=3, col=1)
    fig.update_yaxes(title="Price", row=1, col=1)
    fig.update_yaxes(title="DD", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title="Vol", ticksuffix="%", row=3, col=1)
    return style(fig, 680)


def distribution_chart(paths: pd.DataFrame, target: float, stop: float) -> go.Figure:
    values = paths["return"].dropna() * 100
    fig = go.Figure(go.Histogram(x=values, nbinsx=45, marker_color=BLUE, opacity=.82))
    if target > 0:
        fig.add_vline(x=target * 100, line=dict(color=GREEN, width=2, dash="dash"), annotation_text="Target")
    fig.add_vline(x=-stop * 100, line=dict(color=RED, width=2, dash="dash"), annotation_text="Invalidation")
    fig.update_xaxes(title="Directional holding-period return", ticksuffix="%")
    fig.update_yaxes(title="Observations")
    return style(fig, 430, "closest")


def simulation_chart(wealth: np.ndarray) -> go.Figure:
    bands = np.quantile(wealth, [.05, .50, .95], axis=0)
    days = np.arange(1, wealth.shape[1] + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=(bands[2] - 1) * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=(bands[0] - 1) * 100, fill="tonexty", fillcolor="rgba(68,114,196,.18)", line=dict(width=0), name="5th-95th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=(bands[1] - 1) * 100, line=dict(color=BLUE, width=2.2), name="Median path"))
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    fig.update_xaxes(title="Trading days")
    fig.update_yaxes(title="Portfolio return", ticksuffix="%")
    return style(fig, 470)


def sensitivity_frame(selected_returns: pd.Series, frames: dict[str, pd.DataFrame], direction: str) -> pd.DataFrame:
    labels = {
        "SPY": "US equity beta",
        "QQQ": "Growth / duration equity",
        "TLT": "Long-duration Treasury proxy",
        "UUP": "US dollar proxy",
        "USO": "Oil proxy",
        "^VIX": "Equity volatility",
    }
    sign = 1.0 if direction == "Long" else -1.0
    rows = []
    for ticker, label in labels.items():
        benchmark = close_series(frames, ticker).pct_change(fill_method=None)
        aligned = pd.concat([selected_returns.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna().tail(756)
        if len(aligned) < 40:
            continue
        variance = aligned["benchmark"].var(ddof=1)
        beta = aligned["asset"].cov(aligned["benchmark"]) / variance if variance > 0 else np.nan
        rows.append({
            "Exposure": label,
            "Proxy": ticker,
            "Position correlation": sign * aligned["asset"].corr(aligned["benchmark"]),
            "Position beta": sign * beta,
            "Observations": len(aligned),
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1560)

with st.sidebar:
    st.header("Position setup")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    direction = st.selectbox("Direction", ("Long", "Short"))
    conviction = st.slider("Conviction level", 1, 5, 3, help="1=5%, 2=10%, 3=15%, 4=20%, 5=25% maximum gross exposure.")
    horizon_label = st.selectbox("Intended holding period", tuple(HORIZON_TRADING_DAYS), index=1)
    portfolio_nav = st.number_input("Portfolio NAV", min_value=1_000.0, value=5_000_000.0, step=100_000.0, format="%.0f")
    max_loss_pct = st.number_input("Maximum NAV loss at invalidation", .10, 10.0, 1.25, .05, format="%.2f")
    hold_earnings = st.checkbox("Assume position is held through earnings", value=horizon_label != "1 month")
    with st.expander("Liquidity assumptions"):
        participation = st.slider("Maximum share of median daily dollar volume", 1, 25, 10, format="%d%%")
        liquidation_days = st.slider("Target liquidation window", 1, 10, 3, format="%d days")
    st.markdown("---")
    st.header("About This Tool")
    st.markdown("Conviction establishes the maximum exposure. Historical volatility, tail behavior, earnings reactions, the entered invalidation, and liquidity can only reduce that ceiling.")

render_page_header(PageHeader(
    title=TITLE,
    description="Translate conviction into a 5%-25% exposure ceiling, then pressure-test it against historical volatility, path risk, earnings behavior, liquidity, and explicit invalidation risk.",
    eyebrow="ADFM Risk + Decision Support",
))

if not ticker:
    st.error("Enter a ticker.")
    render_footer()
    st.stop()

symbols = unique_tickers([ticker, *BENCHMARKS])
with st.spinner(f"Loading full available history for {ticker}..."):
    frames, missing = fetch_daily_ohlcv(symbols, period="max")

close = close_series(frames, ticker)
ohlcv = adjusted_frame(frames, ticker)
if len(close) < 63:
    st.error(f"No usable historical series was returned for {ticker}.")
    render_footer()
    st.stop()

latest = float(close.iloc[-1])
horizon_days = HORIZON_TRADING_DAYS[horizon_label]
render_status_line(ticker=ticker, data_through=pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d"), history=f"{len(close):,} sessions", holding_period=horizon_label, source="Yahoo Finance")

render_section_header("Trade structure", "Current target and invalidation distances are applied proportionally to every historical starting date.")
cols = st.columns(3)
entry = cols[0].number_input("Entry price", min_value=.01, value=latest, step=max(.01, latest * .0025), format="%.2f", key=f"entry_{ticker}_{direction}")
target_default = latest * (1.15 if direction == "Long" else .85)
stop_default = latest * (.92 if direction == "Long" else 1.08)
target_price = cols[1].number_input("Target price", min_value=.01, value=target_default, step=max(.01, latest * .0025), format="%.2f", key=f"target_{ticker}_{direction}")
stop_price = cols[2].number_input("Invalidation price", min_value=.01, value=stop_default, step=max(.01, latest * .0025), format="%.2f", key=f"stop_{ticker}_{direction}")

if direction == "Long":
    target_move = target_price / entry - 1
    stop_distance = 1 - stop_price / entry
else:
    target_move = 1 - target_price / entry
    stop_distance = stop_price / entry - 1
if stop_distance <= 0:
    st.error("The invalidation is on the wrong side of the entry for the selected direction.")
    st.stop()
if target_move <= 0:
    st.warning("The target is on the wrong side of the entry. Target-hit statistics are disabled until corrected.")

returns = close.pct_change(fill_method=None).dropna()
rolling_vol = rolling_annualized_volatility(returns, 63)
current_vol = annualized_volatility(returns, 63)
median_vol = float(rolling_vol.dropna().median())
tail_move = historical_tail_move(returns, direction.lower(), 5)
daily_es = expected_shortfall(returns, direction.lower(), .01)
volume = pd.to_numeric(ohlcv["Volume"], errors="coerce")
median_dollar_volume = float((close.reindex(ohlcv.index) * volume).dropna().tail(252).median())

dates = earnings_dates(ticker)
events = earnings_reaction_frame(ohlcv, dates)
if len(events) >= 4:
    event_move = float(events["abs_move"].quantile(.90))
    event_basis = f"90th-percentile absolute earnings reaction across {len(events)} events"
else:
    event_move = daily_gap_proxy(ohlcv, .90)
    event_basis = "90th-percentile overnight gap; earnings history unavailable"
next_earnings = min((date for date in dates if date >= pd.Timestamp(datetime.now(NY_TZ).date())), default=None)

step = {21: 5, 63: 5, 252: 21, 1260: 63}[horizon_days]
paths = historical_windows(close, horizon_days, direction.lower(), step=step)
max_loss = max_loss_pct / 100
sizing = calculate_sizing(
    conviction=conviction,
    max_nav_loss=max_loss,
    stop_distance=stop_distance,
    current_volatility=current_vol,
    historical_median_volatility=median_vol,
    event_move=event_move,
    tail_move=tail_move,
    portfolio_nav=portfolio_nav,
    median_dollar_volume=median_dollar_volume,
    hold_through_earnings=hold_earnings,
    participation_rate=participation / 100,
    liquidation_days=liquidation_days,
)

position_value = portfolio_nav * sizing.suggested_size
shares = position_value / entry
loss_at_stop = sizing.suggested_size * stop_distance
target_nav = sizing.suggested_size * target_move if target_move > 0 else np.nan
event_nav = sizing.suggested_size * event_move
tail_nav = sizing.suggested_size * tail_move
reward_risk = target_move / stop_distance if target_move > 0 else np.nan

if sizing.suggested_size < sizing.conviction_ceiling - 1e-9:
    verdict = f"{sizing.binding_constraint} reduces the {pct(sizing.conviction_ceiling)} conviction ceiling to {pct(sizing.suggested_size)}."
else:
    verdict = "The conviction ceiling remains binding; the selected risk constraints do not force a smaller position."
render_selection_note("Sizing verdict", verdict)
render_kpi_cards([
    ("Conviction ceiling", pct(sizing.conviction_ceiling), f"Conviction {conviction} of 5"),
    ("Suggested exposure", pct(sizing.suggested_size), sizing.binding_constraint),
    ("Position notional", money(position_value), f"Approximately {number(shares, 0)} shares"),
    ("Loss at invalidation", pct(loss_at_stop, 2), f"Budget {pct(max_loss, 2)}"),
    ("NAV impact at target", pct(target_nav, 2), f"Reward / risk {number(reward_risk, 2)}x"),
    ("Historical event loss", pct(event_nav, 2), event_basis),
])

caps = pd.DataFrame([
    ("Conviction ceiling", sizing.conviction_ceiling, f"Conviction {conviction} × 5%"),
    ("Volatility adjustment", sizing.volatility_cap, "Current 63D volatility versus its full-history rolling median"),
    ("Invalidation loss budget", sizing.invalidation_cap, "Maximum NAV loss divided by distance to invalidation"),
    ("Earnings / event risk", sizing.event_cap, event_basis if hold_earnings else "Disabled"),
    ("Historical tail risk", sizing.tail_cap, "Maximum NAV loss divided by 5th-percentile five-day adverse move"),
    ("Liquidity", sizing.liquidity_cap, f"{participation}% of median dollar volume across {liquidation_days} days"),
], columns=["Constraint", "Maximum position", "Method"])
caps["Reduction vs ceiling"] = sizing.conviction_ceiling - caps["Maximum position"]
for column in ("Maximum position", "Reduction vs ceiling"):
    caps[column] *= 100
render_section_header("Sizing attribution", "The final recommendation is the lowest auditable cap; historical inputs never increase exposure above the conviction ceiling.")
st.dataframe(caps, use_container_width=True, hide_index=True, column_config={
    "Maximum position": st.column_config.NumberColumn(format="%.1f%%"),
    "Reduction vs ceiling": st.column_config.NumberColumn(format="%.1f%%"),
})

with st.expander("Position escalation ladder"):
    ladder = pd.DataFrame({
        "Conviction": [1, 2, 3, 4, 5],
        "Exposure ceiling": [5., 10., 15., 20., 25.],
        "Role": ["Initial position", "Evidence improving", "Strong thesis", "High conviction", "Exceptional evidence"],
        "Current": ["Selected" if level == conviction else "" for level in range(1, 6)],
    })
    st.dataframe(ladder, use_container_width=True, hide_index=True, column_config={"Exposure ceiling": st.column_config.NumberColumn(format="%.1f%%")})
    trigger_cols = st.columns(2)
    trigger_cols[0].text_area("Evidence required to increase conviction", placeholder="What observable development justifies moving to the next band?", key=f"add_{ticker}_{direction}")
    trigger_cols[1].text_area("Evidence that forces a reduction", placeholder="What requires cutting size before formal invalidation?", key=f"reduce_{ticker}_{direction}")

summary_tab, paths_tab, events_tab, sensitivity_tab, simulation_tab = st.tabs([
    "Sizing summary", "Historical paths", "Event + tail risk", "Sensitivities", "Path simulation"
])

with summary_tab:
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        render_section_header("Price, drawdown, and realized volatility", "Five years displayed where available; calculations use the full returned series.")
        st.plotly_chart(risk_chart(close, rolling_vol), use_container_width=True)
    with right:
        median_return = float(paths["return"].median()) if not paths.empty else np.nan
        hit_rate = float(paths["return"].gt(0).mean()) if not paths.empty else np.nan
        metrics = pd.DataFrame([
            ("Latest price", money(latest)),
            ("21D realized volatility", pct(annualized_volatility(returns, 21))),
            ("63D realized volatility", pct(current_vol)),
            ("252D realized volatility", pct(annualized_volatility(returns, 252))),
            ("Historical median 63D volatility", pct(median_vol)),
            ("Volatility sizing factor", f"{sizing.volatility_factor:.2f}x"),
            ("Maximum historical drawdown", pct(maximum_drawdown_from_prices(close))),
            ("1% daily expected shortfall", pct(daily_es)),
            ("Five-day tail move", pct(tail_move)),
            ("Median daily dollar volume", money(median_dollar_volume)),
            (f"Median {horizon_label} return", pct(median_return, 1, True)),
            (f"{horizon_label} hit rate", pct(hit_rate, 0)),
        ], columns=["Metric", "Value"])
        render_section_header("Historical risk metrics", "Adjusted daily prices and selected-direction returns.")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

with paths_tab:
    if paths.empty:
        st.warning(f"{ticker} does not have enough history for a full {horizon_label} holding period.")
    else:
        render_section_header(f"Historical {horizon_label} outcomes", f"{len(paths):,} rolling observations sampled every {step} trading days.")
        st.plotly_chart(distribution_chart(paths, max(target_move, 0), stop_distance), use_container_width=True)
        touch = first_touch_statistics(ohlcv, horizon_days, direction.lower(), max(target_move, 0), stop_distance, step=step)
        if touch:
            touch_table = pd.DataFrame([
                ("Target reached first", touch["target_first_rate"] * 100, touch["target_first"]),
                ("Invalidation reached first", touch["stop_first_rate"] * 100, touch["stop_first"]),
                ("Both reached same session", touch["same_day_rate"] * 100, touch["same_day"]),
                ("Neither reached", touch["neither_rate"] * 100, touch["neither"]),
            ], columns=["Outcome", "Rate", "Observations"])
            st.dataframe(touch_table, use_container_width=True, hide_index=True, column_config={"Rate": st.column_config.NumberColumn(format="%.1f%%")})
            render_selection_note("First-touch read", f"Target first: {pct(float(touch['target_first_rate']))}; invalidation first: {pct(float(touch['stop_first_rate']))}.")
        dataframe_download("Download historical paths", paths, f"{ticker.lower()}_{horizon_label.replace(' ', '_')}_paths.csv")

with events_tab:
    render_kpi_cards([
        ("Event move", pct(event_move), event_basis),
        ("Event loss", pct(event_nav, 2), "Suggested exposure × event move"),
        ("Five-day tail", pct(tail_move), "Historical 5th-percentile move"),
        ("Tail loss", pct(tail_nav, 2), "Suggested exposure × tail move"),
        ("Next earnings", next_earnings.strftime("%Y-%m-%d") if next_earnings is not None else "N/A", "Yahoo schedule when available"),
        ("Earnings sample", str(len(events)), "Matched historical events"),
    ])
    if events.empty:
        st.info("No earnings history was returned. The event adjustment uses the historical overnight-gap distribution.")
    else:
        event_display = events.sort_values("date", ascending=False).copy()
        event_display["date"] = pd.to_datetime(event_display["date"]).dt.strftime("%Y-%m-%d")
        for column in ("gap", "session", "close_to_close", "abs_move"):
            event_display[column] *= 100
        st.dataframe(event_display, use_container_width=True, hide_index=True, column_config={
            "gap": st.column_config.NumberColumn("Overnight gap", format="%.1f%%"),
            "session": st.column_config.NumberColumn("Session", format="%.1f%%"),
            "close_to_close": st.column_config.NumberColumn("Close-to-close", format="%.1f%%"),
            "abs_move": st.column_config.NumberColumn("Absolute move", format="%.1f%%"),
        })

with sensitivity_tab:
    sensitivity = sensitivity_frame(returns, frames, direction)
    render_section_header("Cross-asset sensitivity", "Three-year daily correlation and beta, translated into the selected long or short direction.")
    if sensitivity.empty:
        st.info("Insufficient overlapping data for sensitivity analysis.")
    else:
        st.dataframe(sensitivity, use_container_width=True, hide_index=True, column_config={
            "Position correlation": st.column_config.NumberColumn(format="%.2f"),
            "Position beta": st.column_config.NumberColumn(format="%.2f"),
        })

with simulation_tab:
    render_section_header("Block-bootstrap historical paths", "1,000 paths resample five-day blocks from observed returns and hold exposure constant through the selected horizon.")
    wealth, endings, drawdowns = bootstrap_portfolio_paths(
        returns,
        direction=direction.lower(),
        position_size=sizing.suggested_size,
        horizon_days=horizon_days,
        n_paths=1000,
        block_size=5,
        seed=conviction * 100 + horizon_days,
    )
    if wealth.size == 0:
        st.info("Insufficient history to run the simulation.")
    else:
        render_kpi_cards([
            ("Median ending NAV impact", pct(float(np.median(endings)), 1, True), f"At {pct(sizing.suggested_size)} exposure"),
            ("5th-percentile ending impact", pct(float(np.quantile(endings, .05)), 1, True), "Historical block bootstrap"),
            ("Median maximum drawdown", pct(float(np.median(drawdowns))), "Position contribution to NAV"),
            ("Loss-budget breach", pct(float(np.mean(endings < -max_loss))), f"Budget {pct(max_loss, 2)}"),
            ("Drawdown worse than 10%", pct(float(np.mean(drawdowns < -.10))), "Portfolio NAV contribution"),
            ("Drawdown worse than 25%", pct(float(np.mean(drawdowns < -.25))), "Portfolio NAV contribution"),
        ])
        st.plotly_chart(simulation_chart(wealth), use_container_width=True)

if not missing.empty:
    with st.expander("Data diagnostics"):
        st.dataframe(missing, use_container_width=True, hide_index=True)

render_footer(data_note="Primary inputs: Yahoo Finance adjusted daily OHLCV, available earnings dates, and liquid market proxies. Historical calculations are descriptive and do not estimate the probability that the investment thesis is correct.")
