"""Current-snapshot options positioning, skew, and volatility explorer."""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.market_data import (
    adjusted_ohlcv,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
    unique_tickers,
)
from adfm_core.options_positioning import (
    add_cross_sectional_ranks,
    option_snapshot,
)
from adfm_core.options_sources import (
    expirations_from_cboe,
    fetch_cboe_delayed_options,
    select_cboe_expiry,
)
from adfm_core.palette import PASTEL
from adfm_core.provider_calls import CBOE_OPTIONS, YAHOO_OPTIONS
from adfm_core.relative_volatility import annualized_realized_volatility
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
    render_sidebar_about,
    render_status_line,
)

TITLE = "Options Positioning Compass"
DEFAULT_UNIVERSE = "SPY, QQQ, IWM, DIA, TLT, GLD, USO, SMH, EEM, HYG, LQD"
NY_TZ = ZoneInfo("America/New_York")
SELECTED_COLOR = PASTEL["rose"]
PEER_COLOR = PASTEL["lavender"]


def normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()


def parse_universe(value: str, selected: str) -> tuple[str, ...]:
    raw = str(value or "").replace("\n", ",").split(",")
    return unique_tickers([selected, *raw])


def fetch_expirations(symbol: str, *, deadline: float | None = None) -> tuple[str, ...]:
    try:
        return YAHOO_OPTIONS.call(("expirations", symbol), lambda: tuple(yf.Ticker(symbol).options),
            deadline=min(deadline if deadline is not None else float("inf"), time.perf_counter() + 3), valid=bool)
    except Exception:
        return ()


def fetch_cboe_snapshot(
    symbol: str,
    *, deadline: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    return CBOE_OPTIONS.call(("snapshot", symbol), lambda: fetch_cboe_delayed_options(symbol, timeout=6),
        deadline=min(deadline if deadline is not None else float("inf"), time.perf_counter() + 6),
        valid=lambda result: not result[0].empty)


def _yahoo_chain(symbol: str, expiry: str):
    # yfinance creates its Options namedtuple inside option_chain; normalize it
    # before caching because that transient type cannot be pickled.
    chain = yf.Ticker(symbol).option_chain(expiry)
    calls = chain.calls if isinstance(chain.calls, pd.DataFrame) else pd.DataFrame()
    puts = chain.puts if isinstance(chain.puts, pd.DataFrame) else pd.DataFrame()
    underlying = dict(chain.underlying) if isinstance(chain.underlying, Mapping) else {}
    return calls, puts, underlying


def fetch_chain(
    symbol: str, expiry: str, *, deadline: float | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    str | None,
    str,
    str,
]:
    try:
        calls, puts, underlying = YAHOO_OPTIONS.call(("chain", symbol, expiry), lambda: _yahoo_chain(symbol, expiry),
            deadline=min(deadline if deadline is not None else float("inf"), time.perf_counter() + 3),
            valid=lambda result: not result[0].empty and not result[1].empty)
        if not calls.empty and not puts.empty:
            return (
                calls.copy(),
                puts.copy(),
                dict(underlying),
                None,
                "Yahoo Finance",
                "",
            )
        yahoo_error = "Empty Yahoo option chain"
    except Exception as exc:
        yahoo_error = str(exc)

    try:
        cboe_frame, underlying, timestamp = fetch_cboe_snapshot(symbol, deadline=deadline)
        calls, puts = select_cboe_expiry(cboe_frame, expiry)
        if calls.empty or puts.empty:
            raise ValueError("No matching Cboe expiration")
        return calls, puts, underlying, None, "Cboe delayed quotes", timestamp
    except Exception as exc:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            f"Yahoo: {yahoo_error}; Cboe: {exc}",
            "",
            "",
        )


def fetch_cboe_expirations(symbol: str, *, deadline: float | None = None) -> tuple[str, ...]:
    try:
        frame, _, _ = fetch_cboe_snapshot(symbol, deadline=deadline)
        return expirations_from_cboe(frame)
    except Exception:
        return ()


def available_expirations(symbol: str, *, deadline: float | None = None) -> tuple[str, ...]:
    """Use Yahoo's calendar when available and Cboe's when Yahoo is blocked."""
    return fetch_expirations(symbol, deadline=deadline) or fetch_cboe_expirations(symbol, deadline=deadline)


def nearest_expiry(expirations: tuple[str, ...], target_dte: int, as_of: date) -> str | None:
    eligible = []
    for expiry in expirations:
        try:
            dte = (pd.Timestamp(expiry).date() - as_of).days
        except Exception:
            continue
        if dte >= 2:
            eligible.append((abs(dte - target_dte), dte, expiry))
    return min(eligible)[2] if eligible else None


def close_series(raw_frames: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    frame = raw_frames.get(ticker)
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    adjusted = adjusted_ohlcv(frame)
    return pd.to_numeric(adjusted.get("Close"), errors="coerce").dropna()


def latest_value(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def fmt(value: float, suffix: str = "", digits: int = 1) -> str:
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "N/A"


def quadrant_label(price_return: float, richness_percentile: float) -> str:
    if not np.isfinite(price_return) or not np.isfinite(richness_percentile):
        return "Unavailable"
    direction = "Up" if price_return >= 0 else "Down"
    valuation = "Expensive" if richness_percentile >= 50 else "Cheap"
    return f"{direction} + {valuation}"


def compass_chart(
    frame: pd.DataFrame,
    selected: str,
    *,
    return_column: str,
    return_label: str,
) -> go.Figure:
    plot = frame.dropna(subset=[return_column, "iv_richness_percentile"]).copy()
    if plot.empty:
        return go.Figure()

    plot["price_return_pct"] = pd.to_numeric(plot[return_column], errors="coerce") * 100.0
    max_abs = float(np.nanmax(np.abs(plot["price_return_pct"]))) if not plot.empty else 5.0
    x_extent = max(5.0, max_abs * 1.22)

    fig = go.Figure()
    quadrants = (
        (-x_extent, 0, 50, 100, "rgba(192,80,77,.13)"),
        (0, x_extent, 50, 100, "rgba(255,192,0,.12)"),
        (-x_extent, 0, 0, 50, "rgba(91,155,213,.11)"),
        (0, x_extent, 0, 50, "rgba(112,173,71,.13)"),
    )
    for x0, x1, y0, y1, color in quadrants:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    fig.add_hline(y=50, line=dict(color="#666666", width=1))
    fig.add_vline(x=0, line=dict(color="#666666", width=1))

    point_colors = {
        "Up + Cheap": PASTEL["sage"],
        "Up + Expensive": PASTEL["amber"],
        "Down + Cheap": PASTEL["periwinkle"],
        "Down + Expensive": PASTEL["rose"],
    }
    plot["quadrant"] = [
        quadrant_label(float(ret), float(rank))
        for ret, rank in zip(plot[return_column], plot["iv_richness_percentile"], strict=False)
    ]

    fig.add_trace(
        go.Scatter(
            x=plot["price_return_pct"],
            y=plot["iv_richness_percentile"],
            text=plot["ticker"],
            customdata=np.column_stack(
                [
                    plot["atm_iv"] * 100.0,
                    plot["realized_vol_21d"] * 100.0,
                    plot["iv_richness"] * 100.0,
                    plot["expiry"],
                    plot["quadrant"],
                ]
            ),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=[16 if ticker == selected else 11 for ticker in plot["ticker"]],
                color=[
                    SELECTED_COLOR if ticker == selected else point_colors.get(quadrant, PEER_COLOR)
                    for ticker, quadrant in zip(plot["ticker"], plot["quadrant"], strict=False)
                ],
                line=dict(
                    color=["#111111" if ticker == selected else "#ffffff" for ticker in plot["ticker"]],
                    width=[2.0 if ticker == selected else 1.0 for ticker in plot["ticker"]],
                ),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[4]}"
                f"<br>{return_label}: %{{x:+.1f}}%"
                "<br>Options richness rank: %{y:.0f}"
                "<br>ATM IV: %{customdata[0]:.1f}%"
                "<br>21D realized vol: %{customdata[1]:.1f}%"
                "<br>IV - realized: %{customdata[2]:+.1f} vol pts"
                "<br>Expiry: %{customdata[3]}<extra></extra>"
            ),
        )
    )

    annotations = (
        (-x_extent * 0.52, 91, "DOWN + EXPENSIVE"),
        (x_extent * 0.52, 91, "UP + EXPENSIVE"),
        (-x_extent * 0.52, 9, "DOWN + CHEAP"),
        (x_extent * 0.52, 9, "UP + CHEAP"),
    )
    for x, y, label in annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, color="#4b5563"),
        )

    fig.update_xaxes(
        title=f"{return_label} price return",
        ticksuffix="%",
        range=[-x_extent, x_extent],
        zeroline=False,
        showgrid=False,
    )
    fig.update_yaxes(
        title="Options richness percentile vs loaded universe",
        range=[0, 100],
        tickvals=[0, 25, 50, 75, 100],
        showgrid=False,
    )
    fig.update_layout(
        height=610,
        template="plotly_white",
        margin=dict(l=58, r=28, t=24, b=55),
        showlegend=False,
        hovermode="closest",
        font=dict(family="Arial, sans-serif", color="#1f2937"),
    )
    return fig



st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1560)

with st.sidebar:
    render_sidebar_about("16_Options_Positioning_Compass.py")
    st.header("Compass setup")
    selected = normalize_ticker(st.text_input("Highlight ticker", value="QQQ"))
    universe_text = st.text_area(
        "Comparison universe",
        value=DEFAULT_UNIVERSE,
        height=105,
        help="Comma-separated liquid tickers. Cheap/expensive is ranked only within this loaded universe.",
    )
    momentum_horizon = st.selectbox("Price direction", ("1 month", "3 months"), index=0)
    target_dte = st.slider("Options tenor", min_value=14, max_value=120, value=45, step=1, format="%d DTE")
    risk_free_rate = 0.04

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "See which markets are rising or falling while their options screen expensive or cheap relative to the rest of the loaded universe."
        ),
        eyebrow="ADFM Options Intelligence",
    )
)

if not selected:
    st.error("Enter a focus ticker.")
    render_footer()
    st.stop()

as_of_date = datetime.now(NY_TZ).date()
universe = parse_universe(universe_text, selected)
if len(universe) < 2:
    st.error("Add at least one comparison ticker so the cross-sectional ranks are meaningful.")
    render_footer()
    st.stop()

raw_prices, price_failures = fetch_daily_ohlcv(universe, period="1y")
price_metrics: dict[str, dict[str, float]] = {}
for symbol in universe:
    close = close_series(raw_prices, symbol)
    rvol = annualized_realized_volatility(close, 21)
    price_metrics[symbol] = {
        "spot": latest_value(close),
        "realized_vol_21d": latest_value(rvol) / 100.0,
        "return_5d": float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else np.nan,
        "return_21d": float(close.iloc[-1] / close.iloc[-22] - 1.0) if len(close) >= 22 else np.nan,
        "return_63d": float(close.iloc[-1] / close.iloc[-64] - 1.0) if len(close) >= 64 else np.nan,
    }

universe_rows: list[dict[str, object]] = []
provider_errors: list[dict[str, str]] = []
options_deadline = time.perf_counter() + 20.0
with st.spinner("Loading current option-chain snapshots…"):
    for symbol in universe:
        expirations = available_expirations(symbol, deadline=options_deadline)
        expiry = nearest_expiry(expirations, target_dte, as_of_date)
        if expiry is None:
            provider_errors.append({"Ticker": symbol, "Issue": "No eligible option expiration returned"})
            continue
        calls, puts, underlying, error, source, source_timestamp = fetch_chain(
            symbol, expiry, deadline=options_deadline
        )
        if error or calls.empty or puts.empty:
            provider_errors.append({"Ticker": symbol, "Issue": error or "Empty option chain"})
            continue
        spot = price_metrics[symbol]["spot"]
        if not np.isfinite(spot) or spot <= 0:
            spot = float(pd.to_numeric(underlying.get("regularMarketPrice", np.nan), errors="coerce"))
        if not np.isfinite(spot) or spot <= 0:
            provider_errors.append({"Ticker": symbol, "Issue": "No valid underlying price"})
            continue
        snapshot = option_snapshot(
            calls,
            puts,
            spot=spot,
            expiry=expiry,
            as_of=as_of_date,
            risk_free_rate=float(risk_free_rate),
        )
        if not np.isfinite(snapshot["atm_iv"]):
            provider_errors.append({"Ticker": symbol, "Issue": "No valid strikes and implied volatility in the option chain"})
            continue
        universe_rows.append(
            {
                "ticker": symbol,
                "chain_source": source,
                "source_timestamp": source_timestamp,
                **price_metrics[symbol],
                **snapshot,
            }
        )

universe_frame = add_cross_sectional_ranks(pd.DataFrame(universe_rows)) if universe_rows else pd.DataFrame()
if universe_frame.empty:
    st.error("No usable option chains were returned for the current universe.")
    if provider_errors:
        st.dataframe(pd.DataFrame(provider_errors), hide_index=True, width="stretch")
    render_footer()
    st.stop()

loaded_tickers = set(universe_frame.get("ticker", []))
highlight_ticker = selected if selected in loaded_tickers else ""
if selected and not highlight_ticker:
    st.caption(f"{selected} did not return a usable chain on this run; the peer map is still shown.")
return_column = "return_21d" if momentum_horizon == "1 month" else "return_63d"
return_label = "1M" if momentum_horizon == "1 month" else "3M"

universe_frame["price_return"] = pd.to_numeric(
    universe_frame[return_column], errors="coerce"
)
universe_frame["quadrant"] = [
    quadrant_label(float(ret), float(rank))
    for ret, rank in zip(
        universe_frame["price_return"],
        universe_frame["iv_richness_percentile"],
        strict=False,
    )
]

source_names = sorted(
    {
        str(value)
        for value in universe_frame.get("chain_source", pd.Series(dtype=str)).dropna()
        if str(value)
    }
)
render_status_line(
    as_of=as_of_date.isoformat(),
    trend=f"{return_label} return",
    options_tenor=f"~{target_dte} DTE",
    coverage=f"{len(universe_frame)}/{len(universe)} tickers",
    source=", ".join(source_names) if source_names else "Current option chains",
)

render_section_header(
    "Price direction vs options richness",
    (
        f"Horizontal axis is {return_label} price return. Vertical axis ranks ATM implied volatility minus "
        "21-day realized volatility across the currently loaded universe. Above 50 = expensive vs peers; "
        "below 50 = cheap vs peers."
    ),
)
st.plotly_chart(
    compass_chart(
        universe_frame,
        highlight_ticker,
        return_column=return_column,
        return_label=return_label,
    ),
    width="stretch",
    config={"displaylogo": False},
)

display = universe_frame[
    [
        "ticker",
        "quadrant",
        "price_return",
        "atm_iv",
        "realized_vol_21d",
        "iv_richness",
        "iv_richness_percentile",
        "expiry",
        "dte",
    ]
].copy()
display = display.rename(
    columns={
        "ticker": "Ticker",
        "quadrant": "Quadrant",
        "price_return": f"{return_label} Return",
        "atm_iv": "ATM IV",
        "realized_vol_21d": "21D Realized",
        "iv_richness": "IV - Realized",
        "iv_richness_percentile": "Richness Rank",
        "expiry": "Expiry",
        "dte": "DTE",
    }
)
quadrant_order = {
    "Up + Cheap": 0,
    "Up + Expensive": 1,
    "Down + Cheap": 2,
    "Down + Expensive": 3,
    "Unavailable": 4,
}
display["_order"] = display["Quadrant"].map(quadrant_order).fillna(4)
display = display.sort_values(
    ["_order", f"{return_label} Return"],
    ascending=[True, False],
).drop(columns="_order")

quadrant_colors = {
    "Up + Cheap": "#237a3b",
    "Up + Expensive": "#9a6700",
    "Down + Cheap": "#2f5597",
    "Down + Expensive": "#b13030",
}
styled = display.style.format(
    {
        f"{return_label} Return": "{:+.1%}",
        "ATM IV": "{:.1%}",
        "21D Realized": "{:.1%}",
        "IV - Realized": "{:+.1%}",
        "Richness Rank": "{:.0f}",
        "DTE": "{:.0f}",
    },
    na_rep="N/A",
).map(
    lambda value: f"color: {quadrant_colors.get(str(value), '#171717')}; font-weight: 700",
    subset=["Quadrant"],
)
st.dataframe(styled, hide_index=True, width="stretch", height="auto")

if provider_errors:
    st.caption(
        f"{len(provider_errors)} ticker(s) did not return a usable chain inside the request window and are excluded from the map."
    )

with st.expander("Methodology & coverage", expanded=False):
    st.markdown(
        f"""
        **How to read the four quadrants**

        - **Up + Expensive:** {return_label} price return is positive and IV richness ranks above the peer median.
        - **Up + Cheap:** {return_label} price return is positive and IV richness ranks below the peer median.
        - **Down + Expensive:** {return_label} price return is negative and IV richness ranks above the peer median.
        - **Down + Cheap:** {return_label} price return is negative and IV richness ranks below the peer median.

        **Options richness** is ATM implied volatility minus annualized 21-session realized volatility. The chart uses the
        cross-sectional percentile of that spread within the tickers successfully loaded on this run. It is therefore a
        relative current-snapshot measure, not a historical IV rank.

        The option expiration shown for each ticker is the available expiry nearest the selected {target_dte}-day tenor.
        Current option chains come from Yahoo Finance when available, with Cboe delayed quotes as fallback. The 4% risk-free
        assumption is used only in chain calculations that require estimated option delta; it does not determine the
        cheap/expensive quadrant.
        """
    )
    if provider_errors:
        st.dataframe(
            pd.DataFrame(provider_errors).drop_duplicates(),
            hide_index=True,
            width="stretch",
        )

render_footer()
