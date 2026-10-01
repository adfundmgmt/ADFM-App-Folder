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

TITLE = "Options Relative Value Compass"
DEFAULT_UNIVERSE = "SPY, QQQ, IWM, DIA, TLT, GLD, USO, SMH, EEM, HYG, LQD"
NY_TZ = ZoneInfo("America/New_York")


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
    trend = "Positive Trend" if price_return >= 0 else "Negative Trend"
    relative_value = "IV Rich" if richness_percentile >= 50 else "IV Cheap"
    return f"{trend} · {relative_value}"


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
    x_extent = max(5.0, max_abs * 1.18)

    fig = go.Figure()
    quadrants = (
        (-x_extent, 0, 50, 100, "rgba(192,80,77,.055)"),
        (0, x_extent, 50, 100, "rgba(255,192,0,.050)"),
        (-x_extent, 0, 0, 50, "rgba(91,155,213,.045)"),
        (0, x_extent, 0, 50, "rgba(112,173,71,.050)"),
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

    fig.add_hline(y=50, line=dict(color="#7a7a7a", width=1))
    fig.add_vline(x=0, line=dict(color="#7a7a7a", width=1))

    point_colors = {
        "Positive Trend · IV Cheap": PASTEL["sage"],
        "Positive Trend · IV Rich": PASTEL["amber"],
        "Negative Trend · IV Cheap": PASTEL["periwinkle"],
        "Negative Trend · IV Rich": PASTEL["rose"],
    }
    plot["quadrant"] = [
        quadrant_label(float(ret), float(rank))
        for ret, rank in zip(plot[return_column], plot["iv_richness_percentile"], strict=False)
    ]
    display_labels = [
        f"<b>{ticker}</b>" if ticker == selected else str(ticker)
        for ticker in plot["ticker"]
    ]
    text_positions = [
        "bottom center" if float(rank) >= 90 else "top center"
        for rank in plot["iv_richness_percentile"]
    ]

    fig.add_trace(
        go.Scatter(
            x=plot["price_return_pct"],
            y=plot["iv_richness_percentile"],
            text=display_labels,
            customdata=np.column_stack(
                [
                    plot["ticker"],
                    plot["atm_iv"] * 100.0,
                    plot["realized_vol_21d"] * 100.0,
                    plot["iv_richness"] * 100.0,
                    plot["expiry"],
                    plot["quadrant"],
                ]
            ),
            mode="markers+text",
            textposition=text_positions,
            textfont=dict(size=11, color="#172033"),
            marker=dict(
                size=[16 if ticker == selected else 10 for ticker in plot["ticker"]],
                color=[
                    point_colors.get(quadrant, PASTEL["slate_blue"])
                    for quadrant in plot["quadrant"]
                ],
                opacity=0.95,
                line=dict(
                    color=["#111111" if ticker == selected else "#ffffff" for ticker in plot["ticker"]],
                    width=[2.2 if ticker == selected else 1.0 for ticker in plot["ticker"]],
                ),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[5]}"
                f"<br>{return_label} return: %{{x:+.1f}}%"
                "<br>IV richness percentile: %{y:.0f}"
                "<br>ATM IV: %{customdata[1]:.1f}%"
                "<br>21D realized vol: %{customdata[2]:.1f}%"
                "<br>IV-RV spread: %{customdata[3]:+.1f} vol pts"
                "<br>Expiry: %{customdata[4]}<extra></extra>"
            ),
        )
    )

    quadrant_tags = (
        (0.015, 0.975, "NEGATIVE TREND<br><b>IV RICH</b>", "left", "top"),
        (0.985, 0.975, "POSITIVE TREND<br><b>IV RICH</b>", "right", "top"),
        (0.015, 0.025, "NEGATIVE TREND<br><b>IV CHEAP</b>", "left", "bottom"),
        (0.985, 0.025, "POSITIVE TREND<br><b>IV CHEAP</b>", "right", "bottom"),
    )
    for x, y, label, xanchor, yanchor in quadrant_tags:
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor=xanchor,
            yanchor=yanchor,
            align=xanchor,
            font=dict(size=10, color="#5f6672"),
            bgcolor="rgba(255,255,255,.78)",
            bordercolor="rgba(120,120,120,.18)",
            borderwidth=1,
            borderpad=4,
        )

    fig.update_xaxes(
        title=f"{return_label} return",
        ticksuffix="%",
        range=[-x_extent, x_extent],
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,.12)",
        tickfont=dict(size=11, color="#667085"),
        title_font=dict(size=12, color="#667085"),
    )
    fig.update_yaxes(
        title="IV richness percentile",
        range=[0, 100],
        tickvals=[0, 25, 50, 75, 100],
        showgrid=True,
        gridcolor="rgba(148,163,184,.12)",
        tickfont=dict(size=11, color="#667085"),
        title_font=dict(size=12, color="#667085"),
    )
    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(l=62, r=30, t=22, b=56),
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
    st.header("Relative value setup")
    selected = normalize_ticker(st.text_input("Highlight ticker", value="QQQ"))
    universe_text = st.text_area(
        "Comparison universe",
        value=DEFAULT_UNIVERSE,
        height=105,
        help="Comma-separated liquid tickers. IV richness is ranked cross-sectionally within the successfully loaded universe.",
    )
    momentum_horizon = st.selectbox("Trend horizon", ("1 month", "3 months"), index=0)
    target_dte = st.slider("Target options tenor", min_value=14, max_value=120, value=45, step=1, format="%d DTE")
    risk_free_rate = 0.04

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Cross-sectional relative-value map of underlying price trend versus volatility premium. Identify where implied volatility is rich or cheap to recent realized volatility while the underlying trend is strengthening or weakening."
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
    "Trend vs Volatility Relative Value",
    (
        f"The horizontal axis is {return_label} underlying return. The vertical axis is the cross-sectional percentile "
        "of ATM implied volatility minus 21-day realized volatility. Read the map as trend on the x-axis and "
        "relative option richness on the y-axis."
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
    config={"displayModeBar": False, "displaylogo": False, "responsive": True},
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
        "quadrant": "Regime",
        "price_return": f"{return_label} Return",
        "atm_iv": "ATM IV",
        "realized_vol_21d": "21D Realized",
        "iv_richness": "IV-RV Spread",
        "iv_richness_percentile": "IV Richness Pctl",
        "expiry": "Expiry",
        "dte": "DTE",
    }
)
quadrant_order = {
    "Positive Trend · IV Cheap": 0,
    "Positive Trend · IV Rich": 1,
    "Negative Trend · IV Cheap": 2,
    "Negative Trend · IV Rich": 3,
    "Unavailable": 4,
}
display["_order"] = display["Regime"].map(quadrant_order).fillna(4)
display = display.sort_values(
    ["_order", f"{return_label} Return"],
    ascending=[True, False],
).drop(columns="_order")

quadrant_colors = {
    "Positive Trend · IV Cheap": "#237a3b",
    "Positive Trend · IV Rich": "#9a6700",
    "Negative Trend · IV Cheap": "#2f5597",
    "Negative Trend · IV Rich": "#b13030",
}
styled = display.style.format(
    {
        f"{return_label} Return": "{:+.1%}",
        "ATM IV": "{:.1%}",
        "21D Realized": "{:.1%}",
        "IV-RV Spread": "{:+.1%}",
        "IV Richness Pctl": "{:.0f}",
        "DTE": "{:.0f}",
    },
    na_rep="N/A",
).map(
    lambda value: f"color: {quadrant_colors.get(str(value), '#171717')}; font-weight: 700",
    subset=["Regime"],
)
st.dataframe(styled, hide_index=True, width="stretch", height="auto")

if provider_errors:
    st.caption(
        f"{len(provider_errors)} ticker(s) did not return a usable chain inside the request window and are excluded from the map."
    )

with st.expander("Methodology & coverage", expanded=False):
    st.markdown(
        f"""
        **Interpretation framework**

        - **Positive Trend · IV Rich:** the underlying is advancing, while the volatility premium sits above the peer median.
        - **Positive Trend · IV Cheap:** the underlying is advancing, while the volatility premium sits below the peer median.
        - **Negative Trend · IV Rich:** the underlying is declining, while the volatility premium sits above the peer median.
        - **Negative Trend · IV Cheap:** the underlying is declining, while the volatility premium sits below the peer median.

        **IV richness** is defined as ATM implied volatility minus annualized 21-session realized volatility. The vertical
        axis ranks that spread across the tickers successfully loaded on the current run. A high percentile means implied
        volatility carries a larger premium to recent realized volatility than most peers; a low percentile means that
        premium is relatively compressed. This is a current cross-sectional relative-value measure, not historical IV rank.

        The option expiration shown for each ticker is the available expiry nearest the selected {target_dte}-day tenor.
        Current option chains come from Yahoo Finance when available, with Cboe delayed quotes as fallback. The 4% risk-free
        assumption is used only in chain calculations that require estimated option delta; it does not determine the
        relative-value classification.
        """
    )
    if provider_errors:
        st.dataframe(
            pd.DataFrame(provider_errors).drop_duplicates(),
            hide_index=True,
            width="stretch",
        )

render_footer()
