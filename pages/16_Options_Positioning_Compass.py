"""Current-snapshot options positioning, skew, and volatility explorer."""

from __future__ import annotations

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
    build_positioning_commentary,
    option_snapshot,
    ordinal,
    prepare_chain,
)
from adfm_core.options_sources import (
    expirations_from_cboe,
    fetch_cboe_delayed_options,
    select_cboe_expiry,
)
from adfm_core.palette import PASTEL
from adfm_core.relative_volatility import annualized_realized_volatility
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_section_header,
    render_sidebar_about,
    render_selection_note,
    render_status_line,
)

TITLE = "Options Positioning Compass"
DEFAULT_UNIVERSE = "SPY, QQQ, IWM, DIA, TLT, GLD, USO, SMH, EEM, HYG, LQD"
NY_TZ = ZoneInfo("America/New_York")
GRID_COLOR = "rgba(148,163,184,0.23)"
PRIMARY_COLOR = PASTEL["blue"]
PUT_COLOR = PASTEL["coral"]
CALL_COLOR = PASTEL["periwinkle"]
SELECTED_COLOR = PASTEL["rose"]
PEER_COLOR = PASTEL["lavender"]


def normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()


def parse_universe(value: str, selected: str) -> tuple[str, ...]:
    raw = str(value or "").replace("\n", ",").split(",")
    return unique_tickers([selected, *raw])


@st.cache_data(ttl=900, show_spinner=False)
def fetch_expirations(symbol: str) -> tuple[str, ...]:
    try:
        return tuple(yf.Ticker(symbol).options)
    except Exception:
        return ()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_cboe_snapshot(
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    return fetch_cboe_delayed_options(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_chain(
    symbol: str, expiry: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    str | None,
    str,
    str,
]:
    try:
        chain = yf.Ticker(symbol).option_chain(expiry)
        underlying = chain.underlying if isinstance(chain.underlying, Mapping) else {}
        if not chain.calls.empty and not chain.puts.empty:
            return (
                chain.calls.copy(),
                chain.puts.copy(),
                dict(underlying),
                None,
                "Yahoo Finance",
                "",
            )
        yahoo_error = "Empty Yahoo option chain"
    except Exception as exc:
        yahoo_error = str(exc)

    try:
        cboe_frame, underlying, timestamp = fetch_cboe_snapshot(symbol)
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_cboe_expirations(symbol: str) -> tuple[str, ...]:
    try:
        frame, _, _ = fetch_cboe_snapshot(symbol)
        return expirations_from_cboe(frame)
    except Exception:
        return ()


def available_expirations(symbol: str) -> tuple[str, ...]:
    """Use Yahoo's calendar when available and Cboe's when Yahoo is blocked."""
    return fetch_expirations(symbol) or fetch_cboe_expirations(symbol)


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


def money(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.0f}"


def compass_chart(frame: pd.DataFrame, selected: str) -> go.Figure:
    plot = frame.dropna(subset=["put_skew_percentile", "iv_richness_percentile"])
    fig = go.Figure()
    quadrant_colors = (
        (0, 50, 50, 100, "rgba(83,196,174,.20)"),
        (50, 100, 50, 100, "rgba(199,112,169,.20)"),
        (0, 50, 0, 50, "rgba(98,180,168,.12)"),
        (50, 100, 0, 50, "rgba(204,112,169,.13)"),
    )
    for x0, x1, y0, y1, color in quadrant_colors:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
    fig.add_hline(y=50, line=dict(color="#475569", width=1))
    fig.add_vline(x=50, line=dict(color="#475569", width=1))
    fig.add_trace(
        go.Scatter(
            x=plot["put_skew_percentile"],
            y=plot["iv_richness_percentile"],
            text=plot["ticker"],
            customdata=np.column_stack(
                [plot["atm_iv"] * 100.0, plot["put_skew"] * 100.0, plot["expiry"]]
            ),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=[15 if ticker == selected else 10 for ticker in plot["ticker"]],
                color=[SELECTED_COLOR if ticker == selected else PEER_COLOR for ticker in plot["ticker"]],
                line=dict(color="white", width=1.2),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>IV richness rank: %{y:.0f}<br>Put-skew rank: %{x:.0f}"
                "<br>ATM IV: %{customdata[0]:.1f}%<br>Put skew: %{customdata[1]:+.1f} vol pts"
                "<br>Expiry: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    annotations = (
        (24, 88, "Rich IV<br>Call/upside skew"),
        (76, 88, "Rich IV<br>Put/downside skew"),
        (24, 12, "Cheap IV<br>Call/upside skew"),
        (76, 12, "Cheap IV<br>Put/downside skew"),
    )
    for x, y, text in annotations:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=12, color="#64748b"))
    fig.update_xaxes(title="25-delta put-skew percentile in selected universe", range=[-4, 104], showgrid=False)
    fig.update_yaxes(title="IV-minus-realized percentile in selected universe", range=[-4, 104], showgrid=False)
    fig.update_layout(
        height=620,
        template="plotly_white",
        margin=dict(l=55, r=25, t=25, b=55),
        showlegend=False,
        hovermode="closest",
        font=dict(family="Arial, sans-serif", color="#1f2937"),
    )
    return fig



def term_structure_chart(frame: pd.DataFrame) -> go.Figure:
    plot = frame.sort_values("dte")
    fig = go.Figure()
    for column, label, color, dash in (
        ("atm_iv", "ATM IV", PRIMARY_COLOR, "solid"),
        ("put_25d_iv", "25-delta put IV", PUT_COLOR, "dash"),
        ("call_25d_iv", "25-delta call IV", CALL_COLOR, "dot"),
    ):
        fig.add_trace(
            go.Scatter(
                x=plot["dte"],
                y=plot[column] * 100.0,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate="%{x:.0f} DTE<br>%{y:.1f}%<extra></extra>",
            )
        )
    fig.update_xaxes(title="Days to expiration", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title="Implied volatility", ticksuffix="%", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_layout(
        height=445,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=45, r=25, t=30, b=45),
        legend=dict(orientation="h", y=1.04, x=0),
    )
    return fig


def iv_surface_chart(
    term_chains: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame]],
    risk_free_rate: float,
) -> go.Figure:
    grid = np.arange(80.0, 120.1, 2.5)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for snapshot, calls, puts in term_chains:
        spot = float(snapshot["spot"])
        time_years = max(float(snapshot["dte"]), 1.0) / 365.0
        call_frame = prepare_chain(
            calls,
            "call",
            spot=spot,
            time_years=time_years,
            risk_free_rate=risk_free_rate,
        )
        put_frame = prepare_chain(
            puts,
            "put",
            spot=spot,
            time_years=time_years,
            risk_free_rate=risk_free_rate,
        )
        call_frame["moneyness"] = call_frame["strike"] / spot * 100.0
        put_frame["moneyness"] = put_frame["strike"] / spot * 100.0
        otm = pd.concat(
            [
                put_frame.loc[put_frame["moneyness"].le(100.0)],
                call_frame.loc[call_frame["moneyness"].gt(100.0)],
            ],
            ignore_index=True,
        ).dropna(subset=["moneyness", "impliedVolatility"])
        otm = otm.loc[otm["impliedVolatility"].between(0.02, 5.0)].sort_values("moneyness")
        otm = otm.groupby("moneyness", as_index=False)["impliedVolatility"].median()
        if len(otm) < 2:
            continue
        values = np.interp(grid, otm["moneyness"], otm["impliedVolatility"] * 100.0, left=np.nan, right=np.nan)
        rows.append(values)
        labels.append(f"{snapshot['expiry']} · {int(float(snapshot['dte']))}D")
    fig = go.Figure(
        go.Heatmap(
            x=grid,
            y=labels,
            z=np.asarray(rows),
            colorscale="RdBu_r",
            colorbar=dict(title="IV %"),
            hovertemplate="%{y}<br>Moneyness: %{x:.1f}%<br>IV: %{z:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=100, line=dict(color="#111827", width=1.5))
    fig.update_xaxes(title="Strike / spot", ticksuffix="%")
    fig.update_yaxes(title="Expiration", autorange="reversed")
    fig.update_layout(height=max(390, 70 * len(labels)), template="plotly_white", margin=dict(l=55, r=35, t=25, b=50))
    return fig


st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1560)

with st.sidebar:
    render_sidebar_about("16_Options_Positioning_Compass.py")
    st.header("Compass setup")
    selected = normalize_ticker(st.text_input("Focus ticker", value="QQQ"))
    universe_text = st.text_area(
        "Comparison universe",
        value=DEFAULT_UNIVERSE,
        height=105,
        help="Comma-separated Yahoo Finance tickers. Ranks are calculated only within this current list.",
    )
    target_dte = st.slider("Target expiration", min_value=14, max_value=120, value=45, step=1, format="%d DTE")
    term_count = st.slider("Term-structure expirations", min_value=3, max_value=10, value=6)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    st.caption("The rate is used only to estimate option deltas for the 25-delta skew comparison.")

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Map implied-volatility richness and downside skew across a liquid watchlist, then inspect the selected ticker's term structure, IV surface, and largest option activity."
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
    }

universe_rows: list[dict[str, object]] = []
provider_errors: list[dict[str, str]] = []
with st.spinner("Loading current option-chain snapshots…"):
    for symbol in universe:
        expirations = available_expirations(symbol)
        expiry = nearest_expiry(expirations, target_dte, as_of_date)
        if expiry is None:
            provider_errors.append({"Ticker": symbol, "Issue": "No eligible option expiration returned"})
            continue
        calls, puts, underlying, error, source, source_timestamp = fetch_chain(
            symbol, expiry
        )
        if error or calls.empty or puts.empty:
            provider_errors.append({"Ticker": symbol, "Issue": error or "Empty option chain"})
            continue
        spot = price_metrics[symbol]["spot"]
        if not np.isfinite(spot):
            spot = float(underlying.get("regularMarketPrice", np.nan))
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
        universe_rows.append(
            {
                "ticker": symbol,
                "chain_source": source,
                "source_timestamp": source_timestamp,
                **snapshot,
                **price_metrics[symbol],
            }
        )

universe_frame = add_cross_sectional_ranks(pd.DataFrame(universe_rows)) if universe_rows else pd.DataFrame()
if universe_frame.empty or selected not in set(universe_frame.get("ticker", [])):
    st.error(f"Neither Yahoo nor Cboe returned a usable option chain for {selected}.")
    if provider_errors:
        st.dataframe(pd.DataFrame(provider_errors), hide_index=True, width="stretch")
    render_footer()
    st.stop()

selected_row = universe_frame.loc[universe_frame["ticker"].eq(selected)].iloc[0]
selected_expiry = str(selected_row["expiry"])
selected_calls, selected_puts, _, _, selected_source, selected_timestamp = fetch_chain(
    selected, selected_expiry
)
source_detail = selected_source
if selected_timestamp:
    source_detail = f"{selected_source} · snapshot {selected_timestamp} UTC"
render_status_line(
    as_of=as_of_date.isoformat(),
    focus=selected,
    target_expiration=f"{selected_expiry} ({int(selected_row['dte'])} DTE)",
    rank_basis=f"{len(universe_frame)} option chains",
    source=source_detail,
)

render_selection_note("Current positioning read", build_positioning_commentary(selected_row))
render_kpi_cards(
    [
        ("ATM implied vol", fmt(float(selected_row["atm_iv"]) * 100.0, "%"), f"{int(selected_row['dte'])} DTE expiration"),
        ("21D realized vol", fmt(float(selected_row["realized_vol_21d"]) * 100.0, "%"), "Annualized close-to-close"),
        ("IV richness rank", ordinal(float(selected_row["iv_richness_percentile"])), "Current selected universe"),
        ("25D put skew", fmt(float(selected_row["put_skew"]) * 100.0, " vol", 1), "Put IV minus call IV"),
        ("Put/call volume", fmt(float(selected_row["put_call_volume"]), "x", 2), "Aggregate current chain"),
        ("Premium activity", money(float(selected_row["premium_activity"])), "Mid/last × volume × 100"),
    ]
)

compass_tab, structure_tab, activity_tab, data_tab, methodology_tab = st.tabs(
    ["Compass", "Term structure + surface", "Premium activity", "Data", "Methodology"]
)

with compass_tab:
    render_section_header(
        "Current-snapshot volatility compass",
        "Both axes are peer ranks within the tickers successfully loaded above. IV richness is ATM IV minus 21-session realized volatility; skew is 25-delta put IV minus call IV.",
    )
    st.plotly_chart(compass_chart(universe_frame, selected), width="stretch", config={"displaylogo": False})
    compass_table = universe_frame[
        [
            "ticker",
            "expiry",
            "dte",
            "atm_iv",
            "realized_vol_21d",
            "iv_richness",
            "iv_richness_percentile",
            "put_skew",
            "put_skew_percentile",
            "put_call_volume",
            "put_call_oi",
            "return_5d",
            "chain_source",
            "source_timestamp",
        ]
    ].sort_values("put_skew_percentile", ascending=False)
    st.dataframe(
        compass_table.style.format(
            {
                "dte": "{:.0f}",
                "atm_iv": "{:.1%}",
                "realized_vol_21d": "{:.1%}",
                "iv_richness": "{:+.1%}",
                "iv_richness_percentile": "{:.0f}",
                "put_skew": "{:+.1%}",
                "put_skew_percentile": "{:.0f}",
                "put_call_volume": "{:.2f}",
                "put_call_oi": "{:.2f}",
                "return_5d": "{:+.1%}",
            },
            na_rep="N/A",
        ),
        hide_index=True,
        width="stretch",
    )

selected_expirations = available_expirations(selected)
eligible_terms = [
    expiry
    for expiry in selected_expirations
    if 2 <= (pd.Timestamp(expiry).date() - as_of_date).days <= 365
][:term_count]
term_rows: list[dict[str, object]] = []
term_chains: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame]] = []
for expiry in eligible_terms:
    calls, puts, _, error, _, _ = fetch_chain(selected, expiry)
    if error or calls.empty or puts.empty:
        continue
    snapshot = option_snapshot(
        calls,
        puts,
        spot=float(selected_row["spot"]),
        expiry=expiry,
        as_of=as_of_date,
        risk_free_rate=float(risk_free_rate),
    )
    term_rows.append(snapshot)
    term_chains.append((snapshot, calls, puts))
term_frame = pd.DataFrame(term_rows)

with structure_tab:
    if term_frame.empty:
        st.info("No additional expirations were available for the term-structure view.")
    else:
        render_section_header(
            f"{selected} implied-volatility term structure",
            "ATM and estimated 25-delta volatility by expiration. Delta uses the sidebar risk-free-rate assumption and no dividend-yield adjustment.",
        )
        st.plotly_chart(term_structure_chart(term_frame), width="stretch", config={"displaylogo": False})
        render_section_header(
            "Fixed-moneyness IV surface",
            "OTM puts are used below spot and OTM calls above spot; values are linearly interpolated only within observed strikes.",
        )
        st.plotly_chart(
            iv_surface_chart(term_chains, float(risk_free_rate)),
            width="stretch",
            config={"displaylogo": False},
        )
        st.dataframe(
            term_frame.style.format(
                {
                    "dte": "{:.0f}",
                    "spot": "${:,.2f}",
                    "atm_iv": "{:.1%}",
                    "put_25d_iv": "{:.1%}",
                    "call_25d_iv": "{:.1%}",
                    "put_skew": "{:+.1%}",
                    "put_call_volume": "{:.2f}",
                    "put_call_oi": "{:.2f}",
                },
                na_rep="N/A",
            ),
            hide_index=True,
            width="stretch",
        )

with activity_tab:
    render_section_header(
        f"Largest estimated premium activity · {selected_expiry}",
        "Contract premium is estimated as midquote × reported volume × 100 (last price is used when no valid two-sided quote exists). This is aggregate activity, not a tape of individual trades.",
    )
    selected_time_years = max(float(selected_row["dte"]), 1.0) / 365.0
    activity = pd.concat(
        [
            prepare_chain(
                selected_calls,
                "call",
                spot=float(selected_row["spot"]),
                time_years=selected_time_years,
                risk_free_rate=float(risk_free_rate),
            ),
            prepare_chain(
                selected_puts,
                "put",
                spot=float(selected_row["spot"]),
                time_years=selected_time_years,
                risk_free_rate=float(risk_free_rate),
            ),
        ],
        ignore_index=True,
    )
    activity["expiry"] = selected_expiry
    activity["moneyness"] = activity["strike"] / float(selected_row["spot"])
    activity = activity.sort_values("premium_activity", ascending=False).head(30)
    display_activity = activity[
        [
            "contractSymbol",
            "type",
            "strike",
            "moneyness",
            "lastPrice",
            "bid",
            "ask",
            "mid",
            "impliedVolatility",
            "iv_source",
            "volume",
            "openInterest",
            "premium_activity",
            "lastTradeDate",
        ]
    ]
    st.dataframe(
        display_activity.style.format(
            {
                "strike": "${:,.2f}",
                "moneyness": "{:.1%}",
                "lastPrice": "${:,.2f}",
                "bid": "${:,.2f}",
                "ask": "${:,.2f}",
                "mid": "${:,.2f}",
                "impliedVolatility": "{:.1%}",
                "volume": "{:,.0f}",
                "openInterest": "{:,.0f}",
                "premium_activity": "${:,.0f}",
            },
            na_rep="N/A",
        ),
        hide_index=True,
        width="stretch",
        height=670,
    )
    st.warning(
        "Public Yahoo chains do not reveal whether volume was bought or sold, opening or closing, or part of a multi-leg spread. The table must not be read as directional trade flow."
    )

with data_tab:
    render_section_header("Downloadable current snapshot", "Numeric values remain in decimal units in the downloads.")
    dataframe_download("Download compass snapshot", universe_frame, "options_positioning_compass.csv")
    if not term_frame.empty:
        dataframe_download("Download selected term structure", term_frame, f"{selected}_options_term_structure.csv")
    dataframe_download("Download premium activity", display_activity, f"{selected}_{selected_expiry}_premium_activity.csv")
    diagnostics = provider_errors.copy()
    for row in price_failures.to_dict("records"):
        diagnostics.append({"Ticker": str(row.get("Ticker", "")), "Issue": str(row.get("Reason", "Price history unavailable"))})
    if diagnostics:
        st.markdown("**Provider diagnostics**")
        st.dataframe(pd.DataFrame(diagnostics).drop_duplicates(), hide_index=True, width="stretch")

with methodology_tab:
    st.markdown(
        """
        **What is directly observed**

        Expirations, strikes, bid, ask, last price, reported contract volume, open interest, and implied volatility come from Yahoo Finance when available. If Yahoo is unavailable, the same option-chain fields come from Cboe delayed quotes. The page never replaces these measurements with a price-volatility proxy. When a provider returns an obviously invalid IV below 2% or above 500%, the page solves Black-Scholes IV from the quote midpoint, or from the latest option price when no two-sided quote exists; those rows are labeled `Solved from price`.

        **What is calculated**

        - ATM IV averages the valid call and put IV at each side's strike nearest spot.
        - 25-delta contracts are selected by Black-Scholes delta using the configured risk-free rate, no dividend yield, and calendar days to expiration. This is an estimate, not an exchange-supplied Greek.
        - Put skew is 25-delta put IV minus 25-delta call IV. A positive reading means downside puts are richer.
        - IV richness is ATM IV minus annualized 21-session close-to-close realized volatility.
        - Percentiles are mid-ranks across the currently loaded universe. They are **not historical IV rank or historical skew percentile**.
        - Estimated premium activity is quote midpoint × reported contract volume × 100; last price substitutes when there is no valid two-sided quote.

        **Important limitations**

        Yahoo provides a current aggregate chain rather than a complete historical options tape. The page cannot infer buyer versus seller, opening versus closing, spread linkage, dealer gamma, or institutional intent. Quotes can be delayed, stale, crossed, or missing. Generated commentary describes the measurements only and is not an investment recommendation.
        """
    )

render_footer()
