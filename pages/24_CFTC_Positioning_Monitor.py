"""Cross-asset CFTC futures positioning scanner and historical explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.cftc_positioning import (
    COHORTS,
    DEFAULT_COHORT,
    REPORT_LABELS,
    add_metrics,
    build_scanner,
    estimate_notional,
    fetch_contract_history,
    fetch_recent,
    percentile_rank,
    positioning_signal,
    price_proxy,
    rolling_metrics,
)
from adfm_core.market_data import adjusted_ohlcv, configure_yfinance_cache, fetch_daily_ohlcv
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

TITLE = "CFTC Positioning Monitor"
POSITION_COLOR = "#d62728"
PRICE_COLOR = "#111111"
GRID_COLOR = "rgba(148,163,184,0.22)"


@st.cache_data(ttl=21_600, show_spinner=False)
def load_report(report_type: str) -> tuple[pd.DataFrame, str]:
    try:
        return fetch_recent(report_type, years=5), ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@st.cache_data(ttl=21_600, show_spinner=False)
def load_history(report_type: str, contract_code: str) -> tuple[pd.DataFrame, str]:
    try:
        return fetch_contract_history(report_type, contract_code), ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@st.cache_data(ttl=3_600, show_spinner=False)
def load_price(ticker: str) -> tuple[pd.Series, str]:
    frames, failures = fetch_daily_ohlcv((ticker,), period="max")
    frame = frames.get(ticker)
    warning = ""
    if failures is not None and not failures.empty:
        matched = failures.loc[failures["Ticker"].eq(ticker), "Reason"]
        if not matched.empty:
            warning = str(matched.iloc[0])
    if frame is None or frame.empty:
        return pd.Series(dtype=float), warning or "No price history returned"
    close = pd.to_numeric(adjusted_ohlcv(frame).get("Close"), errors="coerce").dropna()
    return close, warning


def fmt_number(value: float, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:+,.0f}" if signed else f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.1%}" if np.isfinite(value) else "N/A"


def fmt_money(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:,.1f}B"
    return f"{sign}${value / 1_000_000:,.1f}M"


def selected_read(percentile: float, change: float, cohort: str) -> str:
    signal = positioning_signal(percentile)
    if np.isfinite(change):
        verb = "added" if change > 0 else "cut" if change < 0 else "held"
        weekly = f"{verb} {abs(change):,.0f} net contracts" if change else "held its net position unchanged"
    else:
        weekly = "has no usable weekly change"
    rank = f"{percentile:,.0f}th" if np.isfinite(percentile) else "unavailable"
    return f"{cohort} positioning is {signal.lower()} at the {rank} percentile of the selected lookback and {weekly} in the latest week."


def positioning_chart(history: pd.DataFrame, price: pd.Series, metric: str, market: str, cohort: str, price_label: str | None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not price.empty:
        fig.add_trace(
            go.Scatter(x=price.index, y=price.values, name=price_label or "Price", mode="lines", line=dict(color=PRICE_COLOR, width=1.4)),
            secondary_y=False,
        )
    mapping = {
        "Net contracts": ("net_contracts", 1.0, "Net contracts"),
        "Net % open interest": ("net_pct_oi", 100.0, "Net % of open interest"),
        "Rolling z-score": ("rolling_zscore", 1.0, "Rolling z-score"),
        "Rolling percentile": ("rolling_percentile", 1.0, "Rolling percentile"),
        "Estimated $ notional": ("net_notional", 1 / 1_000_000_000, "Estimated net notional ($B)"),
    }
    column, scale, axis_title = mapping[metric]
    values = pd.to_numeric(history[column], errors="coerce") * scale
    fig.add_trace(
        go.Scatter(
            x=history["report_date"],
            y=values,
            name=f"{cohort} positioning",
            mode="lines",
            line=dict(color=POSITION_COLOR, width=2),
            fill="tozeroy" if metric in {"Net contracts", "Net % open interest", "Estimated $ notional"} else None,
            fillcolor="rgba(214,39,40,0.08)",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), secondary_y=True)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text=price_label or "Price", showgrid=False, secondary_y=False)
    fig.update_yaxes(title_text=axis_title, showgrid=True, gridcolor=GRID_COLOR, secondary_y=True)
    fig.update_layout(
        height=600,
        template="plotly_white",
        margin=dict(l=45, r=65, t=35, b=45),
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
        title=dict(text=market, font=dict(size=15), x=0.01),
    )
    return fig


def cohort_chart(history: pd.DataFrame, report_type: str) -> go.Figure:
    fig = go.Figure()
    palette = ("#111111", "#d62728", "#6b7280", "#9ca3af", "#4b5563")
    for i, cohort in enumerate(COHORTS[report_type]):
        series = add_metrics(history, report_type, cohort)
        fig.add_trace(
            go.Scatter(
                x=series["report_date"],
                y=series["net_pct_oi"] * 100,
                name=cohort,
                mode="lines",
                line=dict(color=palette[i % len(palette)], width=1.6),
            )
        )
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1))
    fig.update_layout(height=500, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(title="Net position as % of open interest", ticksuffix="%", gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    return fig


st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1600)

with st.sidebar:
    st.header("Positioning setup")
    lookback_label = st.select_slider("Crowding lookback", options=["1Y", "2Y", "3Y", "5Y"], value="3Y")
    lookback_weeks = {"1Y": 52, "2Y": 104, "3Y": 156, "5Y": 260}[lookback_label]
    tff_cohort = st.selectbox("Financial futures cohort", list(COHORTS["TFF"]), index=list(COHORTS["TFF"]).index(DEFAULT_COHORT["TFF"]))
    disagg_cohort = st.selectbox("Physical futures cohort", list(COHORTS["Disaggregated"]), index=list(COHORTS["Disaggregated"]).index(DEFAULT_COHORT["Disaggregated"]))
    market_filter = st.text_input("Filter contracts", placeholder="NASDAQ, gold, yen, crude…")
    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        - Scans public CFTC financial and physical futures reports for crowding and weekly positioning shifts.
        - Normalizes net positioning by open interest and ranks it over a selectable trailing history.
        - Overlays mapped futures prices and only estimates dollar notional where the contract multiplier is explicit.

        **Timing:** COT is a weekly Tuesday position snapshot, normally released Friday. It is not a real-time flow feed.
        """
    )

render_page_header(
    PageHeader(
        title=TITLE,
        description="Scan CFTC financial and physical futures for crowded longs, crowded shorts, and sharp positioning shifts, then inspect each contract against its market price.",
        eyebrow="ADFM Positioning + Flows",
    )
)

with st.spinner("Loading CFTC positioning history…"):
    tff, tff_error = load_report("TFF")
    disagg, disagg_error = load_report("Disaggregated")

parts = []
if not tff.empty:
    parts.append(build_scanner(tff, "TFF", tff_cohort, lookback_weeks))
if not disagg.empty:
    parts.append(build_scanner(disagg, "Disaggregated", disagg_cohort, lookback_weeks))
scanner = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
if scanner.empty:
    st.error("CFTC Public Reporting did not return usable positioning data.")
    if tff_error:
        st.caption(f"TFF: {tff_error}")
    if disagg_error:
        st.caption(f"Disaggregated: {disagg_error}")
    render_footer()
    st.stop()

asset_classes = sorted(scanner["asset_class"].dropna().unique())
with st.sidebar:
    asset_filter = st.multiselect("Asset class", asset_classes)

latest_dates = []
if not tff.empty:
    latest_dates.append(f"TFF {tff['report_date'].max().date().isoformat()}")
if not disagg.empty:
    latest_dates.append(f"Disaggregated {disagg['report_date'].max().date().isoformat()}")
render_status_line(cftc_data_through=" · ".join(latest_dates), lookback=lookback_label, source="CFTC Public Reporting Environment")
if tff_error or disagg_error:
    st.warning("One CFTC report failed to load, so the scanner is running on partial coverage.")

filtered = scanner.copy()
if asset_filter:
    filtered = filtered.loc[filtered["asset_class"].isin(asset_filter)]
if market_filter.strip():
    query = market_filter.strip().casefold()
    haystack = (filtered["market"].fillna("") + " " + filtered["commodity"].fillna("") + " " + filtered["contract_code"]).str.casefold()
    filtered = filtered.loc[haystack.str.contains(query, regex=False)]

scan_tab, detail_tab, cohorts_tab, data_tab, method_tab = st.tabs(["Crowding scanner", "Selected market", "Cohort history", "Data", "Methodology"])

with scan_tab:
    c1, c2 = st.columns(2)
    with c1:
        sort_mode = st.selectbox("Rank scanner", ["Most crowded shorts", "Most crowded longs", "Largest 1W change", "Largest 4W change", "Largest absolute z-score"])
    with c2:
        signals = st.multiselect("Signals", ["Extreme Short", "Crowded Short", "Neutral", "Crowded Long", "Extreme Long"])
    if signals:
        filtered = filtered.loc[filtered["signal"].isin(signals)]
    if sort_mode == "Most crowded shorts":
        filtered = filtered.sort_values("percentile")
    elif sort_mode == "Most crowded longs":
        filtered = filtered.sort_values("percentile", ascending=False)
    elif sort_mode == "Largest 1W change":
        filtered = filtered.assign(_rank=filtered["one_week_change"].abs()).sort_values("_rank", ascending=False).drop(columns="_rank")
    elif sort_mode == "Largest 4W change":
        filtered = filtered.assign(_rank=filtered["four_week_change"].abs()).sort_values("_rank", ascending=False).drop(columns="_rank")
    else:
        filtered = filtered.assign(_rank=filtered["zscore"].abs()).sort_values("_rank", ascending=False).drop(columns="_rank")

    render_section_header("Cross-asset crowding scanner", "Percentiles and z-scores use net contracts as a share of each contract's open interest so market size does not mechanically determine the crowding signal.")
    display = filtered[["asset_class", "market", "report_type", "report_date", "net_contracts", "net_pct_oi", "one_week_change", "four_week_change", "percentile", "zscore", "history_weeks", "signal", "record", "contract_code"]].rename(columns={
        "asset_class": "Asset class", "market": "Market", "report_type": "Report", "report_date": "As of", "net_contracts": "Net contracts", "net_pct_oi": "Net % OI", "one_week_change": "1W Δ", "four_week_change": "4W Δ", "percentile": f"{lookback_label} %ile", "zscore": f"{lookback_label} z", "history_weeks": "History", "signal": "Signal", "record": "Extreme", "contract_code": "CFTC code",
    })
    styled_display = display.style.format(
        {
            "As of": lambda value: pd.Timestamp(value).date().isoformat() if pd.notna(value) else "N/A",
            "Net contracts": "{:,.0f}",
            "Net % OI": "{:+.1%}",
            "1W Δ": "{:+,.0f}",
            "4W Δ": "{:+,.0f}",
            f"{lookback_label} %ile": "{:.0f}",
            f"{lookback_label} z": "{:+.2f}",
            "History": "{:,.0f}W",
        },
        na_rep="N/A",
    )
    st.dataframe(styled_display, hide_index=True, width="stretch", height=650)
    dataframe_download("Download scanner CSV", display, "adfm_cftc_positioning_scanner.csv")

selection = scanner.sort_values(["asset_class", "market"]).reset_index(drop=True)
selection["key"] = selection["report_type"] + "|" + selection["contract_code"]
selection["label"] = selection["market"] + " · " + selection["report_type"] + " · " + selection["contract_code"]
default_key = "TFF|209742"
default_index = int(selection.index[selection["key"].eq(default_key)][0]) if selection["key"].eq(default_key).any() else 0
selected_label = st.sidebar.selectbox("Selected contract", selection["label"].tolist(), index=default_index)
row = selection.loc[selection["label"].eq(selected_label)].iloc[0]
report_type = str(row["report_type"])
contract_code = str(row["contract_code"])
market_name = str(row["market"])
cohort = st.sidebar.selectbox("Selected-market cohort", list(COHORTS[report_type]), index=list(COHORTS[report_type]).index(DEFAULT_COHORT[report_type]))
history_raw, history_error = load_history(report_type, contract_code)

if history_raw.empty:
    with detail_tab:
        st.error(f"No historical CFTC data returned for {market_name}.")
        if history_error:
            st.caption(history_error)
else:
    history = rolling_metrics(history_raw, report_type, cohort, lookback_weeks)
    rank_history = history["net_pct_oi"].tail(lookback_weeks).dropna()
    percentile = percentile_rank(rank_history) if len(rank_history) >= 26 else np.nan
    latest = history.iloc[-1]
    net = history["net_contracts"].dropna()
    one_week = float(net.iloc[-1] - net.iloc[-2]) if len(net) >= 2 else np.nan
    four_week = float(net.iloc[-1] - net.iloc[-5]) if len(net) >= 5 else np.nan

    proxy = price_proxy(contract_code)
    price = pd.Series(dtype=float)
    price_warning = ""
    if proxy:
        ticker, price_label, multiplier = proxy
        price, price_warning = load_price(ticker)
        if not price.empty and multiplier is not None:
            weekly_price = price.reindex(pd.DatetimeIndex(history["report_date"]), method="ffill")
            weekly_price.index = history.index
            history["net_notional"] = estimate_notional(history["net_contracts"], weekly_price, multiplier)
        else:
            history["net_notional"] = np.nan
    else:
        ticker, price_label, multiplier = "", None, None
        history["net_notional"] = np.nan

    metric_options = ["Net contracts", "Net % open interest", "Rolling z-score", "Rolling percentile"]
    if history["net_notional"].notna().any():
        metric_options.append("Estimated $ notional")
    metric = st.sidebar.selectbox("Positioning metric", metric_options, index=1)

    with detail_tab:
        render_selection_note("Current positioning read", selected_read(percentile, one_week, cohort))
        latest_notional = float(history["net_notional"].dropna().iloc[-1]) if history["net_notional"].notna().any() else np.nan
        render_kpi_cards([
            ("Net position", fmt_number(float(latest["net_contracts"]), signed=True), f"{cohort} contracts"),
            ("Net / open interest", fmt_pct(float(latest["net_pct_oi"])), positioning_signal(percentile)),
            (f"{lookback_label} percentile", f"{percentile:,.0f}th" if np.isfinite(percentile) else "N/A", "Crowding rank"),
            (f"{lookback_label} z-score", f"{float(latest['rolling_zscore']):+.2f}" if pd.notna(latest["rolling_zscore"]) else "N/A", "Normalized positioning"),
            ("1W change", fmt_number(one_week, signed=True), f"4W {fmt_number(four_week, signed=True)}"),
            ("Estimated notional", fmt_money(latest_notional), price_label or "No mapped multiplier"),
        ])
        render_section_header(f"{market_name} · price vs positioning", "Black is the mapped market price where available. Red is the selected CFTC cohort; use the metric selector to switch between contracts, % of open interest, percentile, z-score and mapped notional.")
        st.plotly_chart(positioning_chart(history, price, metric, market_name, cohort, price_label), width="stretch", config={"displaylogo": False})
        if proxy is None:
            st.caption("No Yahoo continuous-futures price proxy is mapped for this CFTC contract yet. Positioning history remains available.")
        elif price_warning:
            st.caption(f"Price proxy warning: {price_warning}")

    with cohorts_tab:
        render_section_header(f"{market_name} cohort decomposition", f"Net long minus short as a share of open interest for every public cohort in the {REPORT_LABELS[report_type]} report.")
        st.plotly_chart(cohort_chart(history_raw, report_type), width="stretch", config={"displaylogo": False})

    with data_tab:
        data = history[["report_date", "market_name", "contract_code", "open_interest", "cohort_long", "cohort_short", "net_contracts", "net_pct_oi", "rolling_zscore", "rolling_percentile", "net_notional"]].tail(520)
        st.dataframe(data, hide_index=True, width="stretch", height=620)
        dataframe_download("Download selected history CSV", data, f"adfm_cftc_{report_type.lower()}_{contract_code}.csv")

with method_tab:
    render_section_header("Methodology", "The first version is built entirely from public CFTC weekly positioning and mapped Yahoo continuous-futures price proxies.")
    st.markdown(
        """
        **Financial futures:** Traders in Financial Futures separates Dealer/Intermediary, Asset Manager/Institutional, Leveraged Funds and Other Reportables. The default combines Asset Managers and Leveraged Funds, while each cohort remains selectable.

        **Physical futures:** Disaggregated COT separates Producer/Merchant, Swap Dealers, Managed Money and Other Reportables. Managed Money is the default speculative cohort.

        **Crowding:** net contracts = longs − shorts. The scanner divides net contracts by open interest, then ranks the latest observation against the selected trailing history. Extreme Short is ≤2.5th percentile and Extreme Long is ≥97.5th percentile.

        **Timing:** COT is a weekly Tuesday position snapshot, normally published Friday. It is a positioning-regime input, not a real-time flow feed.

        **Price/notional:** price overlays use mapped continuous futures. Estimated notional is only shown where an explicit multiplier is mapped, so the app does not manufacture dollar exposure for contracts whose quoting convention needs additional handling.
        """
    )

render_footer(data_note="Primary inputs: CFTC TFF Futures Only (gpe5-46if), CFTC Disaggregated Futures Only (72hh-3qpy), and Yahoo Finance continuous-futures price proxies where mapped.")
