"""PM-first CFTC positioning dashboard with advanced drill-downs."""

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
    fetch_contract_history,
    fetch_recent,
    percentile_rank,
    positioning_signal,
    price_proxy,
    rolling_metrics,
)
from adfm_core.market_data import adjusted_ohlcv, configure_yfinance_cache, fetch_daily_ohlcv
from adfm_core.palette import EXCEL, PASTEL_20
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
PRICE_COLOR = "#111111"
POSITION_COLOR = EXCEL["rose"]
GRID_COLOR = "rgba(127,140,141,0.20)"
LOOKBACKS = {"1Y": 52, "2Y": 104, "3Y": 156, "5Y": 260}


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


def fmt_pp(value: float) -> str:
    return f"{value * 100:+.1f}pp" if np.isfinite(value) else "N/A"


def pm_read(market: str, percentile: float, weekly_shift: float, lookback_label: str) -> str:
    signal = positioning_signal(percentile)
    rank = f"{percentile:,.0f}th percentile" if np.isfinite(percentile) else "an unavailable percentile"
    if np.isfinite(weekly_shift):
        if weekly_shift > 0:
            weekly = f"net positioning became more bullish by {abs(weekly_shift) * 100:.1f} percentage points of open interest this week"
        elif weekly_shift < 0:
            weekly = f"net positioning became more bearish by {abs(weekly_shift) * 100:.1f} percentage points of open interest this week"
        else:
            weekly = "net positioning was unchanged this week"
    else:
        weekly = "the latest weekly change is unavailable"
    return f"{market} is {signal.lower()} at the {rank} of the past {lookback_label}, and {weekly}."


def positioning_chart(
    history: pd.DataFrame,
    price: pd.Series,
    market: str,
    cohort: str,
    price_label: str | None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not price.empty:
        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price.values,
                name=price_label or "Price",
                mode="lines",
                line=dict(color=PRICE_COLOR, width=1.4),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=history["report_date"],
            y=pd.to_numeric(history["net_pct_oi"], errors="coerce") * 100,
            name=f"{cohort} positioning",
            mode="lines",
            line=dict(color=POSITION_COLOR, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(192,80,77,0.08)",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, line=dict(color=EXCEL["slate_blue"], width=1), secondary_y=True)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text=price_label or "Price", showgrid=False, secondary_y=False)
    fig.update_yaxes(
        title_text="Net position as % of open interest",
        ticksuffix="%",
        showgrid=True,
        gridcolor=GRID_COLOR,
        secondary_y=True,
    )
    fig.update_layout(
        height=540,
        template="plotly_white",
        margin=dict(l=45, r=65, t=25, b=45),
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
        title=dict(text=market, font=dict(size=15), x=0.01),
    )
    return fig


def cohort_chart(history: pd.DataFrame, report_type: str) -> go.Figure:
    fig = go.Figure()
    for i, cohort in enumerate(COHORTS[report_type]):
        series = add_metrics(history, report_type, cohort)
        fig.add_trace(
            go.Scatter(
                x=series["report_date"],
                y=series["net_pct_oi"] * 100,
                name=cohort,
                mode="lines",
                line=dict(color=PASTEL_20[i % len(PASTEL_20)], width=1.7),
            )
        )
    fig.add_hline(y=0, line=dict(color=EXCEL["slate_blue"], width=1))
    fig.update_layout(
        height=470,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=45, r=25, t=20, b=40),
    )
    fig.update_yaxes(title="Net position as % of open interest", ticksuffix="%", gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    return fig


def compact_signal_table(frame: pd.DataFrame, lookback_label: str) -> pd.io.formats.style.Styler:
    view = frame[["market", "net_pct_oi", "percentile", "one_week_oi_shift", "signal"]].copy()
    view.columns = ["Market", "Net % OI", f"{lookback_label} %ile", "1W shift", "Signal"]
    return view.style.format(
        {
            "Net % OI": "{:+.1%}",
            f"{lookback_label} %ile": "{:.0f}",
            "1W shift": lambda value: f"{value * 100:+.1f}pp" if pd.notna(value) else "N/A",
        },
        na_rep="N/A",
    )


def full_scanner_table(frame: pd.DataFrame, lookback_label: str) -> pd.DataFrame:
    return frame[
        [
            "asset_class",
            "market",
            "report_type",
            "report_date",
            "net_contracts",
            "net_pct_oi",
            "one_week_oi_shift",
            "four_week_change",
            "percentile",
            "zscore",
            "signal",
            "record",
            "contract_code",
        ]
    ].rename(
        columns={
            "asset_class": "Asset class",
            "market": "Market",
            "report_type": "Report",
            "report_date": "As of",
            "net_contracts": "Net contracts",
            "net_pct_oi": "Net % OI",
            "one_week_oi_shift": "1W shift / OI",
            "four_week_change": "4W Δ contracts",
            "percentile": f"{lookback_label} %ile",
            "zscore": f"{lookback_label} z",
            "signal": "Signal",
            "record": "Extreme",
            "contract_code": "CFTC code",
        }
    )


st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1600)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Default view:** what is crowded, what changed this week, and which market deserves a closer look.

        Financial futures default to **Asset Managers + Leveraged Funds**. Physical commodities default to **Managed Money**. Crowding uses a **3-year** history unless changed below.

        **Timing:** COT is a weekly Tuesday position snapshot, normally released Friday. It is not a real-time flow feed.
        """
    )
    with st.expander("Advanced Controls", expanded=False):
        lookback_label = st.select_slider(
            "Crowding lookback",
            options=list(LOOKBACKS),
            value="3Y",
        )
        tff_cohort = st.selectbox(
            "Financial futures cohort",
            list(COHORTS["TFF"]),
            index=list(COHORTS["TFF"]).index(DEFAULT_COHORT["TFF"]),
        )
        disagg_cohort = st.selectbox(
            "Physical futures cohort",
            list(COHORTS["Disaggregated"]),
            index=list(COHORTS["Disaggregated"]).index(DEFAULT_COHORT["Disaggregated"]),
        )

lookback_weeks = LOOKBACKS[lookback_label]

render_page_header(
    PageHeader(
        title=TITLE,
        description="A fast read on futures crowding: crowded longs, crowded shorts, weekly shifts, and one market chart for deeper inspection.",
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

scanner["one_week_oi_shift"] = scanner["one_week_change"] / scanner["open_interest"].replace(0, np.nan)

latest_dates = []
if not tff.empty:
    latest_dates.append(f"TFF {tff['report_date'].max().date().isoformat()}")
if not disagg.empty:
    latest_dates.append(f"Disaggregated {disagg['report_date'].max().date().isoformat()}")
render_status_line(
    cftc_data_through=" · ".join(latest_dates),
    lookback=lookback_label,
    source="CFTC Public Reporting Environment",
)
if tff_error or disagg_error:
    st.warning("One CFTC report failed to load, so the dashboard is running on partial coverage.")

render_section_header(
    "What matters now",
    "Five names per panel. Crowding is ranked on net positioning as a share of open interest; weekly shifts are scaled by current open interest for cross-market comparability.",
)
usable = scanner.dropna(subset=["percentile"]).copy()
crowded_shorts = usable.sort_values("percentile").head(5)
crowded_longs = usable.sort_values("percentile", ascending=False).head(5)
weekly_shifts = (
    scanner.dropna(subset=["one_week_oi_shift"])
    .assign(_abs_shift=lambda frame: frame["one_week_oi_shift"].abs())
    .sort_values("_abs_shift", ascending=False)
    .drop(columns="_abs_shift")
    .head(5)
)

short_col, long_col, shift_col = st.columns(3)
with short_col:
    st.markdown("#### Crowded Shorts")
    st.dataframe(compact_signal_table(crowded_shorts, lookback_label), hide_index=True, width="stretch", height=230)
with long_col:
    st.markdown("#### Crowded Longs")
    st.dataframe(compact_signal_table(crowded_longs, lookback_label), hide_index=True, width="stretch", height=230)
with shift_col:
    st.markdown("#### Largest Weekly Shifts")
    st.dataframe(compact_signal_table(weekly_shifts, lookback_label), hide_index=True, width="stretch", height=230)

selection = scanner.sort_values(["asset_class", "market"]).reset_index(drop=True)
selection["key"] = selection["report_type"] + "|" + selection["contract_code"]
selection["label"] = selection["market"] + " · " + selection["asset_class"]
duplicated = selection["label"].duplicated(keep=False)
selection.loc[duplicated, "label"] = selection.loc[duplicated, "label"] + " · " + selection.loc[duplicated, "contract_code"]
default_key = "TFF|209742"
default_index = int(selection.index[selection["key"].eq(default_key)][0]) if selection["key"].eq(default_key).any() else 0

render_section_header(
    "Inspect one market",
    "The default chart stays simple: market price in black, normalized CFTC positioning in red.",
)
selected_label = st.selectbox(
    "Market to inspect",
    selection["label"].tolist(),
    index=default_index,
    label_visibility="collapsed",
)
row = selection.loc[selection["label"].eq(selected_label)].iloc[0]
report_type = str(row["report_type"])
contract_code = str(row["contract_code"])
market_name = str(row["market"])
cohort = tff_cohort if report_type == "TFF" else disagg_cohort
history_raw, history_error = load_history(report_type, contract_code)

if history_raw.empty:
    st.error(f"No historical CFTC data returned for {market_name}.")
    if history_error:
        st.caption(history_error)
else:
    history = rolling_metrics(history_raw, report_type, cohort, lookback_weeks)
    pct_history = history["net_pct_oi"].tail(lookback_weeks).dropna()
    percentile = percentile_rank(pct_history) if len(pct_history) >= 26 else np.nan
    latest = history.iloc[-1]
    weekly_shift = float(pct_history.iloc[-1] - pct_history.iloc[-2]) if len(pct_history) >= 2 else np.nan

    proxy = price_proxy(contract_code)
    price = pd.Series(dtype=float)
    price_warning = ""
    if proxy:
        ticker, price_label, _ = proxy
        price, price_warning = load_price(ticker)
    else:
        ticker, price_label = "", None

    render_selection_note(
        "PM read",
        pm_read(market_name, percentile, weekly_shift, lookback_label),
    )
    render_kpi_cards(
        [
            ("Signal", positioning_signal(percentile), cohort),
            ("Net / open interest", fmt_pct(float(latest["net_pct_oi"])), "Normalized crowding"),
            (f"{lookback_label} percentile", f"{percentile:,.0f}th" if np.isfinite(percentile) else "N/A", "Historical rank"),
            ("1W shift", fmt_pp(weekly_shift), "More bullish" if weekly_shift > 0 else "More bearish" if weekly_shift < 0 else "Unchanged"),
        ]
    )
    st.plotly_chart(
        positioning_chart(history, price, market_name, cohort, price_label),
        width="stretch",
        config={"displaylogo": False},
    )
    if proxy is None:
        st.caption("No mapped continuous-futures price proxy is available for this contract yet. Positioning history remains available.")
    elif price_warning:
        st.caption(f"Price proxy warning: {price_warning}")

    with st.expander("Advanced Analysis", expanded=False):
        scanner_tab, cohorts_tab, data_tab, method_tab = st.tabs(
            ["Full scanner", "Cohorts", "Raw history", "Methodology"]
        )

        with scanner_tab:
            filter_col, sort_col = st.columns(2)
            with filter_col:
                asset_filter = st.multiselect(
                    "Asset class",
                    sorted(scanner["asset_class"].dropna().unique()),
                )
            with sort_col:
                sort_mode = st.selectbox(
                    "Rank scanner",
                    [
                        "Most crowded shorts",
                        "Most crowded longs",
                        "Largest 1W shift",
                        "Largest 4W contract change",
                        "Largest absolute z-score",
                    ],
                )
            advanced = scanner.copy()
            if asset_filter:
                advanced = advanced.loc[advanced["asset_class"].isin(asset_filter)]
            if sort_mode == "Most crowded shorts":
                advanced = advanced.sort_values("percentile")
            elif sort_mode == "Most crowded longs":
                advanced = advanced.sort_values("percentile", ascending=False)
            elif sort_mode == "Largest 1W shift":
                advanced = advanced.assign(_rank=advanced["one_week_oi_shift"].abs()).sort_values("_rank", ascending=False).drop(columns="_rank")
            elif sort_mode == "Largest 4W contract change":
                advanced = advanced.assign(_rank=advanced["four_week_change"].abs()).sort_values("_rank", ascending=False).drop(columns="_rank")
            else:
                advanced = advanced.assign(_rank=advanced["zscore"].abs()).sort_values("_rank", ascending=False).drop(columns="_rank")

            display = full_scanner_table(advanced, lookback_label)
            styled_display = display.style.format(
                {
                    "As of": lambda value: pd.Timestamp(value).date().isoformat() if pd.notna(value) else "N/A",
                    "Net contracts": "{:,.0f}",
                    "Net % OI": "{:+.1%}",
                    "1W shift / OI": lambda value: f"{value * 100:+.1f}pp" if pd.notna(value) else "N/A",
                    "4W Δ contracts": "{:+,.0f}",
                    f"{lookback_label} %ile": "{:.0f}",
                    f"{lookback_label} z": "{:+.2f}",
                },
                na_rep="N/A",
            )
            st.dataframe(styled_display, hide_index=True, width="stretch", height=560)
            dataframe_download("Download scanner CSV", display, "adfm_cftc_positioning_scanner.csv")

        with cohorts_tab:
            render_section_header(
                f"{market_name} cohort decomposition",
                f"Net long minus short as a share of open interest for every public cohort in the {REPORT_LABELS[report_type]} report.",
            )
            st.plotly_chart(
                cohort_chart(history_raw, report_type),
                width="stretch",
                config={"displaylogo": False},
            )

        with data_tab:
            data = history[
                [
                    "report_date",
                    "market_name",
                    "contract_code",
                    "open_interest",
                    "cohort_long",
                    "cohort_short",
                    "net_contracts",
                    "net_pct_oi",
                    "rolling_zscore",
                    "rolling_percentile",
                ]
            ].tail(520)
            st.dataframe(data, hide_index=True, width="stretch", height=520)
            dataframe_download(
                "Download selected history CSV",
                data,
                f"adfm_cftc_{report_type.lower()}_{contract_code}.csv",
            )

        with method_tab:
            st.markdown(
                """
                **Financial futures:** Traders in Financial Futures separates Dealer/Intermediary, Asset Manager/Institutional, Leveraged Funds and Other Reportables. The default combines Asset Managers and Leveraged Funds.

                **Physical futures:** Disaggregated COT separates Producer/Merchant, Swap Dealers, Managed Money and Other Reportables. Managed Money is the default speculative cohort.

                **Crowding:** net contracts = longs − shorts. The dashboard divides net contracts by open interest and ranks the latest observation against the selected trailing history. Extreme Short is ≤2.5th percentile and Extreme Long is ≥97.5th percentile.

                **Weekly shifts:** the top-panel shift scales the weekly change in net contracts by current open interest for a fast cross-market comparison. The selected-market PM read uses the exact week-over-week change in net positioning as a share of open interest.

                **Timing:** COT is a weekly Tuesday position snapshot, normally published Friday. It is a positioning-regime input, not a real-time flow feed.
                """
            )

render_footer(
    data_note="Primary inputs: CFTC TFF Futures Only (gpe5-46if), CFTC Disaggregated Futures Only (72hh-3qpy), and Yahoo Finance continuous-futures price proxies where mapped."
)
