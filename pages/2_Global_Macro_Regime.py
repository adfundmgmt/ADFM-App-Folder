"""Bond monitor at the established Global Macro Regime URL."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.bond_monitor import GLOBAL_SOVEREIGNS, daily_snapshot, monthly_snapshot
from adfm_core.bond_event_study import HORIZONS, PROFILES, event_dates, event_summary, monthly_history, signal_frame
from adfm_core.global_macro import clean
from adfm_core.sovereign_daily import DAILY_SOVEREIGNS, load_daily_sovereigns
from adfm_core.palette import PASTEL
from adfm_core.primary_data import fetch_fred_symbols
from adfm_core.ui import PageHeader, inject_explorer_style, render_footer, render_page_header, render_sidebar_about


US = (("3-month Treasury", "DGS3MO"), ("2-year Treasury", "DGS2"),
      ("5-year Treasury", "DGS5"), ("10-year Treasury", "DGS10"), ("30-year Treasury", "DGS30"))
REAL = (("5-year real yield", "DFII5"), ("10-year real yield", "DFII10"),
        ("5-year breakeven", "T5YIE"), ("10-year breakeven", "T10YIE"))
CREDIT = (("US investment grade OAS", "BAMLC0A0CM"), ("US BBB OAS", "BAMLC0A4CBBB"),
          ("US high yield OAS", "BAMLH0A0HYM2"))
VIEWS = ("US Treasury", "Real yields & inflation", "Global sovereign", "Credit spreads")
PERIODS = {"Max": None, "1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}


def style():
    inject_explorer_style(max_width_px=1540)
    st.markdown("""
    <style>
    .bond-status {display:flex;flex-wrap:wrap;gap:.45rem 1.1rem;border-top:1px solid #d7d7d7;
        border-bottom:1px solid #d7d7d7;margin:.25rem 0 .75rem;padding:.57rem 0;color:#222;
        font:.78rem/1.35 Arial,Helvetica,sans-serif}
    .bond-status strong {color:#000;font-weight:800}
    .bond-heading {margin:1.05rem 0 .55rem;color:#000;
        font:700 1.25rem/1.2 Georgia,"Times New Roman",serif;letter-spacing:-.018em}
    .bond-wrap {overflow-x:auto;border:1px solid #aeb7bd;background:#fff;margin:.15rem 0 .6rem}
    .bond-table {width:100%;min-width:850px;border-collapse:collapse;table-layout:fixed;
        font:.71rem/1.2 Arial,Helvetica,sans-serif}
    .bond-table th,.bond-table td {border-right:1px solid #aeb7bd;border-bottom:1px solid #aeb7bd;
        padding:.45rem .3rem;text-align:right;white-space:nowrap}
    .bond-table th:first-child,.bond-table td:first-child {text-align:left;width:17%;padding-left:.55rem}
    .bond-table th:last-child,.bond-table td:last-child {border-right:0}
    .bond-table tr:last-child td {border-bottom:0}
    .bond-table tr.selected td:first-child {border-left:4px solid #357f8d;padding-left:calc(.55rem - 4px)}
    .bond-table td.state,.bond-table th:nth-child(2) {text-align:left;width:16%;font-weight:700}
    .bond-table td.study,.bond-table th:nth-child(3) {text-align:left;width:21%}
    .bond-table td small {display:block;color:#66757e;font-size:.66rem;font-weight:400;margin-top:.15rem}
    .bond-table td.state.active {background:#dce9e1}
    .bond-table td.state.watch {background:#fff1cf}
    .bond-table td.state.stale {background:#f4f5f5;color:#777}
    .bond-table th {background:#357f8d;color:white;font-weight:800}
    .bond-table td:first-child {background:#edf0f2;font-weight:800}
    .bond-table td.up {background:#edc9cd}
    .bond-table td.down {background:#dce9e1}
    .bond-table td.flat {background:#e7edf1}
    .bond-table td.na {background:#f4f5f5;color:#777}
    .bond-note {color:#666;font:.73rem/1.45 Arial,Helvetica,sans-serif;margin:.2rem 0 .7rem}
    </style>""", unsafe_allow_html=True)


@st.cache_data(ttl=1800, max_entries=4, show_spinner=False)
def daily_data(symbols: tuple[str, ...]):
    return fetch_fred_symbols(symbols, start="1900-01-01")


@st.cache_data(ttl=21600, max_entries=1, show_spinner=False)
def global_data(symbols: tuple[str, ...]):
    return fetch_fred_symbols(symbols, start="1900-01-01")


@st.cache_data(ttl=1800, max_entries=1, show_spinner=False)
def sovereign_daily_data():
    return load_daily_sovereigns()


def monitor_table(rows: list[dict], monthly: bool, spread: bool, selected: str):
    table = []
    for row in rows:
        snap, summary, latest = row["snapshot"], row["summary"], row["latest"]
        count = summary.loc["Independent N", "3M"]
        controls = summary.loc["Control N", "3M"]
        table.append({"Instrument": row["name"], "Signal state": row["state"],
                      "3M after top (bp)": summary.loc["Signal median", "3M"],
                      "3M matched edge (bp)": summary.loc["Median edge", "3M"],
                      "95% edge low (bp)": summary.loc["Edge CI low", "3M"],
                      "95% edge high (bp)": summary.loc["Edge CI high", "3M"],
                      "Independent N": count, "Control N": controls,
                      "Evidence": "Small sample" if min(count, controls) < 20 else "Retrospective",
                      "Level (%)": snap["Yield"], "1M Δ (bp)": snap["1M"],
                      "3M Δ (bp)": snap["3M"], "YTD Δ (bp)": snap["YTD"],
                      "Move %ile": float(latest["ChangePctile"]) if latest is not None else np.nan,
                      "Observed": snap["Observation"], "Availability proxy": snap.get("Availability", ""), "Status": snap["Status"],
                      "Last top": row["latest_event"], "Source / basis": row["basis"]})
    data = pd.DataFrame(table)
    numeric = data.select_dtypes(include="number").columns
    config = {name: st.column_config.NumberColumn(format="%.1f") for name in numeric}
    def tone(value):
        if pd.isna(value):
            return "background-color: #f4f5f5; color: #777"
        return "background-color: #dce9e1" if value < 0 else "background-color: #edc9cd" if value > 0 else ""
    painted = data.style.map(tone, subset=["3M matched edge (bp)", "1M Δ (bp)", "3M Δ (bp)", "YTD Δ (bp)"])
    st.dataframe(painted, hide_index=True, width="stretch", column_config=config,
                 height=min(800, 38 + 35 * len(data)))
    st.caption("Click a column header to sort. Changes and outcomes are basis points of yield, not bond returns. "
               "Edge and its 95% interval compare matched trend/volatility regimes using non-overlapping windows. "
               "Fewer than 20 independent signals or controls is a small sample.")

def history_chart(series: pd.Series, name: str, years: int | None, monthly: bool, spread: bool,
                  events: pd.DatetimeIndex):
    history = monthly_history(series) if monthly else clean(series)
    if history.empty:
        st.info("No history is available for this instrument.")
        return
    if years is not None:
        history = history.loc[history.index >= history.index[-1] - pd.DateOffset(years=years)]
    fig = go.Figure(go.Scatter(x=history.index, y=history, mode="lines",
                               line=dict(color=PASTEL["blue"], width=2), name=name,
                               hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>"))
    marked = history.reindex(events.intersection(history.index)).dropna()
    if not marked.empty:
        fig.add_trace(go.Scatter(x=marked.index, y=marked, mode="markers", name="Yield top signal",
                                 marker=dict(color=PASTEL["rose"], size=9, line=dict(color="white", width=1)),
                                 hovertemplate="Yield top signal<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>"))
    fig.update_layout(height=430, margin=dict(l=35, r=20, t=10, b=25), paper_bgcolor="white",
                      plot_bgcolor="white", showlegend=False, font=dict(family="Arial", color="#222", size=12),
                      xaxis=dict(showgrid=False, linecolor="#aeb7bd"),
                      yaxis=dict(title="Spread (%)" if spread else "Yield (%)", gridcolor="#e7edf1", zeroline=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"{name} · {'monthly average; month-end availability proxy' if monthly else 'daily observation'} · "
               f"{len(marked)} signal markers in view · latest observation {history.index[-1]:%Y-%m-%d}")


def display_number(value: float, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if np.isfinite(value) else "—"


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def study_row(series: pd.Series, frequency: str, profile: str, period: str):
    diagnostics = signal_frame(series, frequency, profile)
    full_events = event_dates(diagnostics, 6 if frequency == "monthly" else 63)
    history = monthly_history(series) if frequency == "monthly" else clean(series)
    if history.empty:
        start = pd.Timestamp.now().normalize()
    else:
        start = (history.index[0] if period == "Max" else
                 history.index[-1] - pd.DateOffset(years=PERIODS[period]))
    events = full_events[full_events >= start]
    summary, outcomes = event_summary(history.loc[history.index >= start], events, frequency)
    return diagnostics, events, summary, outcomes


def render():
    style()
    render_page_header(PageHeader(
        title="Global Bond Monitor",
        description="Find extended yield moves, identify potential tops, and compare what happened after historical signals.",
        eyebrow="ADFM Rates & Credit",
        source_note="Federal Reserve / FRED · official daily sovereign curves · OECD monthly via FRED",
    ))
    with st.sidebar:
        render_sidebar_about("2_Global_Macro_Regime.py")
        st.caption("Daily official 10-year curves retain their published basis. OECD long history uses monthly averages separately.")

    a, b, c, d = st.columns([1.45, 1.55, 1.35, .7], gap="small")
    with a:
        view = st.selectbox("Bond market", VIEWS, key="bond_view")
    global_view = view == "Global sovereign"
    sovereign_frequency = (st.sidebar.radio("Sovereign history", ("Daily official", "Monthly OECD"),
                           index=0, key="bond_sovereign_frequency") if global_view else "")
    monthly = global_view and sovereign_frequency == "Monthly OECD"
    official_daily = global_view and not monthly
    if official_daily:
        names = [item[0] for item in DAILY_SOVEREIGNS]
        default = "United States"
    elif monthly:
        names = [name for name, _ in GLOBAL_SOVEREIGNS]
        default = "United States"
    else:
        definitions = US if view == "US Treasury" else REAL if view == "Real yields & inflation" else CREDIT
        names = [name for name, _ in definitions]
        default = "10-year Treasury" if view == "US Treasury" else names[0]
    with b:
        selected = st.selectbox("Bond / yield series", names, index=names.index(default), key="bond_instrument")
    with c:
        profile = st.selectbox("Top signal", PROFILES, index=1, key="bond_profile")
    with d:
        period = st.selectbox("Lookback", tuple(PERIODS), index=0, key="bond_period")

    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    if monthly:
        with st.spinner("Loading sovereign yield history…"):
            panel, status = global_data(tuple(symbol for _, symbol in GLOBAL_SOVEREIGNS))
        items = [(name, symbol, panel[symbol] if symbol in panel else pd.Series(dtype=float))
                 for name, symbol in GLOBAL_SOVEREIGNS]
        problems = [(str(row.get("symbol", "FRED")), str(row["error"]))
                    for _, row in status.iterrows() if row.get("error")] if not status.empty and "error" in status else []
    elif official_daily:
        with st.spinner("Loading official daily 10-year history…"):
            panel, status = sovereign_daily_data()
        items = [(country, source, panel[country] if country in panel else pd.Series(dtype=float))
                 for country, _, source, _, _ in DAILY_SOVEREIGNS]
        problems = [(row["country"], row["error"]) for _, row in status.iterrows() if row["error"]]
    else:
        symbols = tuple(symbol for _, symbol in definitions)
        with st.spinner("Loading bond market history…"):
            panel, status = daily_data(symbols)
        items = [(name, symbol, panel[symbol] if symbol in panel else pd.Series(dtype=float))
                 for name, symbol in definitions]
        problems = [(str(row.get("symbol", "FRED")), str(row["error"]))
                    for _, row in status.iterrows() if row.get("error")] if not status.empty and "error" in status else []

    # Incomplete monthly averages never enter signal calculation. Daily future
    # dates likewise cannot become a current signal even if a provider returns them.
    items = [(name, symbol, clean(series).loc[lambda x: (x.index.to_period("M") < today.to_period("M"))
              if monthly else (x.index.normalize() <= today)]) for name, symbol, series in items]
    basis_map = {country: basis for country, _, _, basis, _ in DAILY_SOVEREIGNS}
    rows = [{"name": name, "symbol": symbol, "series": series,
             "basis": basis_map.get(name, "Official daily 10Y") if official_daily else
                      "OECD monthly average" if monthly else "FRED daily yield / OAS",
             "snapshot": monthly_snapshot(series, today) if monthly else daily_snapshot(series, today)}
            for name, symbol, series in items]
    if official_daily and problems:
        st.warning("Some official daily sources failed validation or refresh: " + ", ".join(country for country, _ in problems) + ". Valid histories remain available with their original dates.")
    current = [row["snapshot"]["Observation"] for row in rows if row["snapshot"]["Status"] == "Current"]
    if not current:
        st.warning("No current observations were returned. Historical levels remain visible with their original dates.")
    chosen = next(row for row in rows if row["name"] == selected)
    frequency = "monthly" if monthly else "daily"
    with st.spinner("Studying bond signals…"):
        for row in rows:
            diagnostics, events, summary, history = study_row(row["series"], frequency, profile, period)
            latest = diagnostics.iloc[-1] if not diagnostics.empty else None
            row["latest"] = latest
            row["events"] = events
            row["summary"] = summary
            row["history"] = history
            row["state"] = "Unavailable" if latest is None or row["snapshot"]["Status"] != "Current" else (
                "Yield top signal" if latest["Signal"] else "Potential yield top" if latest["Setup"]
                else "Exhaustion watch" if latest["Watch"] else "Normal")
            row["latest_event"] = (events[-1].strftime("%b %d, %Y")
                                   if len(events) else "None")
    latest = chosen["latest"]
    events = chosen["events"]
    history = chosen["history"]
    first = clean(chosen["series"]).index.min()
    history_from = (first.strftime("%Y-%m" if monthly else "%Y-%m-%d")
                    if pd.notna(first) else "Unavailable")
    st.markdown('<div class="bond-status">'
                f'<span><strong>{escape(selected)}</strong> · {escape(view)}</span>'
                f'<span><strong>Signal</strong> {escape(profile)}</span>'
                f'<span><strong>Current</strong> {escape(chosen["state"])}</span>'
                f'<span><strong>Move pctile</strong> {display_number(float(latest["ChangePctile"])) if latest is not None else "—"}</span>'
                f'<span><strong>Trend ext.</strong> {display_number(float(latest["TrendZ"])) if latest is not None else "—"}</span>'
                f'<span><strong>Yield RSI</strong> {display_number(float(latest["RSI"])) if latest is not None else "—"}</span>'
                f'<span><strong>Vol pctile</strong> {display_number(float(latest["VolPctile"])) if latest is not None else "—"}</span>'
                f'<span><strong>Events</strong> {len(events)}</span>'
                f'<span><strong>Latest event</strong> {escape(chosen["latest_event"])}</span>'
                f'<span><strong>Coverage</strong> {len(current)}/{len(rows)} current</span>'
                f'<span><strong>Data through</strong> {escape(chosen["snapshot"]["Observation"] or "Unavailable")}</span>'
                f'<span><strong>History from</strong> {history_from}</span>'
                '</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bond-heading">{escape(selected)} · yield top signals</div>', unsafe_allow_html=True)
    history_chart(chosen["series"], selected, PERIODS[period], monthly,
                  view == "Credit spreads", events)
    st.markdown('<div class="bond-heading">Bond signal monitor</div>', unsafe_allow_html=True)
    monitor_table(rows, monthly, view == "Credit spreads", selected)
    with st.expander("Historical statistics and chronological holdout"):
        st.dataframe(chosen["summary"], width="stretch")
        st.caption("Earliest 70% of observations form the training period; latest 30% form the holdout. "
                   "Outcome windows crossing the split are purged. Controls remain within the same partition. "
                   "Independent N includes every completed, de-overlapped signal; matched N excludes unknown regimes "
                   "and signals without an eligible control. Edge uses matched samples. "
                   "95% percentile bootstrap intervals resample both signals and controls; they are descriptive, "
                   "and do not correct for parameter selection or residual serial dependence.")
        st.caption("Adverse excursion is the largest forward yield rise; favorable excursion is the largest "
                   "forward yield fall, including zero at entry. All endpoints and intervening monthly observations "
                   "must exist. Holdout results are chronological descriptive checks, not a prospective live test.")
    with st.expander("Historical yield top signals"):
        if history.empty:
            st.write("No historical events met this profile for the available series.")
        else:
            st.dataframe(history.sort_values("Date", ascending=False), hide_index=True, width="stretch",
                         column_config={label: st.column_config.NumberColumn(format="%.0f bp")
                                        for label in HORIZONS[frequency]})
    with st.expander("Sources and definitions"):
        if monthly:
            st.write("OECD 10-year long-term interest rates distributed by FRED. Monthly averages are charted and signaled "
                     "at month-end, an earliest availability proxy rather than a verified publication date. Actual releases "
                     "can be later; historical values may be revised. These are retrospective monthly studies and cannot "
                     "reconstruct a contemporaneous trading decision. Current incomplete months and missing intervals are excluded.")
        elif official_daily:
            st.write("Official daily 10-year observations; each country's exact curve basis is shown in the main table. "
                     "UK and Swiss curves are spot rates; US/Japanese curves are constant-maturity references. "
                     "Neither spot curves nor euro-area composites are relabeled as benchmark bonds. Monthly OECD "
                     "averages are never spliced into these histories. Direct sources refresh at most every 30 minutes; "
                     "validated persisted histories retain their dates on failure. The New Zealand source transitioned "
                     "from mid to closing observations in 2025; historical basis changes must be considered.")
        else:
            st.write("Treasury constant-maturity yields, TIPS real yields and inflation compensation use Federal Reserve series distributed by FRED. Credit uses ICE BofA option-adjusted spread indices distributed by FRED. Values are end-of-day observations, not executable bond prices.")
        st.write("Changes require a baseline near the requested daily horizon or the exact prior month. Daily observations older than seven calendar days and monthly averages older than four reporting periods are stale; their changes are withheld. YTD uses the prior December for monthly data and the prior year-end for daily data.")
        st.write("Max uses each provider series from its earliest available observation. U.S. daily FRED series use validated snapshots refreshed by the scheduled source writer; missing history is requested directly; OECD sovereign series are monthly and follow their publication schedule. Observation dates, rather than download times, determine freshness. If a refresh fails, the last validated history is retained with its original dates.")
        st.write("Yield top profiles use trailing yield-change percentile, distance above the long moving average measured in yield-change volatility, yield RSI, and volatility percentile. Exhaustion watch means at least two of these four readings are elevated; it is not an event. Early Warning requires a high change percentile plus two other extremes; Confirmed Exhaustion requires a subsequent downward reversal; Failed Breakout requires a return below the original long-window high breached during its confirmation window. No positioning proxy is inferred. Monthly windows are counted in months and missing months cannot be filled by a later observation. Markers and forward tables describe historical yield behavior, not forecasts or bond total returns.")
        if problems:
            st.dataframe(pd.DataFrame(problems, columns=["Series", "Provider status"]), hide_index=True, width="stretch")
        if official_daily:
            definition = next(item for item in DAILY_SOVEREIGNS if item[0] == selected)
            st.link_button("View official source", definition[4])
            st.caption(f"10Y · {definition[3]} · {definition[2]}")
        else:
            st.link_button("View source series on FRED", f"https://fred.stlouisfed.org/series/{chosen['symbol']}")
    render_footer()


st.set_page_config(page_title="Global Bond Monitor", layout="wide", initial_sidebar_state="expanded")
render()
