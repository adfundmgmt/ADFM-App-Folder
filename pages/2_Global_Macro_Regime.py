"""Bond monitor at the established Global Macro Regime URL."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.bond_monitor import GLOBAL_SOVEREIGNS, daily_snapshot, monthly_snapshot, spread_series
from adfm_core.bond_event_study import HORIZONS, PROFILES, event_dates, event_summary, signal_frame
from adfm_core.global_macro import clean
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


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def daily_data(symbols: tuple[str, ...]):
    return fetch_fred_symbols(symbols, start="2015-01-01")


@st.cache_data(ttl=21600, max_entries=1, show_spinner=False)
def global_data(symbols: tuple[str, ...]):
    return fetch_fred_symbols(symbols, start="2015-01-01")


def monitor_table(rows: list[dict], monthly: bool, spread: bool, selected: str):
    horizons = ("1M", "3M", "YTD")

    def cell(value, change=False):
        if not np.isfinite(value):
            return '<td class="na">—</td>'
        if not change:
            return f"<td>{value:.2f}%</td>"
        tone = "flat" if abs(value) < .5 else "up" if value > 0 else "down"
        return f'<td class="{tone}">{value:+.0f} bp</td>'

    head = ("<th>Instrument</th><th>Signal state</th><th>3M after top · edge</th><th>Level</th>"
            + "".join(f"<th>{h} Δ</th>" for h in horizons)
            + "<th>Move %ile</th><th>Observed</th>")
    body = []
    for row in rows:
        snap = row["snapshot"]
        date = escape(snap["Observation"] or "—")
        label = date + (" · stale" if snap["Status"] == "Stale" else "")
        latest = row["latest"]
        state = row["state"]
        state_tone = ("active" if state == "Yield top signal" else "watch" if state in
                      ("Potential yield top", "Exhaustion watch") else "stale" if state == "Unavailable" else "")
        summary = row["summary"]
        median = summary.loc["Signal median", "3M"]
        edge = summary.loc["Median edge", "3M"]
        lift = summary.loc["Hit-rate lift", "3M"]
        count = summary.loc["Independent N", "3M"]

        tone = "na" if not np.isfinite(edge) else "flat" if edge == 0 else "down" if edge < 0 else "up"
        result = (f'{median:+.0f} / {edge:+.0f} bp' if np.isfinite(median) and np.isfinite(edge)
                  else "—")
        detail = (f'{lift:+.0f} pp lift · n={count:.0f}' if np.isfinite(lift) and count else
                  f'n={count:.0f}' if count else "No completed events")

        body.append(f'<tr class="{"selected" if row["name"] == selected else ""}"><td>{escape(row["name"])}</td>'
                    + f'<td class="state {state_tone}">{escape(state)}'
                    + f'<small>Last top: {escape(row["latest_event"])}</small></td>'
                    + f'<td class="study {tone}">{result}<small>{detail}</small></td>'
                    + cell(snap["Yield"]) + "".join(cell(snap[h], True) for h in horizons)
                    + f'<td>{display_number(float(latest["ChangePctile"])) if latest is not None else "—"}</td>'
                    + f'<td>{label}</td></tr>')
    st.markdown('<div class="bond-wrap"><table class="bond-table"><thead><tr>' + head
                + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>', unsafe_allow_html=True)
    unit = "spread" if spread else "yield"
    frequency = "monthly average" if monthly else "daily observation"
    st.markdown(f'<div class="bond-note">Level is {unit} in %. Changes are basis points; '
                f'{frequency} dates are shown per instrument. 3M after top is median yield change / edge '
                'versus non-overlapping baseline windows. Lift is the difference in lower-yield hit rate '
                '(percentage points); n is independent 3M signals. '
                'Green favors a yield top; red opposes it. These are yield outcomes, not bond returns.</div>',
                unsafe_allow_html=True)


def history_chart(series: pd.Series, name: str, years: int | None, monthly: bool, spread: bool,
                  events: pd.DatetimeIndex):
    history = clean(series)
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
    st.caption(f"{name} · {'monthly average' if monthly else 'daily observation'} · "
               f"{len(marked)} signal markers in view · latest observation {history.index[-1]:%Y-%m-%d}")


def display_number(value: float, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if np.isfinite(value) else "—"


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def study_row(series: pd.Series, frequency: str, profile: str, period: str):
    diagnostics = signal_frame(series, frequency, profile)
    full_events = event_dates(diagnostics, 6 if frequency == "monthly" else 63)
    history = clean(series)
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
        source_note="Federal Reserve / FRED · OECD long-term rates via FRED",
    ))
    with st.sidebar:
        render_sidebar_about("2_Global_Macro_Regime.py")
        st.caption("Daily U.S. yields and spreads are end-of-day observations. Global 10-year yields are monthly averages.")

    a, b, c, d = st.columns([1.45, 1.55, 1.35, .7], gap="small")
    with a:
        view = st.selectbox("Bond market", VIEWS, key="bond_view")
    monthly = view == "Global sovereign"
    if monthly:
        names = [name for name, _ in GLOBAL_SOVEREIGNS]
        default = "United States"
    else:
        definitions = US if view == "US Treasury" else REAL if view == "Real yields & inflation" else CREDIT
        names = [name for name, _ in definitions]
        default = "10-year Treasury" if view == "US Treasury" else names[0]
        if view == "US Treasury":
            names += ["2s10s curve", "5s30s curve"]
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
    else:
        symbols = tuple(symbol for _, symbol in definitions)
        with st.spinner("Loading bond market history…"):
            panel, status = daily_data(symbols)
        items = [(name, symbol, panel[symbol] if symbol in panel else pd.Series(dtype=float))
                 for name, symbol in definitions]
        problems = [(str(row.get("symbol", "FRED")), str(row["error"]))
                    for _, row in status.iterrows() if row.get("error")] if not status.empty and "error" in status else []
        if view == "US Treasury":
            indexed = {symbol: data for _, symbol, data in items}
            items += [("2s10s curve", "DGS10-DGS2", spread_series(indexed["DGS10"], indexed["DGS2"])),
                      ("5s30s curve", "DGS30-DGS5", spread_series(indexed["DGS30"], indexed["DGS5"]))]

    rows = [{"name": name, "symbol": symbol, "series": series,
             "snapshot": monthly_snapshot(series, today) if monthly else daily_snapshot(series, today)}
            for name, symbol, series in items]
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
            row["latest_event"] = (events[-1].strftime("%b %Y" if monthly else "%b %d, %Y")
                                   if len(events) else "None")
    latest = chosen["latest"]
    events = chosen["events"]
    history = chosen["history"]
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
                '</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bond-heading">{escape(selected)} · yield top signals</div>', unsafe_allow_html=True)
    history_chart(chosen["series"], selected, PERIODS[period], monthly,
                  view == "Credit spreads" or "curve" in selected, events)
    st.markdown('<div class="bond-heading">Bond signal monitor</div>', unsafe_allow_html=True)
    monitor_table(rows, monthly, view == "Credit spreads", selected)
    with st.expander("Historical yield top signals"):
        if history.empty:
            st.write("No historical events met this profile for the available series.")
        else:
            st.dataframe(history.sort_values("Date", ascending=False), hide_index=True, width="stretch",
                         column_config={label: st.column_config.NumberColumn(format="%.0f bp")
                                        for label in HORIZONS[frequency]})
    with st.expander("Sources and definitions"):
        if monthly:
            st.write("OECD 10-year long-term interest rates distributed by FRED. These are monthly averages, not tradable bond prices or intraday quotes. Missing prior months are never bridged.")
        else:
            st.write("Treasury constant-maturity yields, TIPS real yields and inflation compensation use Federal Reserve series distributed by FRED. Credit uses ICE BofA option-adjusted spread indices distributed by FRED. Values are end-of-day observations, not executable bond prices.")
            st.write("Treasury curve spreads subtract yields observed on the same date. A positive 2s10s level means the 10-year yield exceeds the 2-year yield.")
        st.write("Changes require a baseline near the requested daily horizon or the exact prior month. Daily observations older than seven calendar days and monthly averages older than four reporting periods are stale; their changes are withheld. YTD uses the prior December for monthly data and the prior year-end for daily data.")
        st.write("Yield top profiles use trailing yield-change percentile, distance above the long moving average measured in yield-change volatility, yield RSI, and volatility percentile. Exhaustion watch means at least two of these four readings are elevated; it is not an event. Early Warning requires a high change percentile plus two other extremes; Confirmed Exhaustion requires a subsequent downward reversal; Failed Breakout requires a prior long-window high to fail. No positioning proxy is inferred. Monthly windows are counted in months and missing months cannot be filled by a later observation. Markers and forward tables describe historical yield behavior, not forecasts or bond total returns.")
        if problems:
            st.dataframe(pd.DataFrame(problems, columns=["Series", "Provider status"]), hide_index=True, width="stretch")
        if "curve" not in selected:
            st.link_button("View source series on FRED", f"https://fred.stlouisfed.org/series/{chosen['symbol']}")
    render_footer()


st.set_page_config(page_title="Global Bond Monitor", layout="wide", initial_sidebar_state="expanded")
render()
