"""Bond monitor at the established Global Macro Regime URL."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.bond_monitor import GLOBAL_SOVEREIGNS, daily_snapshot, monthly_snapshot, spread_series
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
PERIODS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}


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
        font:.76rem/1.2 Arial,Helvetica,sans-serif}
    .bond-table th,.bond-table td {border-right:1px solid #aeb7bd;border-bottom:1px solid #aeb7bd;
        padding:.57rem .48rem;text-align:right;white-space:nowrap}
    .bond-table th:first-child,.bond-table td:first-child {text-align:left;width:25%;padding-left:.72rem}
    .bond-table th:last-child,.bond-table td:last-child {border-right:0}
    .bond-table tr:last-child td {border-bottom:0}
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


def comparison_table(rows: list[dict], monthly: bool, spread: bool):
    horizons = ("1M", "3M", "YTD") if monthly else ("1D", "1W", "1M", "3M", "YTD")

    def cell(value, change=False):
        if not np.isfinite(value):
            return '<td class="na">—</td>'
        if not change:
            return f"<td>{value:.2f}%</td>"
        tone = "flat" if abs(value) < .5 else "up" if value > 0 else "down"
        return f'<td class="{tone}">{value:+.0f} bp</td>'

    head = "<th>Instrument</th><th>Level</th>" + "".join(f"<th>{h} Δ</th>" for h in horizons) + "<th>Observed</th>"
    body = []
    for row in rows:
        snap = row["snapshot"]
        date = escape(snap["Observation"] or "—")
        label = date + (" · stale" if snap["Status"] == "Stale" else "")
        body.append("<tr><td>" + escape(row["name"]) + "</td>" + cell(snap["Yield"])
                    + "".join(cell(snap[h], True) for h in horizons) + f"<td>{label}</td></tr>")
    st.markdown('<div class="bond-wrap"><table class="bond-table"><thead><tr>' + head
                + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>', unsafe_allow_html=True)
    unit = "spread" if spread else "yield"
    frequency = "monthly average" if monthly else "daily observation"
    st.markdown(f'<div class="bond-note">Level is {unit} in %. Changes are basis points; '
                f'{frequency} dates are shown for every instrument. Red = higher/wider, '
                'green = lower/tighter. Missing comparisons are blank.</div>', unsafe_allow_html=True)


def history_chart(series: pd.Series, name: str, years: int, monthly: bool, spread: bool):
    history = clean(series)
    if history.empty:
        st.info("No history is available for this instrument.")
        return
    history = history.loc[history.index >= history.index[-1] - pd.DateOffset(years=years)]
    fig = go.Figure(go.Scatter(x=history.index, y=history, mode="lines",
                               line=dict(color=PASTEL["blue"], width=2), name=name,
                               hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>"))
    fig.update_layout(height=430, margin=dict(l=35, r=20, t=10, b=25), paper_bgcolor="white",
                      plot_bgcolor="white", showlegend=False, font=dict(family="Arial", color="#222", size=12),
                      xaxis=dict(showgrid=False, linecolor="#aeb7bd"),
                      yaxis=dict(title="Spread (%)" if spread else "Yield (%)", gridcolor="#e7edf1", zeroline=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"{name} · {'monthly average' if monthly else 'daily observation'} · latest observation {history.index[-1]:%Y-%m-%d}")


def render():
    style()
    render_page_header(PageHeader(
        title="Global Bond Monitor",
        description="Track sovereign yields, the Treasury curve, inflation compensation and credit spreads across horizons.",
        eyebrow="ADFM Rates & Credit",
        source_note="Federal Reserve / FRED · OECD long-term rates via FRED",
    ))
    with st.sidebar:
        render_sidebar_about("2_Global_Macro_Regime.py")
        st.caption("Daily U.S. yields and spreads are end-of-day observations. Global 10-year yields are monthly averages.")

    a, b, c = st.columns([1.75, 1.65, .7], gap="small")
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
        selected = st.selectbox("Chart instrument", names, index=names.index(default), key="bond_instrument")
    with c:
        period = st.selectbox("Chart history", tuple(PERIODS), index=1, key="bond_period")

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
    st.markdown('<div class="bond-status">'
                f'<span><strong>Market</strong> {escape(view)}</span>'
                f'<span><strong>Coverage</strong> {len(current)}/{len(rows)} current</span>'
                f'<span><strong>Frequency</strong> {"Monthly average" if monthly else "Daily"}</span>'
                f'<span><strong>Latest period</strong> {escape(max(current) if current else "Unavailable")}</span>'
                '<span><strong>Changes</strong> Basis points</span></div>', unsafe_allow_html=True)
    if not current:
        st.warning("No current observations were returned. Historical levels remain visible with their original dates.")
    chosen = next(row for row in rows if row["name"] == selected)
    st.markdown(f'<div class="bond-heading">{escape(selected)} · history</div>', unsafe_allow_html=True)
    history_chart(chosen["series"], selected, PERIODS[period], monthly,
                  view == "Credit spreads" or "curve" in selected)
    st.markdown('<div class="bond-heading">Market comparison</div>', unsafe_allow_html=True)
    comparison_table(rows, monthly, view == "Credit spreads")

    with st.expander("Sources and definitions"):
        if monthly:
            st.write("OECD 10-year long-term interest rates distributed by FRED. These are monthly averages, not tradable bond prices or intraday quotes. Missing prior months are never bridged.")
        else:
            st.write("Treasury constant-maturity yields, TIPS real yields and inflation compensation use Federal Reserve series distributed by FRED. Credit uses ICE BofA option-adjusted spread indices distributed by FRED. Values are end-of-day observations, not executable bond prices.")
            st.write("Treasury curve spreads subtract yields observed on the same date. A positive 2s10s level means the 10-year yield exceeds the 2-year yield.")
        st.write("Changes require a baseline near the requested daily horizon or the exact prior month. Daily observations older than seven calendar days and monthly averages older than four reporting periods are stale; their changes are withheld. YTD uses the prior December for monthly data and the prior year-end for daily data.")
        if problems:
            st.dataframe(pd.DataFrame(problems, columns=["Series", "Provider status"]), hide_index=True, width="stretch")
        if "curve" not in selected:
            st.link_button("View source series on FRED", f"https://fred.stlouisfed.org/series/{chosen['symbol']}")
    render_footer()


st.set_page_config(page_title="Global Bond Monitor", layout="wide", initial_sidebar_state="expanded")
render()
