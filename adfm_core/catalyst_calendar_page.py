from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.data_registry import SeriesDefinition
from adfm_core.palette import PASTEL, PASTEL_DIVERGING_SCALE
from adfm_core.primary_data import fetch_fred_series
from adfm_core.ui import (
    PageHeader,
    inject_institutional_tool_finish,
    render_footer,
    render_page_header,
)

TITLE = "Catalyst Calendar"
MARKET_TICKERS: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "UUP", "GLD", "^VIX")
ASSET_LABELS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "TLT": "Long Duration",
    "UUP": "Dollar",
    "GLD": "Gold",
    "^VIX": "VIX",
}
EVENT_WEIGHTS = {
    "Fed": 100,
    "Inflation": 96,
    "Labor": 92,
    "Treasury": 84,
    "Growth": 78,
    "Options": 76,
    "Earnings": 68,
    "Quarter-End": 64,
    "Custom": 75,
}
EXPOSURE_MAP = {
    "Fed": "Rates, USD, curve, duration factor, growth equities",
    "Inflation": "Rates, USD, TLT, Nasdaq duration factor, commodities",
    "Labor": "Rates, USD, cyclicals, small caps, index beta",
    "Treasury": "TLT, curve, term premium, liquidity, dollar",
    "Growth": "Cyclicals, small caps, credit beta, dollar",
    "Options": "Index gamma, vol supply, dealer positioning, intraday liquidity",
    "Earnings": "Single-name gaps, revisions, factor leadership",
    "Quarter-End": "Rebalance flows, liquidity, pension/CTA pressure",
    "Custom": "User-defined exposure",
}
TYPE_COLORS = {
    "Fed": PASTEL["lavender"],
    "Inflation": PASTEL["rose"],
    "Labor": PASTEL["blue"],
    "Treasury": PASTEL["amber"],
    "Growth": PASTEL["sage"],
    "Options": PASTEL["teal"],
    "Earnings": PASTEL["coral"],
    "Quarter-End": PASTEL["mauve"],
    "Custom": PASTEL["slate_blue"],
}
RISK_COLORS = {
    "low": PASTEL["sage"],
    "medium": PASTEL["amber"],
    "high": PASTEL["rose"],
    "neutral": PASTEL["slate_blue"],
}

MACRO_SERIES: Tuple[SeriesDefinition, ...] = (
    SeriesDefinition("cpi", "CPI", "CPIAUCSL", "Federal Reserve FRED / BLS", "Inflation", "Headline CPI index.", 45),
    SeriesDefinition("core_cpi", "Core CPI", "CPILFESL", "Federal Reserve FRED / BLS", "Inflation", "CPI excluding food and energy.", 45),
    SeriesDefinition("pce", "PCE", "PCEPI", "Federal Reserve FRED / BEA", "Inflation", "Headline PCE price index.", 50),
    SeriesDefinition("core_pce", "Core PCE", "PCEPILFE", "Federal Reserve FRED / BEA", "Inflation", "PCE excluding food and energy.", 50),
    SeriesDefinition("payrolls", "Payrolls", "PAYEMS", "Federal Reserve FRED / BLS", "Labor", "Total nonfarm payroll employment.", 45),
    SeriesDefinition("unemployment", "Unemployment", "UNRATE", "Federal Reserve FRED / BLS", "Labor", "Civilian unemployment rate.", 45),
    SeriesDefinition("ahe", "Average Hourly Earnings", "CES0500000003", "Federal Reserve FRED / BLS", "Labor", "Average hourly earnings, total private.", 45),
    SeriesDefinition("retail", "Retail Sales", "RSAFS", "Federal Reserve FRED / Census", "Growth", "Advance retail sales.", 45),
    SeriesDefinition("claims", "Initial Claims", "ICSA", "Federal Reserve FRED / DOL", "Labor", "Initial unemployment insurance claims.", 14),
    SeriesDefinition("jolts", "JOLTS Openings", "JTSJOL", "Federal Reserve FRED / BLS", "Labor", "Total nonfarm job openings.", 60),
    SeriesDefinition("gdp", "Real GDP", "GDPC1", "Federal Reserve FRED / BEA", "Growth", "Real gross domestic product.", 120),
)


def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _previous_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _first_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    return _first_weekday(year, month, weekday) + timedelta(days=7 * (n - 1))


def _first_business_day(year: int, month: int) -> date:
    return _next_weekday(date(year, month, 1))


def _nth_business_day(year: int, month: int, n: int) -> date:
    d = _first_business_day(year, month)
    count = 1
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


def _last_day_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _last_business_day(year: int, month: int) -> date:
    return _previous_weekday(_last_day_of_month(year, month))


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)


def _normalize_event_type(raw: object) -> str:
    val = str(raw).strip()
    aliases = {
        "cpi": "Inflation", "ppi": "Inflation", "pce": "Inflation", "inflation": "Inflation",
        "payroll": "Labor", "payrolls": "Labor", "claims": "Labor", "jolts": "Labor", "jobs": "Labor", "labor": "Labor",
        "fomc": "Fed", "fed": "Fed", "treasury": "Treasury", "refunding": "Treasury",
        "growth": "Growth", "ism": "Growth", "retail": "Growth", "gdp": "Growth",
        "opex": "Options", "options": "Options", "earnings": "Earnings",
        "quarter-end": "Quarter-End", "quarter end": "Quarter-End", "custom": "Custom",
    }
    if not val:
        return "Custom"
    return aliases.get(val.lower(), val if val in EVENT_WEIGHTS else "Custom")


def _build_rule_calendar(start: date, horizon_days: int, include_fed: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    end = start + timedelta(days=horizon_days)
    cursor = date(start.year, start.month, 1)
    fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}

    while cursor <= end:
        y, m = cursor.year, cursor.month
        rows.extend([
            {"Date": _first_weekday(y, m, 4), "Event": "Payrolls / Employment Situation", "Type": "Labor", "Region": "U.S.", "Why It Matters": "Growth, wages, Fed pricing, USD, and equity beta.", "Precision": "Estimated"},
            {"Date": _nth_weekday(y, m, 2, 2), "Event": "CPI Inflation Window", "Type": "Inflation", "Region": "U.S.", "Why It Matters": "Rates, USD, duration, Nasdaq duration factor, and real income.", "Precision": "Estimated"},
            {"Date": _nth_weekday(y, m, 3, 2), "Event": "PPI Inflation Window", "Type": "Inflation", "Region": "U.S.", "Why It Matters": "Pipeline inflation, margins, and input costs.", "Precision": "Estimated"},
            {"Date": _last_business_day(y, m), "Event": "PCE Inflation Window", "Type": "Inflation", "Region": "U.S.", "Why It Matters": "Fed-preferred inflation gauge, real rates, USD, and duration.", "Precision": "Estimated"},
            {"Date": _first_weekday(y, m, 1), "Event": "JOLTS Job Openings Window", "Type": "Labor", "Region": "U.S.", "Why It Matters": "Labor demand, wage pressure, and Fed pricing.", "Precision": "Estimated"},
            {"Date": _first_business_day(y, m), "Event": "ISM Manufacturing Window", "Type": "Growth", "Region": "U.S.", "Why It Matters": "Cyclical growth, rates, commodities, and small-cap beta.", "Precision": "Estimated"},
            {"Date": _nth_business_day(y, m, 3), "Event": "ISM Services Window", "Type": "Growth", "Region": "U.S.", "Why It Matters": "Services inflation, labor demand, and broad growth.", "Precision": "Estimated"},
            {"Date": _nth_weekday(y, m, 4, 3), "Event": "Monthly Options Expiration", "Type": "Options", "Region": "U.S.", "Why It Matters": "Gamma decay, dealer hedging, vol supply, and liquidity.", "Precision": "Rule"},
            {"Date": _next_weekday(date(y, m, 15)), "Event": "Retail Sales Window", "Type": "Growth", "Region": "U.S.", "Why It Matters": "Consumption impulse, soft-landing narrative, cyclicals, and rates.", "Precision": "Estimated"},
        ])
        if include_fed and m in fomc_months:
            rows.append({"Date": _nth_weekday(y, m, 2, 3), "Event": "FOMC Decision Window", "Type": "Fed", "Region": "U.S.", "Why It Matters": "Policy path, financial conditions, USD, curve, and growth duration.", "Precision": "Estimated"})
        if m in {1, 4, 7, 10}:
            rows.extend([
                {"Date": _nth_weekday(y, m, 3, 4), "Event": "GDP Release Window", "Type": "Growth", "Region": "U.S.", "Why It Matters": "Growth regime, real-rate pricing, cyclicals, USD, and earnings expectations.", "Precision": "Estimated"},
                {"Date": _nth_weekday(y, m, 0, 2), "Event": "Earnings Season Ramp", "Type": "Earnings", "Region": "U.S.", "Why It Matters": "Guidance, revisions, index concentration, and factor leadership.", "Precision": "Estimated"},
            ])
        if m in {2, 5, 8, 11}:
            rows.append({"Date": _first_weekday(y, m, 2), "Event": "Quarterly Treasury Refunding Window", "Type": "Treasury", "Region": "U.S.", "Why It Matters": "Coupon supply, term premium, curve pressure, and duration risk.", "Precision": "Estimated"})
        if m in {3, 6, 9, 12}:
            rows.append({"Date": _last_day_of_month(y, m), "Event": "Quarter-End Rebalance", "Type": "Quarter-End", "Region": "Global", "Why It Matters": "Rebalance flow, liquidity, dealer balance sheet, and positioning.", "Precision": "Rule"})
        cursor = _add_months(cursor, 1)

    d = start
    while d.weekday() != 3:
        d += timedelta(days=1)
    while d <= end:
        rows.append({"Date": d, "Event": "Initial Jobless Claims", "Type": "Labor", "Region": "U.S.", "Why It Matters": "High-frequency labor deterioration or resilience ahead of payrolls.", "Precision": "Rule"})
        d += timedelta(days=7)

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Type"] = df["Type"].map(_normalize_event_type)
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def _parse_custom_events(text: str) -> pd.DataFrame:
    cols = ["Date", "Event", "Type", "Region", "Why It Matters", "Precision"]
    if not text.strip():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(StringIO(text.strip()))
    except Exception as exc:
        st.warning(f"Could not parse custom CSV: {exc}")
        return pd.DataFrame(columns=cols)
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Date", "Event"}.issubset(df.columns):
        st.warning("Custom calendar requires Date and Event columns.")
        return pd.DataFrame(columns=cols)
    if "Type" not in df.columns:
        df["Type"] = "Custom"
    if "Region" not in df.columns:
        df["Region"] = "Global"
    if "Why It Matters" not in df.columns:
        df["Why It Matters"] = "User-defined catalyst."
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[parsed.notna()].copy()
    df["Date"] = parsed.loc[parsed.notna()].dt.date
    df["Type"] = df["Type"].map(_normalize_event_type)
    df["Precision"] = "Custom"
    return df[cols]


def _close_from_yfinance(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        close = data.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().dropna(how="all").ffill()


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_market(start_iso: str) -> pd.DataFrame:
    try:
        raw = yf.download(list(MARKET_TICKERS), start=start_iso, interval="1d", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    return _close_from_yfinance(raw)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_macro(start_iso: str, end_iso: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_fred_series(MACRO_SERIES, start=start_iso, end=end_iso)


def _trailing_return(s: pd.Series, periods: int) -> float:
    s = s.dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-periods - 1] - 1)


def _build_market_table(close: pd.DataFrame, today_: date) -> pd.DataFrame:
    rows = []
    for ticker in MARKET_TICKERS:
        if ticker not in close.columns:
            continue
        s = close[ticker].dropna()
        if s.empty:
            continue
        latest = float(s.iloc[-1])
        if ticker == "^VIX":
            vals = {"Today": latest - float(s.iloc[-2]) if len(s) > 1 else np.nan, "1W": latest - float(s.iloc[-6]) if len(s) > 5 else np.nan, "1M": latest - float(s.iloc[-22]) if len(s) > 21 else np.nan, "3M": latest - float(s.iloc[-64]) if len(s) > 63 else np.nan}
            year = s[s.index >= pd.Timestamp(date(today_.year, 1, 1))]
            vals["YTD"] = latest - float(year.iloc[0]) if not year.empty else np.nan
            unit = "pts"
        else:
            vals = {"Today": _trailing_return(s, 1), "1W": _trailing_return(s, 5), "1M": _trailing_return(s, 21), "3M": _trailing_return(s, 63)}
            year = s[s.index >= pd.Timestamp(date(today_.year, 1, 1))]
            vals["YTD"] = float(year.iloc[-1] / year.iloc[0] - 1) if len(year) > 1 else np.nan
            unit = "%"
        rows.append({"Ticker": ticker, "Asset": ASSET_LABELS.get(ticker, ticker), "Last": latest, **vals, "Unit": unit})
    return pd.DataFrame(rows)


def _market_stress(close: pd.DataFrame) -> Tuple[float, str]:
    if close.empty or "^VIX" not in close.columns or close["^VIX"].dropna().empty:
        return 0.0, "No VIX signal"
    vix = float(close["^VIX"].dropna().iloc[-1])
    if vix >= 30:
        return 8.0, f"VIX stress: {vix:.1f}"
    if vix >= 24:
        return 5.0, f"VIX elevated: {vix:.1f}"
    if vix >= 19:
        return 2.5, f"VIX firm: {vix:.1f}"
    return 0.0, f"VIX calm: {vix:.1f}"


def _action(row: pd.Series) -> str:
    if int(row["Days"]) <= 3 and float(row["Risk Score"]) >= 82:
        return "Cut low-conviction gross; keep only trades with clear asymmetry."
    if row["Type"] in {"Fed", "Inflation", "Labor"}:
        return "Check rates, USD, duration, and Nasdaq beta before sizing."
    if row["Type"] == "Treasury":
        return "Watch curve, auction tone, term premium, and TLT sensitivity."
    if row["Type"] == "Options":
        return "Separate gamma flow from fundamental signal."
    return "Let price confirm the catalyst before changing risk."


def _score_events(events: pd.DataFrame, today_: date, stress_bonus: float) -> pd.DataFrame:
    if events.empty:
        return events
    df = events.copy()
    df["Days"] = df["Date"].map(lambda d: (d - today_).days)
    same_day = df.groupby("Date")["Event"].transform("count")
    nearby = []
    for d in df["Date"]:
        near = df[(df["Date"] >= d - timedelta(days=2)) & (df["Date"] <= d + timedelta(days=2))]
        nearby.append(max(0, len(near) - 1))
    df["Cluster"] = same_day.sub(1).clip(lower=0).astype(int)
    df["Nearby"] = nearby
    def calc(row: pd.Series) -> float:
        base = EVENT_WEIGHTS.get(str(row["Type"]), 70)
        proximity = max(0.0, 1.0 - max(0, int(row["Days"])) / 30.0)
        cluster_bonus = min(12.0, float(row["Cluster"]) * 5.0 + float(row["Nearby"]) * 1.5)
        return round(float(np.clip(base * 0.62 + base * 0.30 * proximity + cluster_bonus + stress_bonus, 0, 100)), 1)
    df["Risk Score"] = df.apply(calc, axis=1)
    df["Exposure"] = df["Type"].map(EXPOSURE_MAP).fillna("User-defined exposure")
    df["Action"] = df.apply(_action, axis=1)
    return df.sort_values(["Date", "Risk Score"], ascending=[True, False]).reset_index(drop=True)


def _risk_label(score: float) -> str:
    if score >= 82:
        return "High"
    if score >= 65:
        return "Medium"
    return "Low"


def _metric_card(label: str, value: str, footnote: str, color: str) -> None:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{color};'>{value}</div><div class='metric-footnote'>{footnote}</div></div>", unsafe_allow_html=True)


def _timeline(events: pd.DataFrame, today_: date) -> go.Figure:
    fig = go.Figure()
    plot = events.copy()
    plot["PlotDate"] = pd.to_datetime(plot["Date"])
    for event_type, group in plot.groupby("Type", sort=False):
        fig.add_trace(go.Scatter(x=group["PlotDate"], y=group["Risk Score"], mode="markers", name=str(event_type), marker=dict(size=np.clip(group["Risk Score"] / 4.7, 9, 22), color=TYPE_COLORS.get(str(event_type), RISK_COLORS["neutral"]), opacity=0.86, line=dict(width=1, color="white")), text=group["Event"], hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<br>Risk: %{y:.0f}<extra></extra>"))
    today_ts = pd.Timestamp(today_)
    fig.add_shape(type="line", x0=today_ts, x1=today_ts, y0=35, y1=103, xref="x", yref="y", line=dict(color="#0f172a", width=1, dash="dot"))
    fig.add_vrect(x0=today_ts, x1=today_ts + pd.Timedelta(days=7), fillcolor="#f1f5f9", opacity=0.55, line_width=0, layer="below")
    fig.update_layout(height=390, margin=dict(l=12, r=12, t=18, b=12), yaxis=dict(title="Risk score", range=[35, 103], gridcolor="#eef2f7"), xaxis=dict(title="", gridcolor="#f8fafc"), legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def _heatmap(perf: pd.DataFrame) -> go.Figure:
    perf = perf[perf["Ticker"] != "^VIX"].copy()
    if perf.empty:
        return go.Figure()
    windows = ["Today", "1W", "1M", "3M", "YTD"]
    z, text = [], []
    for _, row in perf.iterrows():
        values, labels = [], []
        for w in windows:
            val = row[w]
            values.append(val * 100 if np.isfinite(val) else np.nan)
            labels.append("N/A" if not np.isfinite(val) else f"{val:+.2%}")
        z.append(values)
        text.append(labels)
    fig = go.Figure(data=go.Heatmap(z=z, x=windows, y=perf["Asset"].tolist(), text=text, texttemplate="%{text}", colorscale=PASTEL_DIVERGING_SCALE, zmid=0, colorbar=dict(title="%", len=0.85), hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=12, r=12, t=18, b=12), plot_bgcolor="white", paper_bgcolor="white", xaxis=dict(side="top"))
    return fig


def _latest_pair(series: pd.Series) -> Tuple[pd.Timestamp | None, float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, np.nan, np.nan
    return pd.Timestamp(s.index[-1]), float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else np.nan


def _fmt_period(ts: pd.Timestamp | None, frequency: str) -> str:
    if ts is None:
        return "N/A"
    if frequency == "quarterly":
        return f"Q{ts.quarter} {ts.year}"
    if frequency == "weekly":
        return ts.strftime("%b %d, %Y")
    return ts.strftime("%b %Y")


def _fmt_macro(value: float, fmt: str) -> str:
    if not np.isfinite(value):
        return "N/A"
    if fmt == "pct":
        return f"{value:.1f}%"
    if fmt == "k":
        return f"{value:+.0f}k"
    if fmt == "claims":
        return f"{value / 1000:.0f}k"
    if fmt == "m":
        return f"{value / 1000:.2f}m"
    return f"{value:.2f}"


def _macro_prints(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows = []
    def add(name: str, key: str, transform: str, fmt: str, frequency: str, source: str) -> None:
        if key not in panel.columns or panel[key].dropna().empty:
            return
        raw = panel[key].dropna()
        if transform == "yoy":
            s = raw.pct_change(12, fill_method=None) * 100
        elif transform == "mom":
            s = raw.pct_change(fill_method=None) * 100
        elif transform == "diff":
            s = raw.diff()
        elif transform == "qoq_ann":
            s = ((raw / raw.shift(1)) ** 4 - 1) * 100
        else:
            s = raw
        period, latest, previous = _latest_pair(s)
        rows.append({"Catalyst": name, "Latest": _fmt_macro(latest, fmt), "Previous": _fmt_macro(previous, fmt), "Period": _fmt_period(period, frequency), "Source": source})
    add("Headline CPI YoY", "cpi", "yoy", "pct", "monthly", "BLS / FRED")
    add("Core CPI YoY", "core_cpi", "yoy", "pct", "monthly", "BLS / FRED")
    add("Headline PCE YoY", "pce", "yoy", "pct", "monthly", "BEA / FRED")
    add("Core PCE YoY", "core_pce", "yoy", "pct", "monthly", "BEA / FRED")
    add("Nonfarm Payrolls", "payrolls", "diff", "k", "monthly", "BLS / FRED")
    add("Unemployment Rate", "unemployment", "level", "pct", "monthly", "BLS / FRED")
    add("Average Hourly Earnings YoY", "ahe", "yoy", "pct", "monthly", "BLS / FRED")
    add("Retail Sales MoM", "retail", "mom", "pct", "monthly", "Census / FRED")
    add("Initial Jobless Claims", "claims", "level", "claims", "weekly", "DOL / FRED")
    add("JOLTS Job Openings", "jolts", "level", "m", "monthly", "BLS / FRED")
    add("Real GDP QoQ SAAR", "gdp", "qoq_ann", "pct", "quarterly", "BEA / FRED")
    return pd.DataFrame(rows)


def render_catalyst_calendar() -> None:
    st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
        .block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1580px;}
        .metric-card {background: linear-gradient(180deg,#fff 0%,#fafafa 100%); border:1px solid #e5e7eb; border-radius:14px; padding:13px 15px 10px; min-height:98px; box-shadow:0 1px 4px rgba(15,23,42,.05);}
        .metric-label {font-size:.70rem; color:#64748b; text-transform:uppercase; letter-spacing:.055em; margin-bottom:.42rem;}
        .metric-value {font-size:1.12rem; font-weight:760; color:#0f172a; line-height:1.18;}
        .metric-footnote {font-size:.76rem; color:#94a3b8; margin-top:.42rem; line-height:1.35;}
        .section-title {font-size:1.03rem; font-weight:760; color:#0f172a; margin-top:.85rem; margin-bottom:.45rem;}
        .section-note {font-size:.78rem; color:#64748b; margin-top:-.20rem; margin-bottom:.55rem; line-height:1.4;}
    </style>
    """, unsafe_allow_html=True)
    inject_institutional_tool_finish()

    with st.sidebar:
        st.header("About This Tool")
        st.markdown("Forward catalyst planner for macro releases, options expirations, Treasury supply, quarter-end flows, and user-defined events. Macro dates labeled Estimated are planning windows. Latest-print data comes from BLS, BEA, Census and DOL series distributed through FRED.")
        st.divider()
        st.header("Controls")
        horizon_days = st.select_slider("Event horizon", options=[14, 30, 60, 90, 120, 180], value=90)
        include_estimated = st.checkbox("Include estimated macro windows", value=True)
        include_fed = st.checkbox("Include estimated FOMC windows", value=True)
        hide_low = st.checkbox("Hide low-risk rows", value=False)
        st.divider()
        st.header("Custom Events")
        custom_text = st.text_area("Paste custom event CSV", value="", height=145, placeholder="Date,Event,Type,Region,Why It Matters\n2026-09-16,FOMC Decision,Fed,U.S.,Policy rate and press conference catalyst")

    today = date.today()
    market = _fetch_market(min(date(today.year, 1, 1) - timedelta(days=10), today - timedelta(days=460)).isoformat())
    stress_bonus, stress_label = _market_stress(market)
    macro_panel, macro_status = _fetch_macro(date(today.year - 3, 1, 1).isoformat(), today.isoformat())

    frames: List[pd.DataFrame] = []
    if include_estimated:
        frames.append(_build_rule_calendar(today, horizon_days, include_fed))
    custom = _parse_custom_events(custom_text)
    if not custom.empty:
        frames.append(custom)
    calendar = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not calendar.empty:
        calendar = calendar[(calendar["Date"] >= today) & (calendar["Date"] <= today + timedelta(days=horizon_days))]
        calendar = calendar.drop_duplicates(subset=["Date", "Event", "Type"], keep="last")
        calendar = _score_events(calendar, today, stress_bonus)
        if hide_low:
            calendar = calendar[calendar["Risk Score"] >= 65].reset_index(drop=True)

    perf = _build_market_table(market, today)
    render_page_header(PageHeader(title=TITLE, description="Forward event risk plus the latest macro prints that define the setup going into the next catalyst.", eyebrow="ADFM Risk + Catalysts"))
    if calendar.empty:
        st.info("No events to show. Enable estimated windows or paste a custom CSV.")
        return

    next_event = calendar.iloc[0]
    highest = calendar.sort_values("Risk Score", ascending=False).iloc[0]
    next_week = calendar[calendar["Days"] <= 7]
    cluster_days = int((calendar["Cluster"] > 0).sum())
    cols = st.columns(5)
    with cols[0]:
        _metric_card("Next Catalyst", str(next_event["Event"]), "Today" if int(next_event["Days"]) == 0 else f"{int(next_event['Days'])} days away", TYPE_COLORS.get(str(next_event["Type"]), RISK_COLORS["neutral"]))
    with cols[1]:
        _metric_card("Highest Risk", str(highest["Event"]), f"{_risk_label(float(highest['Risk Score']))} risk, score {float(highest['Risk Score']):.0f}", RISK_COLORS["high"] if float(highest["Risk Score"]) >= 82 else RISK_COLORS["medium"])
    with cols[2]:
        _metric_card("Next 7 Days", str(len(next_week)), f"{int((next_week['Risk Score'] >= 82).sum())} high-risk event(s)", RISK_COLORS["high"] if int((next_week["Risk Score"] >= 82).sum()) else RISK_COLORS["neutral"])
    with cols[3]:
        _metric_card("Clustered Days", str(cluster_days), "Same-day or nearby catalysts", RISK_COLORS["medium"] if cluster_days else RISK_COLORS["neutral"])
    with cols[4]:
        _metric_card("Vol Backdrop", stress_label, f"+{stress_bonus:.1f} added to event score" if stress_bonus else "No risk-score add-on", RISK_COLORS["high"] if stress_bonus >= 5 else RISK_COLORS["neutral"])

    left, right = st.columns([1.12, 0.88])
    with left:
        st.markdown("<div class='section-title'>Catalyst Tape</div>", unsafe_allow_html=True)
        st.plotly_chart(_timeline(calendar, today), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>Market Backdrop: Today, 1W, 1M, 3M, YTD</div>", unsafe_allow_html=True)
        if perf.empty:
            st.info("Market data unavailable.")
        else:
            st.plotly_chart(_heatmap(perf), use_container_width=True)

    st.markdown("<div class='section-title'>Latest Macro Prints</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-note'>The actual numbers behind the recurring catalyst days. Latest and previous values update from primary U.S. data distributed through FRED.</div>", unsafe_allow_html=True)
    macro = _macro_prints(macro_panel)
    if macro.empty:
        st.info("Primary macro data is temporarily unavailable.")
    else:
        st.dataframe(macro, use_container_width=True, hide_index=True, height=420)

    st.markdown("<div class='section-title'>Upcoming Catalyst Rows</div>", unsafe_allow_html=True)
    decision = calendar[["Date", "Days", "Event", "Type", "Risk Score", "Precision", "Exposure", "Action"]].copy()
    decision["Date"] = decision["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
    decision["Risk"] = decision["Risk Score"].map(lambda x: _risk_label(float(x)))
    decision["Risk Score"] = decision["Risk Score"].map(lambda x: f"{float(x):.0f}")
    decision = decision[["Date", "Days", "Event", "Type", "Risk", "Risk Score", "Precision", "Exposure", "Action"]]
    st.dataframe(decision, use_container_width=True, hide_index=True, height=390)

    with st.expander("Full event details"):
        details = calendar.copy()
        details["Date"] = details["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
        st.dataframe(details[["Date", "Days", "Event", "Type", "Region", "Risk Score", "Cluster", "Precision", "Why It Matters", "Exposure", "Action"]], use_container_width=True, hide_index=True)
    with st.expander("Macro data status"):
        if not macro_status.empty:
            st.dataframe(macro_status[["key", "symbol", "provider", "data_through", "status"]], use_container_width=True, hide_index=True)
    render_footer()
