from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TITLE = "Event Risk + Catalyst Calendar"
MARKET_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "UUP", "GLD", "^VIX"]

EVENT_WEIGHTS = {
    "Labor": 90,
    "Inflation": 95,
    "Fed": 100,
    "Treasury": 80,
    "Options": 70,
    "Earnings": 65,
    "Growth": 75,
    "Quarter-End": 60,
    "Custom": 75,
}

COLORS = {
    "green": "#52b788",
    "red": "#e85d5d",
    "amber": "#f59e0b",
    "blue": "#4c78a8",
    "purple": "#8b5cf6",
    "grey": "#8b949e",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {padding-top: 3.0rem; padding-bottom: 2rem; max-width: 1550px;}
        .adfm-title {font-size: 1.9rem; line-height: 1.18; font-weight: 760; margin: 0 0 0.25rem 0; color: #111827;}
        .adfm-subtitle {font-size: 0.96rem; color: #6b7280; margin-bottom: 1.15rem;}
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%); border: 1px solid #e5e7eb; border-radius: 12px; padding: 13px 15px 10px 15px; min-height: 96px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);}
        .metric-label {font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.045em; margin-bottom: 0.42rem;}
        .metric-value {font-size: 1.22rem; font-weight: 750; color: #111827; line-height: 1.18;}
        .metric-footnote {font-size: 0.75rem; color: #9ca3af; margin-top: 0.42rem; line-height: 1.35;}
        .section-title {font-size: 1.03rem; font-weight: 750; color: #111827; margin-top: 0.9rem; margin-bottom: 0.45rem;}
        .note-box {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; color: #475569; font-size: 0.86rem; line-height: 1.45;}
    </style>
    """,
    unsafe_allow_html=True,
)


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d + timedelta(days=14)


def first_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def second_weekday(year: int, month: int, weekday: int) -> date:
    return first_weekday(year, month, weekday) + timedelta(days=7)


def last_day_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def event_score(days_until: int, event_type: str) -> float:
    base = EVENT_WEIGHTS.get(event_type, 65)
    proximity = max(0, 1 - abs(days_until) / 30)
    return round(base * (0.55 + 0.45 * proximity), 1)


def risk_color(score: float) -> str:
    if score >= 82:
        return COLORS["red"]
    if score >= 65:
        return COLORS["amber"]
    return COLORS["green"]


def parse_custom_events(text: str) -> pd.DataFrame:
    cols = ["Date", "Event", "Type", "Region", "Why It Matters", "Precision"]
    if not text.strip():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(StringIO(text))
        required = ["Date", "Event", "Type", "Region", "Why It Matters"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.warning(f"Custom calendar missing columns: {', '.join(missing)}")
            return pd.DataFrame(columns=cols)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df["Precision"] = "Custom"
        return df[cols]
    except Exception as exc:
        st.warning(f"Could not parse custom CSV: {exc}")
        return pd.DataFrame(columns=cols)


def build_rule_calendar(start: date, months_forward: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    end = start + timedelta(days=int(months_forward * 31))
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        y, m = cursor.year, cursor.month
        rows.extend(
            [
                {"Date": first_weekday(y, m, 4), "Event": "Payrolls / Employment Situation", "Type": "Labor", "Region": "U.S.", "Why It Matters": "Growth, wages, rates, and equity-index risk.", "Precision": "Estimated"},
                {"Date": second_weekday(y, m, 2), "Event": "CPI Inflation Window", "Type": "Inflation", "Region": "U.S.", "Why It Matters": "Rates, dollar, duration, and growth-equity catalyst.", "Precision": "Estimated"},
                {"Date": second_weekday(y, m, 3), "Event": "PPI Inflation Window", "Type": "Inflation", "Region": "U.S.", "Why It Matters": "Pipeline inflation and margin-pressure read-through.", "Precision": "Estimated"},
                {"Date": third_friday(y, m), "Event": "Monthly Options Expiration", "Type": "Options", "Region": "U.S.", "Why It Matters": "Dealer positioning, gamma decay, and liquidity shift.", "Precision": "Rule"},
            ]
        )
        if m in [1, 4, 7, 10]:
            rows.append({"Date": second_weekday(y, m, 0), "Event": "Earnings Season Ramp", "Type": "Earnings", "Region": "U.S.", "Why It Matters": "Index revisions, guidance tone, factor leadership.", "Precision": "Estimated"})
        if m in [2, 5, 8, 11]:
            rows.append({"Date": first_weekday(y, m, 2), "Event": "Quarterly Treasury Refunding Window", "Type": "Treasury", "Region": "U.S.", "Why It Matters": "Supply, duration risk, term premium, curve catalyst.", "Precision": "Estimated"})
        if m in [3, 6, 9, 12]:
            rows.append({"Date": last_day_of_month(y, m), "Event": "Quarter-End Rebalance", "Type": "Quarter-End", "Region": "Global", "Why It Matters": "Rebalance flow, liquidity, and positioning risk.", "Precision": "Rule"})
        cursor = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    df = pd.DataFrame(rows)
    return df[(df["Date"] >= start) & (df["Date"] <= end)].sort_values("Date")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: List[str], period: str) -> pd.DataFrame:
    try:
        data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close.dropna(how="all").ffill()


with st.sidebar:
    st.header("Settings")
    horizon_days = st.slider("Event horizon", 14, 180, 90, 7)
    months_forward = max(3, int(np.ceil(horizon_days / 31)) + 1)
    market_period = st.selectbox("Market backdrop", ["3mo", "6mo", "1y", "2y"], index=2)
    include_estimated = st.checkbox("Include estimated macro windows", value=True)

    st.divider()
    st.header("Custom Events")
    st.caption("CSV columns: Date, Event, Type, Region, Why It Matters")
    custom_text = st.text_area(
        "Paste custom event CSV",
        value="",
        height=130,
        placeholder="Date,Event,Type,Region,Why It Matters\n2026-07-29,FOMC Decision,Fed,U.S.,Policy rate and press conference catalyst",
    )

    st.divider()
    st.header("About This Tool")
    st.markdown(
        """
        Forward catalyst planner for known macro/event windows. Rule-based rows are
        estimates for planning; paste official dates as custom events when precision matters.
        """
    )

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Upcoming macro catalysts, options windows, Treasury supply, earnings season, and market backdrop.</div>",
    unsafe_allow_html=True,
)

today = date.today()
events = []
if include_estimated:
    events.append(build_rule_calendar(today, months_forward))
custom = parse_custom_events(custom_text)
if not custom.empty:
    events.append(custom)

calendar = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=["Date", "Event", "Type", "Region", "Why It Matters", "Precision"])
if calendar.empty:
    st.info("No events to show. Enable estimated windows or paste a custom CSV.")
    st.stop()

calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.date
calendar = calendar[(calendar["Date"] >= today) & (calendar["Date"] <= today + timedelta(days=horizon_days))]
calendar["Days"] = calendar["Date"].map(lambda d: (d - today).days)
calendar["Risk Score"] = calendar.apply(lambda r: event_score(int(r["Days"]), str(r["Type"])), axis=1)
calendar = calendar.sort_values(["Date", "Risk Score"], ascending=[True, False]).reset_index(drop=True)

market = fetch_market(MARKET_TICKERS, market_period)
next_event = calendar.iloc[0] if not calendar.empty else None
next_high = calendar.sort_values("Risk Score", ascending=False).iloc[0] if not calendar.empty else None
high_count = int((calendar["Risk Score"] >= 80).sum()) if not calendar.empty else 0
vix_latest = np.nan
if not market.empty and "^VIX" in market and not market["^VIX"].dropna().empty:
    vix_latest = float(market["^VIX"].dropna().iloc[-1])

cards = [
    ("Next Catalyst", str(next_event["Event"]) if next_event is not None else "N/A", "Today" if next_event is not None and next_event["Days"] == 0 else (f"{next_event['Days']} days away" if next_event is not None else "N/A"), COLORS["blue"]),
    ("Highest Risk", str(next_high["Event"]) if next_high is not None else "N/A", f"Score {next_high['Risk Score']:.0f}" if next_high is not None else "N/A", risk_color(float(next_high["Risk Score"])) if next_high is not None else COLORS["grey"]),
    ("High-Risk Events", str(high_count), f"Within {horizon_days} days", COLORS["red"] if high_count else COLORS["green"]),
    ("Event Count", str(len(calendar)), "Visible catalyst rows", COLORS["amber"]),
    ("VIX Backdrop", f"{vix_latest:.1f}" if np.isfinite(vix_latest) else "N/A", market_period, COLORS["purple"]),
]

for col, (label, value, footnote, color) in zip(st.columns(5), cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{value}</div>
                <div class="metric-footnote">{footnote}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([1.05, 0.95])
with left:
    st.markdown("<div class='section-title'>Catalyst Calendar</div>", unsafe_allow_html=True)
    display = calendar.copy()
    display["Date"] = display["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
    display["Risk Score"] = display["Risk Score"].map(lambda x: f"{x:.0f}")
    compact = display[["Date", "Days", "Event", "Type", "Risk Score", "Precision"]]
    st.dataframe(compact, use_container_width=True, hide_index=True, height=385)

with right:
    st.markdown("<div class='section-title'>Event Risk Timeline</div>", unsafe_allow_html=True)
    type_colors = {"Labor": COLORS["blue"], "Inflation": COLORS["red"], "Fed": COLORS["purple"], "Treasury": COLORS["amber"], "Options": COLORS["green"], "Earnings": "#14b8a6", "Growth": "#64748b", "Quarter-End": "#a855f7", "Custom": "#111827"}
    timeline = go.Figure()
    for event_type, group in calendar.groupby("Type"):
        timeline.add_trace(go.Scatter(x=group["Date"], y=group["Risk Score"], mode="markers", name=str(event_type), marker=dict(size=np.clip(group["Risk Score"] / 5, 9, 18), color=type_colors.get(event_type, COLORS["grey"]), opacity=0.82), text=group["Event"], hovertemplate="%{text}<br>%{x}<br>Risk %{y:.0f}<extra></extra>"))
    timeline.update_layout(height=385, margin=dict(l=12, r=12, t=15, b=15), yaxis=dict(title="Risk score", range=[35, 105]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(timeline, use_container_width=True)

st.markdown("<div class='section-title'>Market Backdrop Into Events</div>", unsafe_allow_html=True)
if market.empty:
    st.info("Market data unavailable.")
else:
    rebased = market.dropna(how="all").copy()
    rebased = rebased / rebased.ffill().bfill().iloc[0] * 100
    fig = go.Figure()
    for ticker in ["SPY", "QQQ", "IWM", "TLT", "UUP", "GLD"]:
        if ticker in rebased:
            fig.add_trace(go.Scatter(x=rebased.index, y=rebased[ticker], mode="lines", name=ticker))
    fig.update_layout(height=390, margin=dict(l=12, r=12, t=15, b=15), yaxis_title="Rebased to 100", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Event details"):
    details = calendar.copy()
    details["Date"] = details["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
    st.dataframe(details[["Date", "Days", "Event", "Type", "Region", "Risk Score", "Precision", "Why It Matters"]], use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="note-box">
        Estimated rows are planning placeholders based on recurring macro windows. Use the custom CSV field for exact official release dates.
    </div>
    """,
    unsafe_allow_html=True,
)
