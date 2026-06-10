from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


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
    "Fed": "#7c3aed",
    "Inflation": "#dc2626",
    "Labor": "#2563eb",
    "Treasury": "#d97706",
    "Growth": "#475569",
    "Options": "#059669",
    "Earnings": "#0891b2",
    "Quarter-End": "#9333ea",
    "Custom": "#111827",
}

RISK_COLORS = {
    "low": "#16a34a",
    "medium": "#d97706",
    "high": "#dc2626",
    "neutral": "#64748b",
}


st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 2rem;
            max-width: 1580px;
        }

        .adfm-title {
            font-size: 1.85rem;
            line-height: 1.18;
            font-weight: 780;
            margin: 0 0 0.20rem 0;
            color: #0f172a;
        }

        .adfm-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.0rem;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 13px 15px 10px 15px;
            min-height: 98px;
            box-shadow: 0 1px 4px rgba(15,23,42,0.05);
        }

        .metric-label {
            font-size: 0.70rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            margin-bottom: 0.42rem;
        }

        .metric-value {
            font-size: 1.12rem;
            font-weight: 760;
            color: #0f172a;
            line-height: 1.18;
        }

        .metric-footnote {
            font-size: 0.76rem;
            color: #94a3b8;
            margin-top: 0.42rem;
            line-height: 1.35;
        }

        .section-title {
            font-size: 1.03rem;
            font-weight: 760;
            color: #0f172a;
            margin-top: 0.85rem;
            margin-bottom: 0.45rem;
        }

        .note-box {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 11px 13px;
            color: #475569;
            font-size: 0.85rem;
            line-height: 1.45;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)


def is_weekday(d: date) -> bool:
    return d.weekday() < 5


def next_weekday(d: date) -> date:
    while not is_weekday(d):
        d += timedelta(days=1)
    return d


def first_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month, 1)

    while d.weekday() != weekday:
        d += timedelta(days=1)

    return d


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    return first_weekday(year, month, weekday) + timedelta(days=7 * (n - 1))


def first_business_day(year: int, month: int) -> date:
    return next_weekday(date(year, month, 1))


def nth_business_day(year: int, month: int, n: int) -> date:
    d = first_business_day(year, month)
    count = 1

    while count < n:
        d += timedelta(days=1)
        if is_weekday(d):
            count += 1

    return d


def third_friday(year: int, month: int) -> date:
    return nth_weekday(year, month, 4, 3)


def last_day_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)

    return date(year, month + 1, 1) - timedelta(days=1)


def normalize_event_type(raw: object) -> str:
    val = str(raw).strip()

    if not val:
        return "Custom"

    aliases = {
        "cpi": "Inflation",
        "ppi": "Inflation",
        "inflation": "Inflation",
        "payroll": "Labor",
        "payrolls": "Labor",
        "jobs": "Labor",
        "labor": "Labor",
        "fomc": "Fed",
        "fed": "Fed",
        "treasury": "Treasury",
        "refunding": "Treasury",
        "growth": "Growth",
        "ism": "Growth",
        "retail": "Growth",
        "opex": "Options",
        "options": "Options",
        "earnings": "Earnings",
        "quarter-end": "Quarter-End",
        "quarter end": "Quarter-End",
        "custom": "Custom",
    }

    return aliases.get(val.lower(), val if val in EVENT_WEIGHTS else "Custom")


def build_rule_calendar(start: date, horizon_days: int, include_fed_windows: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    end = start + timedelta(days=horizon_days)
    cursor = date(start.year, start.month, 1)

    fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}

    while cursor <= end:
        y, m = cursor.year, cursor.month

        rows.extend(
            [
                {
                    "Date": first_weekday(y, m, 4),
                    "Event": "Payrolls / Employment Situation",
                    "Type": "Labor",
                    "Region": "U.S.",
                    "Why It Matters": "Growth, wages, Fed pricing, USD, and equity-index beta.",
                    "Precision": "Estimated",
                },
                {
                    "Date": nth_weekday(y, m, 2, 2),
                    "Event": "CPI Inflation Window",
                    "Type": "Inflation",
                    "Region": "U.S.",
                    "Why It Matters": "Rates, dollar, duration, Nasdaq duration factor, and real-income impulse.",
                    "Precision": "Estimated",
                },
                {
                    "Date": nth_weekday(y, m, 3, 2),
                    "Event": "PPI Inflation Window",
                    "Type": "Inflation",
                    "Region": "U.S.",
                    "Why It Matters": "Pipeline inflation, margins, and input-cost pressure.",
                    "Precision": "Estimated",
                },
                {
                    "Date": first_business_day(y, m),
                    "Event": "ISM Manufacturing Window",
                    "Type": "Growth",
                    "Region": "U.S.",
                    "Why It Matters": "Cyclical growth impulse, rates, commodities, and small-cap beta.",
                    "Precision": "Estimated",
                },
                {
                    "Date": nth_business_day(y, m, 3),
                    "Event": "ISM Services Window",
                    "Type": "Growth",
                    "Region": "U.S.",
                    "Why It Matters": "Services inflation, labor demand, and broad growth signal.",
                    "Precision": "Estimated",
                },
                {
                    "Date": third_friday(y, m),
                    "Event": "Monthly Options Expiration",
                    "Type": "Options",
                    "Region": "U.S.",
                    "Why It Matters": "Gamma decay, dealer hedging, vol supply, and liquidity shift.",
                    "Precision": "Rule",
                },
            ]
        )

        retail_window = next_weekday(date(y, m, min(15, last_day_of_month(y, m).day)))

        rows.append(
            {
                "Date": retail_window,
                "Event": "Retail Sales Window",
                "Type": "Growth",
                "Region": "U.S.",
                "Why It Matters": "Consumption impulse, soft landing narrative, cyclicals, and rates.",
                "Precision": "Estimated",
            }
        )

        if include_fed_windows and m in fomc_months:
            rows.append(
                {
                    "Date": nth_weekday(y, m, 2, 3),
                    "Event": "FOMC Decision Window",
                    "Type": "Fed",
                    "Region": "U.S.",
                    "Why It Matters": "Policy path, financial conditions, USD, curve, and growth duration.",
                    "Precision": "Estimated",
                }
            )

        if m in {1, 4, 7, 10}:
            rows.append(
                {
                    "Date": nth_weekday(y, m, 0, 2),
                    "Event": "Earnings Season Ramp",
                    "Type": "Earnings",
                    "Region": "U.S.",
                    "Why It Matters": "Guidance, revisions, index concentration, and factor leadership.",
                    "Precision": "Estimated",
                }
            )

        if m in {2, 5, 8, 11}:
            rows.append(
                {
                    "Date": first_weekday(y, m, 2),
                    "Event": "Quarterly Treasury Refunding Window",
                    "Type": "Treasury",
                    "Region": "U.S.",
                    "Why It Matters": "Coupon supply, term premium, curve pressure, and duration risk.",
                    "Precision": "Estimated",
                }
            )

        if m in {3, 6, 9, 12}:
            rows.append(
                {
                    "Date": last_day_of_month(y, m),
                    "Event": "Quarter-End Rebalance",
                    "Type": "Quarter-End",
                    "Region": "Global",
                    "Why It Matters": "Rebalance flow, liquidity, dealer balance sheet, and positioning.",
                    "Precision": "Rule",
                }
            )

        cursor = add_months(cursor, 1)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Type"] = df["Type"].map(normalize_event_type)

    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def parse_custom_events(text: str) -> pd.DataFrame:
    cols = ["Date", "Event", "Type", "Region", "Why It Matters", "Precision"]

    if not text.strip():
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(StringIO(text.strip()))
    except Exception as exc:
        st.warning(f"Could not parse custom CSV: {exc}")
        return pd.DataFrame(columns=cols)

    df.columns = [str(c).strip() for c in df.columns]

    required = ["Date", "Event"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(f"Custom calendar missing required columns: {', '.join(missing)}")
        return pd.DataFrame(columns=cols)

    for optional, default in {
        "Type": "Custom",
        "Region": "Global",
        "Why It Matters": "User-defined catalyst.",
    }.items():
        if optional not in df.columns:
            df[optional] = default

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    bad = parsed_dates.isna().sum()

    if bad:
        st.warning(f"Dropped {bad} custom event row(s) with invalid dates.")

    df = df.loc[parsed_dates.notna()].copy()
    df["Date"] = parsed_dates.loc[parsed_dates.notna()].dt.date
    df["Event"] = df["Event"].astype(str).str.strip()
    df["Type"] = df["Type"].map(normalize_event_type)
    df["Region"] = df["Region"].fillna("Global").astype(str).str.strip()
    df["Why It Matters"] = df["Why It Matters"].fillna("User-defined catalyst.").astype(str).str.strip()
    df["Precision"] = "Custom"

    return df[cols]


def close_from_yfinance(data: pd.DataFrame, tickers: Tuple[str, ...]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        if "Close" in level0:
            close = data["Close"].copy()
        elif "Close" in level1:
            close = data.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" in data.columns and len(tickers) == 1:
            close = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close = data.copy()

    close = close.loc[:, [c for c in close.columns if str(c) in tickers]]
    close = close.rename(columns={c: str(c) for c in close.columns})
    close.index = pd.to_datetime(close.index).tz_localize(None)

    return close.sort_index().dropna(how="all").ffill()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: Tuple[str, ...], start_iso: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            list(tickers),
            start=start_iso,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    return close_from_yfinance(raw, tickers)


def trailing_return(series: pd.Series, periods: int) -> float:
    s = series.dropna()

    if len(s) <= periods:
        return np.nan

    return float(s.iloc[-1] / s.iloc[-periods - 1] - 1)


def ytd_return(series: pd.Series, today_: date) -> float:
    s = series.dropna()

    if s.empty:
        return np.nan

    year_start = pd.Timestamp(date(today_.year, 1, 1))
    ytd = s[s.index >= year_start]

    if len(ytd) < 2:
        return np.nan

    return float(ytd.iloc[-1] / ytd.iloc[0] - 1)


def build_market_table(close: pd.DataFrame, today_: date) -> pd.DataFrame:
    rows = []

    for ticker in MARKET_TICKERS:
        if ticker not in close.columns:
            continue

        s = close[ticker].dropna()

        if s.empty:
            continue

        latest = float(s.iloc[-1])

        if ticker == "^VIX":
            year_start = pd.Timestamp(date(today_.year, 1, 1))
            ytd_slice = s[s.index >= year_start]

            today_change = latest - float(s.iloc[-2]) if len(s) > 1 else np.nan
            one_week = latest - float(s.iloc[-6]) if len(s) > 5 else np.nan
            one_month = latest - float(s.iloc[-22]) if len(s) > 21 else np.nan
            three_month = latest - float(s.iloc[-64]) if len(s) > 63 else np.nan
            ytd = latest - float(ytd_slice.iloc[0]) if not ytd_slice.empty else np.nan
            unit = "pts"
        else:
            today_change = trailing_return(s, 1)
            one_week = trailing_return(s, 5)
            one_month = trailing_return(s, 21)
            three_month = trailing_return(s, 63)
            ytd = ytd_return(s, today_)
            unit = "%"

        rows.append(
            {
                "Ticker": ticker,
                "Asset": ASSET_LABELS.get(ticker, ticker),
                "Last": latest,
                "Today": today_change,
                "1W": one_week,
                "1M": one_month,
                "3M": three_month,
                "YTD": ytd,
                "Unit": unit,
            }
        )

    return pd.DataFrame(rows)


def market_stress_bonus(close: pd.DataFrame) -> Tuple[float, str]:
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


def action_text(row: pd.Series) -> str:
    days = int(row["Days"])
    score = float(row["Risk Score"]) if "Risk Score" in row else 0
    event_type = str(row["Type"])

    if days == 0 and score >= 80:
        return "Do not add blind risk; define invalidation before the print."
    if days <= 3 and score >= 82:
        return "Cut low-conviction gross; keep only trades with clear asymmetry."
    if event_type in {"Fed", "Inflation", "Labor"}:
        return "Check rates, USD, duration, and Nasdaq beta before sizing."
    if event_type == "Treasury":
        return "Watch curve, auction tone, term premium, and TLT sensitivity."
    if event_type == "Options":
        return "Expect positioning noise; separate gamma flow from fundamental signal."
    if event_type == "Earnings":
        return "Avoid crowded single-name gaps without defined downside."

    return "Monitor tape reaction and adjust only if price confirms the catalyst."


def add_event_scores(events: pd.DataFrame, today_: date, stress_bonus: float) -> pd.DataFrame:
    if events.empty:
        return events

    df = events.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Days"] = df["Date"].map(lambda d: (d - today_).days)
    df["Type"] = df["Type"].map(normalize_event_type)

    same_day_counts = df.groupby("Date")["Event"].transform("count")
    nearby_counts = []

    for d in df["Date"]:
        near = df[(df["Date"] >= d - timedelta(days=2)) & (df["Date"] <= d + timedelta(days=2))]
        nearby_counts.append(max(0, len(near) - 1))

    df["Cluster"] = same_day_counts.sub(1).clip(lower=0).astype(int)
    df["Nearby"] = nearby_counts

    def score_row(row: pd.Series) -> float:
        base = EVENT_WEIGHTS.get(str(row["Type"]), 70)
        days = max(0, int(row["Days"]))
        proximity = max(0.0, 1.0 - days / 30.0)
        cluster_bonus = min(12.0, float(row["Cluster"]) * 5.0 + float(row["Nearby"]) * 1.5)

        score = base * 0.62 + base * 0.30 * proximity + cluster_bonus + stress_bonus

        return round(float(np.clip(score, 0, 100)), 1)

    df["Risk Score"] = df.apply(score_row, axis=1)
    df["Exposure"] = df["Type"].map(EXPOSURE_MAP).fillna("User-defined exposure")
    df["Action"] = df.apply(action_text, axis=1)

    return df.sort_values(["Date", "Risk Score"], ascending=[True, False]).reset_index(drop=True)


def risk_label(score: float) -> str:
    if score >= 82:
        return "High"
    if score >= 65:
        return "Medium"

    return "Low"


def risk_color(score: float) -> str:
    if score >= 82:
        return RISK_COLORS["high"]
    if score >= 65:
        return RISK_COLORS["medium"]

    return RISK_COLORS["low"]


def fmt_return(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "N/A"

    if unit == "pts":
        return f"{value:+.1f}"

    return f"{value:+.2%}"


def metric_card(label: str, value: str, footnote: str, color: str) -> None:
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


def make_timeline(events: pd.DataFrame, today_: date) -> go.Figure:
    fig = go.Figure()

    plot_events = events.copy()
    plot_events["PlotDate"] = pd.to_datetime(plot_events["Date"])

    for event_type, group in plot_events.groupby("Type", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["PlotDate"],
                y=group["Risk Score"],
                mode="markers",
                name=str(event_type),
                marker=dict(
                    size=np.clip(group["Risk Score"] / 4.7, 9, 22),
                    color=TYPE_COLORS.get(str(event_type), RISK_COLORS["neutral"]),
                    opacity=0.86,
                    line=dict(width=1, color="white"),
                ),
                customdata=np.stack(
                    [
                        group["Days"].astype(str),
                        group["Precision"].astype(str),
                        group["Exposure"].astype(str),
                    ],
                    axis=-1,
                ),
                text=group["Event"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "%{x|%Y-%m-%d}<br>"
                    "Days: %{customdata[0]}<br>"
                    "Risk: %{y:.0f}<br>"
                    "Precision: %{customdata[1]}<br>"
                    "%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    today_ts = pd.Timestamp(today_)

    fig.add_shape(
        type="line",
        x0=today_ts,
        x1=today_ts,
        y0=35,
        y1=103,
        xref="x",
        yref="y",
        line=dict(
            color="#0f172a",
            width=1,
            dash="dot",
        ),
    )

    fig.add_annotation(
        x=today_ts,
        y=101,
        xref="x",
        yref="y",
        text="Today",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11, color="#0f172a"),
    )

    fig.add_vrect(
        x0=today_ts,
        x1=today_ts + pd.Timedelta(days=7),
        fillcolor="#f1f5f9",
        opacity=0.55,
        line_width=0,
        layer="below",
    )

    fig.update_layout(
        height=390,
        margin=dict(l=12, r=12, t=18, b=12),
        yaxis=dict(title="Risk score", range=[35, 103], gridcolor="#eef2f7"),
        xaxis=dict(title="", gridcolor="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def make_return_heatmap(perf: pd.DataFrame) -> go.Figure:
    if perf.empty:
        return go.Figure()

    perf = perf[perf["Ticker"] != "^VIX"].copy()

    if perf.empty:
        return go.Figure()

    assets = perf["Asset"].tolist()
    windows = ["Today", "1W", "1M", "3M", "YTD"]

    z = []
    text = []

    for _, row in perf.iterrows():
        values = []
        labels = []

        for w in windows:
            val = row[w]
            values.append(val * 100 if np.isfinite(val) else np.nan)
            labels.append(fmt_return(val, "%"))

        z.append(values)
        text.append(labels)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=windows,
            y=assets,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="%", len=0.85),
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=12, r=12, t=18, b=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(side="top"),
    )

    return fig


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Lightweight catalyst planner for macro windows, options expirations, Treasury supply,
        quarter-end flows, and user-defined events.

        Estimated rows are for planning, not official calendars. Paste exact dates in the
        custom CSV box when precision matters.
        """
    )

    st.divider()
    st.header("Controls")

    horizon_days = st.select_slider(
        "Event horizon",
        options=[14, 30, 60, 90, 120, 180],
        value=90,
    )

    include_estimated = st.checkbox("Include estimated macro windows", value=True)
    include_fed_windows = st.checkbox("Include estimated FOMC windows", value=True)
    hide_low_risk = st.checkbox("Hide low-risk rows", value=False)

    st.divider()
    st.header("Custom Events")
    st.caption("Required: Date, Event. Optional: Type, Region, Why It Matters.")

    custom_text = st.text_area(
        "Paste custom event CSV",
        value="",
        height=145,
        placeholder=(
            "Date,Event,Type,Region,Why It Matters\n"
            "2026-07-29,FOMC Decision,Fed,U.S.,Policy rate and press conference catalyst"
        ),
    )


today = date.today()
market_start = min(date(today.year, 1, 1) - timedelta(days=10), today - timedelta(days=460))

market = fetch_market(MARKET_TICKERS, market_start.isoformat())
stress_bonus, stress_label = market_stress_bonus(market)

event_frames: List[pd.DataFrame] = []

if include_estimated:
    event_frames.append(build_rule_calendar(today, horizon_days, include_fed_windows))

custom = parse_custom_events(custom_text)

if not custom.empty:
    event_frames.append(custom)

if event_frames:
    calendar = pd.concat(event_frames, ignore_index=True)
else:
    calendar = pd.DataFrame(columns=["Date", "Event", "Type", "Region", "Why It Matters", "Precision"])

if not calendar.empty:
    calendar["Date"] = pd.to_datetime(calendar["Date"], errors="coerce").dt.date
    calendar = calendar.dropna(subset=["Date", "Event"])
    calendar = calendar[(calendar["Date"] >= today) & (calendar["Date"] <= today + timedelta(days=horizon_days))]
    calendar = calendar.drop_duplicates(subset=["Date", "Event", "Type"], keep="last")
    calendar = add_event_scores(calendar, today, stress_bonus)

    if hide_low_risk:
        calendar = calendar[calendar["Risk Score"] >= 65].reset_index(drop=True)

perf = build_market_table(market, today)

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Visual-first event risk, tape backdrop, and the next catalyst cluster worth respecting.</div>",
    unsafe_allow_html=True,
)

if calendar.empty:
    st.info("No events to show. Enable estimated windows or paste a custom CSV.")
    st.stop()

next_event = calendar.iloc[0]
highest_risk = calendar.sort_values("Risk Score", ascending=False).iloc[0]
next_week = calendar[calendar["Days"] <= 7].copy()
cluster_days = int((calendar["Cluster"] > 0).sum())

cols = st.columns(5)

with cols[0]:
    metric_card(
        "Next Catalyst",
        str(next_event["Event"]),
        "Today" if int(next_event["Days"]) == 0 else f"{int(next_event['Days'])} days away",
        TYPE_COLORS.get(str(next_event["Type"]), RISK_COLORS["neutral"]),
    )

with cols[1]:
    metric_card(
        "Highest Risk",
        str(highest_risk["Event"]),
        f"{risk_label(float(highest_risk['Risk Score']))} risk, score {float(highest_risk['Risk Score']):.0f}",
        risk_color(float(highest_risk["Risk Score"])),
    )

with cols[2]:
    metric_card(
        "Next 7 Days",
        str(len(next_week)),
        f"{int((next_week['Risk Score'] >= 82).sum())} high-risk event(s)",
        RISK_COLORS["high"] if int((next_week["Risk Score"] >= 82).sum()) else RISK_COLORS["neutral"],
    )

with cols[3]:
    metric_card(
        "Clustered Days",
        str(cluster_days),
        "Same-day or nearby catalysts",
        RISK_COLORS["medium"] if cluster_days else RISK_COLORS["neutral"],
    )

with cols[4]:
    metric_card(
        "Vol Backdrop",
        stress_label,
        f"+{stress_bonus:.1f} added to event score" if stress_bonus else "No risk-score add-on",
        RISK_COLORS["high"] if stress_bonus >= 5 else RISK_COLORS["neutral"],
    )

top_left, top_right = st.columns([1.12, 0.88])

with top_left:
    st.markdown("<div class='section-title'>Catalyst Tape</div>", unsafe_allow_html=True)
    st.plotly_chart(make_timeline(calendar, today), use_container_width=True)

with top_right:
    st.markdown("<div class='section-title'>Market Backdrop: Today, 1W, 1M, 3M, YTD</div>", unsafe_allow_html=True)

    if perf.empty:
        st.info("Market data unavailable.")
    else:
        st.plotly_chart(make_return_heatmap(perf), use_container_width=True)

st.markdown("<div class='section-title'>Decision Rows</div>", unsafe_allow_html=True)

decision = calendar[["Date", "Days", "Event", "Type", "Risk Score", "Precision", "Exposure", "Action"]].copy()
decision["Date"] = decision["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
decision["Risk"] = decision["Risk Score"].map(lambda x: risk_label(float(x)))
decision["Risk Score"] = decision["Risk Score"].map(lambda x: f"{float(x):.0f}")

decision = decision[["Date", "Days", "Event", "Type", "Risk", "Risk Score", "Precision", "Exposure", "Action"]]

st.dataframe(
    decision,
    use_container_width=True,
    hide_index=True,
    height=360,
)

st.markdown("<div class='section-title'>Asset Performance Table</div>", unsafe_allow_html=True)

if perf.empty:
    st.info("Market data unavailable.")
else:
    formatted = perf.copy()

    formatted["Last"] = formatted.apply(
        lambda r: f"{r['Last']:.2f}" if r["Ticker"] != "^VIX" else f"{r['Last']:.1f}",
        axis=1,
    )

    for col in ["Today", "1W", "1M", "3M", "YTD"]:
        formatted[col] = formatted.apply(lambda r: fmt_return(r[col], r["Unit"]), axis=1)

    st.dataframe(
        formatted[["Ticker", "Asset", "Last", "Today", "1W", "1M", "3M", "YTD"]],
        use_container_width=True,
        hide_index=True,
        height=285,
    )

with st.expander("Full event details"):
    details = calendar.copy()
    details["Date"] = details["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
    details["Risk Score"] = details["Risk Score"].map(lambda x: f"{float(x):.1f}")

    st.dataframe(
        details[
            [
                "Date",
                "Days",
                "Event",
                "Type",
                "Region",
                "Risk Score",
                "Cluster",
                "Precision",
                "Why It Matters",
                "Exposure",
                "Action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
