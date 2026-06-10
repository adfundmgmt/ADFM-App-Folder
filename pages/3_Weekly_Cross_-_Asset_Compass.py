
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Cross-Asset Tape",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px;}
        section[data-testid="stSidebar"] {border-right: 1px solid rgba(49,51,63,0.10);}
        .stMetric {
            background: #f7f8fa;
            border: 1px solid rgba(49,51,63,0.08);
            padding: 14px 16px;
            border-radius: 16px;
        }
        h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)


UNIVERSE: Dict[str, Dict[str, str]] = {
    "SPY": {"name": "S&P 500", "sleeve": "Equities"},
    "QQQ": {"name": "Nasdaq 100", "sleeve": "Equities"},
    "IWM": {"name": "Russell 2000", "sleeve": "Equities"},
    "RSP": {"name": "Equal Weight S&P", "sleeve": "Equities"},
    "EFA": {"name": "DM ex-US", "sleeve": "Equities"},
    "EEM": {"name": "Emerging Markets", "sleeve": "Equities"},
    "KWEB": {"name": "China Internet", "sleeve": "Equities"},
    "SMH": {"name": "Semiconductors", "sleeve": "Equities"},
    "XLF": {"name": "Financials", "sleeve": "Equities"},
    "XLE": {"name": "Energy Equities", "sleeve": "Equities"},

    "SHY": {"name": "1-3Y Treasuries", "sleeve": "Rates"},
    "IEF": {"name": "7-10Y Treasuries", "sleeve": "Rates"},
    "TLT": {"name": "20Y+ Treasuries", "sleeve": "Rates"},
    "TIP": {"name": "TIPS", "sleeve": "Rates"},
    "LQD": {"name": "Investment Grade Credit", "sleeve": "Credit"},
    "HYG": {"name": "High Yield Credit", "sleeve": "Credit"},

    "UUP": {"name": "US Dollar", "sleeve": "FX"},
    "FXE": {"name": "Euro", "sleeve": "FX"},
    "FXY": {"name": "Japanese Yen", "sleeve": "FX"},

    "GLD": {"name": "Gold", "sleeve": "Commodities"},
    "SLV": {"name": "Silver", "sleeve": "Commodities"},
    "USO": {"name": "Oil", "sleeve": "Commodities"},
    "UNG": {"name": "Natural Gas", "sleeve": "Commodities"},
    "DBC": {"name": "Broad Commodities", "sleeve": "Commodities"},

    "VIXY": {"name": "VIX Short-Term Futures", "sleeve": "Volatility"},
    "BTC-USD": {"name": "Bitcoin", "sleeve": "Crypto"},
    "ETH-USD": {"name": "Ethereum", "sleeve": "Crypto"},
}

CORE_TAPE = ["SPY", "QQQ", "IWM", "TLT", "HYG", "UUP", "GLD", "USO", "VIXY", "BTC-USD"]


@dataclass
class AnchorDates:
    latest: pd.Timestamp
    prior: pd.Timestamp
    last_friday: pd.Timestamp


def pct_fmt(x: float | int | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x * 100:+.{decimals}f}%"


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / std


def get_close_prices(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        second = raw.columns.get_level_values(1)

        if "Close" in top:
            close = raw["Close"].copy()
        elif "Adj Close" in top:
            close = raw["Adj Close"].copy()
        elif "Close" in second:
            close = raw.xs("Close", level=1, axis=1).copy()
        elif "Adj Close" in second:
            close = raw.xs("Adj Close", level=1, axis=1).copy()
        else:
            raise ValueError("Could not locate close prices.")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[col]].copy()
        close.columns = tickers[:1]

    close = close.loc[:, [c for c in close.columns if c in tickers]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().dropna(how="all")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_daily_prices(tickers: List[str], period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return get_close_prices(raw, tickers).ffill()


@st.cache_data(ttl=60 * 5, show_spinner=False)
def load_intraday_prices(tickers: List[str]) -> Tuple[pd.Series, pd.Timestamp | None]:
    raw = yf.download(
        tickers=tickers,
        period="5d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.Series(dtype=float), None

    close = get_close_prices(raw, tickers)
    if close.empty:
        return pd.Series(dtype=float), None

    last_ts = close.dropna(how="all").index[-1]
    last_prices = close.ffill().iloc[-1].dropna()
    return last_prices, last_ts


def merge_latest_prices(daily: pd.DataFrame, latest_prices: pd.Series, latest_ts: pd.Timestamp | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds current intraday prices only for tickers that actually have a live price.

    The previous version forward-filled missing ETF prices into today's row when only crypto had
    fresh prices. That created fake +0.0% readings. This version leaves missing current prices
    blank instead of fabricating a flat move.
    """
    out = daily.copy()
    status_rows = []

    if latest_prices.empty or latest_ts is None:
        for ticker in out.columns:
            status_rows.append({"ticker": ticker, "asof": out[ticker].dropna().index[-1], "source": "daily_close"})
        return out, pd.DataFrame(status_rows)

    current_date = pd.Timestamp(latest_ts.date())

    for ticker in out.columns:
        s = out[ticker].dropna()
        if s.empty:
            status_rows.append({"ticker": ticker, "asof": pd.NaT, "source": "missing"})
            continue

        if ticker in latest_prices.index and pd.notna(latest_prices[ticker]):
            if current_date not in out.index:
                out.loc[current_date, ticker] = np.nan
            out.loc[current_date, ticker] = float(latest_prices[ticker])
            status_rows.append({"ticker": ticker, "asof": current_date, "source": "intraday"})
        else:
            status_rows.append({"ticker": ticker, "asof": s.index[-1], "source": "daily_close"})

    return out.sort_index(), pd.DataFrame(status_rows)


def get_anchor_dates(close: pd.DataFrame) -> AnchorDates:
    idx = close.dropna(how="all").index
    latest = idx[-1]
    prior = idx[-2] if len(idx) >= 2 else idx[-1]
    friday_candidates = idx[idx.weekday == 4]

    if len(friday_candidates) == 0:
        last_friday = prior
    else:
        last_friday = friday_candidates[friday_candidates <= latest][-1]

    return AnchorDates(latest=latest, prior=prior, last_friday=last_friday)


def safe_change(series: pd.Series, periods: int) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    base = s.iloc[-1 - periods]
    if pd.isna(base) or base == 0:
        return np.nan
    return float(s.iloc[-1] / base - 1)


def build_feature_table(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    latest_calendar_date = close.dropna(how="all").index[-1]
    current_year = latest_calendar_date.year

    for ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) < 65:
            continue

        last_date = s.index[-1]
        has_current_price = last_date == latest_calendar_date

        ret_1d = safe_change(s, 1) if has_current_price else np.nan
        ret_1w = safe_change(s, 5)
        ret_1m = safe_change(s, 21)
        ret_3m = safe_change(s, 63)

        ytd_slice = s[s.index.year == current_year]
        ret_ytd = float(s.iloc[-1] / ytd_slice.iloc[0] - 1) if not ytd_slice.empty else np.nan

        ma20 = s.rolling(20).mean().iloc[-1]
        ma50 = s.rolling(50).mean().iloc[-1]
        vol_1m = s.pct_change().tail(21).std() * np.sqrt(252)

        dist_20 = float(s.iloc[-1] / ma20 - 1) if pd.notna(ma20) and ma20 != 0 else np.nan
        dist_50 = float(s.iloc[-1] / ma50 - 1) if pd.notna(ma50) and ma50 != 0 else np.nan

        rows.append(
            {
                "ticker": ticker,
                "asset": UNIVERSE[ticker]["name"],
                "sleeve": UNIVERSE[ticker]["sleeve"],
                "ret_1d": ret_1d,
                "ret_1w": ret_1w,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_ytd": ret_ytd,
                "dist_20": dist_20,
                "dist_50": dist_50,
                "vol_1m": float(vol_1m) if pd.notna(vol_1m) else np.nan,
                "data_asof": last_date,
                "has_current_price": has_current_price,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["score"] = (
        0.20 * zscore(df["ret_1d"].fillna(0))
        + 0.25 * zscore(df["ret_1w"])
        + 0.25 * zscore(df["ret_1m"])
        + 0.20 * zscore(df["ret_3m"])
        + 0.10 * zscore(df["dist_50"])
        - 0.10 * zscore(df["vol_1m"])
    )

    return df.sort_values(["sleeve", "ret_1w"], ascending=[True, False]).reset_index(drop=True)


def get_value(df: pd.DataFrame, ticker: str, col: str) -> float:
    row = df.loc[df["ticker"] == ticker, col]
    if row.empty:
        return np.nan
    return float(row.iloc[0])


def regime_summary(features: pd.DataFrame) -> Tuple[str, str]:
    eq = np.nanmean([get_value(features, x, "ret_1w") for x in ["SPY", "QQQ", "IWM", "RSP"]])
    credit = get_value(features, "HYG", "ret_1w") - get_value(features, "LQD", "ret_1w")
    rates = get_value(features, "TLT", "ret_1w")
    dollar = get_value(features, "UUP", "ret_1w")
    vol = get_value(features, "VIXY", "ret_1w")

    score = 0
    score += 1 if eq > 0 else -1
    score += 1 if credit > 0 else -1
    score += 1 if rates > 0 else -1
    score += 1 if dollar < 0 else -1
    score += 1 if vol < 0 else -1

    if score >= 3:
        return "Risk-on", "The one-week tape shows easier conditions across equities, credit, rates, dollar, and volatility."
    if score <= -3:
        return "Risk-off", "The one-week tape shows tightening pressure across equities, credit, rates, dollar, and volatility."
    return "Mixed", "Cross-asset confirmation is weak. Dispersion matters more than broad beta."


def make_heatmap(features: pd.DataFrame) -> go.Figure:
    cols = ["ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_ytd"]
    labels = ["Today", "1W", "1M", "3M", "YTD"]
    data = features.sort_values(["sleeve", "ret_1w"], ascending=[True, False]).copy()

    z = data[cols].to_numpy() * 100
    text = np.empty_like(z, dtype=object)

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            text[i, j] = "" if pd.isna(val) else f"{val:+.1f}%"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=[f"{t}  {a}" for t, a in zip(data["ticker"], data["asset"])],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
            colorscale="RdYlGn",
            zmid=0,
            colorbar_title="%",
        )
    )
    fig.update_layout(height=max(560, 24 * len(data)), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def make_horizon_bar(features: pd.DataFrame, horizon_col: str, label: str) -> go.Figure:
    data = features.sort_values(horizon_col, ascending=True).copy()
    data["horizon_pct"] = data[horizon_col] * 100

    fig = px.bar(
        data,
        x="horizon_pct",
        y="ticker",
        color="sleeve",
        orientation="h",
        hover_data={
            "asset": True,
            "horizon_pct": ":.1f",
            "ret_1d": ":.2%",
            "ret_1w": ":.2%",
            "ret_1m": ":.2%",
            "ret_3m": ":.2%",
            "ret_ytd": ":.2%",
        },
        labels={"horizon_pct": f"{label} return (%)", "ticker": ""},
    )
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def make_tape_chart(close: pd.DataFrame, tickers: List[str], anchors: AnchorDates, lookback: int) -> go.Figure:
    tickers = [t for t in tickers if t in close.columns]
    data = close[tickers].dropna(how="all").tail(lookback)
    norm = data.div(data.iloc[0]).mul(100)

    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[ticker],
                mode="lines",
                name=ticker,
                line=dict(width=2),
                hovertemplate=f"{ticker}<br>%{{x|%b %d, %Y}}<br>Indexed: %{{y:.1f}}<extra></extra>",
            )
        )

    if anchors.last_friday in norm.index:
        fig.add_vline(x=anchors.last_friday, line_width=1.3, line_dash="dash")
        fig.add_annotation(
            x=anchors.last_friday,
            y=1.02,
            yref="paper",
            text=f"Friday {anchors.last_friday.strftime('%b %d')}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
        )

    if anchors.latest in norm.index:
        fig.add_vline(x=anchors.latest, line_width=1.3)
        fig.add_annotation(
            x=anchors.latest,
            y=1.02,
            yref="paper",
            text=f"Latest {anchors.latest.strftime('%b %d')}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
        )

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        yaxis_title="Indexed to 100",
        xaxis_title="",
        hovermode="x unified",
    )
    return fig


def make_regime_scatter(features: pd.DataFrame) -> go.Figure:
    data = features.copy()
    data["ret_1w_pct"] = data["ret_1w"] * 100
    data["dist_50_pct"] = data["dist_50"] * 100
    data["vol_1m_pct"] = data["vol_1m"] * 100

    fig = px.scatter(
        data,
        x="dist_50_pct",
        y="ret_1w_pct",
        color="sleeve",
        size="vol_1m_pct",
        hover_name="ticker",
        hover_data={
            "asset": True,
            "ret_1d": ":.2%",
            "ret_1w": ":.2%",
            "ret_1m": ":.2%",
            "ret_3m": ":.2%",
            "ret_ytd": ":.2%",
            "vol_1m_pct": False,
        },
        labels={"dist_50_pct": "Distance from 50D MA (%)", "ret_1w_pct": "1W return (%)"},
    )
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    return fig


with st.sidebar:
    st.markdown("## About This Tool")
    st.caption(
        "A visual cross-asset tape for weekly triage. The heatmap shows Today, 1W, 1M, 3M, and YTD. "
        "Today is left blank when a fresh current-session price is unavailable, instead of showing a fake flat move."
    )

    st.markdown("---")
    st.markdown("## Controls")
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)

    sleeves = sorted(set(v["sleeve"] for v in UNIVERSE.values()))
    selected_sleeves = st.multiselect("Sleeves", sleeves, default=sleeves)

    default_assets = [t for t, meta in UNIVERSE.items() if meta["sleeve"] in selected_sleeves]
    include = st.multiselect("Assets", list(UNIVERSE.keys()), default=default_assets)

    tape_focus = st.multiselect(
        "Tape chart focus",
        include,
        default=[t for t in CORE_TAPE if t in include],
    )

    lookback = st.slider("Tape lookback", min_value=60, max_value=252, value=126, step=21)
    bar_horizon_label = st.selectbox("Ranking bar horizon", ["Today", "1W", "1M", "3M", "YTD"], index=1)

    st.markdown("---")
    st.caption("Data source: Yahoo Finance via yfinance. Use this for triage, not official marks.")


st.title("Cross-Asset Tape")
st.caption("Today, 1W, 1M, 3M, and YTD across equities, rates, credit, FX, commodities, volatility, and crypto.")

if not include:
    st.warning("Select at least one asset.")
    st.stop()

with st.spinner("Loading data..."):
    daily_close = load_daily_prices(include, period)
    latest_prices, latest_ts = load_intraday_prices(include)
    close, data_status = merge_latest_prices(daily_close, latest_prices, latest_ts)

if close.empty:
    st.error("No price data returned.")
    st.stop()

features = build_feature_table(close)
if features.empty:
    st.error("Not enough data to build signals.")
    st.stop()

anchors = get_anchor_dates(close)
regime, note = regime_summary(features)

if latest_ts is None:
    st.warning("Intraday prices are unavailable. Today will be blank unless the daily bar is already current.")
else:
    stale_today = features.loc[~features["has_current_price"], "ticker"].tolist()
    st.caption(
        f"Latest intraday timestamp: {latest_ts.strftime('%b %d, %Y %H:%M')}. "
        f"Blank Today cells mean that ticker did not have a fresh current-session price."
    )
    if stale_today:
        st.caption("No current-session price for: " + ", ".join(stale_today))

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SPY Today", pct_fmt(get_value(features, "SPY", "ret_1d")), pct_fmt(get_value(features, "SPY", "ret_1w")))
m2.metric("QQQ Today", pct_fmt(get_value(features, "QQQ", "ret_1d")), pct_fmt(get_value(features, "QQQ", "ret_1w")))
m3.metric("TLT Today", pct_fmt(get_value(features, "TLT", "ret_1d")), pct_fmt(get_value(features, "TLT", "ret_1w")))
m4.metric("HY-IG 1W", pct_fmt(get_value(features, "HYG", "ret_1w") - get_value(features, "LQD", "ret_1w")))
m5.metric("Dollar 1W", pct_fmt(get_value(features, "UUP", "ret_1w")))
m6.metric("Oil 1W", pct_fmt(get_value(features, "USO", "ret_1w")))

st.markdown(f"### {regime}")
st.write(note)

left, right = st.columns([1.05, 0.95])
with left:
    st.plotly_chart(make_heatmap(features), use_container_width=True)
with right:
    horizon_map = {
        "Today": ("ret_1d", "Today"),
        "1W": ("ret_1w", "1W"),
        "1M": ("ret_1m", "1M"),
        "3M": ("ret_3m", "3M"),
        "YTD": ("ret_ytd", "YTD"),
    }
    horizon_col, horizon_label = horizon_map[bar_horizon_label]
    st.plotly_chart(make_horizon_bar(features, horizon_col, horizon_label), use_container_width=True)

st.markdown("## Tape")
st.plotly_chart(make_tape_chart(close, tape_focus, anchors, lookback), use_container_width=True)

st.markdown("## Regime Map")
st.plotly_chart(make_regime_scatter(features), use_container_width=True)
