from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TITLE = "Alert Engine"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.caption("Multi-signal alert monitor for fast risk-on / risk-off reads.")

RISK_SECTOR_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
CORE_TICKERS = ["SPY", "^VIX", "HYG", "LQD", "XLY", "XLP"] + RISK_SECTOR_TICKERS


def _to_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            s = pd.Series(dtype=float)
    else:
        s = df.get(ticker, pd.Series(dtype=float))

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(period: str) -> pd.DataFrame:
    raw = yf.download(
        CORE_TICKERS,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    out = pd.DataFrame(index=pd.to_datetime(raw.index).tz_localize(None))
    for t in CORE_TICKERS:
        out[t] = _to_series(raw, t)
    return out.dropna(how="all")


def pct_above_ma(df: pd.DataFrame, tickers: list[str], window: int) -> pd.Series:
    valid = []
    for t in tickers:
        if t not in df.columns:
            continue
        s = df[t].dropna()
        if s.empty:
            continue
        ma = s.rolling(window).mean()
        valid.append((s > ma).rename(t))

    if not valid:
        return pd.Series(dtype=float)

    panel = pd.concat(valid, axis=1)
    return panel.mean(axis=1) * 100


def zscore(s: pd.Series, window: int = 126) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd


with st.sidebar:
    st.header("Alert Settings")
    period = st.selectbox("History", ["1y", "2y", "3y", "5y"], index=2)
    rv_window = st.slider("Realized vol window", 10, 63, 21)
    breadth_ma = st.slider("Breadth MA window", 20, 200, 50)
    vix_high = st.slider("VIX high threshold", 15, 45, 25)
    breadth_low = st.slider("Breadth low threshold (%)", 20, 80, 45)
    min_triggers = st.slider("Min triggers for alert", 1, 4, 2)

prices = load_prices(period)

if prices.empty or prices["SPY"].dropna().empty:
    st.error("Unable to load market data right now. Please refresh and try again.")
    st.stop()

spy = prices["SPY"].dropna()
vix = prices["^VIX"].reindex(spy.index).ffill()
hyg_lqd = (prices["HYG"] / prices["LQD"]).reindex(spy.index)
xly_xlp = (prices["XLY"] / prices["XLP"]).reindex(spy.index)

returns = spy.pct_change()
realized_vol = returns.rolling(rv_window).std() * np.sqrt(252) * 100
vol_z = zscore(realized_vol)

breadth = pct_above_ma(prices, RISK_SECTOR_TICKERS, breadth_ma).reindex(spy.index)

credit_trend = hyg_lqd / hyg_lqd.rolling(50).mean() - 1
cyc_def_trend = xly_xlp / xly_xlp.rolling(50).mean() - 1

signals = pd.DataFrame(index=spy.index)
signals["VIX High"] = (vix > vix_high).astype(int)
signals["Breadth Weak"] = (breadth < breadth_low).astype(int)
signals["Credit Weak"] = (credit_trend < 0).astype(int)
signals["Cyclical Weak"] = (cyc_def_trend < 0).astype(int)
signals["Triggered Count"] = signals.sum(axis=1)
signals["Alert"] = (signals["Triggered Count"] >= min_triggers).astype(int)

latest_date = signals.dropna().index.max()
latest = signals.loc[latest_date]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Date", str(latest_date.date()))
col2.metric("Triggers", int(latest["Triggered Count"]))
col3.metric("Alert State", "ON" if latest["Alert"] == 1 else "OFF")
col4.metric("Breadth % > MA", f"{breadth.loc[latest_date]:.1f}%")

st.subheader("Current Trigger Board")
board = pd.DataFrame(
    {
        "Signal": ["VIX High", "Breadth Weak", "Credit Weak", "Cyclical Weak"],
        "State": [
            "ON" if latest["VIX High"] else "OFF",
            "ON" if latest["Breadth Weak"] else "OFF",
            "ON" if latest["Credit Weak"] else "OFF",
            "ON" if latest["Cyclical Weak"] else "OFF",
        ],
    }
)
st.dataframe(board, use_container_width=True, hide_index=True)

st.subheader("Alert History")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=signals.index,
        y=signals["Triggered Count"],
        name="Triggered Count",
        mode="lines",
        line=dict(width=2),
    )
)
fig.add_hline(y=min_triggers, line_dash="dash", annotation_text="Alert threshold")
fig.update_layout(height=360, yaxis_title="# of Triggers", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)

show_cols = ["VIX High", "Breadth Weak", "Credit Weak", "Cyclical Weak", "Triggered Count", "Alert"]
st.dataframe(signals[show_cols].tail(30), use_container_width=True)

with st.expander("Method"):
    st.markdown(
        """
        - **VIX High:** `^VIX` above user threshold.
        - **Breadth Weak:** % of sector ETFs above moving average falls below threshold.
        - **Credit Weak:** `HYG/LQD` ratio below its 50-day average.
        - **Cyclical Weak:** `XLY/XLP` ratio below its 50-day average.
        - **Alert ON:** number of active triggers >= required minimum.
        """
    )

st.caption(f"Data source: Yahoo Finance | Observations: {len(signals.dropna())}")
