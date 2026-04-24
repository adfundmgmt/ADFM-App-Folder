from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TITLE = "Consensus Regime Score"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.caption("Composite 0-100 market regime score across trend, vol, credit, and breadth.")

SECTOR_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
TICKERS = ["SPY", "^VIX", "HYG", "LQD"] + SECTOR_TICKERS


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
        TICKERS,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    out = pd.DataFrame(index=pd.to_datetime(raw.index).tz_localize(None))
    for t in TICKERS:
        out[t] = _to_series(raw, t)
    return out.dropna(how="all")


def minmax_score(s: pd.Series, window: int = 252, invert: bool = False) -> pd.Series:
    roll_min = s.rolling(window).min()
    roll_max = s.rolling(window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    score = 100 * (s - roll_min) / denom
    if invert:
        score = 100 - score
    return score.clip(0, 100)


def pct_above_ma(df: pd.DataFrame, tickers: list[str], window: int = 50) -> pd.Series:
    bits = []
    for t in tickers:
        if t not in df.columns:
            continue
        s = df[t].dropna()
        if s.empty:
            continue
        bits.append((s > s.rolling(window).mean()).rename(t))

    if not bits:
        return pd.Series(dtype=float)

    panel = pd.concat(bits, axis=1)
    return panel.mean(axis=1) * 100


with st.sidebar:
    st.header("Score Settings")
    period = st.selectbox("History", ["2y", "3y", "5y", "10y"], index=1)
    trend_window = st.slider("Trend MA", 50, 250, 200)
    breadth_window = st.slider("Breadth MA", 20, 200, 50)

    st.subheader("Component Weights")
    w_trend = st.slider("Trend", 0.0, 1.0, 0.35, 0.05)
    w_vol = st.slider("Volatility", 0.0, 1.0, 0.25, 0.05)
    w_credit = st.slider("Credit", 0.0, 1.0, 0.20, 0.05)
    w_breadth = st.slider("Breadth", 0.0, 1.0, 0.20, 0.05)

    risk_on_cutoff = st.slider("Risk-on threshold", 50, 90, 65)
    risk_off_cutoff = st.slider("Risk-off threshold", 10, 50, 40)

weights = np.array([w_trend, w_vol, w_credit, w_breadth], dtype=float)
if weights.sum() == 0:
    weights = np.array([0.35, 0.25, 0.20, 0.20])
weights = weights / weights.sum()

prices = load_prices(period)
if prices.empty or prices["SPY"].dropna().empty:
    st.error("Unable to load sufficient market data right now.")
    st.stop()

spy = prices["SPY"].dropna()
vix = prices["^VIX"].reindex(spy.index).ffill()
credit_ratio = (prices["HYG"] / prices["LQD"]).reindex(spy.index)
breadth = pct_above_ma(prices, SECTOR_TICKERS, breadth_window).reindex(spy.index)

trend_raw = 100 * (spy / spy.rolling(trend_window).mean() - 1)
trend_score = minmax_score(trend_raw, window=252)
vol_score = minmax_score(vix, window=252, invert=True)
credit_score = minmax_score(credit_ratio, window=252)
breadth_score = breadth.clip(0, 100)

components = pd.DataFrame(
    {
        "Trend": trend_score,
        "Volatility": vol_score,
        "Credit": credit_score,
        "Breadth": breadth_score,
    }
).dropna()

composite = (
    components["Trend"] * weights[0]
    + components["Volatility"] * weights[1]
    + components["Credit"] * weights[2]
    + components["Breadth"] * weights[3]
)

regime = pd.Series("Neutral", index=composite.index)
regime[composite >= risk_on_cutoff] = "Risk-On"
regime[composite <= risk_off_cutoff] = "Risk-Off"

latest_date = composite.index.max()
latest_score = float(composite.loc[latest_date])
latest_regime = regime.loc[latest_date]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Date", str(latest_date.date()))
col2.metric("Composite Score", f"{latest_score:.1f}")
col3.metric("Regime", latest_regime)
col4.metric("Breadth", f"{components.loc[latest_date, 'Breadth']:.1f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=composite.index, y=composite, name="Composite", line=dict(width=3)))
fig.add_hline(y=risk_on_cutoff, line_dash="dash", annotation_text="Risk-On threshold")
fig.add_hline(y=risk_off_cutoff, line_dash="dash", annotation_text="Risk-Off threshold")
fig.update_layout(height=380, yaxis_title="Score (0-100)", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Component Snapshot")
latest_components = components.loc[latest_date].rename("Score").to_frame()
latest_components["Weight"] = weights
st.dataframe(latest_components.style.format({"Score": "{:.1f}", "Weight": "{:.2f}"}), use_container_width=True)

st.subheader("Recent Regime History")
recent = pd.DataFrame({"Composite": composite, "Regime": regime}).tail(30)
st.dataframe(recent, use_container_width=True)

with st.expander("Method"):
    st.markdown(
        """
        - **Trend:** SPY distance from its moving average, scaled to 0-100 over a rolling 1-year range.
        - **Volatility:** Inverse VIX score (high VIX lowers score).
        - **Credit:** HYG/LQD relative strength scaled 0-100.
        - **Breadth:** % of major sector ETFs above selected moving average.
        - Composite score is a weighted average of the four components.
        """
    )

st.caption(f"Data source: Yahoo Finance | Observations in composite: {len(composite)}")
