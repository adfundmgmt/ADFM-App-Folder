import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Sentiment & Positioning Composite", layout="wide")
st.title("Sentiment & Positioning Composite")
st.caption("A market-based sentiment gauge built from risk-appetite and defensive positioning proxies.")

TICKERS = ["SPY", "QQQ", "IWM", "XLU", "HYG", "LQD", "^VIX"]

with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y"], index=1)
    z_window = st.slider("Z-score window", 21, 252, 126, step=21)


@st.cache_data(ttl=3600)
def fetch_prices(period: str) -> pd.DataFrame:
    raw = yf.download(TICKERS, period=period, interval="1d", progress=False, auto_adjust=True, group_by="column")
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": TICKERS[0]})
    return close.sort_index().ffill().dropna(how="all")


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


prices = fetch_prices(lookback)
required = [t for t in TICKERS if t in prices.columns]
prices = prices[required]

if prices.empty or len(required) < 6:
    st.error("Insufficient data to build sentiment composite.")
    st.stop()

signals = pd.DataFrame(index=prices.index)
signals["Beta Appetite (IWM/SPY)"] = zscore(prices["IWM"] / prices["SPY"], z_window)
signals["Growth vs Defensives (QQQ/XLU)"] = zscore(prices["QQQ"] / prices["XLU"], z_window)
signals["Credit Risk Appetite (HYG/LQD)"] = zscore(prices["HYG"] / prices["LQD"], z_window)
signals["Trend Extension (SPY vs 200D)"] = zscore(prices["SPY"] / prices["SPY"].rolling(200).mean() - 1, z_window)
signals["Volatility Aversion (inverse VIX)"] = zscore(-prices["^VIX"], z_window)

signals = signals.dropna()
composite = signals.mean(axis=1)

latest = composite.iloc[-1]
regime = "Risk-On" if latest > 0.5 else "Neutral" if latest > -0.5 else "Risk-Off"

c1, c2, c3 = st.columns(3)
c1.metric("Composite Score", f"{latest:.2f} z")
c2.metric("Regime", regime)
c3.metric("1M Change", f"{(latest - composite.iloc[-22]):.2f} z" if len(composite) > 22 else "n/a")

fig = go.Figure()
fig.add_trace(go.Scatter(x=composite.index, y=composite, name="Sentiment Composite", line=dict(width=2, color="#2563eb")))
fig.add_hline(y=0.5, line_dash="dash", line_color="#16a34a")
fig.add_hline(y=-0.5, line_dash="dash", line_color="#dc2626")
fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
fig.update_layout(title="Sentiment & Positioning Composite (z-score)", yaxis_title="z-score", height=420)
st.plotly_chart(fig, use_container_width=True)

latest_signals = signals.iloc[-1].sort_values(ascending=False).round(2)
fig_bar = go.Figure(go.Bar(x=latest_signals.values, y=latest_signals.index, orientation="h"))
fig_bar.update_layout(title="Current Sub-Signal Readings", xaxis_title="z-score", height=380)
st.plotly_chart(fig_bar, use_container_width=True)

st.info(
    "Interpretation guide: sustained readings above +0.5 usually indicate broad risk appetite and supportive positioning. "
    "Below -0.5 suggests defensive behavior, tighter risk budgets, and elevated downside sensitivity."
)
