import warnings
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Market Internals Breadth Monitor", layout="wide")
st.title("Market Internals Breadth Monitor")
st.caption("Breadth, participation, and concentration diagnostics to validate headline index strength.")

UNIVERSE = ["SPY", "RSP", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLY", "XLP", "XLI", "XLV", "XLE", "XLU", "XLB", "XLC", "XLRE"]

with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y"], index=1)
    short_ma = st.selectbox("Fast MA", [20, 50, 100], index=1)
    long_ma = st.selectbox("Slow MA", [100, 150, 200], index=2)


@st.cache_data(ttl=3600)
def fetch_prices(tickers: list[str], period: str) -> pd.DataFrame:
    data = yf.download(tickers=tickers, period=period, interval="1d", progress=False, auto_adjust=True, group_by="column")
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})

    closes = closes.sort_index().ffill().dropna(how="all")
    return closes


prices = fetch_prices(UNIVERSE, lookback)
if prices.empty:
    st.error("No price data available right now.")
    st.stop()

valid_tickers = [t for t in UNIVERSE if t in prices.columns]
prices = prices[valid_tickers].dropna(how="all")

above_short = prices.gt(prices.rolling(short_ma).mean())
above_long = prices.gt(prices.rolling(long_ma).mean())

breadth_short = (above_short.sum(axis=1) / len(valid_tickers) * 100).dropna()
breadth_long = (above_long.sum(axis=1) / len(valid_tickers) * 100).dropna()

rolling_high = prices.rolling(63).max()
rolling_low = prices.rolling(63).min()
new_highs = (prices >= rolling_high).sum(axis=1)
new_lows = (prices <= rolling_low).sum(axis=1)

spy_rsp_ratio = (prices["SPY"] / prices["RSP"]).dropna() if {"SPY", "RSP"}.issubset(prices.columns) else pd.Series(dtype=float)
qqq_spy_ratio = (prices["QQQ"] / prices["SPY"]).dropna() if {"QQQ", "SPY"}.issubset(prices.columns) else pd.Series(dtype=float)

today = prices.index.max()

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"% Above {short_ma}D", f"{breadth_short.iloc[-1]:.1f}%")
c2.metric(f"% Above {long_ma}D", f"{breadth_long.iloc[-1]:.1f}%")
c3.metric("63D New High Count", f"{int(new_highs.loc[today])}")
c4.metric("63D New Low Count", f"{int(new_lows.loc[today])}")

fig_breadth = go.Figure()
fig_breadth.add_trace(go.Scatter(x=breadth_short.index, y=breadth_short, name=f"% > {short_ma}D", line=dict(width=2)))
fig_breadth.add_trace(go.Scatter(x=breadth_long.index, y=breadth_long, name=f"% > {long_ma}D", line=dict(width=2)))
fig_breadth.update_layout(title="Participation Breadth", yaxis_title="Percent of Universe", height=420)
st.plotly_chart(fig_breadth, use_container_width=True)

fig_nh_nl = go.Figure()
fig_nh_nl.add_trace(go.Bar(x=new_highs.index, y=new_highs, name="New Highs (63D)"))
fig_nh_nl.add_trace(go.Bar(x=new_lows.index, y=new_lows, name="New Lows (63D)"))
fig_nh_nl.update_layout(title="Expansion vs Deterioration", barmode="group", height=360)
st.plotly_chart(fig_nh_nl, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    if not spy_rsp_ratio.empty:
        fig_ratio_1 = go.Figure(go.Scatter(x=spy_rsp_ratio.index, y=spy_rsp_ratio, name="SPY/RSP", line=dict(color="#2563eb", width=2)))
        fig_ratio_1.update_layout(title="Cap-Weight vs Equal-Weight (SPY/RSP)", height=320)
        st.plotly_chart(fig_ratio_1, use_container_width=True)

with col_b:
    if not qqq_spy_ratio.empty:
        fig_ratio_2 = go.Figure(go.Scatter(x=qqq_spy_ratio.index, y=qqq_spy_ratio, name="QQQ/SPY", line=dict(color="#7c3aed", width=2)))
        fig_ratio_2.update_layout(title="Concentration Proxy (QQQ/SPY)", height=320)
        st.plotly_chart(fig_ratio_2, use_container_width=True)

st.info(
    "Interpretation guide: broad participation + rising highs + stable SPY/RSP usually confirms healthy risk-on tapes. "
    "If headline indexes rise while breadth falls and SPY/RSP surges, concentration risk is increasing."
)
