import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---- Constants
SECTOR_ETFS = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
SPY_TICKER = "SPY"

st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")

# ---- UI: Sector Selection
sector_names = list(SECTOR_ETFS.keys())
selected_sector = st.selectbox("Select sector to compare with S&P 500:", sector_names)

# ---- Download all at once
tickers = list(SECTOR_ETFS.values()) + [SPY_TICKER]
start = (datetime.today() - timedelta(days=370)).strftime("%Y-%m-%d")
end = datetime.today().strftime("%Y-%m-%d")
data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

# Clean up: only use tickers with full data, drop rows with any NaN
data = data.dropna(axis=1, how='any').dropna(axis=0, how='any')

# ---- Require at least one sector + SPY
if SPY_TICKER not in data.columns or len(data.columns) < 2:
    st.error("Not enough data. Check your internet or try later.")
    st.stop()

# ---- Sector Relative Strength vs SPY
sector_ticker = SECTOR_ETFS[selected_sector]
if sector_ticker in data.columns:
    rel = data[sector_ticker] / data[SPY_TICKER]
    st.subheader(f"{selected_sector} / S&P 500 Relative Strength")
    st.plotly_chart(px.line(rel, labels={"value": "Ratio", "index": "Date"}), use_container_width=True)

# ---- Sector Rotation Quadrant (1M vs 3M returns)
try:
    returns_1m = data.pct_change(21).iloc[-1]
    returns_3m = data.pct_change(63).iloc[-1]
    quadrant = pd.DataFrame({
        "1M": returns_1m,
        "3M": returns_3m
    }).drop(index=SPY_TICKER)
    quadrant["Sector"] = [k for k,v in SECTOR_ETFS.items() if v in quadrant.index]
    fig = px.scatter(quadrant, x="1M", y="3M", text="Sector",
                     labels={"1M":"1M Return", "3M":"3M Return"}, height=360)
    fig.update_traces(marker=dict(size=16, color="#1f77b4"))
    st.subheader("Sector Rotation Quadrant (1M vs 3M Returns)")
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.warning("Sector rotation chart unavailable.")

# ---- Breadth Table: % Above 20/50/100/200 DMA
try:
    def above_ma(s, d): return s.iloc[-1] > s.rolling(d).mean().iloc[-1]
    rows = []
    for sector, ticker in SECTOR_ETFS.items():
        if ticker in data.columns:
            s = data[ticker]
            rows.append({
                "Sector": sector,
                "Above 20D": above_ma(s, 20),
                "Above 50D": above_ma(s, 50),
                "Above 100D": above_ma(s, 100),
                "Above 200D": above_ma(s, 200)
            })
    breadth = pd.DataFrame(rows).set_index("Sector")
    st.subheader("Breadth: Above Moving Averages")
    st.dataframe(breadth.replace({True: "✅", False: ""}), use_container_width=True)
except Exception:
    st.warning("Breadth table unavailable.")

st.caption("© 2025 AD Fund Management LP")
