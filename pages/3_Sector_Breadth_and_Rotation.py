import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---- Sector ETF Mapping ----
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
st.caption("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
Sector Relative Strength vs. S&P 500: Ratio of each sector ETF to SPY.
Sector Rotation Quadrant: 1M vs 3M returns for all sectors.
Breadth Table: % above 20/50/100/200-day MAs.
""")

# ---- User selection ----
sector_list = list(SECTOR_ETFS.keys())
selected_sector = st.selectbox("Select sector to compare with S&P 500:", sector_list, index=0)

# ---- Fetch price data (12 months) ----
end_date = datetime.today()
start_date = end_date - timedelta(days=366)
tickers = list(SECTOR_ETFS.values()) + [SPY_TICKER]

@st.cache_data(ttl=3600)
def fetch_sector_prices():
    data, failed = {}, []
    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if "Close" in df and not df["Close"].empty:
                data[t] = df["Close"].dropna()
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
    if not data:
        return None, failed
    df = pd.DataFrame(data)
    df = df.dropna(axis=0, how='any')
    return df, failed

prices, failed = fetch_sector_prices()
if not prices or prices.empty or SPY_TICKER not in prices.columns:
    st.error("No sector or SPY data found. Check Yahoo Finance or your internet connection.")
    st.stop()
if failed:
    st.sidebar.warning("Missing data for: " + ", ".join(failed))

# ---- 1. Sector Relative Strength vs SPY ----
if SECTOR_ETFS[selected_sector] in prices.columns:
    rel = prices[SECTOR_ETFS[selected_sector]] / prices[SPY_TICKER]
    st.subheader(f"{selected_sector} Relative Strength vs SPY")
    fig1 = px.line(rel, labels={"value": "Ratio", "index": "Date"})
    fig1.update_traces(line_color="#d95f02")
    fig1.update_layout(showlegend=False, height=340, margin=dict(l=30, r=30, t=40, b=30))
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning(f"No data for {selected_sector}")

# ---- 2. Sector Rotation Quadrant (1M vs 3M returns) ----
try:
    returns_1m = prices.pct_change(21).iloc[-1]
    returns_3m = prices.pct_change(63).iloc[-1]
    rotation_df = pd.DataFrame({
        "Sector": [k for k,v in SECTOR_ETFS.items() if v in prices.columns],
        "1M Return": [returns_1m[v] for v in SECTOR_ETFS.values() if v in prices.columns],
        "3M Return": [returns_3m[v] for v in SECTOR_ETFS.values() if v in prices.columns],
    })
    fig2 = px.scatter(rotation_df, x="1M Return", y="3M Return", text="Sector",
                      color=(rotation_df["Sector"] == selected_sector),
                      color_discrete_map={True: "#d95f02", False: "#7570b3"},
                      labels={"1M Return":"1M", "3M Return":"3M"},
                      height=340)
    fig2.update_layout(showlegend=False, margin=dict(l=30, r=30, t=40, b=30),
                      title="Sector Rotation Quadrant: 1M vs 3M Returns")
    st.subheader("Sector Rotation Quadrant (1M vs 3M Returns)")
    st.plotly_chart(fig2, use_container_width=True)
except Exception:
    st.warning("Could not compute sector rotation quadrant.")

# ---- 3. Breadth Table ----
try:
    def is_above(s, d): return "✅" if s.iloc[-1] > s.rolling(d).mean().iloc[-1] else ""
    breadth = {}
    for k, v in SECTOR_ETFS.items():
        if v in prices.columns:
            s = prices[v]
            breadth[k] = {
                "Above 20D": is_above(s, 20),
                "Above 50D": is_above(s, 50),
                "Above 100D": is_above(s, 100),
                "Above 200D": is_above(s, 200),
            }
    breadth_df = pd.DataFrame(breadth).T
    st.subheader("% Above 20, 50, 100, 200-Day Moving Averages")
    st.dataframe(breadth_df, use_container_width=True)
except Exception:
    st.warning("Could not compute breadth table.")

st.caption("© 2025 AD Fund Management LP")
