import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import StringIO
import requests
from datetime import datetime, timedelta

# --- Dashboard Config ---
st.set_page_config(page_title="ETF & Factor Flows Dashboard", layout="wide")
st.title("🔀 ETF & Factor Flows Dashboard")

st.sidebar.header("About This Tool")
st.sidebar.info("""
Monitor daily fund flows, crowding, and price/factor trends across ETFs, sectors, and factor proxies.

- **Data:** ETF.com (flows), Yahoo Finance (price)
- **Crowding:** Highlight top/bottom 5% flows as crowding/extreme events
- **Compare:** Select multiple ETFs or factor proxies
""")

# --- ETF List (can be expanded/modified) ---
FACTOR_ETFS = [
    # Index/factor proxies
    "SPY", "QQQ", "IWM", "DIA",        # US indices
    "MTUM", "VLUE", "QUAL", "USMV",    # Factors: Momentum, Value, Quality, MinVol
    "VTV", "VUG",                      # Value, Growth
    "XLF", "XLK", "XLY", "XLE", "XLV", "XLP", "XLI", "XLB", "XLU", "XLRE", # Sectors
    "ARKK", "TLT", "LQD", "HYG", "GLD" # Thematic/bond/commodity
]

# --- User Selection ---
etfs = st.sidebar.multiselect("Select ETFs/factor proxies:", FACTOR_ETFS, default=["SPY", "QQQ", "IWM"])
lookback = st.sidebar.selectbox("Flow lookback (days)", [5, 10, 21, 63], index=1)  # default: 10 days

# --- Fetch ETF Flows (Simulated by ETFDB daily flows CSV) ---
@st.cache_data(ttl=12*3600)
def fetch_etf_flows():
    url = "https://etfdb.com/data-tools/flows/"
    csv_url = "https://etfdb.com/data-tools/flows/?download=csv"
    try:
        r = requests.get(csv_url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"Failed to fetch ETF flows: {e}")
        return pd.DataFrame()

flows = fetch_etf_flows()
if flows.empty:
    st.stop()

# --- Filter for Selected ETFs ---
latest_date = flows["Date"].max()
date_cut = latest_date - timedelta(days=60)
df = flows[flows["Date"] >= date_cut]
df = df[df["Ticker"].isin(etfs)]

# --- Price Data from Yahoo Finance ---
@st.cache_data(ttl=6*3600)
def fetch_prices(tickers, start):
    data = yf.download(list(set(tickers)), start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

price_data = fetch_prices(etfs, start=(latest_date - timedelta(days=90)).strftime("%Y-%m-%d"))

# --- Main Panels ---
st.subheader(f"Latest ETF Flows & Price Chart (Last 60 Days)")
col1, col2 = st.columns([3, 1])

with col1:
    # Plot price + flow chart for each ETF
    for etf in etfs:
        df_etf = df[df["Ticker"] == etf].sort_values("Date")
        if df_etf.empty or etf not in price_data.columns:
            continue

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.plot(price_data.index, price_data[etf], color='black', lw=2, label="Price")
        ax1.set_ylabel(f"{etf} Price", color="black", weight="bold")
        ax2 = ax1.twinx()
        # Bar: daily flows (inflow = green, outflow = red)
        inflow = df_etf["Net Flows ($, mm)"].apply(lambda x: x if x > 0 else np.nan)
        outflow = df_etf["Net Flows ($, mm)"].apply(lambda x: x if x < 0 else np.nan)
        ax2.bar(df_etf["Date"], inflow, color="mediumseagreen", alpha=0.55, width=1.8, label="Inflow")
        ax2.bar(df_etf["Date"], outflow, color="indianred", alpha=0.55, width=1.8, label="Outflow")
        ax2.set_ylabel("Net Flows ($, mm)", color="gray")
        ax2.axhline(0, ls="--", color="gray", lw=0.7)
        # Crowding lines (top/bottom 5%)
        crowding = df_etf["Net Flows ($, mm)"]
        p95 = np.nanpercentile(crowding, 95)
        p5  = np.nanpercentile(crowding, 5)
        ax2.axhline(p95, ls=":", color="green", lw=1.1, label="95th pct")
        ax2.axhline(p5,  ls=":", color="red", lw=1.1, label="5th pct")
        # Title
        ax1.set_title(f"{etf} — Price & Fund Flows (Last 60 Days)", fontsize=15, weight="bold", pad=8)
        fig.tight_layout()
        ax1.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)
        st.pyplot(fig)
        st.caption(f"**{etf}**: Top 5% flow = ${p95:,.0f}mm; Bottom 5% flow = ${p5:,.0f}mm")

with col2:
    st.markdown("#### Recent Flow Heatmap")
    heatmap = df.pivot(index="Date", columns="Ticker", values="Net Flows ($, mm)")
    st.dataframe(heatmap.tail(20).style.background_gradient(axis=0, cmap="RdYlGn"), use_container_width=True)
    st.markdown("Top 5% inflow = crowding / institutional chasing\n\nBottom 5% = capitulation/outflow risk.")

# --- Analytics/Warnings ---
st.subheader("Crowding Alerts (Last 2 Weeks)")
recent = df[df["Date"] >= (latest_date - timedelta(days=14))]
alerts = []
for etf in etfs:
    ser = recent[recent["Ticker"] == etf]["Net Flows ($, mm)"]
    if len(ser) < 5:
        continue
    p95 = np.nanpercentile(ser, 95)
    p5  = np.nanpercentile(ser, 5)
    if ser.iloc[-1] >= p95:
        alerts.append(f"🟢 **{etf}**: Latest daily inflow in the top 5% ({ser.iloc[-1]:,.1f}mm) — crowded long.")
    elif ser.iloc[-1] <= p5:
        alerts.append(f"🔴 **{etf}**: Latest daily outflow in bottom 5% ({ser.iloc[-1]:,.1f}mm) — capitulation risk.")
if alerts:
    st.warning("\n".join(alerts))
else:
    st.info("No extreme flow/crowding signals in the last 2 weeks.")

st.caption("Data: ETFDB, Yahoo Finance. For institutional-grade flow data, connect to Bloomberg, FactSet, or ETFLogic APIs.")

