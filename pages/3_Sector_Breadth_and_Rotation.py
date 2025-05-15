# pages/3_Sector_Breadth_and_Rotation.py

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---- Sector ETF Definitions ----
SECTOR_ETFS = {
    "S&P 500": "SPY",
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

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.caption("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

# ---- Sidebar ----
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
**Sector Relative Strength vs. S&P 500:**  
Compares a sector ETF's price performance relative to the S&P 500 (SPY) over the last 12 months.
""")

# ---- Select Sector ----
sector_list = list(SECTOR_ETFS.keys())
default_sector = sector_list[1]  # Technology by default
selected_sector = st.selectbox("Select sector to compare with S&P 500:", sector_list[1:], index=0)

# ---- Set Fixed 12-Month Window ----
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# ---- Download Data ----
def load_price(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if "Close" in df.columns:
        return df["Close"]
    return pd.Series(dtype="float64")

@st.cache_data(ttl=86400)
def fetch_12m_data():
    price_data = {}
    for name, ticker in SECTOR_ETFS.items():
        s = load_price(ticker)
        if not s.empty:
            price_data[name] = s
    if "S&P 500" not in price_data or selected_sector not in price_data:
        return pd.DataFrame()  # Defensive: avoid errors if missing
    # Join by date
    df = pd.DataFrame(price_data)
    df = df[["S&P 500", selected_sector]].dropna()
    return df

price_df = fetch_12m_data()
if price_df.empty:
    st.error("Failed to load sufficient data for this sector or SPY. Try again later.")
    st.stop()

# ---- Compute Sector Relative Strength (Sector / SPY) ----
sector_rel = price_df[selected_sector] / price_df["S&P 500"]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sector_rel.index, sector_rel, color="orange", linewidth=2.5, label=f"{selected_sector} / SPY")
ax.set_title(f"{selected_sector} Relative Strength vs. S&P 500 (Last 12 Months)", fontsize=16, weight="bold", loc="left")
ax.set_ylabel("Sector / SPY (Ratio)")
ax.set_xlabel("Date")
ax.grid(alpha=0.25, linestyle="--")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# ---- Optionally, display raw data ----
with st.expander("Show Relative Strength Data (Sector/SPY)"):
    df_disp = sector_rel.rename(f"{selected_sector} / SPY").to_frame()
    st.dataframe(df_disp, use_container_width=True)
