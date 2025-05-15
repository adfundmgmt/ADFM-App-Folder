import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---- Sector ETF Definitions ----
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

# ---- Sidebar ----
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
Compares a selected sector ETF's price performance relative to the S&P 500 (SPY) over the last 12 months.
""")

# ---- Select Sector ----
sector_list = list(SECTOR_ETFS.keys())
selected_sector = st.selectbox("Select sector to compare with S&P 500:", sector_list, index=0)

# ---- Fetch Data ----
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data(ttl=86400)
def load_close_series(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    return df["Close"] if "Close" in df.columns and not df["Close"].empty else None

sector_close = load_close_series(SECTOR_ETFS[selected_sector])
spy_close = load_close_series(SPY_TICKER)

if sector_close is None:
    st.error(f"Could not load price data for {selected_sector}.")
    st.stop()
if spy_close is None:
    st.error("Could not load price data for SPY.")
    st.stop()

# ---- Align data ----
aligned = pd.concat([sector_close, spy_close], axis=1, join='inner')
aligned.columns = [selected_sector, "SPY"]
if aligned.empty:
    st.error("No overlapping data between sector and SPY.")
    st.stop()

# ---- Compute and Plot Relative Strength ----
rel_strength = aligned[selected_sector] / aligned["SPY"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rel_strength.index, rel_strength, color="orange", linewidth=2.5, label=f"{selected_sector} / SPY")
ax.set_title(f"{selected_sector} Relative Strength vs. S&P 500 (Last 12 Months)", fontsize=16, weight="bold", loc="left")
ax.set_ylabel("Sector / SPY (Ratio)")
ax.set_xlabel("Date")
ax.grid(alpha=0.25, linestyle="--")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# ---- Optional: Show Raw Data ----
with st.expander("Show Relative Strength Data (Sector/SPY)"):
    st.dataframe(rel_strength.rename(f"{selected_sector}/SPY").to_frame(), use_container_width=True)
