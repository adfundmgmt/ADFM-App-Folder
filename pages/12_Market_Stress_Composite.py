# market_stress_composite_chart.py
# ADFM Analytics Platform — Market Stress Composite
# Visual replication of CNN Fear & Greed chart in MacroMicro style.

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ------------------------------- Page Config -------------------------------
st.set_page_config(page_title="ADFM | Market Stress Composite", layout="wide")
plt.style.use("default")

# ------------------------------- Title -------------------------------------
st.title("Market Stress Composite vs S&P 500")

# ------------------------------- Load Data ---------------------------------
@st.cache_data
def load_data():
    spx = yf.download("^GSPC", start="2020-01-01")
    vix = yf.download("^VIX", start="2020-01-01")
    hy = yf.download("HYG", start="2020-01-01")  # proxy for high yield stress
    move = yf.download("^MOVE", start="2020-01-01")  # bond volatility
    return spx, vix, hy, move

spx, vix, hy, move = load_data()

# ------------------------------- Compute Composite -------------------------
df = pd.DataFrame(index=spx.index)
df["SPX"] = spx["Adj Close"]
df["VIX"] = vix["Adj Close"]
df["MOVE"] = move["Adj Close"]
df["HY"] = hy["Adj Close"]

# Normalize each factor 0-100 (percentile rank)
df = df.fillna(method="ffill")
for col in ["VIX", "MOVE", "HY"]:
    df[col + "_p"] = df[col].rank(pct=True) * 100

# Equal-weight composite
df["Market_Stress"] = df[["VIX_p", "MOVE_p", "HY_p"]].mean(axis=1)

# ------------------------------- Plot --------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(df.index, df["Market_Stress"], color="#4A90E2", label="Market Stress Composite", linewidth=1.8)
ax2.plot(df.index, df["SPX"], color="#D35400", label="S&P 500", linewidth=1.6)

# Aesthetics
ax1.set_ylabel("Composite Index (0–100)", color="#4A90E2", fontsize=10)
ax2.set_ylabel("S&P 500", color="#D35400", fontsize=10)
ax1.set_xlabel("Date", fontsize=9)
ax1.set_title("Market Stress Composite vs S&P 500", fontsize=14, fontweight="bold", pad=15)

ax1.grid(alpha=0.3, linestyle="--")
ax1.set_ylim(0, 100)
ax1.axhline(20, color="gray", linestyle="--", alpha=0.4)
ax1.axhline(80, color="gray", linestyle="--", alpha=0.4)

# Date formatting
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False, fontsize=9)

# Display
st.pyplot(fig)

# ------------------------------- Sidebar -----------------------------------
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
Tracks systemic market stress across volatility, credit, and rates.
**Composite = Equal weight of normalized VIX, MOVE, and HY stress factors.**

**Interpretation:**
- Rising composite = tightening financial conditions
- Falling composite = improving liquidity sentiment
- Horizontal bands at 20 (Greed) and 80 (Fear)
""")
