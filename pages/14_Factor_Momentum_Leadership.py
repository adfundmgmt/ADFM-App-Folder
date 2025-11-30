import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ---------------- Config ----------------
st.set_page_config(page_title="Factor Momentum and Leadership", layout="wide")
plt.style.use("default")
sns.set_style("whitegrid")

PASTELS = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

TEXT = "#222222"
GRID = "#e6e6e6"

# ---------------- Helpers ----------------
def load_prices(tickers, start):
    data = yf.download(tickers, start=start, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Adj Close"]
    return data.dropna(how="all")

def pct_change(series, days):
    return (series.iloc[-1] / series.iloc[-days] - 1.0) if len(series) > days else np.nan

def momentum(series, win=20):
    r = series.pct_change().dropna()
    return r.rolling(win).mean().iloc[-1] if len(r) >= win else np.nan

def zscore(x):
    x = pd.Series(x).dropna()
    if x.std() == 0 or len(x) < 5:
        return x * np.nan
    return (x - x.mean()) / x.std()

def rs(series_a, series_b):
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    return aligned.iloc[:, 0] / aligned.iloc[:, 1] if not aligned.empty else pd.Series(dtype=float)

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def trend_class(series):
    """Uses EMA 10, 20, 40 to classify trend."""
    if len(series) < 50:
        return "Neutral"
    e1 = ema(series, 10).iloc[-1]
    e2 = ema(series, 20).iloc[-1]
    e3 = ema(series, 40).iloc[-1]
    if e1 > e2 > e3:
        return "Up"
    if e1 < e2 < e3:
        return "Down"
    return "Neutral"

def inflection(short_mom, long_mom):
    if pd.isna(short_mom) or pd.isna(long_mom):
        return "Neutral"
    if short_mom > 0 and long_mom < 0:
        return "Turning Up"
    if short_mom < 0 and long_mom > 0:
        return "Turning Down"
    if abs(short_mom) > abs(long_mom):
        return "Strengthening"
    return "Weakening"

# ---------------- Factors ----------------
FACTOR_ETFS = {
    "Growth vs Value": ("VUG", "VTV"),
    "Quality vs Junk": ("QUAL", "JNK"),    
    "High Beta vs Low Vol": ("SPHB", "SPLV"),
    "Small vs Large": ("IWM", "SPY"),
    "Tech vs Broad": ("XLK", "SPY"),
    "Cyclicals vs Defensives": ("XLY", "XLP"),
    "US vs World": ("SPY", "VEA"),
    "Momentum": ("MTUM", None),
    "Equal Weight vs Cap": ("RSP", "SPY"),
}

ALL_TICKERS = sorted(list({t for pair in FACTOR_ETFS.values() for t in pair if t is not None}))

# ---------------- Sidebar ----------------
st.title("Factor Momentum and Leadership Dashboard")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("History start", datetime(2015, 1, 1))
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)
    st.caption("Data from Yahoo Finance. Factors constructed as relative return ratios.")

# ---------------- Load data ----------------
prices = load_prices(ALL_TICKERS, start=str(start_date))
if prices.empty:
    st.error("No data returned.")
    st.stop()

# ---------------- Compute factors ----------------
factor_levels = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        if up in prices:
            factor_levels[name] = prices[up]
        continue
    if up in prices and down in prices:
        factor_levels[name] = rs(prices[up], prices[down])

factor_df = pd.DataFrame(factor_levels).dropna(how="all")

# ---------------- Momentum snapshots ----------------
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < lookback_long + 5:
        continue
    r5 = pct_change(s, 5)
    r20 = pct_change(s, lookback_short)
    r60 = pct_change(s, lookback_long)
    mom = momentum(s, win=lookback_short)
    tclass = trend_class(s)
    infl = inflection(r20, r60)
    rows.append([f, r5, r20, r60, mom, tclass, infl])

mom_df = pd.DataFrame(rows, columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"])
mom_df = mom_df.set_index("Factor").sort_values("Short", ascending=False)

st.subheader("Factor Momentum Snapshot")
st.dataframe(mom_df.round(3), use_container_width=True)

# ---------------- Breadth Index ----------------
st.subheader("Factor Breadth Index")

trend_vals = mom_df["Trend"].value_counts()
num_up = trend_vals.get("Up", 0)
num_down = trend_vals.get("Down", 0)
breadth = num_up / len(mom_df) * 100

st.metric("Breadth Index (Up Trends)", f"{breadth:.1f}%")

# ---------------- Regime Score ----------------
st.subheader("Factor Regime Score")

# Score 0 to 100 based on:
# 40 percent momentum
# 30 percent inflection
# 30 percent trend classification

score = (
    40 * (mom_df["Short"].mean()) +
    30 * ( (mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean() ) +
    30 * ( (mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean() )
)

score = max(0, min(100, 50 + 50 * (score / 5)))

st.metric("Factor Regime Score", f"{score:.1f}")

# ---------------- Leadership matrix ----------------
st.subheader("Leadership Matrix (Short vs Long momentum)")

mat = mom_df[["Short", "Long"]].copy()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    mat,
    cmap="RdYlGn",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar=False,
    ax=ax
)
ax.set_title("Factor Leadership Heatmap", color=TEXT)
st.pyplot(fig)

# ---------------- Time series panels ----------------
st.subheader("Factor Time Series")

ncols = 3
cols = st.columns(ncols)
i = 0

for f in factor_df.columns:
    with cols[i % ncols]:
        fig, ax = plt.subplots(figsize=(5, 3))
        s = factor_df[f].dropna()
        ax.plot(s.index, s.values, color=PASTELS[i % len(PASTELS)], linewidth=2)
        ax.set_title(f, color=TEXT)
        ax.grid(color=GRID)
        st.pyplot(fig)
    i += 1

# ---------------- Cross factor correlation ----------------
st.subheader("Cross Factor Correlation Matrix")

corr = factor_df.pct_change().corr()

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    cmap="PuBuGn",
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    ax=ax2
)
ax2.set_title("Correlation Matrix", color=TEXT)
st.pyplot(fig2)

# ---------------- Relative Strength vs SPY ----------------
st.subheader("Relative Strength vs SPY")

if "SPY" in prices:
    rs_df = {}
    spy = prices["SPY"]

    for f, s in factor_levels.items():
        if isinstance(s, pd.Series):
            rel = rs(s, spy)
            rs_df[f] = rel

    rs_df = pd.DataFrame(rs_df).dropna(how="all")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for i, f in enumerate(rs_df.columns):
        ax3.plot(rs_df.index, rs_df[f], label=f, color=PASTELS[i % len(PASTELS)], linewidth=1.8)
    ax3.grid(color=GRID)
    ax3.legend(fontsize=8)
    ax3.set_title("Factor Relative Strength vs SPY", color=TEXT)
    st.pyplot(fig3)

st.caption("ADFM Factor Momentum and Leadership Dashboard")
