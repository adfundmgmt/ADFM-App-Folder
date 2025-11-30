# 18_Cross_Asset_Vol_Surface.py
# ADFM Cross Asset Volatility Surface Monitor
# Pastel theme, decision grade, simple and robust

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

# ---------------- Config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")
plt.style.use("default")

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

GRID = "#e8e8e8"
TEXT = "#222222"

# ---------------- Helpers ----------------
def zscore(x):
    x = pd.Series(x).dropna()
    if x.std() == 0 or x.empty:
        return pd.Series([np.nan]*len(x), index=x.index)
    return (x - x.mean()) / x.std()

def realized_vol(series, window=21, ann=252):
    r = series.pct_change().dropna()
    return r.rolling(window).std() * np.sqrt(ann)

def load_series(tickers, start="2010-01-01"):
    data = yf.download(
        tickers, start=start, progress=False, auto_adjust=False
    )
    out = {}
    for t in tickers:
        if (t,) in data.columns:
            out[t] = data[(t, "Close")].dropna()
        elif t in data.columns:
            out[t] = data[t]["Close"].dropna()
        else:
            continue
    return out

def pastel_cmap():
    return LinearSegmentedColormap.from_list(
        "pastel_scale",
        ["#F1FAEE", "#A8DADC", "#457B9D"],
        N=256
    )

# ---------------- Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    lookback = st.slider("Rolling window (days)", 15, 90, 21)
    show_z = st.checkbox("Show Z score versions", value=True)
    st.caption("Data: Yahoo Finance. Focus is on high signal, low noise volatility markers.")

# ---------------- Data universe ----------------
VOL_IDX = [
    "^VIX",      # S&P implied vol
    "^VIX3M",    # 3M implied vol
    "^VVIX",     # Vol of vol
    "^MOVE",     # Treasury vol
    "^GVZ",      # Gold vol
    "^OVX",      # Oil vol
]

PRICE_UNDERLYING = [
    "^GSPC",
    "TLT",
    "GLD",
    "CL=F"
]

# ---------------- Load data ----------------
series = load_series(VOL_IDX + PRICE_UNDERLYING, start=str(start_date))

# ---------------- Realized vol ----------------
realized = {}
for p in PRICE_UNDERLYING:
    if p in series:
        realized[p] = realized_vol(series[p], window=lookback)

# ---------------- Build table ----------------
rows = []
for idx in VOL_IDX:
    if idx not in series:
        continue
    s = series[idx]
    lvl = s.iloc[-1] if len(s) else np.nan
    z = zscore(s).iloc[-1] if show_z else np.nan

    rows.append([
        idx,
        round(lvl, 2),
        round(z, 2) if show_z and not pd.isna(z) else ""
    ])

for p in PRICE_UNDERLYING:
    if p not in realized:
        continue
    rv = realized[p]
    lvl = rv.iloc[-1] if len(rv) else np.nan
    zr = zscore(rv).iloc[-1] if show_z else np.nan

    rows.append([
        f"{p} RV",
        round(lvl, 2),
        round(zr, 2) if show_z and not pd.isna(zr) else ""
    ])

df = pd.DataFrame(rows, columns=["Series", "Level", "Z"])

# ---------------- Display table ----------------
st.subheader("Current Volatility Levels")
st.dataframe(df, use_container_width=True)

# ---------------- Term structure panel ----------------
def plot_term_structure(vol, vol3m):
    fig, ax = plt.subplots(figsize=(8, 3))
    points = [0.0, 0.25]
    vals = []
    if vol is not None:
        vals.append(vol)
    else:
        vals.append(np.nan)
    if vol3m is not None:
        vals.append(vol3m)
    else:
        vals.append(np.nan)

    ax.plot(points, vals, marker="o", linewidth=2, color=PASTEL[1])
    ax.set_xticks(points)
    ax.set_xticklabels(["1M", "3M"])
    ax.set_title("VIX Term Structure", color=TEXT)
    ax.grid(color=GRID)
    st.pyplot(fig)

st.subheader("VIX Term Structure")
v = series.get("^VIX", None)
v3 = series.get("^VIX3M", None)
plot_term_structure(
    v.iloc[-1] if v is not None else None,
    v3.iloc[-1] if v3 is not None else None
)

# ---------------- Multi panel figs ----------------
st.subheader("Volatility Time Series")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

targets = [
    "^VIX", "^VVIX", "^MOVE",
    "^GVZ", "^OVX", "^VIX3M"
]

for i, t in enumerate(targets):
    ax = axes[i]
    if t not in series:
        ax.set_title(f"{t} missing")
        ax.axis("off")
        continue

    s = series[t]
    ax.plot(s.index, s.values, color=PASTEL[i % len(PASTEL)], linewidth=2)

    if show_z:
        zs = zscore(s)
        ax2 = ax.twinx()
        ax2.plot(s.index, zs.values, color="#999999", linewidth=1, alpha=0.5)
        ax2.axhline(0, color="#999999", linewidth=0.8)

    ax.set_title(t, color=TEXT)
    ax.grid(color=GRID)

plt.tight_layout()
st.pyplot(fig)

# ---------------- Heatmap of daily changes ----------------
st.subheader("Cross Asset Vol Shock Map")

def daily_change_map(series_dict):
    names = []
    vals = []
    for k, v in series_dict.items():
        if len(v) < 2:
            continue
        ch = (v.iloc[-1] - v.iloc[-2]) / v.iloc[-2]
        names.append(k)
        vals.append(ch)

    return pd.DataFrame({"Series": names, "Daily % Change": vals})

heat = daily_change_map(series)
heat = heat.dropna()
heat = heat.set_index("Series")

fig_h, ax_h = plt.subplots(figsize=(6, 4))
cmap = pastel_cmap()
im = ax_h.imshow(heat.values, cmap=cmap, aspect="auto")
ax_h.set_yticks(range(len(heat.index)))
ax_h.set_yticklabels(heat.index)
ax_h.set_xticks([0])
ax_h.set_xticklabels(["Daily %"])
plt.colorbar(im, ax=ax_h)
st.pyplot(fig_h)

st.caption("ADFM Volatility Surface Monitor")

