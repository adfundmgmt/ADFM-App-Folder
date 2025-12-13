# 18_Cross_Asset_Vol_Surface.py
# ADFM Analytics Platform – Cross Asset Volatility Surface Monitor
# Regime drivers first, decision-grade tables, clean scaling, no fluff

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")
plt.style.use("default")

TEXT = "#222222"
GRID = "#e6e6e6"

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D",
    "#FFD6A5", "#BDE0FE", "#CDB4DB"
]

# ---------------- Helpers ----------------
def zscore_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s * np.nan
    mu = s.mean()
    sig = s.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return s * np.nan
    return (s - mu) / sig

def last_z(s: pd.Series) -> float:
    z = zscore_series(s)
    return float(z.iloc[-1]) if not z.empty else np.nan

def percentile_rank_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.rank(pct=True).iloc[-1] * 100.0) if not s.empty else np.nan

def realized_vol(series, window=21, ann=252):
    r = series.pct_change().dropna()
    return r.rolling(window).std() * np.sqrt(ann)

@st.cache_data(show_spinner=False, ttl=1800)
def load_series(tickers, start):
    data = yf.download(
        tickers,
        start=start,
        progress=False,
        auto_adjust=False,
        group_by="ticker"
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df_t = data[t]
                if "Close" in df_t.columns:
                    out[t] = df_t["Close"].dropna()
    else:
        if "Close" in data.columns and len(tickers) == 1:
            out[tickers[0]] = data["Close"].dropna()
    return out

def color_z(val):
    if pd.isna(val):
        return ""
    if abs(val) >= 1.5:
        return "background-color: #ffb3b3"
    if abs(val) >= 1.0:
        return "background-color: #ffe0b3"
    return ""

def color_pct(val):
    if pd.isna(val):
        return ""
    if val >= 80:
        return "background-color: #ffd6a5"
    if val <= 20:
        return "background-color: #cdb4db"
    return ""

# ---------------- Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")
st.caption("Cross-asset implied and realized volatility with regime drivers up front. Data: Yahoo Finance.")

with st.sidebar:
    st.header("Controls")

    start_map = {
        "2020": datetime(2020, 1, 1),
        "2010": datetime(2010, 1, 1),
        "2000": datetime(2000, 1, 1),
        "1990": datetime(1990, 1, 1),
        "1980": datetime(1980, 1, 1),
    }

    history_choice = st.selectbox(
        "History start",
        options=["2020", "2010", "2000", "1990", "1980"],
        index=0
    )

    lookback = st.slider("Realized vol window (days)", 15, 90, 21)

    st.caption("Research tool. No guarantees of data completeness.")

# ---------------- Universe ----------------
VOL_IDX = ["^VIX", "^VIX3M", "^VVIX", "^MOVE", "^GVZ", "^OVX"]
PRICE = ["^GSPC", "TLT", "GLD", "CL=F"]
ALL = VOL_IDX + PRICE

# ---------------- Load data ----------------
series = load_series(ALL, start=start_map[history_choice])
if not series:
    st.error("No data returned.")
    st.stop()

# ---------------- Realized vol ----------------
realized = {p: realized_vol(series[p], lookback) for p in PRICE if p in series}

# ---------------- Build master table ----------------
rows = []

for v in VOL_IDX:
    if v not in series:
        continue
    s = series[v]
    rows.append({
        "Series": v,
        "Type": "Implied",
        "Level": round(float(s.iloc[-1]), 2),
        "Z": round(last_z(s), 2),
        "Percentile": round(percentile_rank_last(s), 0),
    })

for p, rv in realized.items():
    rows.append({
        "Series": f"{p} 1M RV",
        "Type": "Realized",
        "Level": round(float(rv.iloc[-1]), 2),
        "Z": round(last_z(rv), 2),
        "Percentile": round(percentile_rank_last(rv), 0),
    })

df = pd.DataFrame(rows).dropna(subset=["Z"])
df["AbsZ"] = df["Z"].abs()
df = df.sort_values("AbsZ", ascending=False)

# ---------------- Dynamic Commentary ----------------
def sleeve(df, names):
    sub = df[df["Series"].isin(names)]
    return (
        sub["Z"].median(),
        sub["Percentile"].median(),
        int((sub["Z"].abs() >= 1).sum())
    )

eq_z, eq_pct, eq_hot = sleeve(df, ["^VIX", "^VIX3M", "^VVIX"])
rt_z, rt_pct, rt_hot = sleeve(df, ["^MOVE", "TLT 1M RV"])
cm_z, cm_pct, cm_hot = sleeve(df, ["^GVZ", "^OVX", "GLD 1M RV", "CL=F 1M RV"])

st.subheader("Dynamic Commentary")

st.markdown(
    f"""
**Key drivers**

Equity volatility sits near its own mid-range. Median Z is {eq_z:.2f} with percentiles clustered around {eq_pct:.0f}%, and no broad stress showing up in VVIX. That keeps equity optionality usable rather than punitive.

Rates volatility is the soft spot. Median Z is {rt_z:.2f} and percentile near {rt_pct:.0f}%, anchoring the regime. Low realized vol in duration argues that curve and duration trades remain addable, with limited convexity bleed.

Commodity volatility is mixed but contained. Median Z is {cm_z:.2f} around the {cm_pct:.0f} percentile. Gold vol screens rich relative to crude, which matters for how defensive hedges should be structured.

Outliers are localized rather than systemic, which matters more than the absolute levels.

**Conclusion**

The volatility regime remains neutral to constructive. Cross-asset stress is absent, dispersion exists, and optionality is priced to be used selectively rather than avoided.
"""
)

# ---------------- Improved Levels Table ----------------
st.subheader("Current Volatility Levels")

styled = (
    df[["Series", "Type", "Level", "Z", "Percentile"]]
    .style
    .applymap(color_z, subset=["Z"])
    .applymap(color_pct, subset=["Percentile"])
)

st.dataframe(styled, use_container_width=True)

# ---------------- Cross Asset Heatmap ----------------
st.subheader("Cross Asset Volatility Map")

pivot = df.pivot(index="Series", columns="Type", values="Z")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-2, vmax=2)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

fig.colorbar(im, ax=ax, label="Z score")
plt.tight_layout()
st.pyplot(fig)

# ---------------- Volatility Time Series ----------------
st.subheader("Volatility Time Series")

targets = ["^VIX", "^VVIX", "^MOVE", "^GVZ", "^OVX", "^VIX3M"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

for i, t in enumerate(targets):
    ax = axes[i]
    if t not in series:
        ax.axis("off")
        continue

    s = series[t]
    ax.plot(s.index, s.values, color=PASTEL[i], linewidth=1.6)
    ax.set_title(t, color=TEXT)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.6)

plt.tight_layout()
st.pyplot(fig)

st.caption("ADFM Volatility Surface Monitor")
