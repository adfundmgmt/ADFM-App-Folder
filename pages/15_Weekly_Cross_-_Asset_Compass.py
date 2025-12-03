# ---------------------------------------------------------
# Rolling Cross-Asset Correlation Dashboard
# ADFM Edition – Clean, correlation-only, high-signal
# ---------------------------------------------------------

import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

import streamlit as st

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Rolling Cross-Asset Correlations", layout="wide")

plt.rcParams.update({
    "figure.figsize": (7, 3),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
})

# Red → Yellow → Green gradient for correlations
RYG = LinearSegmentedColormap.from_list(
    "RYG",
    [(0.75, 0.2, 0.2), (0.95, 0.95, 0.5), (0.3, 0.7, 0.3)],
    N=256
)

# ---------------------------------------------------------
# EXPANDED UNIVERSE (feel free to expand further)
# ---------------------------------------------------------
UNIVERSE = {
    "US Equities": ["^GSPC", "^NDX", "^RUT", "IWM", "QQQ"],
    "Global Equities": ["^STOXX50E", "^N225", "EEM", "EWJ", "EWZ"],
    "Credit": ["LQD", "HYG", "JNK"],
    "Rates": ["^TNX", "^TYX", "^FVX"],
    "Commodities": ["CL=F", "NG=F", "GC=F", "HG=F", "SI=F"],
    "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "CNY=X"],
}

ALL_TICKERS = list(dict.fromkeys(sum(UNIVERSE.values(), [])))

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def pct_returns(prices):
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t]
            col = "Adj Close" if "Adj Close" in df else "Close"
            frames.append(df[col].rename(t))
        return pd.concat(frames, axis=1).sort_index()

    return pd.DataFrame()

def roll_corr(df, a, b, window):
    if a not in df.columns or b not in df.columns:
        return pd.Series(dtype=float)
    tmp = df[[a, b]].dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp[a].rolling(window).corr(tmp[b])

def smooth(series):
    return series.rolling(3).mean() if series is not None else series

def auto_commentary(cmat, asset_labels):
    if cmat.dropna(how="all").empty:
        return "Correlation matrix empty. No signal."

    # Identify strong positive and negative pairs
    pairs = []
    for i in range(len(asset_labels)):
        for j in range(i):
            v = cmat.iat[i, j]
            if pd.isna(v):
                continue
            pairs.append((asset_labels[i], asset_labels[j], v))

    if not pairs:
        return "No usable correlation pairs."

    # Sort strongest absolute correlations
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    top_pos = [p for p in pairs_sorted if p[2] > 0][:3]
    top_neg = [p for p in pairs_sorted if p[2] < 0][:3]

    text = ""

    if top_pos:
        text += "Highest positive correlations: "
        text += "; ".join([f"{a} with {b} at {v:.0%}" for a, b, v in top_pos]) + ". "
    if top_neg:
        text += "Highest negative correlations: "
        text += "; ".join([f"{a} vs {b} at {v:.0%}" for a, b, v in top_neg]) + ". "

    # Look for correlation regime shifts
    shocks = []
    for a, b, v in pairs_sorted[:8]:
        if abs(v) > 0.7:
            shocks.append(f"{a}/{b} showing regime-level correlation at {v:.0%}")
    if shocks:
        text += "Regime signals: " + "; ".join(shocks) + ". "

    return text.strip()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("About this tool")
st.sidebar.markdown(
    """
Rolling cross-asset correlation dashboard using Yahoo Finance data.
Clean, high-signal, used to understand how equities, credit, rates,
commodities and FX move together across chosen time horizons.
"""
)

window_choice = st.sidebar.selectbox(
    "Analysis window",
    ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
    index=3
)

lookback_map = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "YTD": None,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "10Y": 252 * 10,
}

rolling_win = st.sidebar.selectbox("Rolling window (days)", [21, 63], index=0)
do_smooth = st.sidebar.checkbox("Smooth rolling correlations", value=True)

# ---------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------
end = datetime.now().date()
start = (datetime.now() - timedelta(days=365 * 12)).date()

PR = fetch(ALL_TICKERS, str(start), str(end))
if PR.empty:
    st.error("No data returned from Yahoo Finance.")
    st.stop()

RET = pct_returns(PR)

# Apply analysis window subset
if window_choice == "YTD":
    y0 = datetime(datetime.now().year, 1, 1)
    RETW = RET[RET.index >= y0]
else:
    days = lookback_map[window_choice]
    RETW = RET.iloc[-days:] if len(RET) > days else RET.copy()

if RETW.empty:
    st.error("Insufficient data for the selected window.")
    st.stop()

asset_list = RETW.columns.tolist()

# ---------------------------------------------------------
# ROLLING CORRELATION MATRIX
# ---------------------------------------------------------
st.subheader(f"Rolling Correlation Matrix ({window_choice})")

# Compute last available rolling correlation between all pairs
corr_last = pd.DataFrame(index=asset_list, columns=asset_list, dtype=float)

for a in asset_list:
    for b in asset_list:
        if a == b:
            corr_last.loc[a, b] = np.nan
            continue
        rc = roll_corr(RETW, a, b, rolling_win)
        if do_smooth:
            rc = smooth(rc)
        corr_last.loc[a, b] = rc.dropna().iloc[-1] if not rc.dropna().empty else np.nan

# Mask upper triangle
for i in range(len(asset_list)):
    for j in range(i, len(asset_list)):
        corr_last.iat[i, j] = np.nan

sty = (
    corr_last.style
        .background_gradient(cmap=RYG, vmin=-1, vmax=1)
        .format(lambda x: "" if pd.isna(x) else f"{x:.0%}")
)

st.dataframe(sty, use_container_width=True, height=600)

# ---------------------------------------------------------
# AUTO COMMENTARY
# ---------------------------------------------------------
st.subheader("Correlation Read")
st.markdown(auto_commentary(corr_last, asset_list))

# ---------------------------------------------------------
# PAIRWISE ROLLING CORRELATION PANELS
# ---------------------------------------------------------
st.subheader(f"Rolling Correlations ({rolling_win} day)")

# Create figure grid
n = len(asset_list)
pairs = [(asset_list[i], asset_list[j]) for i in range(n) for j in range(i)]
rows = math.ceil(len(pairs) / 3)

fig_idx = 0

for i in range(rows):
    cols = st.columns(3)
    for j in range(3):
        if fig_idx >= len(pairs):
            break
        a, b = pairs[fig_idx]
        rc = roll_corr(RETW, a, b, rolling_win)
        if do_smooth:
            rc = smooth(rc)

        with cols[j]:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(rc.index, rc.values, color="#336699", linewidth=1.4)
            ax.axhline(0, color="black", linewidth=1.0)
            ax.set_title(f"{a} vs {b}")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        fig_idx += 1

# END OF FILE
