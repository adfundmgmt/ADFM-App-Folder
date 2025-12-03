# ---------------------------------------------------------
# Rolling Cross-Asset Correlations – ADFM Edition
# Clean rebuild. Only rolling correlations. Dynamic x/y scaling.
# Matching original visual style. Max 30 charts.
# ---------------------------------------------------------

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
    "lines.linewidth": 1.6,
})


# ---------------------------------------------------------
# EXPANDED UNIVERSE (curated for quality + variety)
# ---------------------------------------------------------
UNIVERSE = {
    "US Equities": ["^GSPC", "^NDX", "^RUT", "IWM", "QQQ"],
    "Global Equities": ["^STOXX50E", "^N225", "EEM", "EWJ"],
    "Credit": ["LQD", "HYG"],
    "Rates": ["^TNX", "^TYX"],
    "Commodities": ["CL=F", "NG=F", "GC=F", "HG=F"],
    "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"],
}

ALL_TICKERS = list(dict.fromkeys(sum(UNIVERSE.values(), [])))


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def pct_returns(prices):
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_prices(tickers, start, end):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
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


def roll_corr(df, a, b, win):
    if a not in df.columns or b not in df.columns:
        return pd.Series(dtype=float)
    tmp = df[[a, b]].dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp[a].rolling(win).corr(tmp[b])


def auto_commentary(corr_last, pairs):
    if corr_last.empty:
        return "No correlation signal."

    lines = []

    strong_pos = corr_last[corr_last > 0.6].sort_values(ascending=False)
    strong_neg = corr_last[corr_last < -0.6].sort_values()

    if not strong_pos.empty:
        items = [f"{a} vs {b} at {v:.0%}" for (a, b, v) in pairs if v in strong_pos.values]
        lines.append("Strong positive correlations: " + "; ".join(items))

    if not strong_neg.empty:
        items = [f"{a} vs {b} at {v:.0%}" for (a, b, v) in pairs if v in strong_neg.values]
        lines.append("Strong negative correlations: " + "; ".join(items))

    if not lines:
        return "Correlations are mostly mid-range. No strong cross-asset regime signal."

    return " ".join(lines)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("About this tool")
st.sidebar.markdown(
    """
Rolling cross-asset correlations across equities, credit, rates,
commodities, and FX using Yahoo Finance data.  
Designed for regime detection, hedge selection, and factor alignment.
"""
)

window_choice = st.sidebar.selectbox(
    "Analysis window",
    ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
    index=3,
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

rolling_window = st.sidebar.selectbox("Rolling window (days)", [21, 63], index=0)


# ---------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------
end = datetime.now().date()
start = (datetime.now() - timedelta(days=365 * 12)).date()

PR = fetch_prices(ALL_TICKERS, str(start), str(end))
if PR.empty:
    st.error("No price data returned.")
    st.stop()

RET = pct_returns(PR)

# Apply selected analysis window
if window_choice == "YTD":
    y0 = datetime(datetime.now().year, 1, 1)
    RETW = RET[RET.index >= y0]
else:
    days = lookback_map[window_choice]
    RETW = RET.iloc[-days:] if len(RET) > days else RET.copy()

if RETW.empty:
    st.error("Insufficient data for selected window.")
    st.stop()

assets = RETW.columns.tolist()


# ---------------------------------------------------------
# COMPUTE LAST CORRELATIONS FOR COMMENTARY
# ---------------------------------------------------------
pairs = []
corr_last_vals = []

for i in range(len(assets)):
    for j in range(i):
        a, b = assets[i], assets[j]
        rc = roll_corr(RETW, a, b, rolling_window)
        if not rc.dropna().empty:
            v = float(rc.dropna().iloc[-1])
            pairs.append((a, b, v))
            corr_last_vals.append(v)


corr_last_series = pd.Series(
    corr_last_vals,
    index=[f"{a}-{b}" for a, b, _ in pairs],
    dtype=float,
)

# ---------------------------------------------------------
# COMMENTARY
# ---------------------------------------------------------
st.subheader("Correlation Commentary")
st.markdown(auto_commentary(corr_last_series, pairs))


# ---------------------------------------------------------
# ROLLING CORRELATION PANELS (MAX 30)
# ---------------------------------------------------------
st.subheader(f"Rolling Correlations ({rolling_window}-day)")

max_charts = 30
pairs_to_plot = pairs[:max_charts]
n_pairs = len(pairs_to_plot)

rows = math.ceil(n_pairs / 3)
idx = 0

for r in range(rows):
    cols = st.columns(3)
    for c in range(3):
        if idx >= n_pairs:
            break

        a, b, _ = pairs_to_plot[idx]
        rc = roll_corr(RETW, a, b, rolling_window)
        rc = rc.dropna()

        with cols[c]:
            fig, ax = plt.subplots(figsize=(7, 3))

            if not rc.empty:
                ymin = float(rc.min()) - 0.05
                ymax = float(rc.max()) + 0.05
                ymin = max(ymin, -1)
                ymax = min(ymax, 1)

                ax.plot(rc.index, rc.values, color="#336699", linewidth=1.6)
                ax.set_ylim(ymin, ymax)

            ax.axhline(0, color="black", linewidth=1.0)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_title(f"{a} vs {b}")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        idx += 1

# END OF FILE
