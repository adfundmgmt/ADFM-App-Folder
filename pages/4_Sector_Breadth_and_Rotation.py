# sp_sector_rotation_clean_v6.py
# ADFM — S&P 500 Sector Breadth & Rotation Monitor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(
    page_title="S&P 500 Sector Breadth & Rotation Monitor",
    layout="wide"
)
st.title("S&P 500 Sector Breadth & Rotation Monitor")

# ------------------------------- Constants --------------------------------
SECTORS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

LOOKBACK_PERIOD = "1y"
INTERVAL = "1d"
DAYS_1M = 21
DAYS_3M = 63

PASTELS = [
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6",
    "#ffff99", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"
]

# ------------------------------- Data fetch --------------------------------
@st.cache_data(ttl=3600)
def fetch_prices(tickers):
    data = yf.download(
        tickers,
        period=LOOKBACK_PERIOD,
        interval=INTERVAL,
        progress=False,
        group_by="ticker",
        auto_adjust=False,
    )
    out = {}
    for t in tickers:
        try:
            out[t] = data[(t, "Adj Close")].dropna()
        except Exception:
            pass
    return pd.DataFrame(out)

# ------------------------------- Load data ---------------------------------
tickers = list(SECTORS.keys()) + ["SPY"]
prices = fetch_prices(tickers)

if prices.empty or "SPY" not in prices:
    st.error("Data unavailable.")
    st.stop()

prices = prices.ffill().dropna()

# ------------------------------- Returns -----------------------------------
rets_1m = prices[list(SECTORS)].pct_change(DAYS_1M).iloc[-1]
rets_3m = prices[list(SECTORS)].pct_change(DAYS_3M).iloc[-1]

snap = pd.DataFrame({
    "Sector": [SECTORS[t] for t in SECTORS],
    "Ticker": list(SECTORS.keys()),
    "1M": rets_1m.values,
    "3M": rets_3m.values,
})

snap["Angle (deg)"] = np.degrees(np.arctan2(snap["1M"], snap["3M"]))
snap["Speed"] = np.sqrt(snap["1M"]**2 + snap["3M"]**2)

snap["Quadrant"] = np.select(
    [
        (snap["3M"] > 0) & (snap["1M"] > 0),
        (snap["3M"] < 0) & (snap["1M"] > 0),
        (snap["3M"] < 0) & (snap["1M"] < 0),
        (snap["3M"] > 0) & (snap["1M"] < 0),
    ],
    ["Q1 ++", "Q2 -+", "Q3 --", "Q4 +-"],
    default="Neutral"
)

# ------------------------------- Rotation Scatter --------------------------
st.subheader("Current rotation scatter")

fig = go.Figure()
fig.add_hline(y=0, line_width=1, opacity=0.5)
fig.add_vline(x=0, line_width=1, opacity=0.5)

for i, r in snap.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["3M"]],
        y=[r["1M"]],
        mode="markers+text",
        text=[r["Sector"]],
        textposition="top center",
        marker=dict(size=14, color=PASTELS[i]),
        hovertemplate=(
            f"<b>{r['Sector']}</b><br>"
            f"1M: {r['1M']:.2%}<br>"
            f"3M: {r['3M']:.2%}<br>"
            f"Speed: {r['Speed']:.2%}<br>"
            f"Angle: {r['Angle (deg)']:.1f}°<extra></extra>"
        ),
        showlegend=False
    ))

fig.update_layout(
    title="1M vs 3M total return",
    xaxis=dict(
        title="3M return (decimal)",
        tickformat=".3f",
        zeroline=False
    ),
    yaxis=dict(
        title="1M return (decimal)",
        tickformat=".3f",
        zeroline=False
    ),
    height=520,
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Table -------------------------------------
st.subheader("Rotation snapshot table")

sort_col = st.selectbox(
    "Sort by",
    ["Speed", "1M", "3M", "Angle (deg)", "Sector", "Quadrant"],
)

ascending = st.toggle("Ascending", value=False)

snap_sorted = (
    snap.sort_values(sort_col, ascending=ascending)
    .reset_index(drop=True)
)

# Rank must start at 1
snap_sorted.insert(0, "Rank", np.arange(1, len(snap_sorted) + 1))

display_df = snap_sorted[
    ["Rank", "Sector", "Ticker", "Quadrant", "1M", "3M", "Angle (deg)", "Speed"]
]

styler = (
    display_df.style
    .format({
        "1M": "{:.2%}",
        "3M": "{:.2%}",
        "Angle (deg)": "{:.1f}",
        "Speed": "{:.2%}",
    })
    .bar(subset=["1M", "3M"], align="mid")
    .background_gradient(subset=["Speed"], cmap="PuBuGn")
)

# Exact height: header + rows, no padding rows
row_height = 35
table_height = row_height * (len(display_df) + 1)

st.dataframe(
    styler,
    use_container_width=True,
    height=table_height,
    hide_index=True
)

# ------------------------------- Footer ------------------------------------
st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
