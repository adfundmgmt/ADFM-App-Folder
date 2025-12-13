# sp_sector_rotation_clean_v5.py
# ADFM — S&P 500 Sector Breadth & Rotation Monitor
# Improved rotation scatter: labels above points, no toggles.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(page_title="S&P 500 Sector Breadth & Rotation Monitor", layout="wide")
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
MIN_OBS_FOR_RETURNS = DAYS_3M

PASTELS = [
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6",
    "#ffff99", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"
]

# ------------------------------- Data fetch --------------------------------
@st.cache_data(ttl=3600)
def robust_fetch_batch(tickers, period="1y", interval="1d") -> pd.DataFrame:
    try:
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            progress=False,
            group_by="ticker",
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()

    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = data[(t, "Adj Close")].dropna()
                if not s.empty:
                    out[t] = s
            except Exception:
                continue
    else:
        if "Adj Close" in data.columns:
            out[tickers[0]] = data["Adj Close"].dropna()

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(out).sort_index().dropna(how="all")

# ------------------------------- Load prices -------------------------------
tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch_batch(tickers, LOOKBACK_PERIOD, INTERVAL)

if prices.empty or "SPY" not in prices.columns:
    st.error("Data unavailable or incomplete.")
    st.stop()

prices["SPY"] = prices["SPY"].ffill()
prices = prices[prices["SPY"].notna()]

available_sector_tickers = [t for t in SECTORS if t in prices.columns]
if not available_sector_tickers:
    st.error("No valid sector tickers.")
    st.stop()

# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Track S&P 500 sector leadership, breadth, and rotation.

        • Relative strength vs SPY  
        • 1M vs 3M rotation scatter  
        • Ranked snapshot table  

        Data source: Yahoo Finance.
        """,
        unsafe_allow_html=True,
    )

# ------------------------------- Relative strength -------------------------
relative_strength = prices[available_sector_tickers].div(prices["SPY"], axis=0)

selected_sector = st.selectbox(
    "Select sector:",
    available_sector_tickers,
    format_func=lambda x: SECTORS[x]
)

fig_rs = px.line(
    relative_strength[selected_sector],
    title=f"Relative Strength: {SECTORS[selected_sector]} vs SPY",
    labels={"value": "Ratio", "index": "Date"},
)
fig_rs.update_layout(height=360)
st.plotly_chart(fig_rs, use_container_width=True)

# ------------------------------- Returns snapshot --------------------------
rets_1m = prices[available_sector_tickers].pct_change(DAYS_1M).iloc[-1]
rets_3m = prices[available_sector_tickers].pct_change(DAYS_3M).iloc[-1]

snap = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "Ticker": available_sector_tickers,
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

color_map = {s: PASTELS[i] for i, s in enumerate(snap["Sector"])}

# ------------------------------- Rotation Scatter --------------------------
st.subheader("Current rotation scatter")

fig_xy = go.Figure()
fig_xy.add_hline(y=0, line_width=1, opacity=0.5)
fig_xy.add_vline(x=0, line_width=1, opacity=0.5)

for _, r in snap.iterrows():
    fig_xy.add_trace(go.Scatter(
        x=[r["3M"]],
        y=[r["1M"]],
        mode="markers+text",
        text=[r["Sector"]],
        textposition="top center",
        marker=dict(size=14, color=color_map[r["Sector"]]),
        hovertemplate=(
            f"<b>{r['Sector']}</b><br>"
            f"1M: {r['1M']:.2%}<br>"
            f"3M: {r['3M']:.2%}<br>"
            f"Speed: {r['Speed']:.2%}<br>"
            f"Angle: {r['Angle (deg)']:.1f}°<extra></extra>"
        ),
        showlegend=False
    ))

fig_xy.update_layout(
    title="1M vs 3M total return",
    xaxis=dict(title="3M return", tickformat=".3f"),
    yaxis=dict(title="1M return", tickformat=".3f"),
    height=520,
)
st.plotly_chart(fig_xy, use_container_width=True)

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

snap_sorted.insert(0, "Rank", snap_sorted.index + 1)

styler = (
    snap_sorted[
        ["Rank", "Sector", "Ticker", "Quadrant", "1M", "3M", "Angle (deg)", "Speed"]
    ]
    .style.format({
        "1M": "{:.2%}",
        "3M": "{:.2%}",
        "Angle (deg)": "{:.1f}",
        "Speed": "{:.2%}",
    })
    .bar(subset=["1M", "3M"], align="mid")
    .background_gradient(subset=["Speed"], cmap="PuBuGn")
)

st.dataframe(
    styler,
    use_container_width=True,
    height=40 * len(snap_sorted) + 40
)

# ------------------------------- Footer ------------------------------------
st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
