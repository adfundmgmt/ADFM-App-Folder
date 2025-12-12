# sp_sector_rotation_clean_v6.py
# ADFM — S&P 500 Sector Breadth & Rotation Monitor
# Geometry-correct rotation with volatility-normalized axes

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
MIN_OBS = DAYS_3M

PASTELS = [
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99",
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"
]

# ------------------------------- Data fetch --------------------------------
@st.cache_data(ttl=3600)
def robust_fetch_batch(tickers, period="1y", interval="1d") -> pd.DataFrame:
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        group_by="ticker",
        auto_adjust=False,
    )

    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Adj Close") in data.columns:
                s = data[(t, "Adj Close")].dropna()
                if not s.empty:
                    out[t] = s
    else:
        if "Adj Close" in data.columns:
            out[tickers[0]] = data["Adj Close"].dropna()

    return pd.DataFrame(out).sort_index()

# ------------------------------- Load prices -------------------------------
tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch_batch(tickers, LOOKBACK_PERIOD, INTERVAL)

if prices.empty or "SPY" not in prices:
    st.error("Data unavailable or incomplete.")
    st.stop()

prices["SPY"] = prices["SPY"].ffill()
prices = prices.loc[prices["SPY"].notna()]

valid_cols = [c for c in prices if prices[c].dropna().shape[0] >= MIN_OBS]
prices = prices[valid_cols]

sector_tickers = [t for t in SECTORS if t in prices.columns]
if not sector_tickers:
    st.error("No valid sector data.")
    st.stop()

# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Track S&P 500 sector leadership and rotation.

        • Relative strength vs SPY  
        • 1M vs 3M rotation, volatility-normalized  
        • Speed and angle quantify momentum and trend change  
        """,
        unsafe_allow_html=True,
    )

# ------------------------------- Relative strength -------------------------
relative_strength = prices[sector_tickers].div(prices["SPY"], axis=0)

selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=sector_tickers,
    format_func=lambda x: SECTORS[x],
)

fig_rs = px.line(
    relative_strength[selected_sector].dropna(),
    title=f"Relative Strength: {SECTORS[selected_sector]} vs SPY",
    labels={"value": "Ratio (Sector / SPY)", "index": "Date"},
)
fig_rs.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=30))
st.plotly_chart(fig_rs, use_container_width=True)

# ------------------------------- Returns snapshot --------------------------
rets_1m = prices[sector_tickers].pct_change(DAYS_1M)
rets_3m = prices[sector_tickers].pct_change(DAYS_3M)

r1 = rets_1m.iloc[-1]
r3 = rets_3m.iloc[-1]

snap = pd.DataFrame({
    "Sector": [SECTORS[t] for t in sector_tickers],
    "Ticker": sector_tickers,
    "1M": r1.values,
    "3M": r3.values,
}).dropna()

# Vol-normalize for geometry
vol_1m = snap["1M"].std()
vol_3m = snap["3M"].std()

snap["x"] = snap["3M"] / vol_3m
snap["y"] = snap["1M"] / vol_1m

snap["Angle (deg)"] = np.degrees(np.arctan2(snap["y"], snap["x"]))
snap["Speed"] = np.sqrt(snap["x"]**2 + snap["y"]**2)

snap["Quadrant"] = np.select(
    [
        (snap["x"] > 0) & (snap["y"] > 0),
        (snap["x"] < 0) & (snap["y"] > 0),
        (snap["x"] < 0) & (snap["y"] < 0),
        (snap["x"] > 0) & (snap["y"] < 0),
    ],
    ["Q1: Leading", "Q2: Improving", "Q3: Weak", "Q4: Fading"],
)

snap["Rank (Speed)"] = snap["Speed"].rank(ascending=False, method="min").astype(int)

color_map = {sec: PASTELS[i % len(PASTELS)] for i, sec in enumerate(snap["Sector"])}

# ------------------------------- Rotation Scatter --------------------------
st.subheader("Current rotation scatter (vol-normalized)")

fig_xy = go.Figure()
fig_xy.add_hline(y=0, line_width=1, opacity=0.5)
fig_xy.add_vline(x=0, line_width=1, opacity=0.5)

lim = max(snap["x"].abs().max(), snap["y"].abs().max()) * 1.15
fig_xy.update_xaxes(range=[-lim, lim], title="3M (normalized)")
fig_xy.update_yaxes(range=[-lim, lim], title="1M (normalized)")

for _, r in snap.iterrows():
    fig_xy.add_trace(go.Scatter(
        x=[r["x"]], y=[r["y"]],
        mode="markers+text",
        marker=dict(size=14, color=color_map[r["Sector"]]),
        text=[r["Sector"]],
        textposition="top center",
        hovertemplate=(
            f"<b>{r['Sector']}</b><br>"
            f"1M: {r['1M']:.2%}<br>"
            f"3M: {r['3M']:.2%}<br>"
            f"Speed: {r['Speed']:.2f}<br>"
            f"Angle: {r['Angle (deg)']:.1f}°<extra></extra>"
        ),
        showlegend=False
    ))

fig_xy.update_layout(height=520, margin=dict(l=40, r=40, t=60, b=40))
st.plotly_chart(fig_xy, use_container_width=True)

# ------------------------------- Table -------------------------------------
st.subheader("Rotation snapshot table")

st.dataframe(
    snap[["Sector", "Ticker", "Quadrant", "1M", "3M", "Angle (deg)", "Speed", "Rank (Speed)"]]
    .sort_values("Rank (Speed)")
    .style.format({
        "1M": "{:.2%}", "3M": "{:.2%}", "Angle (deg)": "{:.1f}", "Speed": "{:.2f}"
    }),
    use_container_width=True,
    height=520
)

st.download_button(
    "Download snapshot as CSV",
    data=snap.to_csv(index=False),
    file_name="sector_rotation_snapshot.csv",
    mime="text/csv"
)

st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
