# sp_sector_rotation_clean_v4.py
# ADFM — S&P 500 Sector Breadth & Rotation Monitor
# Keep: Relative-strength vs SPY, polished rotation table
# Add: Interactive X–Y scatter (1M vs 3M) with quadrant shading and label controls

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

# Pastel palette for consistent aesthetics
PASTELS = [
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99",
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"
]

# ------------------------------- Data fetch --------------------------------
@st.cache_data(ttl=3600)
def robust_fetch_batch(tickers, period="1y", interval="1d") -> pd.DataFrame:
    """Batch download Adjusted Close for all tickers, return wide dataframe."""
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
            name = tickers[0] if isinstance(tickers, list) and len(tickers) == 1 else "TICKER"
            out[name] = data["Adj Close"].dropna()

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).sort_index()
    return df.dropna(how="all")

# ------------------------------- Load prices -------------------------------
tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch_batch(tickers, period=LOOKBACK_PERIOD, interval=INTERVAL)

if prices.empty:
    st.error("Downloaded data is empty. Yahoo Finance might be rate limited or unavailable.")
    st.stop()

if "SPY" not in prices.columns:
    st.error("SPY data unavailable. Cannot compute relative measures.")
    st.stop()

# Clean SPY and enforce minimum history
prices["SPY"] = prices["SPY"].ffill()
prices = prices[prices["SPY"].notna()]
enough_history_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_OBS_FOR_RETURNS]
prices = prices[enough_history_cols]

price_cols = [str(c).strip() for c in prices.columns]
available_sector_tickers = [t for t in SECTORS.keys() if t in price_cols]
missing_sectors = [SECTORS[t] for t in SECTORS if t not in price_cols]

if not available_sector_tickers:
    st.error("No sector tickers with sufficient data.")
    st.stop()

# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This dashboard monitors sector relative strength and rotation.

        Views
        • Relative Strength vs SPY
        • Interactive 1M vs 3M scatter with quadrant shading
        • Polished rotation snapshot table

        Data: Yahoo Finance via yfinance, refreshed hourly.
        """
    )
    st.markdown("---")
    if available_sector_tickers:
        st.write("Sectors: " + ", ".join([SECTORS[t] for t in available_sector_tickers]))
    if missing_sectors:
        st.warning("Missing data: " + ", ".join(missing_sectors))

# ------------------------------- Relative strength -------------------------
relative_strength = prices[available_sector_tickers].div(prices["SPY"], axis=0)

default_choice = st.session_state.get("sel_sector", available_sector_tickers[0])
if default_choice not in available_sector_tickers:
    default_choice = available_sector_tickers[0]

selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=available_sector_tickers,
    index=available_sector_tickers.index(default_choice),
    format_func=lambda x: SECTORS[x]
)
st.session_state["sel_sector"] = selected_sector

fig_rs = px.line(
    relative_strength[selected_sector].dropna(),
    title=f"Relative Strength: {SECTORS[selected_sector]} vs SPY",
    labels={"value": "Ratio (Sector / SPY)", "index": "Date"},
)
fig_rs.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=30))
st.plotly_chart(fig_rs, use_container_width=True)

# ------------------------------- Returns snapshot --------------------------
rets_1m_df = prices[available_sector_tickers].pct_change(DAYS_1M)
rets_3m_df = prices[available_sector_tickers].pct_change(DAYS_3M)
returns_1m_latest = rets_1m_df.iloc[-1]
returns_3m_latest = rets_3m_df.iloc[-1]

snap = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "Ticker": available_sector_tickers,
    "1M": [returns_1m_latest.get(t, np.nan) for t in available_sector_tickers],
    "3M": [returns_3m_latest.get(t, np.nan) for t in available_sector_tickers],
})

def quadrant(row):
    if pd.isna(row["1M"]) or pd.isna(row["3M"]):
        return ""
    if row["3M"] > 0 and row["1M"] > 0:
        return "Q1 ++"
    if row["3M"] < 0 and row["1M"] > 0:
        return "Q2 -+"
    if row["3M"] < 0 and row["1M"] < 0:
        return "Q3 --"
    if row["3M"] > 0 and row["1M"] < 0:
        return "Q4 +-"
    return ""

snap["Angle (deg)"] = np.degrees(np.arctan2(snap["1M"], snap["3M"]))
snap["Speed"] = np.sqrt(snap["1M"]**2 + snap["3M"]**2)
snap["Quadrant"] = snap.apply(quadrant, axis=1)
snap["Rank (Speed)"] = snap["Speed"].rank(ascending=False, method="min").astype(int)

# Color map
sector_names = snap["Sector"].tolist()
color_map = {sec: PASTELS[i % len(PASTELS)] for i, sec in enumerate(sector_names)}

# ------------------------------- X–Y Scatter -------------------------------
st.subheader("Current rotation scatter")
c1, c2, c3 = st.columns([2,2,2])
label_topn = c1.slider("Label top N by Speed", 0, len(snap), min(8, len(snap)), step=1)
show_guides = c2.toggle("Show quadrant shading", value=True)
show_tickers = c3.toggle("Use ticker labels on hover", value=False)

fig_xy = go.Figure()

# Optional quadrant shading
if show_guides:
    # Q1
    fig_xy.add_vrect(x0=0, x1=snap["3M"].max()*1.05, y0=0, y1=snap["1M"].max()*1.05,
                     fillcolor="rgba(200, 230, 255, 0.20)", line_width=0)
    # Q3
    fig_xy.add_vrect(x0=snap["3M"].min()*1.05, x1=0, y0=snap["1M"].min()*1.05, y1=0,
                     fillcolor="rgba(255, 220, 220, 0.18)", line_width=0)

# Points
for _, r in snap.iterrows():
    name = r["Sector"]
    hover_name = r["Ticker"] if show_tickers else name
    fig_xy.add_trace(go.Scatter(
        x=[r["3M"]], y=[r["1M"]],
        mode="markers",
        marker=dict(size=14, line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                    color=color_map[name]),
        name=name,
        hovertemplate="<b>"+hover_name+"</b><br>1M: %{y:.1%}<br>3M: %{x:.1%}"
                      "<br>Speed: "+f"{r['Speed']:.1%}"+"<br>Angle: "+f"{r['Angle (deg)']:.1f}°<extra></extra>",
        showlegend=False
    ))

# Zero axes
fig_xy.add_hline(y=0, line_width=1, opacity=0.5)
fig_xy.add_vline(x=0, line_width=1, opacity=0.5)

# Text labels for top N by speed
if label_topn > 0:
    top_names = (
        snap.sort_values("Speed", ascending=False)
            .head(label_topn)["Sector"].tolist()
    )
    for _, r in snap[snap["Sector"].isin(top_names)].iterrows():
        fig_xy.add_trace(go.Scatter(
            x=[r["3M"]], y=[r["1M"]],
            mode="text",
            text=[r["Ticker"] if show_tickers else r["Sector"]],
            textposition="top center",
            showlegend=False
        ))

fig_xy.update_layout(
    title="1M vs 3M total return",
    xaxis_title="3M return",
    yaxis_title="1M return",
    height=520,
    margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig_xy, use_container_width=True)

# ------------------------------- Polished table ----------------------------
st.subheader("Rotation snapshot table")
sort_col = st.selectbox(
    "Sort by", options=["Speed", "1M", "3M", "Angle (deg)", "Sector", "Quadrant", "Rank (Speed)"], index=0
)
ascending = st.toggle("Ascending", value=False)
snap_sorted = snap.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

styler = (
    snap_sorted[["Sector","Ticker","Quadrant","1M","3M","Angle (deg)","Speed","Rank (Speed)"]]
    .style.format({
        "1M": "{:.2%}",
        "3M": "{:.2%}",
        "Angle (deg)": "{:.1f}",
        "Speed": "{:.2%}",
        "Rank (Speed)": "{:d}",
    })
    .set_properties(subset=["Sector","Ticker","Quadrant"], **{"text-align":"left"})
    .set_properties(subset=["Angle (deg)","Speed","Rank (Speed)"], **{"text-align":"right"})
    .bar(subset=["1M","3M"], align="mid")
    .background_gradient(subset=["Speed"], cmap="PuBuGn")
)

st.dataframe(styler, use_container_width=True, height=520)

# Download
csv = snap_sorted.to_csv(index=False).encode()
st.download_button(
    label="Download snapshot as CSV",
    data=csv,
    file_name="sector_rotation_snapshot.csv",
    mime="text/csv"
)

# ------------------------------- Footer ------------------------------------
if not prices.empty:
    st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
