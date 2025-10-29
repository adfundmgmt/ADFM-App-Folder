# sp_sector_rotation_clean_v3.py
# ADFM — S&P 500 Sector Breadth and Rotation Monitor
# Clean version per request: remove "Rotation summary, clutter-free" section.
# Keep relative-strength view and deliver a single polished, sortable snapshot table.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
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

# ------------------------------- Data fetch --------------------------------
@st.cache_data(ttl=3600)
def robust_fetch_batch(tickers, period="1y", interval="1d") -> pd.DataFrame:
    """
    Batch download Adjusted Close for all tickers.
    Returns a DateIndex DataFrame with columns per ticker.
    """
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
    df = df.dropna(how="all")
    return df

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
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard monitors sector relative strength and a compact rotation snapshot.

        Views
        • Relative Strength vs SPY  
        • Polished rotation snapshot table with sort and download

        Data: Yahoo Finance via yfinance, refreshed hourly.
        """
    )
    st.markdown("---")
    if available_sector_tickers:
        st.write("Sectors available: " + ", ".join([SECTORS[t] for t in available_sector_tickers]))
    if missing_sectors:
        st.warning("Missing or insufficient data for: " + ", ".join(missing_sectors))

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
    labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
)
fig_rs.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=30))
st.plotly_chart(fig_rs, use_container_width=True)

# ------------------------------- Returns snapshot --------------------------
rets_1m_df = prices[available_sector_tickers].pct_change(DAYS_1M)
rets_3m_df = prices[available_sector_tickers].pct_change(DAYS_3M)
returns_1m_latest = rets_1m_df.iloc[-1]
returns_3m_latest = rets_3m_df.iloc[-1]

# Build compact snapshot with angle, speed, quadrant, and rank
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

# Sort controls
st.subheader("Rotation snapshot")
sort_col = st.selectbox(
    "Sort table by",
    options=["Speed", "1M", "3M", "Angle (deg)", "Sector", "Quadrant", "Rank (Speed)"],
    index=0
)
ascending = st.toggle("Ascending", value=False)
snap_sorted = snap.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

# Pretty Styler: centered labels, pastel gradients, mid-aligned bars for returns
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
    .bar(subset=["1M","3M"], align="mid")  # intuitive red/green bar around 0
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
