import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG & BRANDING ---
st.set_page_config(
    page_title="S&P 500 Sector Breadth & Rotation Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Replace with your fund logo path
# st.image("logo.png", width=140)
st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.markdown("#### Live Sector Leadership & Rotation | Powered by AD Fund Management LP")
st.divider()

# --- SECTOR DEFINITION ---
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
TICKERS = list(SECTORS.keys()) + ["SPY"]

# --- DATES ---
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=370)

# --- DATA ACQUISITION & CACHING ---
@st.cache_data(ttl=3600, show_spinner="Pulling sector ETF data from Yahoo Finance…")
def fetch_prices(tickers, period="1y", interval="1d"):
    try:
        raw = yf.download(tickers, period=period, interval=interval, progress=False, group_by='ticker')
        # Reformat to DataFrame [date x tickers]
        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" in raw.columns.get_level_values(0):
                prices = raw["Adj Close"]
            else:
                prices = raw["Close"]
        else:
            prices = raw["Adj Close"] if "Adj Close" in raw else raw["Close"]
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(1)
        return prices.dropna(how="all")
    except Exception as e:
        st.warning(f"Data fetch failed: {e}")
        return pd.DataFrame()

prices = fetch_prices(TICKERS)
if prices.empty:
    st.error("Failed to load price data for S&P 500 sectors. Please check Yahoo Finance API or your connection.")
    st.stop()

# --- QUICK METRICS TABLE ---
def sector_returns(prices, windows=[1,5,21,63,126,252]):
    # windows = 1D, 1W, 1M, 3M, 6M, 1Y (approx.)
    last = prices.iloc[-1]
    df = {}
    for w in windows:
        df[f'{w}D'] = prices.pct_change(w).iloc[-1]
    out = pd.DataFrame(df)
    out = out.rename(index=lambda t: SECTORS.get(t, t))
    return out

with st.expander("📈 Sector Performance Table (Daily, Weekly, Monthly, Yearly)"):
    returns_tbl = sector_returns(prices)
    returns_fmt = returns_tbl.applymap(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
    st.dataframe(returns_fmt, use_container_width=True, height=380)

# --- RELATIVE STRENGTH CHART ---
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    **Monitor sector leadership, breadth, and risk-on/risk-off themes in real time.**

    - **Sector Relative Strength**: Track ratio of each sector ETF vs. SPY to see which groups are leading.
    - **Sector Rotation Quadrant**: 1M vs 3M returns map to visualize rotation and breadth.
    - **Performance Table**: See all major sector return horizons at a glance.

    *All data from Yahoo Finance (updated hourly).  
    Built by AD Fund Management LP.*
    """)

st.subheader("Sector Relative Strength vs S&P 500 (SPY)")
# Multi-select for easier side-by-side sector comparison
selected_sectors = st.multiselect(
    "Select sector(s) to compare with SPY:",
    options=sorted(SECTORS.keys()),
    default=["XLK", "XLF"],  # Show Technology + Financials as default
    format_func=lambda x: SECTORS[x]
)

relative_strength = prices[SECTORS.keys()].div(prices["SPY"], axis=0)
fig_rs = go.Figure()
for sector in selected_sectors:
    fig_rs.add_trace(go.Scatter(
        x=relative_strength.index, y=relative_strength[sector],
        mode='lines', name=SECTORS[sector], line=dict(width=2)
    ))
fig_rs.update_layout(
    height=400,
    title="Relative Strength (Sector/ SPY, Last 12M)",
    xaxis_title="Date",
    yaxis_title="Ratio",
    margin=dict(l=40, r=30, t=40, b=30),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_rs, use_container_width=True)

# --- SECTOR ROTATION QUADRANT (Scatter with Quadrants) ---
returns_1m = prices.pct_change(21).iloc[-1]
returns_3m = prices.pct_change(63).iloc[-1]
rot_df = pd.DataFrame({
    "Sector": [SECTORS.get(t, t) for t in SECTORS.keys()],
    "1M Return": returns_1m[SECTORS.keys()].values,
    "3M Return": returns_3m[SECTORS.keys()].values,
})
# Mean lines for quadrants
mean_1m = rot_df["1M Return"].mean()
mean_3m = rot_df["3M Return"].mean()
fig_rot = px.scatter(
    rot_df, x="3M Return", y="1M Return", text="Sector",
    color="Sector", color_discrete_sequence=px.colors.qualitative.Dark24,
    title="Sector Rotation Quadrant (1M vs 3M Total Return)"
)
# Add quadrant lines and background
fig_rot.add_shape(type="line", x0=mean_3m, x1=mean_3m, y0=rot_df["1M Return"].min(), y1=rot_df["1M Return"].max(),
    line=dict(color="grey", dash="dash"), layer="below")
fig_rot.add_shape(type="line", y0=mean_1m, y1=mean_1m, x0=rot_df["3M Return"].min(), x1=rot_df["3M Return"].max(),
    line=dict(color="grey", dash="dash"), layer="below")
fig_rot.update_traces(marker=dict(size=22, opacity=0.85, line=dict(width=2, color='black')))
fig_rot.update_traces(textposition='middle center')
fig_rot.update_layout(
    height=500,
    margin=dict(l=40, r=40, t=60, b=40),
    showlegend=False,
    plot_bgcolor="#fafbfc"
)
st.plotly_chart(fig_rot, use_container_width=True)

# --- SECTOR HEATMAP (Breadth snapshot) ---
st.subheader("Sector Breadth Heatmap (Past Month)")
breadth_df = pd.DataFrame({
    "Sector": [SECTORS.get(t, t) for t in SECTORS.keys()],
    "1M Return": returns_1m[SECTORS.keys()] * 100,
})
heatmap = go.Figure(
    data=go.Heatmap(
        z=[breadth_df["1M Return"].values],
        x=breadth_df["Sector"],
        y=["1M Return"],
        colorscale="RdYlGn",
        colorbar=dict(title="% Return"),
        showscale=True
    )
)
heatmap.update_layout(
    height=180,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis=dict(side="top")
)
st.plotly_chart(heatmap, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Built & © 2025 AD Fund Management LP | Data: Yahoo Finance via yfinance | For informational purposes only.")
