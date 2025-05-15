import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ------------- Sector ETF Mapping -------------
SECTOR_ETFS = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "S&P 500": "SPY",
}

SECTOR_LIST = [k for k in SECTOR_ETFS if k != "S&P 500"]

# ------------- Sidebar -------------
st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Sector Rotation Quadrant (Interactive):**  
        Compares short-term (1M) and medium-term (3M) returns for each S&P 500 sector, visualizing leadership and laggards. The sector highlighted below is the one shown on the Relative Strength chart.

        **Sector Relative Strength vs. S&P 500 (Interactive):**  
        Plots the ratio of each sector ETF to the S&P 500 (SPY) for trend and leadership signals.

        **Performance Heatmap & Breadth Table:**  
        See which sectors are outperforming, and what percentage of each is above their 20, 50, and 200-day moving averages.
        """
    )

st.title("S&P 500 Sector Breadth & Rotation Monitor")

# ------------- Data Fetching -------------
@st.cache_data(ttl=60*60*4)
def fetch_sector_data():
    tickers = list(SECTOR_ETFS.values())
    end = datetime.today()
    start = end - timedelta(days=365 * 1.5)  # ~1.5 years for MA calculations
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    df = df.ffill().dropna()
    return df

data = fetch_sector_data()

# ------------- Returns Table for Rotation Quadrant -------------
returns_1m = data.pct_change(21).iloc[-1]  # Approx 1M trading days
returns_3m = data.pct_change(63).iloc[-1]  # Approx 3M trading days
rotation_df = pd.DataFrame({
    "1M Return": returns_1m,
    "3M Return": returns_3m
}).loc[[SECTOR_ETFS[k] for k in SECTOR_LIST]]
rotation_df.index = SECTOR_LIST

# ------------- Interactive Selection -------------
highlight_sector = st.selectbox(
    "Highlight sector (for overlay below):", SECTOR_LIST, index=0, key="sector_highlight"
)

# ------------- Sector Rotation Quadrant (Interactive) -------------
fig_rot = go.Figure()
fig_rot.add_trace(go.Scatter(
    x=rotation_df["3M Return"]*100, y=rotation_df["1M Return"]*100,
    mode='markers+text',
    text=rotation_df.index,
    textposition='top center',
    marker=dict(
        size=[18 if s==highlight_sector else 12 for s in rotation_df.index],
        color=["orange" if s==highlight_sector else "royalblue" for s in rotation_df.index],
        line=dict(width=2, color="black"),
        opacity=0.9
    ),
    name="Sectors"
))
fig_rot.add_shape(type="line", x0=0, x1=0, y0=rotation_df["1M Return"].min()*100, y1=rotation_df["1M Return"].max()*100, line=dict(dash="dash", color="gray"))
fig_rot.add_shape(type="line", x0=rotation_df["3M Return"].min()*100, x1=rotation_df["3M Return"].max()*100, y0=0, y1=0, line=dict(dash="dash", color="gray"))
fig_rot.update_layout(
    title="Sector Rotation Quadrant (Short vs. Medium Term Returns)",
    xaxis_title="3M Return (%)",
    yaxis_title="1M Return (%)",
    height=500, width=900,
    margin=dict(l=30,r=30,t=60,b=30)
)
st.plotly_chart(fig_rot, use_container_width=True)

# ------------- Relative Strength Overlay Chart -------------
spy = data[SECTOR_ETFS["S&P 500"]]
fig_rel = go.Figure()
for sector in SECTOR_LIST:
    rel_strength = data[SECTOR_ETFS[sector]] / spy
    color = "orange" if sector==highlight_sector else "#1f77b4"
    width = 3 if sector==highlight_sector else 1.2
    fig_rel.add_trace(go.Scatter(
        x=rel_strength.index, y=rel_strength,
        mode="lines",
        line=dict(color=color, width=width),
        name=sector,
        opacity=0.92 if sector==highlight_sector else 0.5
    ))
fig_rel.update_layout(
    title=f"{highlight_sector} Highlighted — Sector Relative Strength vs. S&P 500",
    xaxis_title="Date",
    yaxis_title="Sector / SPY (Ratio)",
    height=440,
    width=900,
    showlegend=True,
    margin=dict(l=30,r=30,t=60,b=30)
)
st.plotly_chart(fig_rel, use_container_width=True)

# ------------- Performance Heatmap & Breadth Table -------------
st.subheader("Sector Performance & Breadth Snapshot")
perf_window = {"1W": 5, "1M": 21, "3M": 63, "YTD": data.index[-1].year - data.index[0].year + 1}
perf = pd.DataFrame(index=SECTOR_LIST)
for k, n in zip(["1W", "1M", "3M"], [5,21,63]):
    perf[k] = data[[SECTOR_ETFS[x] for x in SECTOR_LIST]].pct_change(n).iloc[-1]*100
perf["YTD"] = (
    data[[SECTOR_ETFS[x] for x in SECTOR_LIST]].iloc[-1].values /
    data[[SECTOR_ETFS[x] for x in SECTOR_LIST]].loc[data.index[-1].replace(month=1, day=2)].values - 1
) * 100

ma_windows = [20, 50, 200]
breadth = {}
for win in ma_windows:
    breadth[f"Above {win}D"] = [
        100 * (data[SECTOR_ETFS[sector]].iloc[-1] > data[SECTOR_ETFS[sector]].rolling(win).mean().iloc[-1])
        for sector in SECTOR_LIST
    ]

# Merge all
heatmap_df = perf.copy()
for key in breadth:
    heatmap_df[key] = breadth[key]

st.dataframe(
    heatmap_df.style.format("{:.2f}").background_gradient(cmap="RdYlGn", subset=["1W", "1M", "3M", "YTD"]),
    use_container_width=True,
    height=440
)

st.caption("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

