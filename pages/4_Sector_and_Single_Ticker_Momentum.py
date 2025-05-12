"""
Monthly Seasonality Surface — 3-D Explorer (v1)
Author: ChatGPT / Arya Deniz

Purpose
───────
Visualise a security’s **monthly return seasonality** across years in an
interactive 3-D surface:

* **x-axis**  → Month (Jan … Dec)
* **y-axis**  → Calendar year
* **z-axis**  → Total return for that month (%)

Actionable uses:
• Spot persistent positive/negative months (fade or front-run flows)  
• Identify regime shifts when seasonality flips sign  
• Compare assets quickly — just change the ticker.

Dependencies (add to `requirements.txt`)
• streamlit>=1.30  
• yfinance>=0.2  
• plotly>=5.20  
• pandas>=2.2
"""

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Monthly Seasonality 3-D", layout="wide")
st.title("📅 Monthly Seasonality — 3-D Surface")

# ═════════ Sidebar ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    first_year = st.number_input("Start year", min_value=1980, max_value=2024, value=2000)
    metric: Literal["Total Return", "Hit Rate"] = st.radio(
        "Metric", ["Total Return", "Hit Rate"], index=0,
        help="Hit Rate = % of years the month was positive"
    )

    st.markdown("""
### How to read
* **X-axis** = Month (1-12)
* **Y-axis** = Year
* **Z-axis** = Monthly % return (or hit-rate 0-1)

Rotate to view amplitude; hover for exact numbers.
""")

# ═════════ Data ─ download & transform ═══════════════════════════════════════
@lru_cache(maxsize=8)
def load_price(sym: str) -> pd.Series:
    df = yf.download(sym, period="max", auto_adjust=True, progress=False)["Close"]
    return df.dropna()

prices = load_price(ticker)
prices = prices[prices.index.year >= first_year]

if prices.empty:
    st.error("No data available for chosen start year.")
    st.stop()

# monthly end prices & pct change
mclose = prices.resample("M").last()
ret = mclose.pct_change().dropna()

# pivot: rows=year, cols=month
ret_df = ret.to_frame("ret")
ret_df["Year"] = ret_df.index.year
ret_df["Month"] = ret_df.index.month
pivot = ret_df.pivot(index="Year", columns="Month", values="ret")

# ensure full month columns 1-12
pivot = pivot.reindex(columns=range(1, 13))

if metric == "Hit Rate":
    hit = pivot.apply(lambda col: (col > 0).astype(int))
    z_mat = hit.cumsum(axis=0) / (np.arange(len(hit))[:, None] + 1)
else:
    z_mat = pivot.values  # raw monthly returns

years = pivot.index.to_numpy()
months = np.arange(1, 13)

# 3-D grid matrices
x_mat = np.tile(months, (len(years), 1))
y_mat = np.tile(years[:, None], (1, len(months)))

# ═════════ Plot ══════════════════════════════════════════════════════════════
color_title = "Hit Rate" if metric == "Hit Rate" else "Return (%)"
fig = go.Figure(
    go.Surface(
        z=z_mat if metric == "Hit Rate" else z_mat * 100,
        x=x_mat,
        y=y_mat,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=color_title),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>%s: %{z:.2f}%s<extra></extra>" % (
            "Hit" if metric == "Hit Rate" else "Return",
            "" if metric == "Hit Rate" else "%",
        ),
    )
)

fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        xaxis=dict(title="Month", tickmode="array", tickvals=months, ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
        yaxis=dict(title="Year"),
        zaxis=dict(title=color_title),
        aspectratio=dict(x=1.2, y=2.0, z=0.6),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Ticker: {ticker} │ Metric: {metric} │ Data: Yahoo Finance adjusted close")
