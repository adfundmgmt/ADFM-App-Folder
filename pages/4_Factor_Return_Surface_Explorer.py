"""
3‑D Factor Return Surface Explorer
Author: ChatGPT / Arya Deniz

Streamlit app that pulls Fama‑French factor returns from the Kenneth French Data
Library and plots a rotatable 3‑D surface: **Factor ➜ Date ➜ Cumulative Return**.

Requirements (add to requirements.txt):
  pandas_datareader>=0.10
  plotly>=5.20.0
  streamlit>=1.30
"""

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas_datareader import data as web

st.set_page_config(page_title="3‑D Factor Surface", layout="wide")

st.title("📊 3‑D Factor Cumulative Return Surface (Fama‑French)")

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    years_back = st.slider("Years of history", 1, 50, value=15)
    freq_label = st.selectbox("Sampling frequency", ["Monthly", "Quarterly", "Annual"], index=0)

freq_map = {"Monthly": "M", "Quarterly": "Q", "Annual": "A"}

# ── Data download & cache ─────────────────────────────────────────────────────
FACTOR_MAP = {
    "Mkt‑RF": "Mkt-RF",
    "SMB": "SMB",
    "HML": "HML",
    "RMW": "RMW",
    "CMA": "CMA",
}

@lru_cache(maxsize=2)
def load_factors() -> pd.DataFrame:
    """Load the 5‑factor 2×3 data (monthly returns, %)."""
    ds = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")
    df = ds[0]  # monthly DataFrame, index = YYYYMM int
    # convert index to datetime at month end
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df[[*FACTOR_MAP.values()]]  # keep only factor columns (drop RF)
    return df / 100  # convert % to decimal

raw = load_factors()

# restrict date range
dt_end = raw.index.max()
dt_start = dt_end - pd.DateOffset(years=years_back)
data = raw.loc[dt_start:dt_end]

# resample
if freq_label != "Monthly":
    data = data.resample(freq_map[freq_label]).sum()

# cumulative returns
data_cum = (1 + data).cumprod() - 1

# ── Prepare meshgrid for surface plot ─────────────────────────────────────────
# x: factor numeric index (0‑4), y: date numeric, z: cumulative return
factors = list(FACTOR_MAP.keys())
x = np.arange(len(factors))
y_dates = data_cum.index
x_mesh, y_mesh = np.meshgrid(x, y_dates)
z = data_cum.values

# convert y axis to POSIX for plotly but keep tick text readable
y_num = y_dates.astype(np.int64) // 10 ** 9

# ── Plotly surface ────────────────────────────────────────────────────────────
fig = go.Figure(
    data=[
        go.Surface(
            z=z,
            x=x_mesh,
            y=y_num[:, None] * np.ones_like(x_mesh),
            colorscale="Viridis",
            showscale=True,
            name="Factor Cumulative Returns",
        )
    ]
)

fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="Factor",
            tickmode="array",
            tickvals=x,
            ticktext=factors,
        ),
        yaxis=dict(
            title="Date",
            tickmode="array",
            tickvals=y_num[:: max(1, len(y_num) // 10)],
            ticktext=[d.strftime("%Y-%m-%d") for d in y_dates][:: max(1, len(y_dates) // 10)],
        ),
        zaxis_title="Cumulative Return",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Data: Kenneth R. French – Fama‑French 5 Factor 2×3 (monthly). Surface shows (1+return) cumprod‑1 for each factor. Rotate, pan, and zoom to explore factor cycles, spreads, and regime shifts."
)
