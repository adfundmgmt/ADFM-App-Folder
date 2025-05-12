"""
3‑D Factor Return Surface Explorer (Robust Index Parsing)
Author: ChatGPT / Arya Deniz

Streamlit page that visualises cumulative Fama‑French factor returns as a 3‑D
Plotly surface (Factor ⟶ Date ⟶ Cumulative Return).  This revision fixes the
index‑parsing ValueError by coercing non‑numeric index entries before converting
YYYYMM integers to month‑end timestamps.

Add/ensure in requirements.txt:
  pandas_datareader>=0.10  # Fama‑French loader
  plotly>=5.20.0           # 3‑D surface
  streamlit>=1.30          # web app
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
    """Load monthly Fama‑French 5‑factor returns (decimal)."""
    ds = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")
    df = ds[0]  # monthly returns, index: YYYYMM ints + possible footers

    # ── Robust index‑to‑datetime conversion ───────────────────────────────────
    idx_num = pd.to_numeric(df.index, errors="coerce")  # non‑numeric rows → NaN
    df = df.loc[~idx_num.isna()].copy()
    idx_num = idx_num.dropna().astype(int)
    df.index = pd.to_datetime(idx_num.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)

    df = df[[*FACTOR_MAP.values()]].astype(float) / 100  # % → decimal
    return df

raw = load_factors()

# restrict date range
dt_end = raw.index.max()
dt_start = dt_end - pd.DateOffset(years=years_back)
data = raw.loc[dt_start:dt_end]

# resample (aggregating simple returns)
if freq_label != "Monthly":
    data = ((1 + data).resample(freq_map[freq_label]).prod() - 1)

# cumulative returns
data_cum = (1 + data).cumprod() - 1

# ── Meshgrid for surface ──────────────────────────────────────────────────────
factors = list(FACTOR_MAP.keys())
num_factors = len(factors)

x_vals = np.arange(num_factors)
y_dates = data_cum.index

# repeat x across rows to match y length
a = np.tile(x_vals, (len(y_dates), 1))

z = data_cum.values  # shape (n_dates, n_factors)

y_num = y_dates.astype(np.int64) // 10 ** 9  # POSIX seconds

fig = go.Figure(
    data=[
        go.Surface(
            z=z,
            x=a,
            y=y_num[:, None] * np.ones(num_factors),
            colorscale="Viridis",
            showscale=True,
        )
    ]
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title="Factor", tickmode="array", tickvals=x_vals, ticktext=factors),
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
    "Data: Kenneth R. French ‑ Fama‑French 5‑Factor (2×3). Surface = cumulative compounded returns. Rotate to visualise factor cycles & regime shifts."
)
