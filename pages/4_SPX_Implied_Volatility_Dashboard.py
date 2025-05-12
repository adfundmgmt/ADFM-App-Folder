"""
SPX Implied Volatility Surface — 3‑D Explorer (v1)
Author: ChatGPT / Arya Deniz

This Streamlit page pulls option‑chain data for the S&P 500 index (ticker
`^SPX`) from Yahoo Finance via *yfinance* and renders a 3‑D implied‑volatility
surface:

* **x‑axis**  → Moneyness (strike / spot − 1)
* **y‑axis**  → Days‑to‑expiry
* **z‑axis**  → Implied volatility (σ)

Why it’s actionable
──────────────────
• Spot smile/skew differences between near‑dated and far‑dated maturity wings.  
• Identify term‑structure kinks (e.g., event risk priced into a single expiry).  
• Compare surface shape across dates (save a screenshot, rerun tomorrow).

Dependencies
────────────
› streamlit ≥ 1.30  • yfinance ≥ 0.2  • plotly ≥ 5.20  • pandas ≥ 2.2  • numpy
"""

from datetime import datetime
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SPX Volatility Surface", layout="wide")
st.title("📈 SPX Implied Volatility — 3‑D Surface")

# ═════════ Sidebar Controls ═════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    max_expiries = st.slider("Number of expiries", 3, 15, value=8)
    step = st.slider("Moneyness step (%)", 2, 10, value=5)
    width = st.slider("Moneyness band (±% from ATM)", 10, 40, value=20)

    st.markdown("""
### How to read
* **X** = (strike / spot − 1). 0 = ATM; 0.1 ≈ 10 % OTM calls; −0.1 ≈ 10 % OTM puts.
* **Y** = calendar days until option expiry.
* **Z** = Black‑Scholes implied volatility (σ).

Rotate, zoom, and hover to inspect skew and term‑structure.
""")

# ═════════ Data Helpers ═════════════════════════════════════════════════════
@lru_cache(maxsize=2)
def get_spx_price() -> float:
    return yf.download("^SPX", period="1d", interval="1m", progress=False)["Close"].iloc[-1]

@lru_cache(maxsize=1)
def list_expirations() -> List[str]:
    return yf.Ticker("^SPX").options

@lru_cache(maxsize=16)
def fetch_chain(expiry: str) -> pd.DataFrame:
    """Return options chain with implied vols + moneyness."""
    tk = yf.Ticker("^SPX")
    opt = tk.option_chain(expiry)
    spot = get_spx_price()
    chain = pd.concat([opt.calls, opt.puts], ignore_index=True)
    chain = chain.dropna(subset=["impliedVolatility"])
    chain["moneyness"] = chain["strike"] / spot - 1
    chain["dte"] = (pd.to_datetime(expiry) - datetime.utcnow()).days
    return chain

# ═════════ Build Surface Grid ═══════════════════════════════════════════════
spot_px = get_spx_price()
all_exps = list_expirations()[:max_expiries]

frames: list[pd.DataFrame] = []
for exp in all_exps:
    try:
        df = fetch_chain(exp)
        if not df.empty:
            frames.append(df)
    except Exception:
        # ignore expiries that fail to download / parse
        continue

if not frames:
    st.error("No option data available for the selected expiries. Try widening the date range or come back later.")
    st.stop()

raw = pd.concat(frames, ignore_index=True)

# filter by moneyness band
band = width / 100
raw = raw[(raw["moneyness"].abs() <= band)]

# define moneyness grid
step_frac = step / 100
m_grid = np.arange(-band, band + 1e-6, step_frac)

dte_vals = sorted(raw["dte"].unique())

# pivot to rectangular matrix using nearest available IV per cell
pivot = np.full((len(dte_vals), len(m_grid)), np.nan)

for i, d in enumerate(dte_vals):
    subset = raw[raw["dte"] == d]
    if subset.empty:
        continue
    for j, m in enumerate(m_grid):
        # pick strike with moneyness closest to grid point
        row = subset.iloc[(subset["moneyness"] - m).abs().argsort()[:1]]
        if not row.empty:
            pivot[i, j] = row["impliedVolatility"].values[0]

# remove rows without any data
mask = ~np.isnan(pivot).all(axis=1)
pivot = pivot[mask]
dte_vals = np.array(dte_vals)[mask]

if pivot.size == 0:
    st.error("No option data after filtering. Try widening the moneyness band or increasing expiries.")
    st.stop()

# ═════════ Plotly Surface ═══════════════════════════════════════════════════
x_mat = np.tile(m_grid, (len(dte_vals), 1))
y_mat = np.tile(dte_vals[:, None], (1, len(m_grid)))

fig = go.Figure(
    go.Surface(
        z=pivot * 100,  # as %
        x=x_mat,
        y=y_mat,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="IV (%)"),
        hovertemplate="DTE: %{y} days<br>Moneyness: %{x:.2%}<br>IV: %{z:.2f}%<extra></extra>",
    )
)

fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        xaxis=dict(title="Moneyness (strike / spot − 1)", tickformat=".0%"),
        yaxis=dict(title="Days to Expiry"),
        zaxis=dict(title="Implied Vol (%)"),
        aspectratio=dict(x=1.3, y=1.6, z=0.7),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Spot: {spot_px:.2f}  │  Expiries shown: {len(dte_vals)}  │  Data: Yahoo Finance option chain (^SPX)."
)
