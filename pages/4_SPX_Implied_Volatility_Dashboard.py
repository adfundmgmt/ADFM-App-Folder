"""
SPX / SPY Implied Volatility Surface — 3-D Explorer (v2)
Author: ChatGPT / Arya Deniz

Reliable IV-surface builder with automatic fallback:
* Tries index options (ticker `^SPX`) first.  
* If Yahoo doesn’t serve data, falls back to ETF `SPY`.

Axes
────
* **x-axis**  → Moneyness (strike / spot − 1)  
* **y-axis**  → Days-to-expiry (calendar)  
* **z-axis**  → Implied volatility (σ, %)

Sidebar tweaks
──────────────
* Underlying ticker displayed so you know which dataset you’re seeing.  
* Warning badge if we had to fall back to `SPY`.

Dependencies
────────────
streamlit ≥ 1.30 • yfinance ≥ 0.2 • plotly ≥ 5.20 • pandas ≥ 2.2 • numpy
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Volatility Surface 3-D", layout="wide")
st.title("📈 SPX / SPY Implied Volatility — 3-D Surface (v2)")

# ═════════ Sidebar Controls ═════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    max_expiries = st.slider("Number of expiries", 3, 20, value=10)
    step = st.slider("Moneyness step (%)", 1, 10, value=5)
    width = st.slider("Moneyness band (±% from ATM)", 5, 50, value=20)

    st.markdown("""
### How to read
* **X** = strike/spot − 1 (0 = ATM). Positive → OTM calls; negative → OTM puts.
* **Y** = calendar days to expiry.
* **Z** = Black-Scholes implied volatility (%).

Rotate & hover to inspect skew and term-structure.
""")

PRIMARY_UNDERLYING = "^SPX"
FALLBACK_UNDERLYING = "SPY"

# ═════════ Data Helpers ═════════════════════════════════════════════════════
@lru_cache(maxsize=4)
def get_spot(ticker: str) -> float:
    return yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].iloc[-1]

@lru_cache(maxsize=2)
def list_expirations(ticker: str) -> List[str]:
    try:
        return yf.Ticker(ticker).options
    except Exception:
        return []

def choose_underlying() -> str:
    """Return a ticker symbol that has option chains available."""
    for tk in [PRIMARY_UNDERLYING, FALLBACK_UNDERLYING]:
        if list_expirations(tk):
            return tk
    return FALLBACK_UNDERLYING  # default

UNDERLYING = choose_underlying()

@lru_cache(maxsize=32)
def fetch_chain(ticker: str, expiry: str, spot: float) -> pd.DataFrame:
    try:
        opt = yf.Ticker(ticker).option_chain(expiry)
    except Exception:
        return pd.DataFrame()
    chain = pd.concat([opt.calls, opt.puts], ignore_index=True)
    if "impliedVolatility" not in chain.columns:
        return pd.DataFrame()
    chain = chain.dropna(subset=["impliedVolatility"])
    if chain.empty:
        return chain
    chain["moneyness"] = chain["strike"] / spot - 1
    dte = (pd.to_datetime(expiry).tz_localize("US/Eastern").date() - datetime.now(timezone.utc).date()).days
    chain["dte"] = dte
    return chain

# ═════════ Pull chains ══════════════════════════════════════════════════════
spot_px = get_spot(UNDERLYING)
all_exps = list_expirations(UNDERLYING)[:max_expiries]

frames: list[pd.DataFrame] = []
for exp in all_exps:
    df = fetch_chain(UNDERLYING, exp, spot_px)
    if not df.empty:
        frames.append(df)

if not frames:
    st.error("Yahoo did not return any option IV data. Please try again later or reduce filters.")
    st.stop()

raw = pd.concat(frames, ignore_index=True)

# ═════════ Grid construction ════════════════════════════════════════════════
band = width / 100
raw = raw[raw["moneyness"].abs() <= band]

if raw.empty:
    st.warning("No strikes within the selected moneyness band. Expand the band.")
    st.stop()

step_frac = step / 100
m_grid = np.arange(-band, band + 1e-9, step_frac)

dte_vals = sorted(raw["dte"].unique())

pivot = np.full((len(dte_vals), len(m_grid)), np.nan)
for i, d in enumerate(dte_vals):
    sub = raw[raw["dte"] == d]
    if sub.empty:
        continue
    nearest_idx = np.abs(sub["moneyness"].to_numpy()[:, None] - m_grid[None, :]).argmin(axis=0)
    pivot[i, :] = sub.iloc[nearest_idx]["impliedVolatility"].to_numpy()

mask = ~np.isnan(pivot).all(axis=1)
if not mask.any():
    st.error("Unable to build surface — data too sparse. Adjust controls.")
    st.stop()

pivot = pivot[mask]
dte_vals = np.array(dte_vals)[mask]

# ═════════ Plot surface ═════════════════════════════════════════════════════
x_mat = np.tile(m_grid, (len(dte_vals), 1))
y_mat = np.tile(dte_vals[:, None], (1, len(m_grid)))

fig = go.Figure(go.Surface(
    z=pivot * 100,
    x=x_mat,
    y=y_mat,
    colorscale="Viridis",
    showscale=True,
    colorbar=dict(title="IV (%)"),
    hovertemplate="ΔT: %{y}d<br>Mny: %{x:.1%}<br>IV: %{z:.2f}%<extra></extra>",
))

fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        xaxis=dict(title="Moneyness", tickformat=".0%"),
        yaxis=dict(title="Days-to-Expiry"),
        zaxis=dict(title="Implied Vol (%)"),
        aspectratio=dict(x=1.2, y=1.6, z=0.7),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

badge = "Using fallback ETF SPY" if UNDERLYING != PRIMARY_UNDERLYING else "Using index options ^SPX"
st.caption(f"Spot ({UNDERLYING}): {spot_px:.2f} │ {badge} │ Expiries: {len(dte_vals)} | Data: Yahoo Finance")
