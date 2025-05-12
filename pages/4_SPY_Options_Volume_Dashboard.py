"""
SPY Options Volume Surface — 3-D Explorer (v1)
Author: ChatGPT / Arya Deniz

What this tool does
───────────────────
Visualises **where traders are concentrating volume** across the SPY options
surface over the next 30 days:

* Filters strikes **±10 %** around spot.
* Pulls all expiries within a configurable **max-DTE** (default 30 days).
* Aggregates **call + put contract volume** at each (strike, expiry).
* Renders a rotatable 3-D bar surface:
  * **x-axis** → Moneyness (strike/spot − 1)
  * **y-axis** → Days-to-expiry
  * **z-axis** → Volume (number of contracts)

Why it’s actionable
──────────────────
▶ Spot “gravity” zones where pinning risk is highest.  
▶ Identify near-dated expiries with unusually heavy OTM put/call demand.  
▶ Map volume shifts day-to-day to track directional conviction.

How it works
────────────
1. **Data source:** Yahoo Finance option chain for `SPY` (deepest liquidity).  
2. **Processing:** Groups by expiry & strike, sums call + put volume.  
3. **Filtering:** Keeps rows where |moneyness| ≤ 10 % **and** DTE ≤ slider value
   **and** volume ≥ slider threshold.  
4. **Plot:** 3-D Plotly bars — hover for exact contracts.

Controls
────────
* **Max Days-to-Expiry** → Upper DTE limit (3-30 days).  
* **Min Volume** → Hide low-liquidity strikes.  
* **Moneyness step** → Grid granularity (1-10 %).

Dependencies
────────────
streamlit ≥ 1.30 • yfinance ≥ 0.2 • plotly ≥ 5.20 • pandas ≥ 2.2 • numpy
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SPY Volume Surface 3-D", layout="wide")
st.title("📊 SPY Options Volume — 3-D Surface")

# ═════════ Sidebar ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    max_dte = st.slider("Max days-to-expiry", 3, 30, value=30)
    min_vol = st.slider("Min contract volume", 0, 5000, value=100, step=50)
    step = st.slider("Moneyness step (%)", 1, 10, value=5)

    st.markdown(st.session_state.get("__doc_markdown__", ""))

UNDERLYING = "SPY"
MONEYNESS_BAND = 0.10  # ±10 %

# ═════════ Data helpers ═════════════════════════════════════════════════════
@lru_cache(maxsize=4)
def get_spot() -> float:
    return yf.Ticker(UNDERLYING).history(period="1d", interval="1m")["Close"].iloc[-1]

@lru_cache(maxsize=1)
def list_expirations() -> List[str]:
    return yf.Ticker(UNDERLYING).options

@lru_cache(maxsize=32)
def fetch_chain(expiry: str, spot: float) -> pd.DataFrame:
    try:
        opt = yf.Ticker(UNDERLYING).option_chain(expiry)
    except Exception:
        return pd.DataFrame()
    chain = pd.concat([opt.calls, opt.puts], ignore_index=True)
    if "volume" not in chain.columns:
        return pd.DataFrame()
    chain = chain.dropna(subset=["volume"])
    if chain.empty:
        return chain
    chain["moneyness"] = chain["strike"] / spot - 1
    dte = (pd.to_datetime(expiry).tz_localize("US/Eastern").date() - datetime.now(timezone.utc).date()).days
    chain["dte"] = dte
    return chain[["strike", "volume", "moneyness", "dte"]]

# ═════════ Gather chains ════════════════════════════════════════════════════
spot_px = get_spot()
exps = [e for e in list_expirations() if (pd.to_datetime(e) - datetime.now()).days <= max_dte]

frames = []
for exp in exps:
    df = fetch_chain(exp, spot_px)
    if not df.empty:
        frames.append(df)

if not frames:
    st.error("No option data for the selected parameters. Try expanding the DTE window.")
    st.stop()

raw = pd.concat(frames, ignore_index=True)

# ═════════ Filtering ════════════════════════════════════════════════════════
raw = raw[(raw["moneyness"].abs() <= MONEYNESS_BAND) & (raw["volume"] >= min_vol)]

if raw.empty:
    st.warning("No strikes meet the moneyness/volume filters. Lower the min-volume threshold.")
    st.stop()

# Round moneyness to grid
step_frac = step / 100
raw["m_grid"] = (np.round(raw["moneyness"] / step_frac) * step_frac).clip(-MONEYNESS_BAND, MONEYNESS_BAND)

# Aggregate volume per (dte, m_grid)
agg = raw.groupby(["dte", "m_grid"], as_index=False)["volume"].sum()

dte_vals = sorted(agg["dte"].unique())
mono_vals = np.arange(-MONEYNESS_BAND, MONEYNESS_BAND + 1e-9, step_frac)

vol_matrix = np.full((len(dte_vals), len(mono_vals)), np.nan)
for i, d in enumerate(dte_vals):
    sub = agg[agg["dte"] == d]
    idx = {m: v for m, v in zip(sub["m_grid"], sub["volume"]) }
    for j, m in enumerate(mono_vals):
        vol_matrix[i, j] = idx.get(m, np.nan)

# Replace NaN with 0 for volume visualization
vol_matrix = np.nan_to_num(vol_matrix, nan=0.0)

# ═════════ Plot 3-D surface (bars) ══════════════════════════════════════════
x_mat = np.tile(mono_vals, (len(dte_vals), 1))
y_mat = np.tile(np.array(dte_vals)[:, None], (1, len(mono_vals)))

fig = go.Figure(go.Surface(
    z=vol_matrix,
    x=x_mat,
    y=y_mat,
    colorscale="Viridis",
    showscale=True,
    colorbar=dict(title="Contracts"),
    hovertemplate="DTE: %{y}d<br>Mny: %{x:.1%}<br>Vol: %{z}<extra></extra>",
))

fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        xaxis=dict(title="Moneyness", tickformat=".0%"),
        yaxis=dict(title="Days-to-Expiry"),
        zaxis=dict(title="Volume (contracts)"),
        aspectratio=dict(x=1.3, y=1.5, z=0.6),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Spot SPY: {spot_px:.2f} │ Expiries ≤ {max_dte}d │ Volume ≥ {min_vol} contracts │ Data: Yahoo Finance")
