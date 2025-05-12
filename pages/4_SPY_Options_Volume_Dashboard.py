"""
SPY Options Volume Surface — 3-D Explorer (v2)
Author: ChatGPT / Arya Deniz

Overview
────────
This tool reveals **where SPY option traders are concentrating volume** over the
next 30 days. Each point on the 3-D surface is the sum of call + put contracts
at a given moneyness bucket and days-to-expiry (DTE).

Axes
────
* **x-axis** → Moneyness bucket (strike / spot − 1) — buckets stagger by the
  *Moneyness step* slider.
* **y-axis** → Days until option expiration (calendar).
* **z-axis** → Total traded volume (contracts).

Actionable uses
───────────────
* **Pin risk:** Peaks show strike clusters that could magnetise spot into
  expiration.
* **Sentiment skew:** Compare relative heights of OTM calls (+x) versus OTM puts
  (−x).
* **Event premium:** Sudden spikes at single DTE bands flag FOMC/OPEX flows.

How it works
────────────
1. **Source:** Real-time Yahoo Finance option chains for `SPY` (high liquidity).
2. **Filter:** Only strikes with |moneyness| ≤ 10 % and DTE ≤ *Max DTE* slider.
3. **Aggregate:** Volume summed across calls + puts → grouped into equal moneyness
   buckets set by the *Moneyness step* slider.
4. **Visualise:** Plotly 3-D surface with hover read-outs; axes auto-scale based
   on filters so the plot stays legible.

Sidebar Controls
────────────────
* **Max DTE**: cap expiries (3–30 days).
* **Min Volume**: hide illiquid strikes (0–5 000 contracts).
* **Moneyness Step**: bucket width (1–10 %).

Dependencies
────────────
streamlit ≥ 1.30  •  yfinance ≥ 0.2  •  plotly ≥ 5.20  •  pandas ≥ 2.2  •  numpy
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPY Volume Surface 3-D", layout="wide")
st.title("📊 SPY Options Volume — 3-D Surface (v2)")

# ═════════ Sidebar ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    max_dte = st.slider("Max days-to-expiry", 3, 30, value=30)
    min_vol = st.slider("Min contract volume", 0, 5000, value=500, step=50)
    step_pct = st.slider("Moneyness bucket width (%)", 1, 10, value=5)

    st.markdown(__doc__)

UNDERLYING = "SPY"
BAND = 0.10  # ±10 %

# ═════════ Helpers ══════════════════════════════════════════════════════════
@lru_cache(maxsize=4)
def get_spot() -> float:
    return yf.Ticker(UNDERLYING).history(period="1d", interval="1m")["Close"].iloc[-1]

@lru_cache(maxsize=1)
def list_expirations() -> List[str]:
    return yf.Ticker(UNDERLYING).options

@lru_cache(maxsize=32)
def fetch_chain(expiry: str, spot: float) -> pd.DataFrame:
    try:
        oc = yf.Ticker(UNDERLYING).option_chain(expiry)
    except Exception:
        return pd.DataFrame()
    chain = pd.concat([oc.calls, oc.puts], ignore_index=True)
    if chain.empty or "volume" not in chain.columns:
        return pd.DataFrame()
    chain = chain.dropna(subset=["volume"])
    chain["mny"] = chain["strike"] / spot - 1
    dte = (pd.to_datetime(expiry).tz_localize("US/Eastern").date() - datetime.now(timezone.utc).date()).days
    chain["dte"] = dte
    return chain[["volume", "mny", "dte"]]

# ═════════ Data pull ════════════════════════════════════════════════════════
spot = get_spot()
expiries = [e for e in list_expirations() if (pd.to_datetime(e) - datetime.now()).days <= max_dte]

frames = [df for e in expiries if not (df := fetch_chain(e, spot)).empty]

if not frames:
    st.error("No option data available. Try expanding DTE or check connection.")
    st.stop()

data = pd.concat(frames, ignore_index=True)

# Filter by moneyness and min volume
mask = (data["mny"].abs() <= BAND) & (data["volume"] >= min_vol)
data = data[mask]
if data.empty:
    st.warning("No data after filters. Lower Min Volume or widen band.")
    st.stop()

# Bucket moneyness
step = step_pct / 100
bucket = lambda x: np.round(x / step) * step
data["m_bucket"] = data["mny"].apply(bucket).clip(-BAND, BAND)

agg = data.groupby(["dte", "m_bucket"], as_index=False)["volume"].sum()

dtes = np.sort(agg["dte"].unique())
m_buckets = np.arange(-BAND, BAND + 1e-9, step)

vol_mat = np.zeros((len(dtes), len(m_buckets)))
for i, d in enumerate(dtes):
    sub = agg[agg["dte"] == d]
    mapping = dict(zip(sub["m_bucket"], sub["volume"]))
    vol_mat[i] = [mapping.get(m, 0) for m in m_buckets]

# Dynamic aspect ratio: stretch x for wide band, y for many expiries
aspect = dict(x=1.0 + BAND * 4, y=1.0 + len(dtes) / 15, z=0.7)

# ═════════ Plot ═════════════════════════════════════════════════════════════
x_mat = np.tile(m_buckets, (len(dtes), 1))
y_mat = np.tile(dtes[:, None], (1, len(m_buckets)))

fig = go.Figure(go.Surface(
    z=vol_mat,
    x=x_mat,
    y=y_mat,
    colorscale="Viridis",
    showscale=True,
    colorbar=dict(title="Contracts"),
    hovertemplate="DTE: %{y}d<br>Mny: %{x:.1%}<br>Vol: %{z:,}<extra></extra>",
))

fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=1.3, y=1.3, z=0.8)),
        xaxis=dict(title="Moneyness", tickformat=".0%"),
        yaxis=dict(title="Days-to-Expiry"),
        zaxis=dict(title="Volume"),
        aspectratio=aspect,
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Spot SPY {spot:.2f} │ Expiries ≤ {max_dte}d │ Min volume ≥ {min_vol} │ Band ±10 % │ Bucket {step_pct}%"
)
