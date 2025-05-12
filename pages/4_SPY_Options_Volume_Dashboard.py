"""
SPY Options Volume Surface — 3‑D Explorer (v3)
Author: ChatGPT / Arya Deniz

Upgrade highlights
──────────────────
▸ **Log‑Scale Toggle** — flip between raw contracts and `log10(volume+1)` so
   low‑/mid‑tier pockets stop being dwarfed by monster strikes.
▸ **3‑D Bars** — switch from a smooth surface to upright bars for crisper
   pin‑risk visual (optional).
▸ **Dynamic Z cap** — auto‑clips extreme spikes to keep the plot readable.
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SPY Volume Surface 3‑D", layout="wide")
st.title("📊 SPY Options Volume — 3‑D Surface (v3)")

# ═════════ Sidebar ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    max_dte = st.slider("Max days‑to‑expiry", 3, 30, value=30)
    min_vol = st.slider("Min contract volume", 0, 5000, value=200, step=50)
    step_pct = st.slider("Moneyness bucket width (%)", 1, 10, value=2)

    z_scale: Literal["Raw", "Log10"] = st.radio("Z‑scale", ["Raw", "Log10"], index=1,
                                               help="Log scale prevents single strikes from dwarfing others")
    chart_type: Literal["Surface", "Bars"] = st.radio("Chart type", ["Surface", "Bars"], index=1)

    st.markdown("---")
    st.markdown("**Tip:** If the plot still looks flat, tighten *Min volume* or shrink the *bucket width*.")

UNDERLYING = "SPY"
BAND = 0.10

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
    chain = chain.dropna(subset=["volume"])
    if chain.empty:
        return chain
    chain["mny"] = chain["strike"] / spot - 1
    dte = (pd.to_datetime(expiry).tz_localize("US/Eastern").date() - datetime.now(timezone.utc).date()).days
    chain["dte"] = dte
    return chain[["volume", "mny", "dte"]]

# ═════════ Data pull ════════════════════════════════════════════════════════
spot = get_spot()
exps = [e for e in list_expirations() if (pd.to_datetime(e) - datetime.now()).days <= max_dte]
frames = [df for e in exps if not (df := fetch_chain(e, spot)).empty]

if not frames:
    st.error("No chains returned. Try later or expand DTE.")
    st.stop()

data = pd.concat(frames)

mask = (data["mny"].abs() <= BAND) & (data["volume"] >= min_vol)
data = data[mask]
if data.empty:
    st.warning("No data after filters. Relax volume threshold or widen band.")
    st.stop()

step = step_pct / 100
bucket = lambda x: np.round(x / step) * step

data["m_bucket"] = data["mny"].apply(bucket).clip(-BAND, BAND)
agg = data.groupby(["dte", "m_bucket"], as_index=False)["volume"].sum()

dtes = np.sort(agg["dte"].unique())
mbs = np.arange(-BAND, BAND + 1e-9, step)

vol = np.zeros((len(dtes), len(mbs)))
for i, d in enumerate(dtes):
    sub = agg[agg["dte"] == d]
    vol_map = dict(zip(sub["m_bucket"], sub["volume"]))
    vol[i] = [vol_map.get(m, 0) for m in mbs]

# ── Z‑scale and clipping ────────────────────────────────────────────────────
if z_scale == "Log10":
    z_data = np.log10(vol + 1)
    z_title = "log10(Vol + 1)"
else:
    z_data = vol
    z_title = "Volume"

# cap extreme spikes to 95th percentile to improve contrast
cap = np.percentile(z_data, 95)
if cap > 0:
    z_data = np.clip(z_data, 0, cap)

# ═════════ Plot ═════════════════════════════════════════════════════════════
if chart_type == "Surface":
    x_mat = np.tile(mbs, (len(dtes), 1))
    y_mat = np.tile(dtes[:, None], (1, len(mbs)))
    fig = go.Figure(go.Surface(z=z_data, x=x_mat, y=y_mat, colorscale="Viridis", showscale=True,
                               colorbar=dict(title=z_title),
                               hovertemplate="DTE: %{y}d<br>Mny: %{x:.1%}<br>Vol: %{customdata:,}<extra></extra>",
                               customdata=vol))
else:  # Bars (use Mesh3d impostor via scatter3d bar)
    xs, ys, zs, sizes = [], [], [], []
    for i, d in enumerate(dtes):
        for j, m in enumerate(mbs):
            v = z_data[i, j]
            if v <= 0:
                continue
            xs.append(m)
            ys.append(d)
            zs.append(0)
            sizes.append(v)
    fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                                 marker=dict(size=np.array(sizes)/np.max(sizes)*20, color=sizes, colorscale='Viridis', showscale=True, colorbar=dict(title=z_title)),
                                 hovertemplate="DTE: %{y}d<br>Mny: %{x:.1%}<br>Scaled Vol: %{marker.size:.1f}<extra></extra>"))

fig.update_layout(
    scene=dict(
        xaxis=dict(title="Moneyness", tickformat=".0%"),
        yaxis=dict(title="Days‑to‑Expiry"),
        zaxis=dict(title=z_title if chart_type=="Surface" else ""),
        camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
        aspectratio=dict(x=1.2, y=1 + len(dtes)/15, z=0.6),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Spot {spot:.2f} │ Exp ≤ {max_dte}d │ Min vol ≥ {min_vol} │ Buck {step_pct}% │ Z = {z_title}")
