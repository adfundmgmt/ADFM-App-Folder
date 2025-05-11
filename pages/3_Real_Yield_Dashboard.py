import sys, types
from typing import List, Tuple

import packaging.version
try:
    import distutils.version
except ImportError:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = packaging.version.Version
    _d.version = _dv
    sys.modules.update({"distutils": _d, "distutils.version": _dv})

import pandas as pd
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

REAL_SERIES = "DFII10"
NOM_SERIES = "DGS10"
WIN = 63
EASE_BP = 40
TIGHT_BP = -40
NBSP = "\u00A0"

st.set_page_config(page_title="10-Year Yield Dashboard", layout="wide")
st.title("📈 10-Year Nominal & Real Yield Dashboard")

with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    This dashboard visualizes the **10-year nominal and real Treasury yields** alongside the **inverted 63-day momentum** of real yields.

    ### How to interpret:
    - **Top Panel** shows nominal (blue) vs real (orange) yields.
    - **Bottom Panel** shows 63-day inverted momentum: 
        - Positive momentum (green shading) implies real yields are **falling** → easing signal.
        - Negative momentum (red shading) implies real yields are **rising** → tightening signal.
    - Use this to track macro conditions impacting equity P/E multiples, Fed policy risk, or curve structure.
    """)

# ── Helpers ───────────────────────────────────────────────────────────────

def fred(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = pdr.DataReader(series, "fred", start, end)[series]
    return s.ffill().dropna()

def contiguous(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    segs, st = [], None
    for ts, flag in mask.items():
        if flag and st is None:
            st = ts
        elif not flag and st is not None:
            segs.append((st, ts)); st = None
    if st is not None:
        segs.append((st, mask.index[-1]))
    return segs

# ── Main ──────────────────────────────────────────────────────────────────
opt = st.sidebar.selectbox("Lookback", ["2y", "3y", "5y", "10y"], index=2)
end = pd.Timestamp.today().normalize()
lookback = int(opt[:-1])
start = end - pd.DateOffset(years=lookback)
ext = start - pd.Timedelta(days=WIN*2)

real = fred(REAL_SERIES, ext, end)
nom = fred(NOM_SERIES, ext, end)
real, nom = real.loc[start:], nom.loc[start:]
mom = -(real - real.shift(WIN)) * 100

latest = dict(date=real.index[-1], real=real.iloc[-1], nom=nom.iloc[-1], mom=mom.iloc[-1])
regime = "Easing" if latest["mom"] > EASE_BP else "Tightening" if latest["mom"] < TIGHT_BP else "Neutral"

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.05,
    subplot_titles=("Nominal vs Real 10-Yr Yield", "63-Day Inverted Momentum (bp)")
)

fig.add_trace(go.Scatter(x=nom.index, y=nom, name="Nominal", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=real.index, y=real, name="Real", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
fig.update_yaxes(title="Yield (%)", tickformat=".2f", row=1, col=1)

fig.add_trace(go.Scatter(x=mom.index, y=mom, name="Momentum (bp)", line=dict(color="#d62728", dash="dash", width=1.8)), row=2, col=1)
for th, lab in [(EASE_BP, "+40 bp"), (0, "zero"), (TIGHT_BP, "-40 bp")]:
    fig.add_hline(y=th, line=dict(color="grey", dash="dot", width=1), row=2, col=1,
                  annotation_text=lab, annotation_position="right")

for s, e in contiguous(mom > EASE_BP):
    fig.add_vrect(x0=s, x1=e, fillcolor="rgba(0,128,0,0.12)", line_width=0, row=2, col=1)
for s, e in contiguous(mom < TIGHT_BP):
    fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,0,0,0.12)", line_width=0, row=2, col=1)

fig.update_yaxes(title="bp", tickformat=".0f", row=2, col=1)

metrics = f"Nom {latest['nom']:.2f}%{NBSP*3}|{NBSP*3}Real {latest['real']:.2f}%{NBSP*3}|{NBSP*3}Inv Mom {latest['mom']:+.0f} bp{NBSP*3}|{NBSP*3}{regime}"
st.markdown(f"**{latest['date']:%B %d, %Y}** — {metrics}")

fig.update_layout(
    template="plotly_white",
    height=720,
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(orientation="h", x=0, y=1.1, xanchor="left"),
)
fig.update_xaxes(tickformat="%b-%y", row=2, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)
