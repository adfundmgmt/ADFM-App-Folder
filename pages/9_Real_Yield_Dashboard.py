############################################################
# Built by AD Fund Management LP.
############################################################

import sys, types
from typing import List, Tuple

import matplotlib.pyplot as plt
plt.style.use("default")

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

# ── Constants ─────────────────────────────────────────────
REAL_SERIES = "DFII10"
NOM_SERIES = "DGS10"
DEFAULT_WIN = 63
DEFAULT_EASE_BP = 40
DEFAULT_TIGHT_BP = -40
NBSP = "\u00A0"

# ── Streamlit Setup ──────────────────────────────────────
st.set_page_config(page_title="10-Year Yield Dashboard", layout="wide")
st.title("10-Year Nominal & Real Yield Dashboard")

# ── Sidebar: Controls & Info ─────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This dashboard visualizes the **10-year nominal and real Treasury yields** alongside the **inverted 63-day momentum** of real yields.

    ### How to interpret:
    - **Top Panel**: Nominal (blue) vs real (orange) yields.
    - **Bottom Panel**: 63-day inverted momentum — positive (green) = easing, negative (red) = tightening.
    - Use this to track macro conditions impacting equity P/E multiples, Fed policy, and curve structure.
    """)
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback Window", ["2y", "3y", "5y", "10y"], index=2)
    start_years = int(lookback[:-1])
    win = st.number_input("Momentum Window (days)", min_value=21, max_value=252, value=DEFAULT_WIN, step=1, help="Lookback for momentum calculation. 63 ≈ 3 months trading days.")
    ease_bp = st.number_input("Easing Threshold (bp)", min_value=10, max_value=200, value=DEFAULT_EASE_BP, step=5)
    tight_bp = st.number_input("Tightening Threshold (bp)", min_value=-200, max_value=-10, value=DEFAULT_TIGHT_BP, step=5)
    show_regime = st.checkbox("Show Regime Shading", value=True)
    st.markdown("---")
    st.caption("Yields: FRED DGS10 (nominal), DFII10 (real)")

# ── Data Download Helper ────────────────────────────────
def fred(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill().dropna()
    except Exception as e:
        st.error(f"Failed to load FRED series '{series}': {e}")
        return pd.Series(dtype=float)

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

# ── Main Analysis ───────────────────────────────────────
end = pd.Timestamp.today().normalize()
start = end - pd.DateOffset(years=start_years)
ext = start - pd.Timedelta(days=win*2)

real = fred(REAL_SERIES, ext, end)
nom = fred(NOM_SERIES, ext, end)
if real.empty or nom.empty:
    st.stop()

real, nom = real.loc[start:], nom.loc[start:]
mom = -(real - real.shift(win)) * 100

# ── Regime Detection ────────────────────────────────────
latest = dict(
    date=real.index[-1],
    real=real.iloc[-1],
    nom=nom.iloc[-1],
    mom=mom.iloc[-1]
)
regime = (
    "🟢 Easing" if latest["mom"] > ease_bp else
    "🔴 Tightening" if latest["mom"] < tight_bp else
    "⚪️ Neutral"
)

# ── Metrics Box ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns([2,2,2,3])
c1.metric("Nominal 10Y Yield", f"{latest['nom']:.2f}%", help="Current 10-year nominal U.S. Treasury yield.")
c2.metric("Real 10Y Yield", f"{latest['real']:.2f}%", help="Current 10-year real U.S. Treasury yield (inflation-adjusted).")
c3.metric(f"Inverted {win}-Day Real Yield Momentum", f"{latest['mom']:+.0f} bp", help="-(Current real yield - real yield WIN days ago) × 100")
c4.markdown(f"### {regime}")

# ── Main Chart ──────────────────────────────────────────
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.05,
    subplot_titles=("Nominal vs Real 10-Yr Yield", f"{win}-Day Inverted Momentum (bp)")
)
fig.add_trace(go.Scatter(x=nom.index, y=nom, name="Nominal Yield", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=real.index, y=real, name="Real Yield", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
fig.update_yaxes(title="Yield (%)", tickformat=".2f", row=1, col=1)

fig.add_trace(go.Scatter(x=mom.index, y=mom, name=f"Momentum ({win}d)", line=dict(color="#d62728", dash="dash", width=1.8)), row=2, col=1)
for th, lab in [(ease_bp, f"+{ease_bp} bp"), (0, "zero"), (tight_bp, f"{tight_bp} bp")]:
    fig.add_hline(y=th, line=dict(color="grey", dash="dot", width=1), row=2, col=1,
                  annotation_text=lab, annotation_position="right")

# Regime shading
if show_regime:
    for s, e in contiguous(mom > ease_bp):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(0,128,0,0.12)", line_width=0, row=2, col=1)
    for s, e in contiguous(mom < tight_bp):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,0,0,0.12)", line_width=0, row=2, col=1)

fig.update_yaxes(title="bp", tickformat=".0f", row=2, col=1)

fig.update_layout(
    template="plotly_white",
    height=720,
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
)
fig.update_xaxes(tickformat="%b-%y", row=2, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# ── Download & Methodology ──────────────────────────────
with st.expander("Download Data"):
    df_out = pd.DataFrame({
        "Nominal Yield (%)": nom,
        "Real Yield (%)": real,
        f"Inverted {win}-Day Real Yield Momentum (bp)": mom
    })
    st.download_button(
        "Download Yield Data (CSV)",
        df_out.to_csv(index=True),
        file_name="10yr_yield_dashboard.csv",
        mime="text/csv"
    )

with st.expander("Methodology & Notes", expanded=False):
    st.markdown(f"""
    **Yield Sources**:  
    • Nominal: FRED DGS10  
    • Real: FRED DFII10

    **Inverted Momentum**:  
    − (Current real yield − real yield {win} days ago) × 100

    **Regime Definition**:  
    • 🟢 *Easing*: momentum &gt; {ease_bp}bp  
    • ⚪️ *Neutral*: between  
    • 🔴 *Tightening*: momentum &lt; {tight_bp}bp

    **Tips**:  
    • Green/red shading on the lower panel shows macro "easing" and "tightening" streaks.
    • Use different lookbacks and momentum windows to stress-test current regime shifts.

    """)
st.caption("© 2025 AD Fund Management LP")
