############################################################
# Built by AD Fund Management LP. Macro Overlays: Automated.
############################################################

import sys, types
from typing import List, Tuple
import pandas as pd
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

from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests
from bs4 import BeautifulSoup

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
    This dashboard visualizes the **10-year nominal and real Treasury yields** with macro overlays.
    - **Top Panel**: Nominal (blue) vs real (orange) yields.
    - **Bottom Panel**: Inverted 63-day momentum — positive (green) = easing, negative (red) = tightening.
    """)
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback Window", ["2y", "3y", "5y", "10y"], index=2)
    start_years = int(lookback[:-1])
    win = st.number_input("Momentum Window (days)", min_value=21, max_value=252, value=DEFAULT_WIN, step=1)
    ease_bp = st.number_input("Easing Threshold (bp)", min_value=10, max_value=200, value=DEFAULT_EASE_BP, step=5)
    tight_bp = st.number_input("Tightening Threshold (bp)", min_value=-200, max_value=-10, value=DEFAULT_TIGHT_BP, step=5)
    show_regime = st.checkbox("Show Regime Shading", value=True)
    show_spread = st.checkbox("Show Yield Spread (Nominal - Real)", value=False)
    st.markdown("---")
    st.header("Macro Overlays")
    show_fomc = st.checkbox("Show FOMC Meeting Dates", value=True)
    show_cpi = st.checkbox("Show CPI Print Dates", value=False)
    show_recession = st.checkbox("Show Recession Bands", value=True)
    st.markdown("---")
    st.caption("Yields: FRED DGS10 (nominal), DFII10 (real)")

# ── Data Download Helper ────────────────────────────────
def fred(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        s = s.ffill().dropna()
        return s
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

# ── Macro Data Overlays (Automated Scraping) ────────────
@st.cache_data(ttl=86400, show_spinner="Fetching macro overlay dates…")
def fetch_fomc_dates():
    # Official: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        dates = []
        for td in soup.select("td.eventdate"):
            dt = pd.to_datetime(td.text.strip().split()[0], errors='coerce')
            if pd.notnull(dt):
                dates.append(dt)
        return sorted(list(set(dates)))
    except Exception as e:
        st.warning(f"Could not fetch FOMC dates: {e}")
        return []

@st.cache_data(ttl=86400, show_spinner="Fetching CPI print dates…")
def fetch_cpi_dates():
    # Official: https://www.bls.gov/schedule/news_release/cpi.htm
    url = "https://www.bls.gov/schedule/news_release/cpi.htm"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        dates = []
        for td in soup.select("table.nrtable tbody tr td"):
            text = td.text.strip()
            if len(text) >= 6 and text[0:3].isalpha():
                try:
                    dt = pd.to_datetime(text, errors='coerce')
                    if pd.notnull(dt):
                        dates.append(dt)
                except:
                    continue
        return sorted(list(set(dates)))
    except Exception as e:
        st.warning(f"Could not fetch CPI dates: {e}")
        return []

@st.cache_data(ttl=86400, show_spinner="Fetching NBER recession dates…")
def fetch_recession_bands():
    # Official NBER: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
    url = "https://www.nber.org/releases.csv"
    try:
        df = pd.read_csv(url)
        starts = pd.to_datetime(df['Peak'], errors='coerce')
        ends = pd.to_datetime(df['Trough'], errors='coerce')
        bands = [(start, end) for start, end in zip(starts, ends) if pd.notnull(start) and pd.notnull(end)]
        return bands
    except Exception as e:
        # Fallback to last two recessions
        st.warning(f"Could not fetch NBER recessions: {e}")
        return [
            (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30")),
            (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"))
        ]

# ── Main Analysis ───────────────────────────────────────
end = pd.Timestamp.today().normalize()
start = end - pd.DateOffset(years=start_years)
ext = start - pd.Timedelta(days=win*2)

real = fred(REAL_SERIES, ext, end)
nom = fred(NOM_SERIES, ext, end)

if real.empty or nom.empty:
    st.warning("Data download failed or series is empty. Please check FRED or try again later.")
    st.stop()
if abs((real.index[-1] - end).days) > 3:
    st.warning(f"Latest real yield data from FRED is stale (last: {real.index[-1].date()}).")

# --- Align Index ---
df = pd.DataFrame({"Nominal": nom, "Real": real}).dropna()
df = df[df.index >= start]
if df.empty:
    st.warning("No overlapping data in window. Check FRED.")
    st.stop()

mom = -(df['Real'] - df['Real'].shift(win)) * 100
mom = mom.dropna()
df = df.loc[mom.index]
df["Spread"] = df["Nominal"] - df["Real"]

# --- Regime Detection ---
latest = dict(
    date=df.index[-1],
    real=df['Real'].iloc[-1],
    nom=df['Nominal'].iloc[-1],
    spread=df['Spread'].iloc[-1],
    mom=mom.iloc[-1],
    mom_prev=mom.iloc[-2] if len(mom) > 1 else None
)
regime = (
    "🟢 Easing" if latest["mom"] > ease_bp else
    "🔴 Tightening" if latest["mom"] < tight_bp else
    "⚪️ Neutral"
)

# ── Metrics Box ─────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2,2,2,2,3])
c1.metric("Nominal 10Y Yield", f"{latest['nom']:.2f}%", f"{df['Nominal'].iloc[-1] - df['Nominal'].iloc[-2]:+.2f}%", help="Current vs prior day.")
c2.metric("Real 10Y Yield", f"{latest['real']:.2f}%", f"{df['Real'].iloc[-1] - df['Real'].iloc[-2]:+.2f}%", help="Current vs prior day.")
c3.metric(f"Inverted {win}-Day Real Yield Momentum", f"{latest['mom']:+.0f} bp", f"{(latest['mom']-latest['mom_prev']):+.0f} bp" if latest['mom_prev'] is not None else "", help="-(Current real yield - real yield WIN days ago) × 100")
c4.metric("10Y Yield Spread", f"{latest['spread']:.2f}%", help="Nominal minus real yield")
c5.markdown(f"<h3>{regime}</h3><small>as of {latest['date'].date()}</small>", unsafe_allow_html=True)

# ── Chart Setup ─────────────────────────────────────────
nrows = 3 if show_spread else 2
fig = make_subplots(
    rows=nrows,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.40, 0.35] + ([0.25] if show_spread else []),
    vertical_spacing=0.05,
    subplot_titles=(
        "Nominal vs Real 10-Yr Yield",
        f"{win}-Day Inverted Real Yield Momentum (bp)",
        "10Y Yield Spread (Nominal - Real)" if show_spread else None,
    )[:nrows]
)
fig.add_trace(go.Scatter(x=df.index, y=df['Nominal'], name="Nominal Yield", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Real'], name="Real Yield", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
fig.update_yaxes(title="Yield (%)", tickformat=".2f", row=1, col=1)

fig.add_trace(go.Scatter(x=mom.index, y=mom, name=f"Momentum ({win}d)", line=dict(color="#d62728", dash="dash", width=1.8)), row=2, col=1)
for th, lab in [(ease_bp, f"+{ease_bp} bp"), (0, "zero"), (tight_bp, f"{tight_bp} bp")]:
    fig.add_hline(y=th, line=dict(color="grey", dash="dot", width=1), row=2, col=1,
                  annotation_text=lab, annotation_position="right")

# Regime shading with labels
if show_regime:
    for s, e in contiguous(mom > ease_bp):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(0,128,0,0.12)", line_width=0, row=2, col=1, annotation_text="Easing", annotation_position="top left")
    for s, e in contiguous(mom < tight_bp):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,0,0,0.12)", line_width=0, row=2, col=1, annotation_text="Tightening", annotation_position="top left")

fig.update_yaxes(title="bp", tickformat=".0f", row=2, col=1)
fig.update_xaxes(tickformat="%b-%y", row=nrows, col=1, title="Date")

if show_spread:
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread'], name="10Y Spread", line=dict(color="#2ca02c", width=2)), row=3, col=1)
    fig.update_yaxes(title="Spread (%)", tickformat=".2f", row=3, col=1)

# --- Macro Overlays ---
if show_fomc:
    fomc_dates = fetch_fomc_dates()
    for dt in fomc_dates:
        if df.index[0] <= dt <= df.index[-1]:
            fig.add_vline(
                x=dt, line=dict(color="rgba(44,160,44,0.7)", dash="dashdot", width=1.2),
                annotation_text="FOMC", annotation_position="top right", row=1, col=1
            )

if show_cpi:
    cpi_dates = fetch_cpi_dates()
    for dt in cpi_dates:
        if df.index[0] <= dt <= df.index[-1]:
            fig.add_vline(
                x=dt, line=dict(color="rgba(255,127,14,0.3)", dash="dot", width=1),
                annotation_text="CPI", annotation_position="bottom right", row=1, col=1
            )

if show_recession:
    recessions = fetch_recession_bands()
    for start, end in recessions:
        if end >= df.index[0] and start <= df.index[-1]:
            fig.add_vrect(
                x0=max(start, df.index[0]), x1=min(end, df.index[-1]),
                fillcolor="rgba(128,128,128,0.18)", line_width=0,
                annotation_text="Recession", annotation_position="top left"
            )

fig.update_layout(
    template="plotly_white",
    height=820 if show_spread else 720,
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
)

st.plotly_chart(fig, use_container_width=True)

# ── Download & Methodology ──────────────────────────────
with st.expander("Download Data"):
    df_out = df.copy()
    df_out[f"Inverted {win}-Day Real Yield Momentum (bp)"] = mom
    st.download_button(
        "Download Yield Data (CSV)",
        df_out.to_csv(index=True),
        file_name="10yr_yield_dashboard.csv",
        mime="text/csv"
    )

with st.expander("Methodology & Interpretation", expanded=False):
    st.markdown(f"""
    **Yield Sources:**  
    • Nominal: FRED DGS10  
    • Real: FRED DFII10

    **Inverted Momentum:**  
    − (Current real yield − real yield {win} days ago) × 100

    **Regime Definition:**  
    • 🟢 *Easing*: momentum &gt; {ease_bp}bp  
    • ⚪️ *Neutral*: between  
    • 🔴 *Tightening*: momentum &lt; {tight_bp}bp

    **Macro Overlays:**  
    • **FOMC Dates**: Green dashed verticals, major Fed meetings.  
    • **CPI Prints**: Orange dotted verticals, CPI releases.  
    • **Recession Bands**: Gray bands, official NBER cycles.

    These overlays contextualize regime shifts and volatility in real-time.

    """)
st.caption("© 2025 AD Fund Management LP")
