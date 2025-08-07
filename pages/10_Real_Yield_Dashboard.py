############################################################
# Built by AD Fund Management LP. Clean Final Version.
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
NOM_SERIES  = "DGS10"
DEFAULT_WIN = 63         # trading days
DEFAULT_EASE_BP  = 40
DEFAULT_TIGHT_BP = -40

# Macro overlays: always-on defaults
SHOW_FOMC = True
SHOW_CPI = False
SHOW_RECESSION = True

REQ_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AD-Fund-Yield-Dashboard/1.0)"}

# ── Streamlit Setup ──────────────────────────────────────
st.set_page_config(page_title="10-Year Yield Dashboard", layout="wide")
st.title("10-Year Nominal and Real Yield Dashboard")

# ── Sidebar: Controls and Info ───────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    Tracks 10-year U.S. nominal and real Treasury yields with macro context.
    Top: nominal (blue) vs real (orange). Bottom: inverted {win}-day real-yield momentum
    where green suggests easing and red suggests tightening.
    """)
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback Window", ["2y", "3y", "5y", "10y"], index=2)
    start_years = int(lookback[:-1])
    win = st.number_input("Momentum Window (trading days)", min_value=21, max_value=252, value=DEFAULT_WIN, step=1)
    ease_bp = st.number_input("Easing Threshold (bp)", min_value=10, max_value=200, value=DEFAULT_EASE_BP, step=5)
    tight_bp = st.number_input("Tightening Threshold (bp)", min_value=-200, max_value=-10, value=DEFAULT_TIGHT_BP, step=5)
    show_regime = st.checkbox("Show Regime Shading", value=True)
    show_spread = st.checkbox("Show Yield Spread (Nominal - Real)", value=False)
    st.markdown("---")
    st.caption("Yields: FRED DGS10 (nominal), DFII10 (real)")

# ── Data Download Helper ────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fred(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        s = s.ffill().dropna()
        return s
    except Exception:
        return pd.Series(dtype=float)

def contiguous(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    segs, start_dt = [], None
    for ts, flag in mask.items():
        if flag and start_dt is None:
            start_dt = ts
        elif not flag and start_dt is not None:
            segs.append((start_dt, ts))
            start_dt = None
    if start_dt is not None:
        segs.append((start_dt, mask.index[-1]))
    return segs

# ── Macro Data Overlays (Silent Fail) ───────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fomc_dates():
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        r = requests.get(url, headers=REQ_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        dates = []
        for td in soup.select("td.eventdate"):
            txt = td.get_text(strip=True)
            try:
                dt = pd.to_datetime(txt, errors="coerce")
                if pd.notnull(dt):
                    dates.append(dt.normalize())
            except Exception:
                continue
        return sorted(set(dates))
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cpi_dates():
    url = "https://www.bls.gov/schedule/news_release/cpi.htm"
    try:
        r = requests.get(url, headers=REQ_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        dates = []
        for td in soup.select("table.nrtable tbody tr td"):
            text = td.get_text(strip=True)
            try:
                dt = pd.to_datetime(text, errors="coerce")
                if pd.notnull(dt):
                    dates.append(dt.normalize())
            except Exception:
                continue
        return sorted(set(dates))
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_recession_bands():
    url = "https://www.nber.org/releases.csv"
    try:
        df = pd.read_csv(url)
        starts = pd.to_datetime(df["Peak"], errors="coerce").dropna()
        ends   = pd.to_datetime(df["Trough"], errors="coerce").dropna()
        bands = [(s.normalize(), e.normalize()) for s, e in zip(starts, ends)]
        return bands
    except Exception:
        return []

# ── Main Analysis ───────────────────────────────────────
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=start_years)
ext = start - pd.Timedelta(days=DEFAULT_WIN * 2)

real = fred(REAL_SERIES, ext, today)
nom  = fred(NOM_SERIES,  ext, today)

if real.empty or nom.empty:
    st.warning("Failed to load yield data from FRED.")
    st.stop()

df = pd.DataFrame({"Nominal": nom, "Real": real}).dropna()
df = df[df.index >= start]
if df.empty:
    st.warning("No overlapping yield data for the selected lookback.")
    st.stop()

# Inverted momentum in basis points over win observations
mom = -(df["Real"] - df["Real"].shift(win)) * 100
mom = mom.dropna()
df = df.loc[mom.index]
df["Spread"] = df["Nominal"] - df["Real"]

# Latest snapshot with guards
def last_delta(series: pd.Series):
    if series.size >= 2:
        return series.iloc[-1] - series.iloc[-2]
    return pd.NA

latest = dict(
    date=df.index[-1],
    real=df["Real"].iloc[-1],
    nom=df["Nominal"].iloc[-1],
    spread=df["Spread"].iloc[-1],
    mom=mom.iloc[-1],
    mom_prev=mom.iloc[-2] if mom.size > 1 else pd.NA,
    d_nom=last_delta(df["Nominal"]),
    d_real=last_delta(df["Real"]),
)

regime = (
    "🟢 Easing" if latest["mom"] > ease_bp else
    "🔴 Tightening" if latest["mom"] < tight_bp else
    "⚪️ Neutral"
)

# ── Metrics Box ─────────────────────────────────────────
def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def fmt_bp(x):
    return "N/A" if pd.isna(x) else f"{x:+.0f} bp"

c1, c2, c3, c4, c5 = st.columns([2,2,2,2,3])
c1.metric("Nominal 10Y Yield", fmt_pct(latest["nom"]), fmt_pct(latest["d_nom"]))
c2.metric("Real 10Y Yield", fmt_pct(latest["real"]), fmt_pct(latest["d_real"]))
c3.metric(f"Inverted {win}-Trading-Day Real Momentum", f"{latest['mom']:+.0f} bp",
          fmt_bp(latest["mom"] - latest["mom_prev"]) if pd.notna(latest["mom_prev"]) else "N/A",
          help=f"-(Current real yield - real yield {win} observations ago) × 100")
c4.metric("10Y Yield Spread", fmt_pct(latest["spread"]), help="Nominal minus real yield")
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
        "Nominal vs Real 10-Year Yield",
        f"{win}-Trading-Day Inverted Real Yield Momentum (bp)",
        "10Y Yield Spread (Nominal - Real)" if show_spread else None,
    )[:nrows]
)

# Top panel
fig.add_trace(go.Scatter(x=df.index, y=df["Nominal"], name="Nominal Yield",
                         line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Real"], name="Real Yield",
                         line=dict(color="#ff7f0e", width=2)), row=1, col=1)
fig.update_yaxes(title="Yield (%)", tickformat=".2f", row=1, col=1)

# Momentum panel
fig.add_trace(go.Scatter(x=mom.index, y=mom, name=f"Momentum ({win})",
                         line=dict(color="#d62728", dash="dash", width=1.8)), row=2, col=1)

def add_hline_shape(y, row, col, color="grey", dash="dot", width=1, text=None):
    xref = f"x{row}" if row > 1 else "x"
    fig.add_shape(
        type="line",
        xref=xref, yref=f"y{row}" if row > 1 else "y",
        x0=df.index.min(), x1=df.index.max(), y0=y, y1=y,
        line=dict(color=color, width=width, dash=dash),
        row=row, col=col
    )
    if text:
        fig.add_annotation(
            x=df.index.max(), y=y, xref=xref, yref=f"y{row}" if row > 1 else "y",
            text=text, showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=10), row=row, col=col
        )

add_hline_shape(ease_bp, 2, 1, color="grey", dash="dot", width=1, text=f"+{ease_bp} bp")
add_hline_shape(0,       2, 1, color="grey", dash="dot", width=1, text="zero")
add_hline_shape(tight_bp,2, 1, color="grey", dash="dot", width=1, text=f"{tight_bp} bp")

# Regime shading
if show_regime:
    for s, e in contiguous(mom > ease_bp):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(0,128,0,0.12)", line_width=0, row=2, col=1)
    for s, e in contiguous(mom < tight_bp):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,0,0,0.12)", line_width=0, row=2, col=1)

fig.update_yaxes(title="bp", tickformat=".0f", row=2, col=1)
fig.update_xaxes(tickformat="%b-%y", row=nrows, col=1, title="Date")

# Spread panel
if show_spread:
    fig.add_trace(go.Scatter(x=df.index, y=df["Spread"], name="10Y Spread",
                             line=dict(color="#2ca02c", width=2)), row=3, col=1)
    fig.update_yaxes(title="Spread (%)", tickformat=".2f", row=3, col=1)

# Macro overlays (always-on per constants above)
def add_vline_shape(x_dt, row, col, color, dash, width, text=None):
    xref = f"x{row}" if row > 1 else "x"
    fig.add_shape(
        type="line", xref=xref, yref="paper",
        x0=x_dt, x1=x_dt, y0=0, y1=1,
        line=dict(color=color, dash=dash, width=width),
        row=row, col=col
    )
    if text:
        fig.add_annotation(
            x=x_dt, xref=xref, yref="paper", y=1.0,
            text=text, showarrow=False, yanchor="bottom",
            font=dict(size=9), row=row, col=col
        )

if SHOW_FOMC:
    for dt in fetch_fomc_dates():
        if df.index[0] <= dt <= df.index[-1]:
            add_vline_shape(dt, 1, 1, color="rgba(44,160,44,0.7)", dash="dashdot", width=1.2, text="FOMC")

if SHOW_CPI:
    for dt in fetch_cpi_dates():
        if df.index[0] <= dt <= df.index[-1]:
            add_vline_shape(dt, 1, 1, color="rgba(255,127,14,0.3)", dash="dot", width=1.0, text="CPI")

if SHOW_RECESSION:
    for rs, re in fetch_recession_bands():
        if re >= df.index[0] and rs <= df.index[-1]:
            fig.add_vrect(
                x0=max(rs, df.index[0]), x1=min(re, df.index[-1]),
                fillcolor="rgba(128,128,128,0.18)", line_width=0
            )

fig.update_layout(
    template="plotly_white",
    height=820 if show_spread else 720,
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
)

st.plotly_chart(fig, use_container_width=True)

# ── Download and Methodology ────────────────────────────
with st.expander("Download Data"):
    df_out = df.copy()
    df_out[f"Inverted {win}-Trading-Day Real Yield Momentum (bp)"] = mom
    st.download_button(
        "Download Yield Data (CSV)",
        df_out.to_csv(index=True, index_label="Date"),
        file_name="10yr_yield_dashboard.csv",
        mime="text/csv"
    )

with st.expander("Methodology and Interpretation", expanded=False):
    st.markdown(f"""
    Yield Sources:
    • Nominal: FRED DGS10
    • Real: FRED DFII10

    Inverted Momentum:
    - (Current real yield - real yield {win} observations ago) × 100

    Regime Definition:
    • 🟢 Easing: momentum > {ease_bp} bp
    • ⚪️ Neutral: between thresholds
    • 🔴 Tightening: momentum < {tight_bp} bp

    Macro Overlays:
    • FOMC Dates: green dashed verticals
    • CPI Prints: orange dotted verticals
    • Recession Bands: gray bands (NBER)
    """)

st.caption("© 2025 AD Fund Management LP")
