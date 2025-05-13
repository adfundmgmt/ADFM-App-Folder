import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG – must be the first Streamlit command
# --------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard", layout="wide")

# --------------------------------------------------
# SIDEBAR – About This Tool
# --------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **US CPI Dashboard (Streamlit)**  
        • Three Bloomberg-style inflation charts.  
        • Live FRED data: headline CPI, core CPI, NBER recession flag.  
        • Fully vectorised & cached for speed.  
        
        *Fix (v1.1):* `load_series()` returns a Series (no more ambiguous DataFrame errors).
        """
    )

# --------------------------------------------------
# DATA LOADER with CACHING
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Fetch a FRED series and return a forward-filled monthly Series."""
    df = DataReader(series_id, "fred", start)  # returns DataFrame
    s = df[series_id].copy()                   # convert to Series
    s.name = series_id
    return s.asfreq("MS").ffill()

# --------------------------------------------------
# DATA INGESTION
# --------------------------------------------------
START_DATE = "1990-01-01"
headline = load_series("CPIAUCNS", START_DATE)  # Headline CPI (NSA)
core     = load_series("CPILFESL", START_DATE)  # Core CPI (SA)
recess   = load_series("USREC",    START_DATE)  # NBER recession flag

# Transformations: YoY, MoM, 3‑month annualised
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# --------------------------------------------------
# Recession windows extraction
# --------------------------------------------------
def get_recession_periods(flag: pd.Series):
    f = flag.dropna().astype(int)
    if f.empty:
        return []
    diff = f.diff()
    starts = f[(f == 1) & (diff == 1)].index
    if f.iloc[0] == 1:
        starts = starts.insert(0, f.index[0])
    ends = f[(f == 0) & (diff == -1)].index
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([f.index[-1]]))
    return list(zip(starts, ends))

recession_windows = get_recession_periods(recess)

# --------------------------------------------------
# 1. CHART – YoY Headline vs Core CPI
# --------------------------------------------------
fig_yoy = go.Figure()
fig_yoy.add_trace(go.Scatter(x=headline_yoy.index, y=headline_yoy,
                             name="Headline CPI YoY", line=dict(color="#1f77b4")))
fig_yoy.add_trace(go.Scatter(x=core_yoy.index,     y=core_yoy,
                             name="Core CPI YoY",     line=dict(color="#ff7f0e")))
# Shade recessions across entire plot
for start, end in recession_windows:
    fig_yoy.add_vrect(x0=start, x1=end, fillcolor="rgba(200,0,0,0.15)",
                      opacity=0.3, layer="below", line_width=0, yref="paper")
fig_yoy.update_layout(
    title="US CPI YoY – Headline vs Core",
    yaxis_title="% YoY",
    hovermode="x unified"
)

# --------------------------------------------------
# 2. CHART – MoM Bars for Headline & Core
# --------------------------------------------------
fig_mom = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Headline CPI MoM %", "Core CPI MoM %")
)
fig_mom.add_trace(go.Bar(x=headline_mom.index, y=headline_mom,
                         name="Headline MoM", marker_color="#1f77b4"), row=1, col=1)
fig_mom.add_trace(go.Bar(x=core_mom.index,     y=core_mom,
                         name="Core MoM",     marker_color="#ff7f0e"), row=2, col=1)
# Shade each subplot separately
for start, end in recession_windows:
    fig_mom.add_vrect(x0=start, x1=end, row=1, col=1,
                      fillcolor="rgba(200,0,0,0.15)", opacity=0.3,
                      layer="below", line_width=0)
    fig_mom.add_vrect(x0=start, x1=end, row=2, col=1,
                      fillcolor="rgba(200,0,0,0.15)", opacity=0.3,
                      layer="below", line_width=0)
fig_mom.update_yaxes(title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=2, col=1)
fig_mom.update_layout(showlegend=False, hovermode="x unified")

# --------------------------------------------------
# 3. CHART – Core CPI Index & 3‑Month Annualised
# --------------------------------------------------
fig_core = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Core CPI Index", "3‑Month Annualised Core CPI %")
)
fig_core.add_trace(go.Scatter(x=core.index, y=core,
                              name="Core Index", line=dict(color="#ff7f0e")), row=1, col=1)
fig_core.add_trace(go.Scatter(x=core_3m_ann.index, y=core_3m_ann,
                              name="3M Ann.", line=dict(color="#1f77b4")), row=2, col=1)
# Shade each subplot
for start, end in recession_windows:
    fig_core.add_vrect(x0=start, x1=end, row=1, col=1,
                       fillcolor="rgba(200,0,0,0.15)", opacity=0.3,
                       layer="below", line_width=0)
    fig_core.add_vrect(x0=start, x1=end, row=2, col=1,
                       fillcolor="rgba(200,0,0,0.15)", opacity=0.3,
                       layer="below", line_width=0)
fig_core.update_yaxes(title_text="Index Level",      row=1, col=1)
fig_core.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig_core.update_layout(hovermode="x unified")

# --------------------------------------------------
# 4. MAIN LAYOUT
# --------------------------------------------------
st.title("US Inflation Dashboard")

with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** FRED via *pandas-datareader*  
        • Headline CPI (NSA): `CPIAUCNS`  
        • Core CPI (SA): `CPILFESL`  
        • Recessions: `USREC`  
        **3‑Mo annualised formula:** `((CPI_t / CPI_{t-3}) ** 4 – 1) × 100`.
        """
    )

st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

st.caption(f"© {datetime.today().year} • Built with Streamlit & Plotly • Data: FRED")
