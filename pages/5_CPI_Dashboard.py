import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --------------------------------------------------
# 1. CONFIGURATION & CACHING
# --------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Download a FRED series and forward-fill missing values."""
    s = DataReader(series_id, "fred", start)
    s.name = series_id
    s = s.asfreq("MS")  # monthly start-of-month frequency
    return s.ffill()

# --------------------------------------------------
# 2. DATA INGESTION
# --------------------------------------------------
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Headline & Core CPI (NSA for headline, SA for core)
headline = load_series("CPIAUCNS", START_DATE)
core = load_series("CPILFESL", START_DATE)  # Seasonally adjusted core
recession_flag = load_series("USREC", START_DATE)  # 1 during NBER recessions

# YoY % change
headline_yoy = headline.pct_change(12) * 100
core_yoy = core.pct_change(12) * 100

# MoM % change (SA headline calculation uses CPIAUCSL, but keep consistent with headline here)
headline_mom = headline.pct_change(1) * 100
core_mom = core.pct_change(1) * 100

# 3-Month annualised change for core
core_3m_ann = (core / core.shift(3)) ** 4 - 1
core_3m_ann = core_3m_ann * 100

# Helper to extract recession periods

def get_recession_periods(rec_series: pd.Series):
    """Return list of (start, end) tuples where rec_series == 1."""
    rec_series = rec_series.dropna()
    rec_series = rec_series.astype(int)
    transitions = rec_series.diff()
    starts = rec_series[(rec_series == 1) & (transitions == 1)].index
    # If series starts with a recession, include first period
    if rec_series.iloc[0] == 1:
        starts = starts.insert(0, rec_series.index[0])
    ends = rec_series[(rec_series == 0) & (transitions == -1)].index
    # If series ends with a recession, append last date
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([rec_series.index[-1]]))
    return list(zip(starts, ends))

recessions = get_recession_periods(recession_flag)

# --------------------------------------------------
# 3. PLOTTING UTILITIES
# --------------------------------------------------

def add_recession_bands(fig, recessions, yref="paper", color="rgba(200,0,0,0.15)"):
    """Add vertical shaded regions to a Plotly figure."""
    for start, end in recessions:
        fig.add_vrect(
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0,
            yref=yref,
        )

# --------------------------------------------------
# 4. CHART 1 – HEADLINE vs CORE YoY
# --------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=headline_yoy.index, y=headline_yoy, name="Headline CPI YoY", line=dict(color="#1f77b4")))
fig1.add_trace(go.Scatter(x=core_yoy.index, y=core_yoy, name="Core CPI YoY", line=dict(color="#ff7f0e")))
add_recession_bands(fig1, recessions)
fig1.update_layout(
    title="US CPI YoY: Headline vs Core",
    yaxis_title="% YoY",
    hovermode="x unified",
)

# --------------------------------------------------
# 5. CHART 2 – MoM CHANGE BARS (HEADLINE & CORE)
# --------------------------------------------------
fig2 = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Headline CPI MoM (SA) %", "Core CPI MoM (SA) %"),
)
fig2.add_trace(
    go.Bar(x=headline_mom.index, y=headline_mom, name="Headline MoM", marker_color="#1f77b4"),
    row=1, col=1,
)
fig2.add_trace(
    go.Bar(x=core_mom.index, y=core_mom, name="Core MoM", marker_color="#ff7f0e"),
    row=2, col=1,
)
add_recession_bands(fig2, recessions, yref="paper", color="rgba(200,0,0,0.15)")
fig2.update_layout(showlegend=False, hovermode="x unified")
fig2.update_yaxes(title_text="% MoM", row=1, col=1)
fig2.update_yaxes(title_text="% MoM", row=2, col=1)

# --------------------------------------------------
# 6. CHART 3 – CORE INDEX LEVEL + 3-M ANN. % CHANGE
# --------------------------------------------------
fig3 = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Core CPI Index Level", "3-Month Annualised Core CPI % Change"),
)
fig3.add_trace(
    go.Scatter(x=core.index, y=core, name="Core Index", line=dict(color="#ff7f0e")),
    row=1, col=1,
)
fig3.add_trace(
    go.Scatter(x=core_3m_ann.index, y=core_3m_ann, name="3M Annualised", line=dict(color="#1f77b4")),
    row=2, col=1,
)
add_recession_bands(fig3, recessions)
fig3.update_yaxes(title_text="Index Level", row=1, col=1)
fig3.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig3.update_layout(hovermode="x unified")

# --------------------------------------------------
# 7. STREAMLIT LAYOUT
# --------------------------------------------------
st.title("US Inflation Dashboard")
with st.expander("Methodology & Sources", expanded=False):
    st.write(
        """
        **Data:** All series are sourced from FRED via *pandas-datareader* and auto-updated on every app launch.  
        **Recession shading:** NBER recession dates (USREC).  
        **3-Month annualised core CPI:** `((Core_t / Core_{t-3}) ** 4 - 1) × 100`.  
        """
    )

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------
st.caption("© {} • Built with Streamlit & Plotly • Data: FRED".format(datetime.today().year))
