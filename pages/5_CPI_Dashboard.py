import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

"""
US CPI Dashboard (Streamlit)
• Re-creates three Bloomberg-style inflation charts.
• Pulls data live from FRED (headline CPI, core CPI, USREC recession indicator).
• All functions fully vectorised and cached for speed.
Fix (v1.1): ensure `load_series()` always returns a **Series** (not a DataFrame). That removes the ValueError you hit when `get_recession_periods()` tried to index an empty/ambiguous DataFrame.
"""

# --------------------------------------------------
# 1. CONFIG & CACHING
# --------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Download a FRED series and forward-fill missing values, always returning a Series."""
    df = DataReader(series_id, "fred", start)            # DataFrame with one column
    s = df[series_id].copy()                              # select the column → Series
    s.name = series_id
    s = s.asfreq("MS")                                   # monthly frequency (start-of-month)
    return s.ffill()

# --------------------------------------------------
# 2. DATA
# --------------------------------------------------
START_DATE = "1990-01-01"
headline = load_series("CPIAUCNS", START_DATE)           # headline CPI (NSA)
core     = load_series("CPILFESL", START_DATE)           # core CPI (SA)
recess   = load_series("USREC", START_DATE)              # NBER recession flag (0/1)

# YoY / MoM / 3-month annualised
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12) * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# --------------------------------------------------
# 3. RECESSION WINDOWS
# --------------------------------------------------

def get_recession_periods(flag: pd.Series):
    flag = flag.dropna().astype(int)
    if flag.empty:
        return []

    # detect 0→1 and 1→0 transitions
    transitions = flag.diff()
    starts = flag[(flag == 1) & (transitions == 1)].index
    if flag.iloc[0] == 1:  # recession already in progress at start
        starts = starts.insert(0, flag.index[0])

    ends = flag[(flag == 0) & (transitions == -1)].index
    if len(ends) < len(starts):                 # recession running right now
        ends = ends.append(pd.Index([flag.index[-1]]))

    return list(zip(starts, ends))

recession_windows = get_recession_periods(recess)

# helper to shade recessions on any Plotly fig

def add_recession_bands(fig, windows, yref="paper", color="rgba(200,0,0,0.15)"):
    for start, end in windows:
        fig.add_vrect(x0=start, x1=end, y0=0, y1=1, yref=yref,
                       fillcolor=color, opacity=0.3, layer="below", line_width=0)

# --------------------------------------------------
# 4. CHARTS
# --------------------------------------------------
# 4A. YoY headline vs core
fig_yoy = go.Figure()
fig_yoy.add_trace(go.Scatter(x=headline_yoy.index, y=headline_yoy, name="Headline CPI YoY", line=dict(color="#1f77b4")))
fig_yoy.add_trace(go.Scatter(x=core_yoy.index,     y=core_yoy,     name="Core CPI YoY",     line=dict(color="#ff7f0e")))
add_recession_bands(fig_yoy, recession_windows)
fig_yoy.update_layout(title="US CPI YoY – Headline vs Core", yaxis_title="% YoY", hovermode="x unified")

# 4B. MoM bars
fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=("Headline CPI MoM %", "Core CPI MoM %"))
fig_mom.add_trace(go.Bar(x=headline_mom.index, y=headline_mom, name="Headline MoM", marker_color="#1f77b4"), row=1, col=1)
fig_mom.add_trace(go.Bar(x=core_mom.index,     y=core_mom,     name="Core MoM",     marker_color="#ff7f0e"), row=2, col=1)
add_recession_bands(fig_mom, recession_windows, yref="paper")
fig_mom.update_yaxes(title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=2, col=1)
fig_mom.update_layout(showlegend=False, hovermode="x unified")

# 4C. Core index & 3-month annualised change
fig_core = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                         subplot_titles=("Core CPI Index", "3-Month Annualised Core CPI %"))
fig_core.add_trace(go.Scatter(x=core.index, y=core, name="Core Index", line=dict(color="#ff7f0e")), row=1, col=1)
fig_core.add_trace(go.Scatter(x=core_3m_ann.index, y=core_3m_ann, name="3M Ann.", line=dict(color="#1f77b4")), row=2, col=1)
add_recession_bands(fig_core, recession_windows)
fig_core.update_yaxes(title_text="Index Level", row=1, col=1)
fig_core.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig_core.update_layout(hovermode="x unified")

# --------------------------------------------------
# 5. STREAMLIT LAYOUT
# --------------------------------------------------
st.title("US Inflation Dashboard")
with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** FRED via *pandas-datareader* (headline: `CPIAUCNS`, core SA: `CPILFESL`, recessions: `USREC`).  
        **3-Mo annualised formula:** `((CPI_t / CPI_{t-3}) ** 4 – 1) × 100`.  
        **Update cadence:** fresh on every app launch or cache expiry.
        """
    )

st.plotly_chart(fig_yoy, use_container_width=True)
st.plotly_chart(fig_mom, use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

st.caption(f"© {datetime.today().year} • Built with Streamlit & Plotly • Data: FRED")
