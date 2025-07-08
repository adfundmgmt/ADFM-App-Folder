import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="US CPI Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Caching Data Loading ---
@st.cache_data(show_spinner=True)
def load_fred_series(series_id, start="1990-01-01"):
    try:
        df = DataReader(series_id, "fred", start)
        return df[series_id].asfreq("MS").ffill()
    except Exception as e:
        return pd.Series(dtype=float)

# --- Load Data ---
start_date_full = "1990-01-01"
headline = load_fred_series("CPIAUCNS", start_date_full)
core     = load_fred_series("CPILFESL", start_date_full)
recess   = load_fred_series("USREC",    start_date_full)

# --- Data Integrity Check ---
missing = []
for name, ser in [("Headline CPI", headline), ("Core CPI", core), ("Recession", recess)]:
    if ser.empty:
        missing.append(name)
    elif ser.isnull().any():
        missing.append(name + " (missing values)")

if missing:
    st.error(f"Missing or incomplete data: {', '.join(missing)}. Check FRED or try again later.")
    st.stop()

# --- Build canonical index (guaranteed monthly) ---
full_index = pd.date_range(start=headline.index.min(), end=headline.index.max(), freq="MS")
headline = headline.reindex(full_index)
core     = core.reindex(full_index)
recess   = recess.reindex(full_index)

# --- Transformations ---
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# --- Sidebar Period Selector ---
with st.sidebar:
    st.header("About This Tool")
    st.markdown("Explore interactive inflation charts powered by the latest FRED data. View year‑over‑year and month‑over‑month changes for both headline and core CPI, with NBER recessions shaded for context. Data is fetched live and cached for responsiveness.")
    st.subheader("Time Range")
    period = st.selectbox("Select period:", ["1M", "3M", "6M", "9M", "1Y", "3Y", "5Y", "All"], index=7)
    manual_range = st.checkbox("Custom date range")
    reset_button = st.button("Reset to full range", help="Show all data")

# --- Date Window Logic ---
end_date = full_index.max()
if period == "All" or reset_button:
    start_date = full_index.min()
elif period.endswith("M"):
    months = int(period[:-1])
    start_date = end_date - pd.DateOffset(months=months)
elif period.endswith("Y"):
    years = int(period[:-1])
    start_date = end_date - pd.DateOffset(years=years)
else:
    st.error(f"Invalid period selection: {period}")
    st.stop()

if manual_range:
    slider = st.sidebar.slider(
        "Select date range:",
        min_value=full_index.min().to_pydatetime().date(),
        max_value=full_index.max().to_pydatetime().date(),
        value=(start_date.to_pydatetime().date(), end_date.to_pydatetime().date()),
        format="YYYY-MM"
    )
    start_date = pd.Timestamp(slider[0])
    end_date = pd.Timestamp(slider[1])

window_idx = (full_index >= start_date) & (full_index <= end_date)
headline_win = headline[window_idx]
core_win = core[window_idx]
h_yoy = headline_yoy[window_idx]
c_yoy = core_yoy[window_idx]
h_mom = headline_mom[window_idx]
c_mom = core_mom[window_idx]
c_3m = core_3m_ann[window_idx]
recess_win = recess[window_idx]

# --- Recession period calculation (no risk of misalignment) ---
def get_recessions(flag: pd.Series):
    f = flag.fillna(0).astype(int)
    dif = f.diff()
    starts = f[(f==1)&(dif==1)].index
    if f.iloc[0]==1: starts = starts.insert(0,f.index[0])
    ends = f[(f==0)&(dif==-1)].index
    if len(ends)<len(starts): ends = ends.append(pd.Index([f.index[-1]]))
    return [(s, e) for s, e in zip(starts, ends) if e >= s]

def within_window(rec_list, start, end):
    out = []
    for s,e in rec_list:
        if e < start or s > end: continue
        out.append((max(s,start), min(e,end)))
    return out

recs = get_recessions(recess)
recs_window = within_window(recs, start_date, end_date)

# --- OUTLIER DETECTION AND MACRO REGIMES ---
# Outlier thresholds
yoy_std = c_yoy.rolling(120, min_periods=24).std()
yoy_outlier = abs(c_yoy - c_yoy.rolling(120, min_periods=24).mean()) > 1.5*yoy_std
yoy_outlier_idx = c_yoy.index[yoy_outlier.fillna(False)]

mom_outlier = (h_mom > 0.7) | (h_mom < -0.2)
mom_outlier_idx = h_mom.index[mom_outlier.fillna(False)]

# Macro regimes for Core CPI YoY
core_regime = pd.Series("Normal", index=c_yoy.index)
core_regime[c_yoy > 3.0] = "High inflation"
core_regime[c_yoy < 2.0] = "Low inflation"
regime_change_idx = core_regime[core_regime != core_regime.shift(1)].index
# Color band periods
regime_bands = []
current = None
for idx in core_regime.index:
    r = core_regime[idx]
    if r != current:
        if current is not None:
            regime_bands[-1]['end'] = idx
        regime_bands.append({'regime': r, 'start': idx, 'end': idx})
        current = r
    else:
        regime_bands[-1]['end'] = idx
# Only plot bands within window
regime_bands = [b for b in regime_bands if b['end'] >= h_yoy.index[0] and b['start'] <= h_yoy.index[-1]]

# --- Analyst Comment (auto) ---
headline_delta = h_yoy.iloc[-1] - h_yoy.iloc[-2] if len(h_yoy) > 1 else None
core_delta = c_yoy.iloc[-1] - c_yoy.iloc[-2] if len(c_yoy) > 1 else None

regime_summary = ", ".join(
    f"{b['regime']} regime from {b['start'].strftime('%b %Y')}" for b in regime_bands[-3:]
)
outlier_msg = ""
if any(yoy_outlier.fillna(False)):
    last_outlier = yoy_outlier_idx[-1].strftime('%b %Y')
    outlier_msg = f"Last Core CPI YoY outlier detected: {last_outlier}."
if any(mom_outlier.fillna(False)):
    last_mom_outlier = mom_outlier_idx[-1].strftime('%b %Y')
    outlier_msg += f" Last Headline MoM outlier detected: {last_mom_outlier}."

commentary = f"""
**Quick Take:** Headline CPI YoY is at {h_yoy.iloc[-1]:.2f}% ({headline_delta:+.2f}pp vs last month), Core CPI YoY is at {c_yoy.iloc[-1]:.2f}% ({core_delta:+.2f}pp).  
Regimes: {regime_summary if regime_summary else 'Stable'}.  
{outlier_msg}
"""

# --- Main Layout ---
st.title("US Inflation Dashboard")
st.markdown(commentary)
st.caption(f"Last FRED update: {full_index[-1].strftime('%b %Y')}")
st.divider()

# --- KPI Cards ---
cols = st.columns(2)
cols[0].metric("Headline CPI YoY", f"{h_yoy.iloc[-1]:.2f}%", f"{headline_delta:+.2f}pp" if headline_delta else "0.00pp")
cols[1].metric("Core CPI YoY", f"{c_yoy.iloc[-1]:.2f}%", f"{core_delta:+.2f}pp" if core_delta else "0.00pp")

# --- CPI YoY Chart with Outliers and Regimes ---
fig_yoy = go.Figure()
# Regime color bands
for band in regime_bands:
    color = dict(
        Normal="rgba(255,255,255,0)", 
        High="rgba(255,200,0,0.12)", 
        Low="rgba(0,200,255,0.12)",
        **{"High inflation":"rgba(255,100,0,0.13)", "Low inflation":"rgba(0,200,255,0.13)"}
    ).get(band['regime'], "rgba(255,255,255,0)")
    fig_yoy.add_vrect(
        x0=band['start'], x1=band['end'], 
        fillcolor=color, opacity=0.3, layer="below", line_width=0
    )

fig_yoy.add_trace(go.Scatter(
    x=h_yoy.index, y=h_yoy, name="Headline CPI YoY", line_color="#1f77b4", hovertemplate='%{y:.2f}%'
))
fig_yoy.add_trace(go.Scatter(
    x=c_yoy.index, y=c_yoy, name="Core CPI YoY", line_color="#ff7f0e", hovertemplate='%{y:.2f}%'
))
for s, e in recs_window:
    fig_yoy.add_shape(dict(
        type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
        fillcolor="rgba(200,0,0,0.12)", opacity=0.3, layer="below", line_width=0
    ))
# Outlier markers (Core YoY)
fig_yoy.add_trace(go.Scatter(
    x=yoy_outlier_idx, y=c_yoy.loc[yoy_outlier_idx], 
    mode="markers", name="Core YoY Outlier",
    marker=dict(color="red", size=10, symbol="diamond-open"),
    hovertemplate="Outlier: %{y:.2f}%"
))
fig_yoy.update_layout(
    title="US CPI YoY – Headline vs Core (with Macro Regimes & Outliers)",
    yaxis_title="% YoY",
    hovermode="x unified",
    margin=dict(t=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    xaxis=dict(rangeslider=dict(visible=False), showline=True, linewidth=1),
)

# --- MoM Bars Chart with Outliers ---
fig_mom = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Headline CPI MoM %", "Core CPI MoM %"))
fig_mom.add_trace(go.Bar(
    x=h_mom.index, y=h_mom, name="Headline MoM", marker_color="#1f77b4", hovertemplate='%{y:.2f}%'), row=1, col=1)
fig_mom.add_trace(go.Bar(
    x=c_mom.index, y=c_mom, name="Core MoM", marker_color="#ff7f0e", hovertemplate='%{y:.2f}%'), row=2, col=1)
for s,e in recs_window:
    for row in [1, 2]:
        fig_mom.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                               fillcolor="rgba(200,0,0,0.12)", opacity=0.3, layer="below", line_width=0), row=row, col=1)
# Outlier markers for Headline MoM
fig_mom.add_trace(go.Scatter(
    x=mom_outlier_idx, y=h_mom.loc[mom_outlier_idx], mode="markers",
    name="Headline MoM Outlier", marker=dict(color="purple", size=10, symbol="star-open"),
    hovertemplate="Outlier: %{y:.2f}%"
), row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=1, col=1, zeroline=True)
fig_mom.update_yaxes(title_text="% MoM", row=2, col=1, zeroline=True)
fig_mom.update_layout(
    showlegend=True,
    hovermode="x unified",
    margin=dict(t=80),
    xaxis=dict(rangeslider=dict(visible=False)),
    xaxis2=dict(rangeslider=dict(visible=False))
)

# --- Core Index & 3M Ann (unchanged) ---
fig_core = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Core CPI Index", "3‑Mo Annualised Core CPI %"))
fig_core.add_trace(go.Scatter(
    x=core_win.index, y=core_win, name="Core Index", line_color="#ff7f0e", mode="lines+markers", hovertemplate='%{y:.2f}'), row=1, col=1)
fig_core.add_trace(go.Scatter(
    x=c_3m.index, y=c_3m, name="3M Ann.", line_color="#1f77b4", mode="lines+markers", hovertemplate='%{y:.2f}%'), row=2, col=1)
for s,e in recs_window:
    for row in [1, 2]:
        fig_core.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                                fillcolor="rgba(200,0,0,0.12)", opacity=0.3, layer="below", line_width=0), row=row, col=1)
fig_core.update_yaxes(title_text="Index Level", row=1, col=1)
fig_core.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig_core.update_layout(
    hovermode="x unified",
    margin=dict(t=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    xaxis=dict(rangeslider=dict(visible=False)),
    xaxis2=dict(rangeslider=dict(visible=False))
)

st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** FRED via pandas-datareader  
        **Series:** Headline CPI (CPIAUCNS), Core CPI (CPILFESL), Recessions (USREC)  
        **3‑Mo annualised formula:** ((CPI_t / CPI_{t-3}) ** 4 – 1) × 100  
        **Outlier detection:** Core CPI YoY outliers are 1.5 standard deviations above 10-year trailing mean; Headline MoM outliers are moves above +0.7% or below -0.2%. Macro regimes: Core CPI >3% = High, <2% = Low.
        """
    )

st.caption("© 2025 AD Fund Management LP")
