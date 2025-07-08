import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
plt.style.use("default")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard", layout="wide")

# --------------------------------------------------
# SIDEBAR: About & Period Selector
# --------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore interactive inflation charts powered by the latest FRED data. View year‑over‑year and month‑over‑month changes for both headline and core CPI, with NBER recessions shaded for context. Data is fetched live and cached for responsiveness.
        """
    )
    st.subheader("Time Range")
    period = st.selectbox(
        "Select period:",
        ["1M", "3M", "6M", "9M", "1Y", "3Y", "5Y", "All"],
        index=7
    )

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    try:
        df = DataReader(series_id, "fred", start)
        s = df[series_id].copy()
        return s.asfreq("MS").ffill()
    except Exception as e:
        st.error(f"Could not load {series_id} from FRED: {e}")
        return pd.Series(dtype=float)

# Fetch series
start_date_full = "1990-01-01"
headline = load_series("CPIAUCNS", start_date_full)
core     = load_series("CPILFESL", start_date_full)
recess   = load_series("USREC",    start_date_full)

if headline.empty or core.empty or recess.empty:
    st.stop()

# --------------------------------------------------
# TRANSFORMATIONS
# --------------------------------------------------
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# --------------------------------------------------
# DETERMINE DATE WINDOW
# --------------------------------------------------
end_date = headline.index.max()
if period == "All":
    start_date = headline.index.min()
elif period.endswith("M"):
    months = int(period[:-1])
    start_date = end_date - pd.DateOffset(months=months)
elif period.endswith("Y"):
    years = int(period[:-1])
    start_date = end_date - pd.DateOffset(years=years)
else:
    st.error(f"Invalid period selection: {period}")
    st.stop()

# Optional: Allow manual override with a date slider
manual_range = st.sidebar.checkbox("Custom date range")
if manual_range:
    slider = st.sidebar.slider(
        "Select date range:",
        min_value=headline.index.min().to_pydatetime().date(),
        max_value=headline.index.max().to_pydatetime().date(),
        value=(start_date.to_pydatetime().date(), end_date.to_pydatetime().date()),
        format="YYYY-MM"
    )
    start_date = pd.Timestamp(slider[0])
    end_date = pd.Timestamp(slider[1])

# Slice all series for window
idx = (headline.index >= start_date) & (headline.index <= end_date)
h_yoy = headline_yoy.loc[idx]
c_yoy = core_yoy.loc[idx]
h_mom = headline_mom.loc[idx]
c_mom = core_mom.loc[idx]
c_3m  = core_3m_ann.loc[idx]

# Filter recessions overlapping window
def within_window(rec_list, start, end):
    out = []
    for s,e in rec_list:
        if e < start or s > end:
            continue
        out.append((max(s,start), min(e,end)))
    return out

# Recession extraction
def get_recessions(flag: pd.Series):
    f = flag.dropna().astype(int)
    dif = f.diff()
    starts = f[(f==1)&(dif==1)].index
    if f.iloc[0]==1: starts = starts.insert(0,f.index[0])
    ends = f[(f==0)&(dif==-1)].index
    # Patch for rare bug: Recession starts but never ends in data
    if len(ends)<len(starts): ends = ends.append(pd.Index([f.index[-1]]))
    # Patch for instant recessions (same start and end)
    return [(s, e) for s, e in zip(starts, ends) if e >= s]

recs = get_recessions(recess)
recs_window = within_window(recs, start_date, end_date)

# --------------------------------------------------
# BUILD CHARTS
# --------------------------------------------------
# 1) YoY
fig_yoy = go.Figure()
fig_yoy.add_trace(go.Scatter(
    x=h_yoy.index, y=h_yoy, name="Headline CPI YoY", line_color="#1f77b4", hovertemplate='%{y:.2f}%'))
fig_yoy.add_trace(go.Scatter(
    x=c_yoy.index, y=c_yoy, name="Core CPI YoY",     line_color="#ff7f0e", hovertemplate='%{y:.2f}%'))
for s,e in recs_window:
    fig_yoy.add_shape(
        dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
             fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0)
    )
fig_yoy.update_layout(
    title="US CPI YoY – Headline vs Core",
    yaxis_title="% YoY",
    hovermode="x unified",
    margin=dict(t=80),
    xaxis=dict(rangeslider=dict(visible=False), showline=True, linewidth=1),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    )
)

# 2) MoM bars
fig_mom = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Headline CPI MoM %", "Core CPI MoM %"))
fig_mom.add_trace(go.Bar(
    x=h_mom.index, y=h_mom, name="Headline MoM", marker_color="#1f77b4", hovertemplate='%{y:.2f}%'), row=1, col=1)
fig_mom.add_trace(go.Bar(
    x=c_mom.index, y=c_mom, name="Core MoM",     marker_color="#ff7f0e", hovertemplate='%{y:.2f}%'), row=2, col=1)
for s,e in recs_window:
    fig_mom.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                            fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0), row=1, col=1)
    fig_mom.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                            fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0), row=2, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=2, col=1)
fig_mom.update_layout(
    showlegend=False,
    hovermode="x unified",
    margin=dict(t=80),
    xaxis=dict(rangeslider=dict(visible=False)),      # disable side zoom top
    xaxis2=dict(rangeslider=dict(visible=False))      # disable side zoom bottom
)

# 3) Core index & 3M ann
fig_core = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Core CPI Index", "3‑Mo Annualised Core CPI %"))
fig_core.add_trace(go.Scatter(
    x=core.loc[idx].index, y=core.loc[idx], name="Core Index", line_color="#ff7f0e", mode="lines+markers", hovertemplate='%{y:.2f}'), row=1, col=1)
fig_core.add_trace(go.Scatter(
    x=c_3m.index, y=c_3m, name="3M Ann.", line_color="#1f77b4", mode="lines+markers", hovertemplate='%{y:.2f}%'), row=2, col=1)
for s,e in recs_window:
    fig_core.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                             fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0), row=1, col=1)
    fig_core.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                             fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0), row=2, col=1)
fig_core.update_yaxes(title_text="Index Level", row=1, col=1)
fig_core.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig_core.update_layout(
    hovermode="x unified",
    margin=dict(t=80),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    ),
    xaxis=dict(rangeslider=dict(visible=False)),      # disable side zoom top
    xaxis2=dict(rangeslider=dict(visible=False))      # disable side zoom bottom
)

# --------------------------------------------------
# RENDER: HEADLINE + DELTA BOX
# --------------------------------------------------
st.title("US Inflation Dashboard")

# Cool addition: Headline/Core latest values + delta
latest_date = h_yoy.index.max()
headline_latest = h_yoy.iloc[-1]
core_latest = c_yoy.iloc[-1]
headline_prev = h_yoy.iloc[-2] if len(h_yoy) > 1 else None
core_prev = c_yoy.iloc[-2] if len(c_yoy) > 1 else None

cols = st.columns(2)
cols[0].metric("Headline CPI YoY", f"{headline_latest:.2f}%", f"{(headline_latest - headline_prev):+.2f}%" if headline_prev else "0.00%")
cols[1].metric("Core CPI YoY", f"{core_latest:.2f}%", f"{(core_latest - core_prev):+.2f}%" if core_prev else "0.00%")

# Main plots
st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

# Download section
with st.expander("Download Data"):
    combined = pd.DataFrame({
        "Headline CPI": headline.loc[idx],
        "Core CPI": core.loc[idx],
        "Headline YoY (%)": h_yoy,
        "Core YoY (%)": c_yoy,
        "Headline MoM (%)": h_mom,
        "Core MoM (%)": c_mom,
        "Core 3M Ann. (%)": c_3m,
        "Recession Flag": recess.loc[idx],
    })
    st.download_button(
        "Download CSV", combined.to_csv(index=True), file_name="us_cpi_data.csv", mime="text/csv"
    )

with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** FRED via pandas-datareader  
        **Series:** Headline CPI (CPIAUCNS), Core CPI (CPILFESL), Recessions (USREC)  
        **3‑Mo annualised formula:** ((CPI_t / CPI_{t-3}) ** 4 – 1) × 100
        """
    )

st.caption("© 2025 AD Fund Management LP")
