import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

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
        **US Inflation Dashboard**  
        Provides a comprehensive view of U.S. consumer price trends over time. The tool tracks both headline and core CPI on a year-over-year and month-over-month basis, with recession periods shaded for macro context.  
        3-month annualised core inflation is included to better detect near-term momentum shifts. Historical insights are made interactive with selectable time windows.

        """ +
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
    df = DataReader(series_id, "fred", start)
    s = df[series_id].copy()
    return s.asfreq("MS").ffill()

# Fetch series
start_date_full = "1990-01-01"
headline = load_series("CPIAUCNS", start_date_full)
core     = load_series("CPILFESL", start_date_full)
recess   = load_series("USREC",    start_date_full)

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
    if len(ends)<len(starts): ends = ends.append(pd.Index([f.index[-1]]))
    return list(zip(starts, ends))

recs = get_recessions(recess)
recs_window = within_window(recs, start_date, end_date)

# --------------------------------------------------
# BUILD CHARTS
# --------------------------------------------------
# 1) YoY
fig_yoy = go.Figure()
fig_yoy.add_trace(go.Scatter(x=h_yoy.index, y=h_yoy, name="Headline CPI YoY", line_color="#1f77b4"))
fig_yoy.add_trace(go.Scatter(x=c_yoy.index, y=c_yoy, name="Core CPI YoY",     line_color="#ff7f0e"))
for s,e in recs_window:
    fig_yoy.add_shape(
        dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
             fillcolor="rgba(200,0,0,0.15)", opacity=0.3, layer="below", line_width=0)
    )
fig_yoy.update_layout(title="US CPI YoY – Headline vs Core", yaxis_title="% YoY", hovermode="x unified")

# 2) MoM bars
fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=("Headline CPI MoM %", "Core CPI MoM %"))
fig_mom.add_trace(go.Bar(x=h_mom.index, y=h_mom, name="Headline MoM", marker_color="#1f77b4"), row=1, col=1)
fig_mom.add_trace(go.Bar(x=c_mom.index, y=c_mom, name="Core MoM",     marker_color="#ff7f0e"), row=2, col=1)
for s,e in recs_window:
    fig_mom.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                            fillcolor="rgba(200,0,0,0.15)", opacity=0.3, layer="below", line_width=0), row=1, col=1)
    fig_mom.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                            fillcolor="rgba(200,0,0,0.15)", opacity=0.3, layer="below", line_width=0), row=2, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=2, col=1)
fig_mom.update_layout(showlegend=False, hovermode="x unified")

# 3) Core index & 3M ann
fig_core = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                         subplot_titles=("Core CPI Index", "3‑Mo Annualised Core CPI %"))
fig_core.add_trace(go.Scatter(x=c_3m.index, y=c_3m, name="3M Ann.", line_color="#1f77b4"), row=2, col=1)
fig_core.add_trace(go.Scatter(x=core.loc[idx].index, y=core.loc[idx], name="Core Index", line_color="#ff7f0e"), row=1, col=1)
for s,e in recs_window:
    fig_core.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                             fillcolor="rgba(200,0,0,0.15)", opacity=0.3, layer="below", line_width=0), row=1, col=1)
    fig_core.add_shape(dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                             fillcolor="rgba(200,0,0,0.15)", opacity=0.3, layer="below", line_width=0), row=2, col=1)
fig_core.update_yaxes(title_text="Index Level", row=1, col=1)
fig_core.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig_core.update_layout(hovermode="x unified")

# --------------------------------------------------
# CHART 4 & 5: CPI select categories (dynamic fetch)
# --------------------------------------------------
# Map of CPI component names to FRED series IDs
category_series = {
    'All items': 'CPIAUCNS',
    'Natural gas (piped)': 'CUSR0000SEHB01',
    'Tobacco & smoking products': 'CUSR0000SASAC',
    'Meats, poultry, fish, & eggs': 'CUSR0000SEMT',
    'Motor vehicle insurance': 'CUSR0000SEMC1',
    'Motor vehicle maintenance & repair': 'CUSR0000SETR',
    "Owners' equivalent rent": 'CUSR0000SEHA',
    'Rent of primary residence': 'CUSR0000SEH',
    'Food away from home': 'CUSR0000SEFA',
    'Hospital services': 'CUSR0000SEHC',
    'Electricity': 'CUSR0000SEEX',
    'Nonalcoholic beverages': 'CUSR0000SEEC',
    "Physicians' services": 'CUSR0000SEHC1',
    'Alcoholic beverages': 'CUSR0000SEEA',
    'Dairy & related products': 'CUSR0000SEMD',
    'Used cars and trucks': 'CUSR0000SETC',
    'Medical care commodities': 'CUSR0000SEMC',
    'Other food at home': 'CUSR0000SEFO',
    'New vehicles': 'CUSR0000SETT01',
    'Cereal & bakery products': 'CUSR0000SEM',
    'Apparel': 'CUSR0000SEPA',
    'Fruits and vegetables': 'CUSR0000SEFV',
    'Airline fare': 'CUSR0000SEFV01',
    'Fuel oil': 'CUSR0000SEXH01',
    'Gasoline (all types)': 'CUSR0000SETA01'
}

# Fetch and compute current YoY and MoM for each category
yoy_vals = {}
mom_vals = {}
for name, sid in category_series.items():
    series = load_series(sid, start_date_full)
    yoy_vals[name] = series.pct_change(12).iloc[-1] * 100
    mom_vals[name] = series.pct_change(1).iloc[-1] * 100

# Build sorted DataFrames
df_yoy = pd.Series(yoy_vals).sort_values()
df_mom = pd.Series(mom_vals).sort_values()

# Horizontal bar chart function
def make_cat_fig(data: pd.Series, title: str):
    fig = go.Figure(go.Bar(
        x=data.values,
        y=data.index,
        orientation='h',
        marker_color=[('#ff7f0e' if cat=='All items' else '#1f77b4') for cat in data.index]
    ))
    fig.update_layout(
        title=title,
        xaxis_title='% change',
        yaxis=dict(autorange='reversed'),
        height=800,
        margin=dict(l=200)
    )
    return fig

fig_cat_yoy = make_cat_fig(df_yoy, 'CPI select categories (y/y % change)')
fig_cat_mom = make_cat_fig(df_mom, 'CPI select categories (m/m % change)')

# --------------------------------------------------
# RENDER
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
st.title("US Inflation Dashboard")
st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)
# category breakdowns
st.plotly_chart(fig_cat_yoy, use_container_width=True)
st.plotly_chart(fig_cat_mom, use_container_width=True)

with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** FRED via pandas-datareader  
        **Series:** Headline CPI (CPIAUCNS), Core CPI (CPILFESL), Recessions (USREC)  
        **3‑Mo annualised formula:** ((CPI_t / CPI_{t-3}) ** 4 – 1) × 100
        """
    )
st.caption(f"© {datetime.today().year} • Built with Streamlit & Plotly • Data: FRED")
