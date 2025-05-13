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
# CHART 4: CPI select categories (y/y % change)
# --------------------------------------------------
# last values from April 2025 (static for demonstration)
yoy_vals = {
    'Natural gas (piped)': 15.7,
    'Tobacco & smoking products': 7.1,
    'Meats, poultry, fish, & eggs': 7.0,
    'Motor vehicle insurance': 6.4,
    'Motor vehicle maintenance & repair': 5.6,
    "Owners' equivalent rent": 4.3,
    'Rent of primary residence': 4.0,
    'Food away from home': 3.9,
    'Hospital services': 3.6,
    'Electricity': 3.6,
    'Nonalcoholic beverages': 3.2,
    "Physicians' services": 3.1,
    'All items': 2.3,
    'Alcoholic beverages': 1.8,
    'Dairy & related products': 1.6,
    'Used cars and trucks': 1.5,
    'Medical care commodities': 1.0,
    'Other food at home': 0.7,
    'New vehicles': 0.3,
    'Cereal & bakery products': 0.0,
    'Apparel': -0.7,
    'Fruits and vegetables': -0.9,
    'Airline fare': -7.9,
    'Fuel oil': -9.6,
    'Gasoline (all types)': -11.8,
}
df_yoy = pd.Series(yoy_vals).sort_values()
fig_cat_yoy = go.Figure(go.Bar(
    x=df_yoy.values,
    y=df_yoy.index,
    orientation='h',
    marker_color=['#ff7f0e' if name=='All items' else '#1f77b4' for name in df_yoy.index]
))
fig_cat_yoy.update_layout(
    title='CPI select categories (y/y % change)',
    xaxis_title='% change',
    yaxis=dict(autorange='reversed'),
    height=800,
    margin=dict(l=200)
)

# --------------------------------------------------
# CHART 5: CPI select categories (m/m % change)
# --------------------------------------------------
mom_vals = {
    'Natural gas (piped)': 3.7,
    'Electricity': 0.8,
    'Nonalcoholic beverages': 0.7,
    'Motor vehicle maintenance & repair': 0.7,
    'Motor vehicle insurance': 0.6,
    'Hospital services': 0.6,
    'Food away from home': 0.4,
    'Medical care commodities': 0.4,
    "Owners' equivalent rent": 0.4,
    "Physicians' services": 0.3,
    'Rent of primary residence': 0.3,
    'Tobacco & smoking products': 0.3,
    'All items': 0.2,
    'Alcoholic beverages': 0.0,
    'New vehicles': 0.0,
    'Gasoline (all types)': -0.1,
    'Other food at home': -0.1,
    'Dairy & related products': -0.2,
    'Apparel': -0.2,
    'Fruits and vegetables': -0.4,
    'Cereal & bakery products': -0.5,
    'Used cars and trucks': -0.5,
    'Fuel oil': -1.3,
    'Meats, poultry, fish, & eggs': -1.6,
    'Airline fare': -2.8,
}
df_mom = pd.Series(mom_vals).sort_values()
fig_cat_mom = go.Figure(go.Bar(
    x=df_mom.values,
    y=df_mom.index,
    orientation='h',
    marker_color=['#ff7f0e' if name=='All items' else '#1f77b4' for name in df_mom.index]
))
fig_cat_mom.update_layout(
    title='CPI select categories (m/m % change)',
    xaxis_title='% change',
    yaxis=dict(autorange='reversed'),
    height=800,
    margin=dict(l=200)
)

# --------------------------------------------------
# CHART 4: CPI select categories (y/y % change)
# --------------------------------------------------
# Static example values for April 2025; replace with dynamic fetch if desired
```python
# Year-over-year category changes
yoy_vals = {
    'Natural gas (piped)': 15.7,
    'Tobacco & smoking products': 7.1,
    'Meats, poultry, fish, & eggs': 7.0,
    'Motor vehicle insurance': 6.4,
    'Motor vehicle maintenance & repair': 5.6,
    "Owners' equivalent rent": 4.3,
    'Rent of primary residence': 4.0,
    'Food away from home': 3.9,
    'Hospital services': 3.6,
    'Electricity': 3.6,
    'Nonalcoholic beverages': 3.2,
    "Physicians' services": 3.1,
    'All items': 2.3,
    'Alcoholic beverages': 1.8,
    'Dairy & related products': 1.6,
    'Used cars and trucks': 1.5,
    'Medical care commodities': 1.0,
    'Other food at home': 0.7,
    'New vehicles': 0.3,
    'Cereal & bakery products': 0.0,
    'Apparel': -0.7,
    'Fruits and vegetables': -0.9,
    'Airline fare': -7.9,
    'Fuel oil': -9.6,
    'Gasoline (all types)': -11.8,
}

import pandas as pd
from plotly.graph_objects import Figure, Bar

# Prepare horizontal bar chart data
cat_yoy = pd.Series(yoy_vals).sort_values()
fig_cat_yoy = Figure(data=[Bar(
    x=cat_yoy.values,
    y=cat_yoy.index,
    orientation='h',
    marker_color=[('#ff7f0e' if i == 'All items' else '#1f77b4') for i in cat_yoy.index]
)])
fig_cat_yoy.update_layout(
    title='CPI select categories (y/y % change)',
    xaxis_title='% change',
    yaxis=dict(autorange='reversed'),
    height=800,
    margin=dict(l=200)
)
```

# --------------------------------------------------
# CHART 5: CPI select categories (m/m % change)
# --------------------------------------------------
```python
mom_vals = {
    'Natural gas (piped)': 3.7,
    'Electricity': 0.8,
    'Nonalcoholic beverages': 0.7,
    'Motor vehicle maintenance & repair': 0.7,
    'Motor vehicle insurance': 0.6,
    'Hospital services': 0.6,
    'Food away from home': 0.4,
    'Medical care commodities': 0.4,
    "Owners' equivalent rent": 0.4,
    "Physicians' services": 0.3,
    'Rent of primary residence': 0.3,
    'Tobacco & smoking products': 0.3,
    'All items': 0.2,
    'Alcoholic beverages': 0.0,
    'New vehicles': 0.0,
    'Gasoline (all types)': -0.1,
    'Other food at home': -0.1,
    'Dairy & related products': -0.2,
    'Apparel': -0.2,
    'Fruits and vegetables': -0.4,
    'Cereal & bakery products': -0.5,
    'Used cars and trucks': -0.5,
    'Fuel oil': -1.3,
    'Meats, poultry, fish, & eggs': -1.6,
    'Airline fare': -2.8,
}

cat_mom = pd.Series(mom_vals).sort_values()
fig_cat_mom = Figure(data=[Bar(
    x=cat_mom.values,
    y=cat_mom.index,
    orientation='h',
    marker_color=[('#ff7f0e' if i == 'All items' else '#1f77b4') for i in cat_mom.index]
)])
fig_cat_mom.update_layout(
    title='CPI select categories (m/m % change)',
    xaxis_title='% change',
    yaxis=dict(autorange='reversed'),
    height=800,
    margin=dict(l=200)
)
```

# --------------------------------------------------
# RENDER
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
