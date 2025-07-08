# ────────────────────────────────────────────────────────────────────────────────
#  US CPI DASHBOARD  –  No pandas-datareader, no Boolean-mask errors
# ────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="US CPI Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------
# 1.  FRED CSV loader  (cached)
# ------------------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

@st.cache_data(show_spinner=True)
def fred_series(sid: str, start="1990-01-01") -> pd.Series:
    url = FRED_CSV.format(sid=sid)
    df = pd.read_csv(url, parse_dates=["DATE"])
    s = (df
         .set_index("DATE")[sid]
         .replace({"." : None})          # FRED missing value placeholder
         .astype(float)
         .asfreq("MS")                   # monthly skeleton
         .ffill())
    return s.loc[start:]

# ------------------------------------------------------------------------------
# 2.  Fetch series
# ------------------------------------------------------------------------------
headline = fred_series("CPIAUCNS")   # headline CPI
core     = fred_series("CPILFESL")   # core CPI
recess   = fred_series("USREC")      # recession flag (0/1)

if headline.empty or core.empty or recess.empty:
    st.error("Could not download all FRED series. Try again later.")
    st.stop()

# ------------------------------------------------------------------------------
# 3.  Canonical index  +  derived series
# ------------------------------------------------------------------------------
idx_all = pd.date_range(headline.index.min(),
                        headline.index.max(),
                        freq="MS")

headline = headline.reindex(idx_all)
core     = core.reindex(idx_all)
recess   = recess.reindex(idx_all)

headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m      = ((core / core.shift(3))**4 - 1) * 100

# ------------------------------------------------------------------------------
# 4.  Sidebar – window (label slicing only)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Time Range")
    quick = st.selectbox("Preset:", ["1Y","3Y","5Y","10Y","All"], index=2)
    if quick=="All":
        start, end = idx_all.min(), idx_all.max()
    else:
        yrs = int(quick.rstrip("Y"))
        end   = idx_all.max()
        start = end - pd.DateOffset(years=yrs)
    if st.checkbox("Custom range"):
        rng = st.slider("Pick dates",
                        min_value=idx_all.min().date(),
                        max_value=idx_all.max().date(),
                        value=(start.date(), end.date()),
                        format="YYYY-MM")
        start, end = map(pd.Timestamp, rng)

# Window slices (no Boolean masks)
h      = headline.loc[start:end]
c      = core.loc[start:end]
h_yoy  = headline_yoy.loc[start:end]
c_yoy  = core_yoy.loc[start:end]
h_mom  = headline_mom.loc[start:end]
c_mom  = core_mom.loc[start:end]
c_3m_w = core_3m.loc[start:end]
rec_w  = recess.loc[start:end]

# ------------------------------------------------------------------------------
# 5.  Recession spans  (list of tuples)
# ------------------------------------------------------------------------------
def spans(flag: pd.Series):
    flag = flag.fillna(0).astype(int)
    s = flag[(flag==1) & (flag.shift(1,fill_value=0)==0)].index
    e = flag[(flag==0) & (flag.shift(1,fill_value=0)==1)].index
    if flag.iloc[-1]==1: e = e.append(pd.Index([flag.index[-1]]))
    return [(i,j) for i,j in zip(s,e)]

recs = [(i,j) for i,j in spans(recess) if j>=start and i<=end]

# ------------------------------------------------------------------------------
# 6.  Outliers & regimes  (simple)
# ------------------------------------------------------------------------------
sigma = core_yoy.rolling(120,min_periods=24).std()
mu    = core_yoy.rolling(120,min_periods=24).mean()
out_yoy = core_yoy.loc[start:end][(core_yoy-mu).abs() > 1.5*sigma]

out_mom = headline_mom.loc[start:end][(headline_mom>0.7)|(headline_mom<-0.2)]

regime = pd.Series("Normal", index=core_yoy.index)
regime[core_yoy>3] = "High"
regime[core_yoy<2] = "Low"
bands = []
cur = None
for t in regime.index:
    if regime[t]!=cur:
        if cur: bands[-1]["end"]=t
        bands.append({"reg":regime[t],"start":t})
        cur=regime[t]
bands[-1]["end"]=regime.index[-1]
bands=[b for b in bands if b["end"]>=start and b["start"]<=end]

# ------------------------------------------------------------------------------
# 7.  Header & stats
# ------------------------------------------------------------------------------
Δh = h_yoy.iloc[-1]-h_yoy.iloc[-2] if len(h_yoy)>1 else 0
Δc = c_yoy.iloc[-1]-c_yoy.iloc[-2] if len(c_yoy)>1 else 0
st.title("US Inflation Dashboard")
st.markdown(f"**Headline YoY:** {h_yoy.iloc[-1]:.2f}% ({Δh:+.2f} pp)  "
            f"**Core YoY:** {c_yoy.iloc[-1]:.2f}% ({Δc:+.2f} pp)")
st.caption("Diamonds = core YoY outlier • Stars = headline MoM outlier")
st.divider()

# ------------------------------------------------------------------------------
# 8.  Charts
# ------------------------------------------------------------------------------
# – YoY –
fig_yoy = go.Figure()
for b in bands:
    color={"High":"rgba(255,100,0,0.15)",
           "Low":"rgba(0,200,255,0.15)",
           "Normal":"rgba(255,255,255,0)"}[b["reg"]]
    fig_yoy.add_vrect(b["start"], b["end"], fillcolor=color,
                      layer="below", line_width=0)
for s,e in recs:
    fig_yoy.add_shape("rect", x0=s, x1=e, xref="x", yref="paper",
                      y0=0, y1=1, fillcolor="rgba(200,0,0,0.15)",
                      layer="below", line_width=0)
fig_yoy.add_scatter(x=h_yoy.index, y=h_yoy,
                    name="Headline YoY", line=dict(color="#1f77b4"))
fig_yoy.add_scatter(x=c_yoy.index, y=c_yoy,
                    name="Core YoY", line=dict(color="#ff7f0e"))
fig_yoy.add_scatter(x=out_yoy.index, y=out_yoy,
                    mode="markers", name="Core YoY outlier",
                    marker=dict(symbol="diamond-open",color="red",size=9))
fig_yoy.update_layout(title="CPI YoY (%)", hovermode="x unified",
                      legend_orientation="h", margin=dict(t=50))

# – MoM –
fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        subplot_titles=("Headline MoM","Core MoM"))
fig_mom.add_bar(x=h_mom.index, y=h_mom,
                name="Headline MoM", marker_color="#1f77b4", row=1,col=1)
fig_mom.add_bar(x=c_mom.index, y=c_mom,
                name="Core MoM", marker_color="#ff7f0e", row=2,col=1)
for s,e in recs:
    for r in (1,2):
        fig_mom.add_shape("rect", x0=s, x1=e, xref="x", yref="paper",
                          y0=0, y1=1, fillcolor="rgba(200,0,0,0.15)",
                          layer="below", line_width=0, row=r,col=1)
fig_mom.add_scatter(x=out_mom.index, y=out_mom,
                    mode="markers", name="Headline MoM outlier",
                    marker=dict(symbol="star-open",color="purple",size=9),
                    row=1,col=1)
fig_mom.update_yaxes(title_text="% MoM", zeroline=True, row=1,col=1)
fig_mom.update_yaxes(title_text="% MoM", zeroline=True, row=2,col=1)
fig_mom.update_layout(showlegend=True, hovermode="x unified",
                      margin=dict(t=50))

# – Core index & 3-mo ann –
fig_core = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.02,
                         subplot_titles=("Core CPI index","3-mo-ann Core CPI %"))
fig_core.add_scatter(x=c.index, y=c,
                     name="Core index", line=dict(color="#ff7f0e"),
                     row=1,col=1)
fig_core.add_scatter(x=c_3m_w.index, y=c_3m_w,
                     name="3-mo ann.", line=dict(color="#1f77b4"),
                     row=2,col=1)
for s,e in recs:
    for r in (1,2):
        fig_core.add_shape("rect", x0=s, x1=e, xref="x", yref="paper",
                           y0=0, y1=1, fillcolor="rgba(200,0,0,0.15)",
                           layer="below", line_width=0, row=r,col=1)
fig_core.update_layout(hovermode="x unified", legend_orientation="h",
                       margin=dict(t=50))

# Render
st.plotly_chart(fig_yoy, use_container_width=True)
st.plotly_chart(fig_mom, use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

with st.expander("Methodology & thresholds"):
    st.write("""
    **Core-YoY regimes:** High > 3 %   Low < 2 %  
    **Core-YoY outlier:** |Core-YoY − 10-yr μ| > 1.5 σ  
    **Headline-MoM outlier:** MoM > 0.7 % or < -0.2 %  
    **Recession bars:** NBER USREC
    """)
