# -------------  US CPI DASHBOARD  (robust, no boolean-mask bugs)  ----------------
import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="US CPI Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ────────────────────────────────────────────────────────────────────────────────
# Helper: load FRED series with monthly freq
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def fred(series_id: str, start="1990-01-01") -> pd.Series:
    s = DataReader(series_id, "fred", start)[series_id]
    return s.asfreq("MS").ffill()

# ────────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ────────────────────────────────────────────────────────────────────────────────
headline = fred("CPIAUCNS")     # headline CPI
core     = fred("CPILFESL")     # core CPI
recess   = fred("USREC")        # NBER recession flag (0/1)

if headline.empty or core.empty or recess.empty:
    st.error("FRED data unavailable – try later.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# 2. Canonical monthly index  →  re-index everything
# ────────────────────────────────────────────────────────────────────────────────
idx_all = pd.date_range(headline.index.min(),
                        headline.index.max(),
                        freq="MS")

headline = headline.reindex(idx_all)
core     = core.reindex(idx_all)
recess   = recess.reindex(idx_all)

# Derived series – immediately re-indexed (so shapes always match)
headline_yoy = headline.pct_change(12).reindex(idx_all) * 100
core_yoy     = core.pct_change(12).reindex(idx_all)     * 100
headline_mom = headline.pct_change(1).reindex(idx_all)  * 100
core_mom     = core.pct_change(1).reindex(idx_all)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1).reindex(idx_all) * 100

# ────────────────────────────────────────────────────────────────────────────────
# 3. Sidebar: date window
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("US CPI Dashboard")
    quick = st.selectbox("Quick range",
                         ["1Y", "3Y", "5Y", "10Y", "All"],
                         index=2)
    custom = st.checkbox("Custom range")
    if custom:
        rng = st.slider("Pick dates",
                        min_value=idx_all.min().date(),
                        max_value=idx_all.max().date(),
                        value=(idx_all[-60].date(), idx_all.max().date()),
                        format="YYYY-MM")
        start, end = map(pd.Timestamp, rng)
    else:
        end = idx_all.max()
        if quick == "All":
            start = idx_all.min()
        else:
            years = int(quick.rstrip("Y"))
            start = end - pd.DateOffset(years=years)

# Window slices – **pure label slicing** (no Boolean mask arrays)
h      = headline.loc[start:end]
c      = core    .loc[start:end]
h_yoy  = headline_yoy.loc[start:end]
c_yoy  = core_yoy    .loc[start:end]
h_mom  = headline_mom.loc[start:end]
c_mom  = core_mom    .loc[start:end]
c_3m   = core_3m_ann .loc[start:end]
rec_w  = recess.loc[start:end]

# ────────────────────────────────────────────────────────────────────────────────
# 4. Recession periods that overlap the window
# ────────────────────────────────────────────────────────────────────────────────
def rec_periods(flag: pd.Series):
    flag = flag.fillna(0).astype(int)
    starts = flag[(flag==1) & (flag.shift(1, fill_value=0)==0)].index
    ends   = flag[(flag==0) & (flag.shift(1, fill_value=0)==1)].index
    if flag.iloc[-1] == 1:  # unfinished recession
        ends = ends.append(pd.Index([flag.index[-1]]))
    return [(s, e) for s, e in zip(starts, ends)]

recs = [(s, e) for s, e in rec_periods(recess)
        if e >= start and s <= end]

# ────────────────────────────────────────────────────────────────────────────────
# 5. Outliers & regimes  (core YoY basis)
# ────────────────────────────────────────────────────────────────────────────────
roll_std = core_yoy.rolling(120, min_periods=24).std()
yoy_out  = (core_yoy - core_yoy.rolling(120).mean()).abs() > 1.5 * roll_std
yoy_pts  = core_yoy.loc[start:end].where(yoy_out).dropna()

mom_out  = (headline_mom > 0.7) | (headline_mom < -0.2)
mom_pts  = headline_mom.loc[start:end].where(mom_out).dropna()

regime = pd.Series("Normal", index=core_yoy.index)
regime[core_yoy > 3] = "High"
regime[core_yoy < 2] = "Low"

bands = []
cur = None
for t in regime.index:
    r = regime[t]
    if r != cur:
        if cur:
            bands[-1]["end"] = t
        bands.append({"reg": r, "start": t, "end": t})
        cur = r
    else:
        bands[-1]["end"] = t
bands = [b for b in bands if b["end"] >= start and b["start"] <= end]

# ────────────────────────────────────────────────────────────────────────────────
# 6. Quick commentary
# ────────────────────────────────────────────────────────────────────────────────
Δh = h_yoy.iloc[-1] - h_yoy.iloc[-2] if len(h_yoy) > 1 else 0
Δc = c_yoy.iloc[-1] - c_yoy.iloc[-2] if len(c_yoy) > 1 else 0

st.title("US Inflation Dashboard")
st.markdown(
    f"**Headline YoY:** {h_yoy.iloc[-1]:.2f}% ({Δh:+.2f} pp) &nbsp;&nbsp; "
    f"**Core YoY:** {c_yoy.iloc[-1]:.2f}% ({Δc:+.2f} pp)")
st.caption(
    "Bands = regimes (core YoY > 3 % or < 2 %).  "
    "Diamonds = core YoY outliers.  Stars = headline MoM outliers."
)
st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# 7. Charts
# ────────────────────────────────────────────────────────────────────────────────
# YoY
fig_yoy = go.Figure()
for b in bands:
    fig_yoy.add_vrect(x0=b["start"], x1=b["end"],
                      fillcolor={"High":"rgba(255,100,0,0.13)",
                                 "Low":"rgba(0,200,255,0.13)",
                                 "Normal":"rgba(255,255,255,0)"}[b["reg"]],
                      layer="below", line_width=0)
for s,e in recs:
    fig_yoy.add_shape(type="rect", x0=s, x1=e, xref="x", yref="paper",
                      y0=0, y1=1,
                      fillcolor="rgba(200,0,0,0.15)",
                      layer="below", line_width=0)
fig_yoy.add_scatter(x=h_yoy.index, y=h_yoy, name="Headline YoY",
                    line=dict(color="#1f77b4"))
fig_yoy.add_scatter(x=c_yoy.index, y=c_yoy, name="Core YoY",
                    line=dict(color="#ff7f0e"))
fig_yoy.add_scatter(x=yoy_pts.index, y=yoy_pts,
                    mode="markers", name="Core YoY outlier",
                    marker=dict(symbol="diamond-open", size=9, color="red"))
fig_yoy.update_layout(title="CPI YoY (%)",
                      hovermode="x unified",
                      legend_orientation="h",
                      margin=dict(t=60))

# MoM
fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        subplot_titles=("Headline MoM", "Core MoM"))
fig_mom.add_bar(x=h_mom.index, y=h_mom, name="Headline MoM",
                marker_color="#1f77b4", row=1, col=1)
fig_mom.add_bar(x=c_mom.index, y=c_mom, name="Core MoM",
                marker_color="#ff7f0e", row=2, col=1)
for s,e in recs:
    for r in (1,2):
        fig_mom.add_shape(type="rect", x0=s, x1=e, xref="x", yref="paper",
                          y0=0, y1=1,
                          fillcolor="rgba(200,0,0,0.15)",
                          layer="below", line_width=0,
                          row=r, col=1)
fig_mom.add_scatter(x=mom_pts.index, y=mom_pts,
                    mode="markers", name="Headline MoM outlier",
                    marker=dict(symbol="star-open", size=9, color="purple"),
                    row=1, col=1)
fig_mom.update_yaxes(zeroline=True, title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(zeroline=True, title_text="% MoM", row=2, col=1)
fig_mom.update_layout(showlegend=True,
                      hovermode="x unified",
                      margin=dict(t=60))

# Core index & 3-mo ann.
fig_core = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.02,
                         subplot_titles=("Core CPI index",
                                         "3-mo-ann Core CPI %"))
fig_core.add_scatter(x=c.index, y=c,
                     name="Core index", line=dict(color="#ff7f0e"),
                     row=1, col=1)
fig_core.add_scatter(x=c_3m.index, y=c_3m,
                     name="3-mo ann.", line=dict(color="#1f77b4"),
                     row=2, col=1)
for s,e in recs:
    for r in (1,2):
        fig_core.add_shape(type="rect", x0=s, x1=e, xref="x", yref="paper",
                           y0=0, y1=1,
                           fillcolor="rgba(200,0,0,0.15)",
                           layer="below", line_width=0,
                           row=r, col=1)
fig_core.update_layout(hovermode="x unified",
                       legend_orientation="h",
                       margin=dict(t=60))

# Render
st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

with st.expander("Methodology & thresholds"):
    st.write("""
    **Regimes:** Core YoY > 3 % → “High”; < 2 % → “Low”.  
    **Core YoY outlier:** deviation > 1.5× rolling 10-yr σ.  
    **Headline MoM outlier:** move > +0.7 % or < –0.2 %.  
    **Recession bars:** NBER USREC series.
    """)
# -------------------------------------------------------------------------------
