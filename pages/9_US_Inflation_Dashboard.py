import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="US CPI Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------
# 1. Load FRED series (cached)
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fred(series: str, start="1990-01-01") -> pd.Series:
    try:
        s = DataReader(series, "fred", start)[series]
        return s.asfreq("MS").ffill()
    except Exception:
        return pd.Series(dtype=float)

headline = fred("CPIAUCNS")
core     = fred("CPILFESL")
recess   = fred("USREC")

if headline.empty or core.empty or recess.empty:
    st.error("FRED data unavailable — try again later.")
    st.stop()

# --------------------------------------------------------------------
# 2. Canonical monthly index + fill gaps
# --------------------------------------------------------------------
idx_full = pd.date_range(headline.index.min(), headline.index.max(), freq="MS")
headline = headline.reindex(idx_full)
core     = core.reindex(idx_full)
recess   = recess.reindex(idx_full)

# Derived series
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# --------------------------------------------------------------------
# 3. Sidebar – period selection
# --------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Interactive US inflation dashboard with outlier & macro-regime detection."
    )
    period = st.selectbox("Quick range",
                          ["1Y", "3Y", "5Y", "10Y", "All"],
                          index=2)
    custom = st.checkbox("Custom range")
    if custom:
        rng = st.slider("Pick dates",
                        min_value=idx_full.min().date(),
                        max_value=idx_full.max().date(),
                        value=(idx_full[-60].date(), idx_full.max().date()),
                        format="YYYY-MM")
        start, end = map(pd.Timestamp, rng)
    else:
        end = idx_full.max()
        if period == "All":
            start = idx_full.min()
        else:
            yrs = int(period.rstrip("Y"))
            start = end - pd.DateOffset(years=yrs)

# --------------------------------------------------------------------
# 4. Windowed slices (loc — no Boolean mask)
# --------------------------------------------------------------------
h_win  = headline.loc[start:end]
c_win  = core    .loc[start:end]
h_yoy  = headline_yoy.loc[start:end]
c_yoy  = core_yoy    .loc[start:end]
h_mom  = headline_mom.loc[start:end]
c_mom  = core_mom    .loc[start:end]
c_3m   = core_3m_ann .loc[start:end]
rec_win = recess.loc[start:end]

# --------------------------------------------------------------------
# 5. Recession periods inside window
# --------------------------------------------------------------------
def rec_periods(flag):
    f = flag.fillna(0).astype(int)
    shift = f.diff()
    starts = f[(f == 1) & (shift == 1)].index
    if f.iloc[0] == 1:
        starts = starts.insert(0, f.index[0])
    ends = f[(f == 0) & (shift == -1)].index
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([f.index[-1]]))
    return [(s, e) for s, e in zip(starts, ends)]

recs_window = [(s, e) for s, e in rec_periods(recess)
               if e >= start and s <= end]

# --------------------------------------------------------------------
# 6. Outliers & regimes (on **core YoY**)
# --------------------------------------------------------------------
rolling_std = core_yoy.rolling(120, min_periods=24).std()
yoy_outlier = (core_yoy - core_yoy.rolling(120).mean()
               ).abs() > 1.5 * rolling_std
yoy_out_idx = yoy_outlier.loc[start:end][lambda s: s].index

mom_outlier = (headline_mom > 0.7) | (headline_mom < -0.2)
mom_out_idx = mom_outlier.loc[start:end][lambda s: s].index

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

# --------------------------------------------------------------------
# 7. Quick commentary
# --------------------------------------------------------------------
Δh = h_yoy.iloc[-1] - h_yoy.iloc[-2] if len(h_yoy) > 1 else 0
Δc = c_yoy.iloc[-1] - c_yoy.iloc[-2] if len(c_yoy) > 1 else 0
st.title("US Inflation Dashboard")
st.markdown(
    f"**Headline YoY:** {h_yoy.iloc[-1]:.2f}% ({Δh:+.2f} pp) &nbsp;&nbsp; "
    f"**Core YoY:** {c_yoy.iloc[-1]:.2f}% ({Δc:+.2f} pp)")
st.caption(f"Data through {idx_full[-1].strftime('%B %Y')} — "
           "bands=regimes, diamonds=core YoY outliers, stars=headline MoM outliers")
st.divider()

# --------------------------------------------------------------------
# 8. Charts
# --------------------------------------------------------------------
# YoY chart with regimes & outliers
fig_yoy = go.Figure()
for b in bands:
    color = {"High": "rgba(255,100,0,0.12)",
             "Low": "rgba(0,200,255,0.12)",
             "Normal": "rgba(255,255,255,0)"}[b["reg"]]
    fig_yoy.add_vrect(x0=b["start"], x1=b["end"],
                      fillcolor=color, opacity=0.3,
                      layer="below", line_width=0)
for s, e in recs_window:
    fig_yoy.add_shape(type="rect", x0=s, x1=e, xref="x", yref="paper",
                      y0=0, y1=1, fillcolor="rgba(200,0,0,0.15)",
                      layer="below", line_width=0)
fig_yoy.add_scatter(x=h_yoy.index, y=h_yoy, name="Headline YoY",
                    line=dict(color="#1f77b4"))
fig_yoy.add_scatter(x=c_yoy.index, y=c_yoy, name="Core YoY",
                    line=dict(color="#ff7f0e"))
fig_yoy.add_scatter(x=yoy_out_idx, y=c_yoy.loc[yoy_out_idx],
                    mode="markers", name="Core YoY outlier",
                    marker=dict(symbol="diamond-open", size=9,
                                color="red"))
fig_yoy.update_layout(title="CPI YoY (%)",
                      hovermode="x unified",
                      legend_orientation="h",
                      margin=dict(t=60))

# MoM chart with outliers
fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        subplot_titles=("Headline MoM", "Core MoM"))
fig_mom.add_bar(x=h_mom.index, y=h_mom, name="Headline MoM",
                marker_color="#1f77b4", row=1, col=1)
fig_mom.add_bar(x=c_mom.index, y=c_mom, name="Core MoM",
                marker_color="#ff7f0e", row=2, col=1)
for s, e in recs_window:
    for r in (1, 2):
        fig_mom.add_shape(type="rect", xref="x", yref="paper",
                          x0=s, x1=e, y0=0, y1=1,
                          fillcolor="rgba(200,0,0,0.15)",
                          layer="below", line_width=0,
                          row=r, col=1)
fig_mom.add_scatter(x=mom_out_idx, y=h_mom.loc[mom_out_idx],
                    mode="markers", name="Headline MoM outlier",
                    marker=dict(symbol="star-open", size=10,
                                color="purple"),
                    row=1, col=1)
fig_mom.update_yaxes(zeroline=True, title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(zeroline=True, title_text="% MoM", row=2, col=1)
fig_mom.update_layout(showlegend=True, hovermode="x unified",
                      margin=dict(t=60))

# Core index & 3-mo annualised
fig_core = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.02,
                         subplot_titles=("Core CPI index",
                                         "
