# ────────────────────────────────────────────────────────────────────────────────
#   US CPI DASHBOARD — ZERO Boolean-mask indexing  (v2025-07-07)
# ────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# 1. Streamlit page setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------
# 2. Helper → FRED loader (monthly, forward-filled)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fred(series: str, start="1990-01-01") -> pd.Series:
    s = DataReader(series, "fred", start)[series]
    return s.asfreq("MS").ffill()

# ------------------------------------------------------------------------------
# 3. Load series
# ------------------------------------------------------------------------------
headline = fred("CPIAUCNS")   # Headline CPI
core     = fred("CPILFESL")   # Core CPI
recess   = fred("USREC")      # Recession flag 0/1

if headline.empty or core.empty or recess.empty:
    st.error("Could not fetch all FRED series. Please try again later.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. Canonical monthly index & re-index everything up-front
# ------------------------------------------------------------------------------
idx_all = pd.date_range(headline.index.min(),
                        headline.index.max(),
                        freq="MS")

headline = headline.reindex(idx_all)
core     = core.reindex(idx_all)
recess   = recess.reindex(idx_all)

# All derived series carry the **same** index
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# ------------------------------------------------------------------------------
# 5. Sidebar – window selection (label slicing only)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Time Range")
    quick = st.selectbox("Preset:", ["1Y", "3Y", "5Y", "10Y", "All"], index=2)
    if quick == "All":
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

# Label-based slices (no Boolean mask!)
h      = headline.loc[start:end]
c      = core    .loc[start:end]
h_yoy  = headline_yoy.loc[start:end]
c_yoy  = core_yoy    .loc[start:end]
h_mom  = headline_mom.loc[start:end]
c_mom  = core_mom    .loc[start:end]
c_3m   = core_3m_ann .loc[start:end]
rec_w  = recess.loc[start:end]

# ------------------------------------------------------------------------------
# 6. Recession periods as list of tuples  (no masks)
# ------------------------------------------------------------------------------
def recession_spans(flag: pd.Series):
    flag = flag.fillna(0).astype(int)
    starts = flag[(flag==1) & (flag.shift(1, fill_value=0)==0)].index.to_list()
    ends   = flag[(flag==0) & (flag.shift(1, fill_value=0)==1)].index.to_list()
    if flag.iloc[-1] == 1:                       # open-ended recession
        ends.append(flag.index[-1])
    return [(s, e) for s, e in zip(starts, ends)]

recs_in_window = [(s, e) for s, e in recession_spans(recess)
                  if e >= start and s <= end]

# ------------------------------------------------------------------------------
# 7. Outliers & regimes → recorded as **index lists**, not masks
# ------------------------------------------------------------------------------
# Core YoY outlier = |dev| > 1.5×10-yr σ
sigma_10y = core_yoy.rolling(120, min_periods=24).std()
mean_10y  = core_yoy.rolling(120, min_periods=24).mean()
is_outlier = (core_yoy - mean_10y).abs() > 1.5 * sigma_10y
yoy_out_idx = core_yoy.loc[start:end].index[is_outlier.loc[start:end].values]

# Headline MoM outlier
mom_out_idx = headline_mom.loc[start:end].index[
    ((headline_mom > 0.7) | (headline_mom < -0.2)).loc[start:end].values
]

# Core YoY regimes
regime = pd.Series("Normal", index=core_yoy.index)
regime[core_yoy > 3] = "High"
regime[core_yoy < 2] = "Low"

regime_bands = []
cur = None
for t in regime.index:
    if regime[t] != cur:
        if cur is not None:
            regime_bands[-1]["end"] = t
        regime_bands.append({"reg": regime[t], "start": t})
        cur = regime[t]
regime_bands[-1]["end"] = regime.index[-1]
regime_bands = [b for b in regime_bands if b["end"] >= start and b["start"] <= end]

# ------------------------------------------------------------------------------
# 8. Quick stats
# ------------------------------------------------------------------------------
Δh = (h_yoy.iloc[-1] - h_yoy.iloc[-2]) if len(h_yoy) > 1 else 0
Δc = (c_yoy.iloc[-1] - c_yoy.iloc[-2]) if len(c_yoy) > 1 else 0

st.title("US Inflation Dashboard (robust)")
st.markdown(
    f"**Headline YoY:** {h_yoy.iloc[-1]:.2f}% ({Δh:+.2f} pp)  "
    f"**Core YoY:** {c_yoy.iloc[-1]:.2f}% ({Δc:+.2f} pp)"
)
st.caption("Bands = Core-YoY regimes; diamond = Core-YoY outlier; "
           "star = Headline-MoM outlier.")
st.divider()

# ------------------------------------------------------------------------------
# 9. Charts
# ------------------------------------------------------------------------------
# — YoY —
fig_yoy = go.Figure()
# regime bands
for b in regime_bands:
    color = {"High": "rgba(255,100,0,0.15)",
             "Low":  "rgba(0,200,255,0.15)",
             "Normal":"rgba(255,255,255,0)"}[b["reg"]]
    fig_yoy.add_vrect(x0=b["start"], x1=b["end"],
                      fillcolor=color, layer="below", line_width=0)

# Recession overlays
for s, e in recs_in_window:
    fig_yoy.add_shape(type="rect", x0=s, x1=e, xref="x", yref="paper",
                      y0=0, y1=1,
                      fillcolor="rgba(200,0,0,0.15)", layer="below",
                      line_width=0)

fig_yoy.add_scatter(x=h_yoy.index, y=h_yoy,
                    name="Headline YoY", line=dict(color="#1f77b4"))
fig_yoy.add_scatter(x=c_yoy.index, y=c_yoy,
                    name="Core YoY", line=dict(color="#ff7f0e"))
fig_yoy.add_scatter(x=yoy_out_idx, y=c_yoy.loc[yoy_out_idx],
                    mode="markers", name="Core YoY outlier",
                    marker=dict(symbol="diamond-open", color="red", size=9))

fig_yoy.update_layout(title="CPI YoY (%)",
                      hovermode="x unified",
                      legend_orientation="h",
                      margin=dict(t=50))

# — MoM —
fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        subplot_titles=("Headline MoM", "Core MoM"))
fig_mom.add_bar(x=h_mom.index, y=h_mom,
                name="Headline MoM", marker_color="#1f77b4",
                row=1, col=1)
fig_mom.add_bar(x=c_mom.index, y=c_mom,
                name="Core MoM", marker_color="#ff7f0e",
                row=2, col=1)
for s, e in recs_in_window:
    for r in (1, 2):
        fig_mom.add_shape(type="rect", x0=s, x1=e,
                          xref="x", yref="paper",
                          y0=0, y1=1,
                          fillcolor="rgba(200,0,0,0.15)",
                          layer="below", line_width=0,
                          row=r, col=1)

fig_mom.add_scatter(x=mom_out_idx, y=h_mom.loc[mom_out_idx],
                    mode="markers", name="Headline MoM outlier",
                    marker=dict(symbol="star-open", color="purple", size=9),
                    row=1, col=1)

fig_mom.update_yaxes(title_text="% MoM", zeroline=True, row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", zeroline=True, row=2, col=1)
fig_mom.update_layout(showlegend=True,
                      hovermode="x unified",
                      margin=dict(t=50))

# — Core index & 3-mo ann —
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
for s, e in recs_in_window:
    for r in (1, 2):
        fig_core.add_shape(type="rect", x0=s, x1=e,
                           xref="x", yref="paper",
                           y0=0, y1=1,
                           fillcolor="rgba(200,0,0,0.15)",
                           layer="below", line_width=0,
                           row=r, col=1)
fig_core.update_layout(hovermode="x unified",
                       legend_orientation="h",
                       margin=dict(t=50))

# Render
st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

with st.expander("Methodology & thresholds"):
    st.write("""
    * **Core-YoY regimes:** High > 3 %; Low < 2 %  
    * **Core-YoY outlier:** |Core-YoY – 10-yr mean| > 1.5×10-yr σ  
    * **Headline-MoM outlier:** MoM > 0.7 % or < -0.2 %  
    * **Recession bars:** NBER USREC
    """)

# ────────────────────────────────────────────────────────────────────────────────
