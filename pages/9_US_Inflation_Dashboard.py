# ────────────────────────────────────────────────────────────────────────────────
#  US CPI DASHBOARD – rock-solid, no third-party fetch libs, no boolean masks
#  Arya Deniz – July 2025
# ────────────────────────────────────────────────────────────────────────────────
import io, requests, streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta

# ------------------------------------------------------------------------------
# 1. page layout
# ------------------------------------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------
# 2. FRED CSV loader (retries once, cached 6 h)
# ------------------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

@st.cache_data(show_spinner=True, ttl=6*60*60)
def fred_series(sid: str, start="1990-01-01") -> pd.Series:
    def _get() -> pd.DataFrame:
        resp = requests.get(FRED_CSV.format(sid=sid), timeout=10)
        if "text/csv" not in resp.headers.get("Content-Type", ""):
            raise ValueError("FRED did not return CSV")
        return pd.read_csv(io.StringIO(resp.text))
    try:
        df = _get()
    except Exception:
        df = _get()                       # single retry

    if "DATE" not in df or sid not in df:
        raise ValueError(f"Unexpected FRED format for {sid}")

    df["DATE"] = pd.to_datetime(df["DATE"])
    ser = (df.set_index("DATE")[sid]
             .replace({".": None})
             .astype(float)
             .asfreq("MS")
             .ffill())
    return ser.loc[start:]

def safe_load(sid: str, label: str) -> pd.Series:
    try:
        return fred_series(sid)
    except Exception as e:
        st.error(f"Failed to download **{label}** ({sid}): {e}")
        st.stop()

headline = safe_load("CPIAUCNS", "Headline CPI")
core     = safe_load("CPILFESL", "Core CPI")
recess   = safe_load("USREC",    "Recession flag")

# ------------------------------------------------------------------------------
# 3. canonical monthly index & transformations
# ------------------------------------------------------------------------------
idx_all = pd.date_range(headline.index.min(),
                        headline.index.max(), freq="MS")
headline = headline.reindex(idx_all)
core     = core.reindex(idx_all)
recess   = recess.reindex(idx_all)

headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3))**4 - 1) * 100

# ------------------------------------------------------------------------------
# 4. sidebar – window selector (label slices only)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Time Range")
    preset = st.selectbox("Preset", ["1Y","3Y","5Y","10Y","All"], 2)
    if preset == "All":
        start, end = idx_all.min(), idx_all.max()
    else:
        years = int(preset.rstrip("Y"))
        end   = idx_all.max()
        start = end - pd.DateOffset(years=years)
    if st.checkbox("Custom range"):
        rng = st.slider("Pick dates", min_value=idx_all.min().date(),
                        max_value=idx_all.max().date(),
                        value=(start.date(), end.date()),
                        format="YYYY-MM")
        start, end = map(pd.Timestamp, rng)

# windowed series
h      = headline.loc[start:end]
c      = core.loc[start:end]
h_yoy  = headline_yoy.loc[start:end]
c_yoy  = core_yoy.loc[start:end]
h_mom  = headline_mom.loc[start:end]
c_mom  = core_mom.loc[start:end]
c_3m   = core_3m_ann.loc[start:end]
rec_w  = recess.loc[start:end]

# ------------------------------------------------------------------------------
# 5. recession spans (list of tuples)
# ------------------------------------------------------------------------------
def spans(flag: pd.Series):
    flag = flag.fillna(0).astype(int)
    s = flag[(flag==1)&(flag.shift(1,fill_value=0)==0)].index.to_list()
    e = flag[(flag==0)&(flag.shift(1,fill_value=0)==1)].index.to_list()
    if flag.iloc[-1]==1: e.append(flag.index[-1])
    return [(i,j) for i,j in zip(s,e)]

recessions = [(s,e) for s,e in spans(recess)
              if e >= start and s <= end]

# ------------------------------------------------------------------------------
# 6. outliers & regimes (index-based, not masks)
# ------------------------------------------------------------------------------
# Core-YoY outlier
σ_10y = core_yoy.rolling(120, min_periods=24).std()
μ_10y = core_yoy.rolling(120, min_periods=24).mean()
out_yoy = core_yoy[(core_yoy-μ_10y).abs() > 1.5*σ_10y].loc[start:end]

# Headline-MoM outlier
out_mom = headline_mom[((headline_mom>0.7)|(headline_mom<-0.2))].loc[start:end]

# Regimes
reg = pd.Series("Normal", index=core_yoy.index)
reg[core_yoy>3] = "High"
reg[core_yoy<2] = "Low"
bands=[]
curr=None
for t in reg.index:
    if reg[t]!=curr:
        if curr: bands[-1]["end"]=t
        bands.append({"reg":reg[t],"start":t})
        curr=reg[t]
bands[-1]["end"]=reg.index[-1]
bands=[b for b in bands if b["end"]>=start and b["start"]<=end]

# ------------------------------------------------------------------------------
# 7. header & KPIs
# ------------------------------------------------------------------------------
Δh = h_yoy.iloc[-1]-h_yoy.iloc[-2] if len(h_yoy)>1 else 0
Δc = c_yoy.iloc[-1]-c_yoy.iloc[-2] if len(c_yoy)>1 else 0

st.title("US Inflation Dashboard")
st.metric("Headline CPI YoY", f"{h_yoy.iloc[-1]:.2f} %", f"{Δh:+.2f} pp")
st.metric("Core CPI YoY",     f"{c_yoy.iloc[-1]:.2f} %", f"{Δc:+.2f} pp")
st.caption("Bands = Core-YoY regimes • diamonds = Core-YoY outlier • "
           "stars = Headline-MoM outlier")
st.divider()

# ------------------------------------------------------------------------------
# 8. Plotly charts
# ------------------------------------------------------------------------------
# YoY
fig_y = go.Figure()
for b in bands:
    fig_y.add_vrect(b["start"], b["end"],
                    fillcolor={"High":"rgba(255,100,0,0.15)",
                               "Low":"rgba(0,200,255,0.15)",
                               "Normal":"rgba(255,255,255,0)"}[b["reg"]],
                    layer="below", line_width=0)
for s,e in recessions:
    fig_y.add_shape("rect", x0=s, x1=e, xref="x", yref="paper",
                    y0=0, y1=1,
                    fillcolor="rgba(200,0,0,0.15)", layer="below",
                    line_width=0)
fig_y.add_scatter(x=h_yoy.index, y=h_yoy,
                  name="Headline YoY", line=dict(color="#1f77b4"))
fig_y.add_scatter(x=c_yoy.index, y=c_yoy,
                  name="Core YoY", line=dict(color="#ff7f0e"))
fig_y.add_scatter(x=out_yoy.index, y=out_yoy,
                  mode="markers", name="Core YoY outlier",
                  marker=dict(symbol="diamond-open", color="red", size=9))
fig_y.update_layout(title="CPI YoY (%)", hovermode="x unified",
                    legend_orientation="h", margin=dict(t=40))

# MoM
fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      vertical_spacing=0.03,
                      subplot_titles=("Headline MoM","Core MoM"))
fig_m.add_bar(x=h_mom.index, y=h_mom,
              name="Headline MoM", marker_color="#1f77b4", row=1,col=1)
fig_m.add_bar(x=c_mom.index, y=c_mom,
              name="Core MoM", marker_color="#ff7f0e", row=2,col=1)
for s,e in recessions:
    for r in (1,2):
        fig_m.add_shape("rect", x0=s, x1=e, xref="x", yref="paper",
                        y0=0, y1=1,
                        fillcolor="rgba(200,0,0,0.15)",
                        layer="below", line_width=0, row=r,col=1)
fig_m.add_scatter(x=out_mom.index, y=out_mom,
                  mode="markers", name="Headline MoM outlier",
                  marker=dict(symbol="star-open", color="purple", size=9),
                  row=1,col=1)
fig_m.update_yaxes(title_text="% MoM", zeroline=True, row=1,col=1)
fig_m.update_yaxes(title_text="% MoM", zeroline=True, row=2,col=1)
fig_m.update_layout(showlegend=True, hovermode="x unified",
                    margin=dict(t=40))

# Core index & 3-mo ann.
fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      vertical_spacing=0.03,
                      subplot_titles=("Core CPI index",
                                      "3-mo-ann Core CPI %"))
fig_c.add_scatter(x=c.index, y=c,
                  name="Core index", line=dict(color="#ff7f0e"),
                  row=1,col=1)
fig_c.add_scatter(x=c_3m.index, y=c_3m,
                  name="3-mo ann.", line=dict(color="#1f77b4"),
                  row=2,col=1)
for s,e in recessions:
    for r in (1,2):
        fig_c.add_shape("rect", x0=s, x1=e, xref="x", yref="paper",
                        y0=0, y1=1,
                        fillcolor="rgba(200,0,0,0.15)",
                        layer="below", line_width=0,
                        row=r,col=1)
fig_c.update_layout(hovermode="x unified", legend_orientation="h",
                    margin=dict(t=40))

# Render
st.plotly_chart(fig_y, use_container_width=True)
st.plotly_chart(fig_m, use_container_width=True)
st.plotly_chart(fig_c, use_container_width=True)

with st.expander("Methodology & thresholds"):
    st.write("""
    * **Regimes** – Core YoY > 3 % ⇒ High; < 2 % ⇒ Low  
    * **Core YoY outlier** – deviation from 10-yr mean > 1.5 σ  
    * **Headline MoM outlier** – MoM > 0.7 % or < -0.2 %  
    * **Recession bars** – NBER USREC
    """)
