import streamlit as st
import pandas as pd
from pandas_datareader.data import DataReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="US CPI Dashboard", layout="wide")

# --- Config: FRED Series Codes ---
FRED_SERIES = {
    "headline": "CPIAUCNS",
    "core":     "CPILFESL",
    "recess":   "USREC"
}
START_DATE_FULL = "1990-01-01"

# --- Sidebar ---
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore interactive inflation charts powered by the latest FRED data. View year-over-year and month-over-month changes for both headline and core CPI, with NBER recessions shaded for context. Data is fetched live and cached for responsiveness.
        """
    )
    st.subheader("Time Range")
    period = st.selectbox(
        "Select period:",
        ["1M", "3M", "6M", "9M", "1Y", "3Y", "5Y", "All"],
        index=7
    )

# --- Data Loader ---
@st.cache_data(show_spinner=False, ttl=24*60*60)
def load_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    try:
        df = DataReader(series_id, "fred", start)
        s = df[series_id].copy()

        # Ensure monthly frequency aligned to MS without backfilling pre-history
        # Many FRED CPI series already arrive as MS; resample defensively if needed
        try:
            freqstr = s.index.freqstr
        except Exception:
            freqstr = None
        if freqstr != "MS":
            s = s.resample("MS").ffill()

        return s
    except Exception as e:
        st.warning(f"Failed to load series {series_id}: {e}")
        return pd.Series(dtype=float)

headline = load_series(FRED_SERIES["headline"], START_DATE_FULL)
core     = load_series(FRED_SERIES["core"],     START_DATE_FULL)
recess   = load_series(FRED_SERIES["recess"],   START_DATE_FULL)

if headline.empty or core.empty or recess.empty:
    st.error("Critical FRED data failed to load. Please refresh or check your connection.")
    st.stop()

# --- Calculations ---
headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100  # 3-month annualized

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
    st.error("Invalid period selected.")
    st.stop()

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

idx = (headline.index >= start_date) & (headline.index <= end_date)
h_yoy = headline_yoy.loc[idx]
c_yoy = core_yoy.loc[idx]
h_mom = headline_mom.loc[idx]
c_mom = core_mom.loc[idx]
c_3m  = core_3m_ann.loc[idx]

# --- Edge Case Guarding ---
if h_yoy.empty or c_yoy.empty or h_mom.empty or c_mom.empty:
    st.warning("No data is available for the selected range.")
    st.stop()

# --- Recessions ---
def within_window(rec_list, start, end):
    out = []
    for s, e in rec_list:
        if e < start or s > end:
            continue
        out.append((max(s, start), min(e, end)))
    return out

def get_recessions(flag: pd.Series):
    f = flag.dropna().astype(int)
    if f.empty:
        return []
    dif = f.diff()
    starts = f[(f == 1) & (dif == 1)].index
    if f.iloc[0] == 1:
        starts = starts.insert(0, f.index[0])
    ends = f[(f == 0) & (dif == -1)].index
    if len(ends) < len(starts):
        ends = ends.insert(len(ends), f.index[-1])
    return [(s, e) for s, e in zip(starts, ends) if e >= s]

recs = get_recessions(recess)
recs_window = within_window(recs, start_date, end_date)

# --- Plotting Helpers ---
def _add_recession_shapes_single_axis(fig, recs_window):
    # For single-axis figures
    for s, e in recs_window:
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0
        )

def _add_recession_shapes_two_rows(fig, recs_window):
    # For two-row subplots where each row has its own x-axis
    for s, e in recs_window:
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=s, x1=e, y0=0, y1=0.5,
            fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0
        )
        fig.add_shape(
            type="rect", xref="x2", yref="paper",
            x0=s, x1=e, y0=0.5, y1=1,
            fillcolor="rgba(40,40,40,0.35)", opacity=0.35, layer="below", line_width=0
        )

hover_pct = dict(hovertemplate='%{y:.2f}%')

def plot_yoy(h_yoy, c_yoy, recs_window):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h_yoy.index, y=h_yoy, name="Headline CPI YoY", line_color="#1f77b4", **hover_pct))
    fig.add_trace(go.Scatter(
        x=c_yoy.index, y=c_yoy, name="Core CPI YoY",     line_color="#ff7f0e", **hover_pct))
    _add_recession_shapes_single_axis(fig, recs_window)
    fig.update_layout(
        title="US CPI YoY — Headline vs Core",
        yaxis_title="% YoY",
        hovermode="x unified",
        margin=dict(t=80),
        xaxis=dict(rangeslider=dict(visible=False), showline=True, linewidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig

def plot_mom(h_mom, c_mom, recs_window):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.02,
        subplot_titles=("Headline CPI MoM %", "Core CPI MoM %"))
    fig.add_trace(go.Bar(
        x=h_mom.index, y=h_mom, name="Headline MoM", marker_color="#1f77b4", **hover_pct), row=1, col=1)
    fig.add_trace(go.Bar(
        x=c_mom.index, y=c_mom, name="Core MoM",     marker_color="#ff7f0e", **hover_pct), row=2, col=1)

    _add_recession_shapes_two_rows(fig, recs_window)

    fig.update_yaxes(title_text="% MoM", zeroline=True, zerolinewidth=1, row=1, col=1)
    fig.update_yaxes(title_text="% MoM", zeroline=True, zerolinewidth=1, row=2, col=1)
    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=80),
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=False))
    )
    return fig

def plot_core(core_idx, c_3m):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=("Core CPI Index", "3 Mo Annualised Core CPI %"))
    fig.add_trace(go.Scatter(
        x=core_idx.index, y=core_idx, name="Core Index", line_color="#ff7f0e",
        mode="lines+markers", hovertemplate='%{y:.2f}'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=c_3m.index, y=c_3m, name="3M Ann.", line_color="#1f77b4",
        mode="lines+markers", **hover_pct), row=2, col=1)
    fig.update_yaxes(title_text="Index Level", row=1, col=1)
    fig.update_yaxes(title_text="% (annualised)", zeroline=True, zerolinewidth=1, row=2, col=1)
    fig.update_layout(
        hovermode="x unified",
        margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=False))
    )
    return fig

# --- Render UI ---
st.title("US Inflation Dashboard")

# Clean metric handling to avoid NaN display on short windows
def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def safe_delta(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return "N/A"
    return f"{curr - prev:+.2f}%"

h_yoy_disp = h_yoy.dropna()
c_yoy_disp = c_yoy.dropna()

headline_latest = h_yoy_disp.iloc[-1] if not h_yoy_disp.empty else float("nan")
headline_prev   = h_yoy_disp.iloc[-2] if h_yoy_disp.size > 1 else float("nan")
core_latest     = c_yoy_disp.iloc[-1] if not c_yoy_disp.empty else float("nan")
core_prev       = c_yoy_disp.iloc[-2] if c_yoy_disp.size > 1 else float("nan")

cols = st.columns(2)
cols[0].metric("Headline CPI YoY", fmt_pct(headline_latest), safe_delta(headline_latest, headline_prev))
cols[1].metric("Core CPI YoY",     fmt_pct(core_latest),     safe_delta(core_latest, core_prev))

st.plotly_chart(plot_yoy(h_yoy, c_yoy, recs_window),  use_container_width=True)
st.plotly_chart(plot_mom(h_mom, c_mom, recs_window),  use_container_width=True)
st.plotly_chart(plot_core(core.loc[idx], c_3m), use_container_width=True)

# --- Download ---
with st.expander("Download Data"):
    include_rec = st.checkbox("Include recession flag", value=False)
    recession_aligned = recess.reindex(headline.index)
    data_dict = {
        "Headline CPI": headline.loc[idx],
        "Core CPI": core.loc[idx],
        "Headline YoY (%)": h_yoy,
        "Core YoY (%)": c_yoy,
        "Headline MoM (%)": h_mom,
        "Core MoM (%)": c_mom,
        "Core 3M Ann. (%)": c_3m,
    }
    if include_rec:
        data_dict["Recession Flag"] = recession_aligned.loc[idx]
    combined = pd.DataFrame(data_dict)
    st.download_button(
        "Download CSV",
        combined.to_csv(index=True, index_label="Date"),
        file_name="us_cpi_data.csv",
        mime="text/csv"
    )

with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** FRED via pandas-datareader  
        **Series:** Headline CPI (CPIAUCNS), Core CPI (CPILFESL), Recessions (USREC)  
        **3 Mo annualised formula:** ((CPI_t / CPI_{t-3}) ** 4 - 1) × 100
        """
    )

st.caption("© 2025 AD Fund Management LP")
