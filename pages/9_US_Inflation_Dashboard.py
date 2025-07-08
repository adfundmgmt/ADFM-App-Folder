import streamlit as st
import yfinance as yf
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="US CPI Dashboard", layout="wide")

# ---------------------------
# Helper: Yahoo Finance loader
# ---------------------------
@st.cache_data(show_spinner=True)
def load_yahoo_series(ticker: str, start: str = "1990-01-01") -> pd.Series:
    try:
        df = yf.download(ticker, start=start, interval="1mo", auto_adjust=True, progress=False)
        if df.empty:
            st.error(f"No data found for {ticker}.")
            return pd.Series(dtype=float)
        df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
        s = df['Close'].ffill()
        return s
    except Exception as e:
        st.error(f"Could not load {ticker} from Yahoo Finance: {e}")
        return pd.Series(dtype=float)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore interactive inflation charts powered by Yahoo Finance data. View year‑over‑year and month‑over‑month changes for both headline and core CPI. NBER recessions shaded for context.
        """
    )
    st.subheader("Time Range")
    period = st.selectbox(
        "Select period:",
        ["1M", "3M", "6M", "9M", "1Y", "3Y", "5Y", "All"],
        index=7
    )

# Load data from Yahoo Finance
headline = load_yahoo_series("CPIAUCSL")    # Headline CPI (All Urban Consumers)
core     = load_yahoo_series("CPILFESL")    # Core CPI (Less Food & Energy)

# Example: USREC not on Yahoo, so we skip recession bars in this quick version
# (If needed, you can download NBER US recession dates via public CSV)

if headline.empty or core.empty:
    st.error("Failed to load CPI data. Try again later.")
    st.stop()

headline_yoy = headline.pct_change(12) * 100
core_yoy     = core.pct_change(12)     * 100
headline_mom = headline.pct_change(1)  * 100
core_mom     = core.pct_change(1)      * 100
core_3m_ann  = ((core / core.shift(3)) ** 4 - 1) * 100

# Determine window
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

# Plot headline & core YoY
fig_yoy = go.Figure()
fig_yoy.add_trace(go.Scatter(
    x=h_yoy.index, y=h_yoy, name="Headline CPI YoY", line_color="#1f77b4"))
fig_yoy.add_trace(go.Scatter(
    x=c_yoy.index, y=c_yoy, name="Core CPI YoY", line_color="#ff7f0e"))
fig_yoy.update_layout(
    title="US CPI YoY – Headline vs Core",
    yaxis_title="% YoY",
    hovermode="x unified",
    margin=dict(t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
)

# MoM bars
fig_mom = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Headline CPI MoM %", "Core CPI MoM %"))
fig_mom.add_trace(go.Bar(
    x=h_mom.index, y=h_mom, name="Headline MoM", marker_color="#1f77b4"), row=1, col=1)
fig_mom.add_trace(go.Bar(
    x=c_mom.index, y=c_mom, name="Core MoM", marker_color="#ff7f0e"), row=2, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=1, col=1)
fig_mom.update_yaxes(title_text="% MoM", row=2, col=1)
fig_mom.update_layout(
    showlegend=False,
    hovermode="x unified",
    margin=dict(t=60),
    xaxis=dict(rangeslider=dict(visible=False)),
    xaxis2=dict(rangeslider=dict(visible=False))
)

# Core index & 3M ann
fig_core = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    subplot_titles=("Core CPI Index", "3‑Mo Annualised Core CPI %"))
fig_core.add_trace(go.Scatter(
    x=core.loc[idx].index, y=core.loc[idx], name="Core Index", line_color="#ff7f0e", mode="lines+markers"), row=1, col=1)
fig_core.add_trace(go.Scatter(
    x=c_3m.index, y=c_3m, name="3M Ann.", line_color="#1f77b4", mode="lines+markers"), row=2, col=1)
fig_core.update_yaxes(title_text="Index Level", row=1, col=1)
fig_core.update_yaxes(title_text="% (annualised)", row=2, col=1)
fig_core.update_layout(
    hovermode="x unified",
    margin=dict(t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    xaxis=dict(rangeslider=dict(visible=False)),
    xaxis2=dict(rangeslider=dict(visible=False))
)

# Latest values + delta
latest_date = h_yoy.index.max()
headline_latest = h_yoy.iloc[-1]
core_latest = c_yoy.iloc[-1]
headline_prev = h_yoy.iloc[-2] if len(h_yoy) > 1 else None
core_prev = c_yoy.iloc[-2] if len(c_yoy) > 1 else None

cols = st.columns(2)
cols[0].metric("Headline CPI YoY", f"{headline_latest:.2f}%", f"{(headline_latest - headline_prev):+.2f}%" if headline_prev else "0.00%")
cols[1].metric("Core CPI YoY", f"{core_latest:.2f}%", f"{(core_latest - core_prev):+.2f}%" if core_prev else "0.00%")

st.title("US Inflation Dashboard")
st.plotly_chart(fig_yoy,  use_container_width=True)
st.plotly_chart(fig_mom,  use_container_width=True)
st.plotly_chart(fig_core, use_container_width=True)

with st.expander("Download Data"):
    combined = pd.DataFrame({
        "Headline CPI": headline.loc[idx],
        "Core CPI": core.loc[idx],
        "Headline YoY (%)": h_yoy,
        "Core YoY (%)": c_yoy,
        "Headline MoM (%)": h_mom,
        "Core MoM (%)": c_mom,
        "Core 3M Ann. (%)": c_3m,
        # "Recession Flag": ... # not available in Yahoo
    })
    st.download_button(
        "Download CSV", combined.to_csv(index=True), file_name="us_cpi_data.csv", mime="text/csv"
    )

with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data:** Yahoo Finance  
        **Series:** Headline CPI (CPIAUCSL), Core CPI (CPILFESL)  
        **3‑Mo annualised formula:** ((CPI_t / CPI_{t-3}) ** 4 – 1) × 100
        """
    )

st.caption("© 2025 AD Fund Management LP")
