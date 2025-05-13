# -------------------------------------------------------------
#  Market Drawdown Dashboard — v3.0  |  AD Fund Management LP
# -------------------------------------------------------------
#  Enhancements (per Arya):
#    • Remove date-picker – always Year-to-Date (today’s close)
#    • Broaden coverage: add key European, Japanese & Chinese benchmarks
#          – Euro Stoxx 50, FTSE 100, Nikkei 225, TOPIX, FXI (China Large-Cap ETF)
#    • Richer sidebar explainer
#    • Replace simple breadth table with actionable breadth gauges:
#          % constituents above 50-d MA, 200-d MA, near 52-w highs, and beating benchmark YTD
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

TODAY = datetime.today().date()
YEAR_START = datetime(TODAY.year, 1, 1)

# -------------------------------------------------------------
# Page config & theme CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Market Drawdown Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_CSS = """
<style>
body { font-family: 'Helvetica Neue', sans-serif; }
/* KPI */
.css-1ht1j8u .stMetric label{font-size:.9rem}
.css-1ht1j8u .stMetric div{font-size:2.2rem;font-weight:700}
</style>
"""
DARK_CSS = """
<style>
body{font-family:'Helvetica Neue',sans-serif;background:#0e1117;color:#f3f4f6}
.sidebar .sidebar-content{background:#111319}
.css-1ht1j8u .stMetric label{color:#c9d1d9}
</style>
"""

# -------------------------------------------------------------
# Sidebar — controls & explainer
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f4c8.svg", width=40)
    st.markdown("""### Year-to-Date Market Stress Dashboard  
**What you get:**
* **Drawdown matrix** — instant view of how major global equity benchmarks have performed YTD, their bounces off the low, and max pain from the high.
* **Breadth gauges** — holistic read on S&P 500 internals: trend health (50-/200-day MAs), momentum (near 52-week highs) and relative strength versus the index.

Use the selector below to flip the headline benchmark; the breadth gauges always reference the S&P 500 (our cycle bellwether).""")

    theme = st.radio("Theme", ["Light", "Dark"], index=0)
    st.markdown(LIGHT_CSS if theme=="Light" else DARK_CSS, unsafe_allow_html=True)

    st.markdown("---")
    benchmark = st.selectbox(
        "Headline benchmark",
        (
            "S&P 500",
            "Nasdaq 100",
            "Russell 2000",
            "Dow 30",
            "Euro Stoxx 50",
            "FTSE 100",
            "Nikkei 225",
            "TOPIX",
            "FXI (China Large-Cap)"
        ),
        index=0,
    )

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    """Return adjusted close prices dataframe."""
    raw = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=False)
    def adj(df):
        if isinstance(df.columns, pd.MultiIndex):
            lvl = "Adj Close" if "Adj Close" in df.columns.get_level_values(1) else "Close"
            return df.swaplevel(axis=1)[lvl]
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return df[[col]].rename(columns={col: tickers[0]})
    out = adj(raw).sort_index().loc[~raw.index.duplicated(keep="first")]
    return out

# Simple pct return over n days
def pct_ret(series, days):
    return np.nan if len(series)<=days else series.iloc[-1]/series.iloc[-(days+1)] - 1

# -------------------------------------------------------------
# Index universe
# -------------------------------------------------------------
INDEX_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow 30": "^DJI",
    "Euro Stoxx 50": "^STOXX50E",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "TOPIX": "^TOPX",
    "FXI (China Large-Cap)": "FXI",
}

# -------------------------------------------------------------
# Fetch YTD price series for all indices (single pull for performance)
# -------------------------------------------------------------
all_prices = load_prices(list(INDEX_TICKERS.values()), YEAR_START, TODAY + timedelta(days=1))

# Build drawdown DataFrame
rows = []
for name, tk in INDEX_TICKERS.items():
    ser = all_prices[tk]
    rows.append({
        "Index": name,
        "YTD" : ser.iloc[-1]/ser.iloc[0] - 1,
        "From YTD Low" : ser.iloc[-1]/ser.min() - 1,
        "Max DD" : ser.iloc[-1]/ser.max() - 1,
    })

dd = pd.DataFrame(rows)
fmt = {c:"{:+.1%}" for c in dd.columns if c!="Index"}
colorize = lambda v: f"color:{'green' if v>0 else 'red'}" if isinstance(v,(int,float)) else ""

st.markdown("## YTD Drawdowns (Global Benchmarks)")
st.dataframe(
    dd.style.format(fmt).applymap(colorize, subset=["YTD","From YTD Low","Max DD"]),
    use_container_width=True,
    height=340,
)

# -------------------------------------------------------------
# KPI strip for chosen benchmark
# -------------------------------------------------------------
sel_series = all_prices[INDEX_TICKERS[benchmark]]
kp1,kp2,kp3 = st.columns(3, gap="large")
kp1.metric("YTD Return", f"{sel_series.iloc[-1]/sel_series.iloc[0]-1:+.1%}")
kp2.metric("Bounce from YTD Low", f"{sel_series.iloc[-1]/sel_series.min()-1:+.1%}")
kp3.metric("Max Drawdown", f"{sel_series.iloc[-1]/sel_series.max()-1:.1%}")

# -------------------------------------------------------------
# Breadth Gauges (S&P 500 internals)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sp500_members():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url, header=0)[0]["Symbol"].tolist()

sp_tickers = sp500_members()
sp_prices = load_prices(sp_tickers, TODAY - timedelta(days=400), TODAY + timedelta(days=1))
latest = sp_prices.iloc[-1]
ma50 = sp_prices.rolling(50).mean().iloc[-1]
ma200 = sp_prices.rolling(200).mean().iloc[-1]
high52 = sp_prices.rolling(252).max().iloc[-1]

pct_above50 = (latest > ma50).mean()
pct_above200 = (latest > ma200).mean()
pct_near_high = (latest >= high52*0.98).mean()

# Relative YTD outperformance vs SPX
spx_series = all_prices["^GSPC"]
spx_ytd = spx_series.iloc[-1]/spx_series.iloc[0]-1
member_ytd = sp_prices.apply(lambda s: s.iloc[-1]/s.iloc[0]-1)
rel_outperf = (member_ytd > spx_ytd).mean()

st.markdown("### S&P 500 Breadth Snapshot")
colA,colB,colC,colD = st.columns(4)
colA.metric("Above 50-d MA", f"{pct_above50:.0%}")
colB.metric("Above 200-d MA", f"{pct_above200:.0%}")
colC.metric("Near 52-w High (<2%)", f"{pct_near_high:.0%}")
colD.metric("Beating SPX YTD", f"{rel_outperf:.0%}")

# -------------------------------------------------------------
# Price chart for selected benchmark
# -------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=sel_series.index, y=sel_series, mode="lines", name=benchmark))
fig.add_trace(go.Scatter(x=[sel_series.idxmax()], y=[sel_series.max()], mode="markers", marker_symbol="triangle-up", marker_size=12, name="YTD High"))
fig.add_trace(go.Scatter(x=[sel_series.idxmin()], y=[sel_series.min()], mode="markers", marker_symbol="triangle-down", marker_size=12, name="YTD Low"))
fig.update_layout(title=f"{benchmark} — Price YTD", hovermode="x unified", height=420, template="plotly_white" if theme=="Light" else "plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Downloads & footer
# -------------------------------------------------------------
c1,c2 = st.columns(2)
c1.download_button("Download Drawdowns CSV", dd.to_csv(index=False).encode(), "global_drawdowns.csv", "text/csv")
c2.download_button("Download Breadth Metrics CSV", pd.DataFrame({"Metric":["Above50d","Above200d","NearHigh","BeatsYTD"],"Value":[pct_above50,pct_above200,pct_near_high,rel_outperf]}).to_csv(index=False).encode(), "sp500_breadth.csv", "text/csv")

st.markdown("""<div style='text-align:center;font-size:0.75rem;margin-top:2rem;'>Data: Yahoo Finance • Calculations: AD Fund Management LP • Past performance is no guarantee of future results</div>""", unsafe_allow_html=True)
