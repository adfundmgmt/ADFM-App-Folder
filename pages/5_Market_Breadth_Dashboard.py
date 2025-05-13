# -------------------------------------------------------------
#  Market Drawdown & Breadth Dashboard — Streamlit
#  Polished UI v2.0 • AD Fund Management LP
# -------------------------------------------------------------
#  What’s new
#    • Clean headline banner & sidebar explainer
#    • Optional light / dark theme toggle (CSS injection)
#    • Colour-coded tables: greens for gains, reds for pain
#    • Responsive KPI strip, centred
#    • Sticky footer with data disclaimers
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------------------
# Page / Theme utilities
# -------------------------------------------------------------
st.set_page_config(
    page_title="Market Drawdown & Breadth Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS helpers (will be conditionally wrapped for light / dark)
LIGHT_CSS = """
<style>
body { font-family: "Helvetica Neue", sans-serif; }
/* KPI text bigger */
.css-1ht1j8u .stMetric label { font-size:0.9rem; }
.css-1ht1j8u .stMetric div { font-size:2rem; font-weight:700; }
/* Table zebra */
.dataframe tbody tr:nth-child(even) { background: #f8f9fc; }
</style>
"""

DARK_CSS = """
<style>
body { font-family: "Helvetica Neue", sans-serif; background:#0e1117; color:#f3f4f6; }
.sidebar .sidebar-content { background:#111319; }
/* KPI */
.css-1ht1j8u .stMetric label { color:#c9d1d9; }
.dataframe tbody tr:nth-child(even) { background: #161b22; }
</style>
"""

# -------------------------------------------------------------
# Sidebar — controls & explainer
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f4c8.svg", width=40)
    st.markdown("""### Market Drawdown & Breadth Dashboard  
Quick snapshot of **index stress** (YTD drawdowns) and **internals** (% of S&P 500 constituents beating the index).  
Select your benchmark, date, and theme — everything else updates on the fly.""")

    theme_choice = st.radio("Theme", ["Light", "Dark"], index=0)
    if theme_choice == "Light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    else:
        st.markdown(DARK_CSS, unsafe_allow_html=True)

    st.markdown("---")
    index_choice = st.selectbox("Benchmark index", ("S&P 500", "Nasdaq 100", "Russell 2000", "Dow 30"))
    as_of = st.date_input("As-of date", value=datetime.today().date())
    if as_of > datetime.today().date():
        st.error("As-of date cannot be in the future.")
        st.stop()

# -------------------------------------------------------------
# Helper fns
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_index_members(index: str):
    import pandas as pd
    if index == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        return pd.read_html(url, header=0)[0]["Symbol"].tolist()
    if index == "Nasdaq 100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        return pd.read_html(url, header=0, match="Ticker")[0]["Ticker"].str.strip().tolist()
    if index == "Russell 2000":
        return yf.Ticker("IWM").fund_holdings["symbol"].tolist()
    if index == "Dow 30":
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        return pd.read_html(url, header=0, match="Symbol")[1]["Symbol"].tolist()
    return []

@st.cache_data(show_spinner=False)
def get_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=False)
    def adj(d):
        if isinstance(d.columns, pd.MultiIndex):
            lvl = "Adj Close" if "Adj Close" in d.columns.get_level_values(1) else "Close"
            return d.swaplevel(axis=1)[lvl]
        col = "Adj Close" if "Adj Close" in d.columns else "Close"
        return d[[col]].rename(columns={col: tickers[0]})
    out = adj(df).sort_index().loc[~df.index.duplicated(keep="first")]
    return out

def pct_ret(series, days):
    if len(series) <= days: return np.nan
    return series.iloc[-1] / series.iloc[-(days+1)] - 1

# -------------------------------------------------------------
# Data prep
# -------------------------------------------------------------
index_map = {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Russell 2000": "^RUT", "Dow 30": "^DJI"}
year_start = datetime(as_of.year, 1, 1)
idx_series = get_prices([index_map[index_choice]], year_start, as_of + timedelta(days=1)).iloc[:,0]

# -------------------------------------------------------------
# KPI deck
# -------------------------------------------------------------
st.markdown("## Key Stats")
kp1,kp2,kp3 = st.columns(3, gap="large")
kp1.metric("YTD Return", f"{idx_series.iloc[-1]/idx_series.iloc[0]-1:+.1%}")
kp2.metric("Bounce from YTD Low", f"{idx_series.iloc[-1]/idx_series.min()-1:+.1%}")
kp3.metric("Max Drawdown", f"{idx_series.iloc[-1]/idx_series.max()-1:.1%}")

# -------------------------------------------------------------
# Drawdown table (all indices)
# -------------------------------------------------------------
rows = []
for n, tk in index_map.items():
    s = get_prices([tk], year_start, as_of + timedelta(days=1)).iloc[:,0]
    rows.append({
        "Index": n,
        "YTD": s.iloc[-1]/s.iloc[0]-1,
        "From YTD Low": s.iloc[-1]/s.min()-1,
        "Max DD": s.iloc[-1]/s.max()-1,
    })

dd = pd.DataFrame(rows)
fmt_cols = {c:"{:+.1%}" for c in dd.columns if c!="Index"}

def colour(val):
    return f"color:{'green' if val>0 else 'red'}" if isinstance(val,(int,float)) else ""

dd_st = dd.style.format(fmt_cols).applymap(colour, subset=["YTD","From YTD Low","Max DD"])

st.markdown("### Major Indices — YTD Stress")
st.dataframe(dd_st, use_container_width=True, height=240)

# -------------------------------------------------------------
# Breadth table (always S&P 500)
# -------------------------------------------------------------
sp_members = get_index_members("S&P 500")
prices = get_prices(sp_members, as_of - timedelta(days=370), as_of + timedelta(days=1))

horizons = {"1m":21,"2m":42,"3m":63,"4m":84,"5m":105,"6m":126,"1y":252}
rec = []
for lbl, d in horizons.items():
    member_ret = prices.apply(lambda s: pct_ret(s,d))
    idx_ret = pct_ret(idx_series if index_choice=="S&P 500" else get_prices(["^GSPC"], year_start, as_of+timedelta(days=1)).iloc[:,0], d)
    rec.append({"Horizon":lbl, "% Beats": (member_ret>idx_ret).mean()})

breadth = pd.DataFrame(rec)
breadth_st = breadth.style.format({"% Beats":"{:.0%}"}).applymap(lambda v: f"background-color:rgba(0,125,0,{v})" if isinstance(v,float) else "", subset=["% Beats"])

st.markdown("### Breadth — Constituents Outperforming Benchmark")
st.dataframe(breadth_st, use_container_width=True, height=220)

# -------------------------------------------------------------
# Price chart
# -------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx_series.index, y=idx_series, mode="lines", name=index_choice))
fig.add_trace(go.Scatter(x=[idx_series.idxmax()], y=[idx_series.max()], mode="markers", marker_symbol="triangle-up", marker_size=12, name="YTD High"))
fig.add_trace(go.Scatter(x=[idx_series.idxmin()], y=[idx_series.min()], mode="markers", marker_symbol="triangle-down", marker_size=12, name="YTD Low"))
fig.update_layout(title=f"{index_choice} — Price YTD", hovermode="x unified", height=420, template="plotly_white" if theme_choice=="Light" else "plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Downloads + footer
# -------------------------------------------------------------
c1,c2 = st.columns(2)
with c1:
    st.download_button("Download Drawdowns CSV", dd.to_csv(index=False).encode(), "index_drawdowns.csv", "text/csv")
with c2:
    st.download_button("Download Breadth CSV", breadth.to_csv(index=False).encode(), "sp500_breadth.csv", "text/csv")

st.markdown("""<div style='text-align:center; font-size:0.75rem; margin-top:2rem;'>Data: Yahoo Finance • Calculations: AD Fund Management LP • Past performance is no guarantee of future results</div>""", unsafe_allow_html=True)
