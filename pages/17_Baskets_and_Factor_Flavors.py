# streamlit run adfm_bloomberg_panel_v2.py
# ADFM Bloomberg Panel — full version with all flavors, preset ranges, and basket tickers

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="ADFM Bloomberg Panel", layout="wide")

TITLE = "ADFM Bloomberg-Style Basket Dashboard"
SUBTITLE = "Equal-weight baskets with daily indicators, preset ranges, and full flavor visibility."

PASTEL = [
    "#AEC6CF", "#FFB347", "#B39EB5", "#77DD77", "#F49AC2",
    "#CFCFC4", "#DEA5A4", "#C6E2FF", "#FFDAC1", "#E2F0CB",
    "#C7CEEA", "#FFB3BA", "#FFD1DC", "#B5EAD7", "#E7E6F7",
    "#F1E3DD", "#B0E0E6", "#E0BBE4", "#F3E5AB", "#D5E8D4"
]

# -----------------------------------------------------
# Full basket + flavor definitions
# -----------------------------------------------------
FACTOR_FLAVORS = {
    "Growth & Innovation": {
        "Semiconductors": ["SMH"],
        "AI Infrastructure Leaders": ["NVDA","AMD","AVGO","TSM","ASML","ANET","MU"],
        "Hyperscalers & Cloud": ["MSFT","AMZN","GOOGL","META","ORCL"],
        "Quality SaaS": ["ADBE","CRM","NOW","INTU","SNOW"],
        "Cybersecurity": ["PANW","FTNT","CRWD","ZS","OKTA"],
        "Digital Payments": ["V","MA","PYPL","SQ","FI","FIS"],
        "E-Commerce Platforms": ["AMZN","SHOP","MELI","ETSY"],
        "Social & Consumer Internet": ["META","SNAP","PINS","MTCH","GOOGL"],
        "Streaming & Media": ["NFLX","DIS","WBD","PARA","ROKU"],
        "Fintech & Neobanks": ["SQ","PYPL","AFRM","HOOD","SOFI"]
    },
    "AI & Next-Gen Compute": {
        "5G & Networking Infra": ["AMT","CCI","SBAC","ANET","CSCO"],
        "Industrial Automation": ["ROK","ETN","EMR","AME","PH"],
        "Space Economy": ["ARKX","RKLB","IRDM","ASTS"]
    },
    "Energy & Hard Assets": {
        "Energy Majors": ["XOM","CVX","COP","SHEL","BP"],
        "US Shale & E&Ps": ["EOG","DVN","FANG","MRO","OXY"],
        "Oilfield Services": ["SLB","HAL","BKR","NOV","CHX"],
        "Uranium & Fuel Cycle": ["CCJ","UUUU","UEC","URG","UROY"],
        "Battery & Materials": ["ALB","SQM","LTHM","PLL","LAC"],
        "Metals & Mining": ["BHP","RIO","VALE","FCX","NEM"],
        "Gold & Silver Miners": ["GDX","GDXJ","NEM","AEM","PAAS"]
    },
    "Clean Energy Transition": {
        "Solar & Inverters": ["TAN","FSLR","ENPH","SEDG","RUN"],
        "Wind & Renewables": ["ICLN","FAN","FSLR","ENPH","SEDG"],
        "Hydrogen": ["PLUG","BE","BLDP"],
        "Utilities & Power": ["VST","CEG","NEE","DUK","SO"]
    },
    "Health & Longevity": {
        "Large-Cap Biotech": ["AMGN","GILD","REGN","BIIB"],
        "GLP-1 & Metabolic": ["NVO","LLY","PFE","AZN"],
        "MedTech Devices": ["MDT","SYK","ISRG","BSX","ZBH"],
        "Healthcare Payers": ["UNH","HUM","CI","ELV"]
    },
    "Financials & Credit": {
        "Money-Center & IBs": ["JPM","BAC","C","WFC","GS","MS"],
        "Regional Banks": ["KRE","CFG","FITB","TFC","RF"],
        "Brokers & Exchanges": ["IBKR","SCHW","CME","ICE","NDAQ","CBOE"],
        "Alt Managers & PE": ["BX","KKR","APO","CG","ARES"],
        "Mortgage Finance": ["RKT","UWMC","COOP","FNF"]
    },
    "Real Assets & Inflation Beneficiaries": {
        "Homebuilders": ["ITB","DHI","LEN","NVR","PHM","TOL"],
        "REITs Core": ["VNQ","PLD","AMT","EQIX","SPG","O"],
        "Shipping & Logistics": ["FDX","UPS","GXO","XPO","ZIM"],
        "Agriculture & Machinery": ["MOS","NTR","DE","CNHI","ADM","BG"]
    },
    "Consumer Cyclicals": {
        "Retail Discretionary": ["HD","LOW","M","GPS","BBY","TJX"],
        "Restaurants": ["MCD","SBUX","YUM","CMG","DRI"],
        "Travel & Booking": ["BKNG","EXPE","ABNB","TRIP"],
        "Hotels & Casinos": ["MAR","HLT","IHG","MGM","LVS","WYNN"],
        "Airlines": ["AAL","DAL","UAL","LUV","JBLU"],
        "Autos Legacy OEMs": ["TM","HMC","F","GM","STLA"],
        "Electric Vehicles": ["TSLA","RIVN","LCID","NIO","LI","XPEV"]
    },
    "Defensives & Staples": {
        "Retail Staples": ["WMT","COST","TGT","DG","KR"],
        "Telecom & Cable": ["T","VZ","TMUS","CHTR","CMCSA"],
        "Aerospace & Defense": ["LMT","NOC","RTX","GD","HII"]
    },
    "Alternative Assets & Reflexivity Plays": {
        "Crypto Proxies": ["COIN","MSTR","MARA","RIOT","BITO"],
        "China Tech ADRs": ["BABA","BIDU","JD","PDD","BILI","TCEHY"]
    }
}

def build_basket_universe(flavor_dict):
    baskets = {}
    for _, groups in flavor_dict.items():
        for name, tks in groups.items():
            baskets[name] = list(dict.fromkeys(tks))
    return baskets

BASKETS = build_basket_universe(FACTOR_FLAVORS)

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start, end):
    df = yf.download(list(set(tickers)), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.sort_index()

def ew_basket_returns_from_levels(levels, baskets):
    rets = levels.pct_change()
    out = {}
    for b, tks in baskets.items():
        cols = [c for c in tks if c in rets.columns]
        if cols: out[b] = rets[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out).dropna(how="all")

def pct_since(series, start_ts):
    sub = series[series.index >= start_ts]
    return (sub.iloc[-1] / sub.iloc[0]) - 1 if len(sub) else np.nan

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.markdown("### About This Tool")
    st.write("""
Tracks basket-level total returns and technical indicators across ADFM-defined factor themes.  
Metrics: 5D, 1M, YTD, RSI-14D/W, MACD, EMA (4/9/18), and 3M realized volatility.
""")
    st.divider()
    st.markdown("### Controls")

    today = date.today()
    start_of_year = date(today.year, 1, 1)

    preset = st.selectbox("Date Range Preset", ["YTD","1W","1M","3M","1Y","3Y","5Y"])
    if preset == "YTD":
        start_date = start_of_year
    elif preset == "1W":
        start_date = today - timedelta(days=7)
    elif preset == "1M":
        start_date = today - timedelta(days=30)
    elif preset == "3M":
        start_date = today - timedelta(days=90)
    elif preset == "1Y":
        start_date = today - timedelta(days=365)
    elif preset == "3Y":
        start_date = today - timedelta(days=365*3)
    elif preset == "5Y":
        start_date = today - timedelta(days=365*5)
    end_date = today

    chosen_flavor = st.selectbox("Select Flavor", ["All"] + list(FACTOR_FLAVORS.keys()))
    baskets = list(BASKETS.keys()) if chosen_flavor == "All" else list(FACTOR_FLAVORS[chosen_flavor].keys())
    selected_baskets = st.multiselect("Select Baskets", baskets, default=baskets[:10])

if not selected_baskets:
    st.warning("Select at least one basket.")
    st.stop()

# -----------------------------------------------------
# DATA
# -----------------------------------------------------
need = {"SPY"}
for b in selected_baskets: need.update(BASKETS[b])
levels = fetch_prices(list(need), start=pd.to_datetime(start_date), end=pd.to_datetime(end_date) + pd.Timedelta(days=1))
basket_rets = ew_basket_returns_from_levels(levels, BASKETS)
basket_rets = basket_rets[selected_baskets]
basket_cum = (1 + basket_rets).cumprod() - 1
spy_cum = (1 + levels["SPY"].pct_change().dropna()).cumprod() - 1

# -----------------------------------------------------
# BLOOMBERG PANEL
# -----------------------------------------------------
st.subheader("Bloomberg-Style Basket Panel")
rows = []
for b in basket_cum.columns:
    s = (1 + basket_rets[b]).cumprod()
    r5d = (s.iloc[-1]/s.iloc[-6]-1)*100 if len(s)>6 else np.nan
    r1m = pct_since(s, s.index.max()-pd.DateOffset(months=1))*100
    rytd = pct_since(s, pd.Timestamp(year=s.index.max().year, month=1, day=1))*100
    rows.append({"Basket": b, "%5D": round(r5d,1), "%1M": round(r1m,1), "↓ %YTD": round(rytd,1)})
panel = pd.DataFrame(rows).set_index("Basket").sort_values("↓ %YTD", ascending=False)
st.dataframe(panel, use_container_width=True)

# -----------------------------------------------------
# INTERACTIVE CHART
# -----------------------------------------------------
st.subheader("Cumulative Performance (interactive)")
fig = go.Figure()
for i, b in enumerate(basket_cum.columns):
    fig.add_trace(go.Scatter(
        x=basket_cum.index,
        y=basket_cum[b]*100,
        mode="lines",
        line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
        name=b,
        hovertemplate=f"{b}<br>% Cum: %{ '{y:.1f}' }%<extra></extra>"
    ))
fig.add_trace(go.Scatter(
    x=spy_cum.index, y=spy_cum*100,
    mode="lines",
    line=dict(width=2, dash="dash", color="#888"),
    name="SPY",
    hovertemplate="SPY<br>% Cum: %{y:.1f}%<extra></extra>"
))
fig.update_layout(showlegend=False, hovermode="x unified", yaxis_title="Cumulative return, %")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# BASKET CONSTITUENTS
# -----------------------------------------------------
with st.expander("Basket Constituents"):
    for f, groups in FACTOR_FLAVORS.items():
        st.markdown(f"**{f}**")
        for name, tks in groups.items():
            st.write(f"- {name}: {', '.join(tks)}")
