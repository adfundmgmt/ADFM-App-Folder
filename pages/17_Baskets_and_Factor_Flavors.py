# streamlit run adfm_bloomberg_panel.py
# ADFM Bloomberg-Style Basket Dashboard
# By Arya Deniz / AD Fund Management LP

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="ADFM Bloomberg Panel", layout="wide")
TITLE = "ADFM Bloomberg-Style Basket Dashboard"
SUBTITLE = "Equal-weight baskets with daily indicators and interactive cumulative performance."

PASTEL = [
    "#AEC6CF", "#FFB347", "#B39EB5", "#77DD77", "#F49AC2",
    "#CFCFC4", "#DEA5A4", "#C6E2FF", "#FFDAC1", "#E2F0CB",
    "#C7CEEA", "#FFB3BA", "#FFD1DC", "#B5EAD7", "#E7E6F7",
    "#F1E3DD", "#B0E0E6", "#E0BBE4", "#F3E5AB", "#D5E8D4"
]

# -----------------------------------------------------
# BASKET UNIVERSE (SHORTENED EXAMPLE)
# -----------------------------------------------------
FACTOR_FLAVORS = {
    "Growth & Innovation": {
        "Semiconductors": ["SMH"],
        "AI Infrastructure Leaders": ["NVDA","AMD","AVGO","TSM","ASML","ANET","MU"],
        "Hyperscalers & Cloud": ["MSFT","AMZN","GOOGL","META","ORCL"],
        "Cybersecurity": ["PANW","FTNT","CRWD","ZS","OKTA"],
        "Digital Payments": ["V","MA","PYPL","SQ","FI","FIS"],
        "E-Commerce Platforms": ["AMZN","SHOP","MELI","ETSY"],
        "Streaming & Media": ["NFLX","DIS","WBD","PARA","ROKU"],
        "Fintech & Neobanks": ["SQ","PYPL","AFRM","HOOD","SOFI"]
    },
    "Energy & Hard Assets": {
        "Energy Majors": ["XOM","CVX","COP","SHEL","BP"],
        "Uranium & Fuel Cycle": ["CCJ","UUUU","UEC","URG","UROY"],
        "Gold & Silver Miners": ["GDX","GDXJ","NEM","AEM","PAAS"]
    },
    "Financials & Credit": {
        "Money-Center & IBs": ["JPM","BAC","C","WFC","GS","MS"],
        "Alt Managers & PE": ["BX","KKR","APO","CG","ARES"],
        "Mortgage Finance": ["RKT","UWMC","COOP","FNF"]
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
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index()

def ew_basket_returns_from_levels(levels, baskets):
    rets = levels.pct_change()
    out = {}
    for b, tks in baskets.items():
        cols = [c for c in tks if c in rets.columns]
        if len(cols) == 0: continue
        out[b] = rets[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out).dropna(how="all")

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def ema_regime(series):
    e4 = series.ewm(span=4).mean()
    e9 = series.ewm(span=9).mean()
    e18 = series.ewm(span=18).mean()
    last = series.index[-1]
    if e4.loc[last] > e9.loc[last] > e18.loc[last]:
        return "Up"
    elif e4.loc[last] < e9.loc[last] < e18.loc[last]:
        return "Down"
    else:
        return "Neutral"

def momentum_label(hist, lookback=5):
    if hist.empty:
        return "Neutral"
    latest = hist.iloc[-1]
    ref = hist.iloc[-lookback] if len(hist) > lookback else hist.iloc[0]
    base = "Positive" if latest > 0 else "Negative" if latest < 0 else "Neutral"
    if base == "Neutral":
        return "Neutral"
    return f"{base} {'Strengthening' if latest - ref > 0 else 'Weakening'}"

def realized_vol(returns, days=63, ann=252):
    sub = returns.dropna().iloc[-days:]
    return sub.std(ddof=0) * np.sqrt(ann) * 100 if len(sub) else np.nan

def pct_since(levels, start_ts):
    sub = levels[levels.index >= start_ts]
    return (sub.iloc[-1] / sub.iloc[0]) - 1 if len(sub) else np.nan

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    with st.expander("About this tool", expanded=False):
        st.markdown("""
- **Daily data** used for 5D, 1M, YTD, RSI, MACD, EMA, 3M Vol.
- **Interactive cumulative chart** shows xx.x% hover, no legend.
- **Top/Bottom Movers** replaces flavor KPIs.
- Equal-weight basket and flavor averages.
        """)
    st.header("Controls")
    today = date.today()
    start_date = st.date_input("Start date", today - timedelta(days=365*3))
    end_date = st.date_input("End date", today)
    chosen_flavor = st.selectbox("Select Flavor", ["All"] + list(FACTOR_FLAVORS.keys()))
    baskets = list(BASKETS.keys()) if chosen_flavor == "All" else list(FACTOR_FLAVORS[chosen_flavor].keys())
    default_sel = baskets[:10]
    selected_baskets = st.multiselect("Select Baskets", baskets, default=default_sel)

if not selected_baskets:
    st.stop()

# -----------------------------------------------------
# FETCH DATA
# -----------------------------------------------------
need = set(["SPY"])
for b in selected_baskets:
    need.update(BASKETS[b])

levels = fetch_prices(list(need), pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1))
basket_rets = ew_basket_returns_from_levels(levels, BASKETS)
basket_rets = basket_rets[selected_baskets]
basket_lvls = (1 + basket_rets).cumprod() * 100

# -----------------------------------------------------
# BLOOMBERG PANEL
# -----------------------------------------------------
st.subheader("Bloomberg-Style Basket Panel")

rows = []
for b in basket_lvls.columns:
    s = basket_lvls[b].dropna()
    if len(s) < 30:
        continue
    r5d = (s.iloc[-1] / s.iloc[-6]) - 1 if len(s) > 6 else np.nan
    r1m = pct_since(s, s.index.max() - pd.DateOffset(months=1))
    y_start = pd.Timestamp(year=s.index.max().year, month=1, day=1)
    rytd = pct_since(s, y_start)
    rsi_14d = rsi(s, 14).iloc[-1]
    w = s.resample("W-FRI").last().dropna()
    rsi_14w = rsi(w, 14).iloc[-1] if len(w) > 14 else np.nan
    hist = macd_hist(s)
    macd_m = momentum_label(hist)
    ema_tag = ema_regime(s)
    rv = realized_vol(basket_rets[b])
    rows.append({
        "Basket": b,
        "%5D": round(r5d*100,1),
        "%1M": round(r1m*100,1),
        "↓ %YTD": round(rytd*100,1),
        "RSI 14D": round(rsi_14d,2),
        "MACD Momentum": macd_m,
        "EMA 4/9/18": ema_tag,
        "RSI 14W": round(rsi_14w,2),
        "3M Realized Vol": round(rv,1)
    })

panel = pd.DataFrame(rows).set_index("Basket").sort_values("↓ %YTD", ascending=False)
st.dataframe(panel, use_container_width=True)

# -----------------------------------------------------
# INTERACTIVE CUMULATIVE CHART
# -----------------------------------------------------
st.subheader("Cumulative Performance (interactive)")
basket_cum_pct = ((1 + basket_rets).cumprod() - 1) * 100
spy_cum = ((1 + levels["SPY"].pct_change().dropna()).cumprod() - 1) * 100

fig = go.Figure()
for i, b in enumerate(basket_cum_pct.columns):
    fig.add_trace(go.Scatter(
        x=basket_cum_pct.index,
        y=basket_cum_pct[b],
        mode="lines",
        line=dict(width=2, color=PASTEL[i % len(PASTEL)]),
        name=b,
        hovertemplate=f"{b}<br>% Cum: %{ '{y:.1f}' }%<extra></extra>"
    ))
fig.add_trace(go.Scatter(
    x=spy_cum.index,
    y=spy_cum,
    mode="lines",
    line=dict(width=2, dash="dash", color="#888"),
    name="SPY",
    hovertemplate="SPY<br>% Cum: %{y:.1f}%<extra></extra>"
))
fig.update_layout(showlegend=False, hovermode="x unified", yaxis_title="Cumulative return, %")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# FLAVOR MOVERS
# -----------------------------------------------------
st.subheader("Factor-Flavor Movers")

def flavor_returns(basket_ret_df):
    out = {}
    for flavor, groups in FACTOR_FLAVORS.items():
        cols = [c for c in groups.keys() if c in basket_ret_df.columns]
        if cols:
            out[flavor] = basket_ret_df[cols].mean(axis=1)
    return pd.DataFrame(out)

flv = flavor_returns(basket_rets)
flv_lvls = (1 + flv).cumprod()

def flavor_period_table(lvls):
    rows = []
    for f in lvls.columns:
        s = lvls[f]
        one_m = pct_since(s, s.index.max() - pd.DateOffset(months=1))
        ytd = pct_since(s, pd.Timestamp(year=s.index.max().year, month=1, day=1))
        rows.append({"Flavor": f, "1M %": round(one_m*100,1), "YTD %": round(ytd*100,1)})
    df = pd.DataFrame(rows).set_index("Flavor")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 by 1M**")
        st.dataframe(df.sort_values("1M %", ascending=False).head(5))
    with col2:
        st.markdown("**Bottom 5 by 1M**")
        st.dataframe(df.sort_values("1M %", ascending=True).head(5))
    st.markdown("**Top 5 by YTD**")
    st.dataframe(df.sort_values("YTD %", ascending=False).head(5))

flavor_period_table(flv_lvls)
