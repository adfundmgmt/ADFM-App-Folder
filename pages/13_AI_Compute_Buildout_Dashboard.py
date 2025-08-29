# ai_tradeboard.py

"""
AI + Datacenter Tradeboard — zero logins, zero keys
Data: Yahoo Finance via yfinance (prices & volumes only)

What this page does
- Builds a **Core AI/Datacenter universe** (US + key EU/Asia ADRs) and computes actionable signals:
  - Relative strength vs SPY (60d), distance to 52w high, trend state (50/200DMA)
  - Volume z‑score, RSI(14), ATR% (risk proxy), breakouts (20d high with volume surge)
- Creates an **Equal‑Weight Theme Index** and compares it to SPY
- Surfaces **Long candidates** and **Trim/Short candidates** using a transparent scoring model you can tune
- Pairs & Ratios tab to track supply/demand leadership (Vendors ↔ Hypers, NVDA vs peers, REITs vs Power)

Run
pip install streamlit yfinance pandas numpy plotly
streamlit run ai_tradeboard.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI + Datacenter Tradeboard", layout="wide")

# ---------------------
# Universe presets (adjust freely)
# ---------------------
US_HYPERS   = ["AMZN","MSFT","GOOGL","META","ORCL","AAPL"]
US_VENDORS  = ["NVDA","AMD","AVGO","MRVL","SMCI","MU"]
EU_ASIA     = ["TSM","ASML","BABA","TCEHY"]  # ADRs only for reliability
REITS_PWR   = ["EQIX","DLR","VST","NEE","DUK"]
UNIVERSE_DEFAULT = US_HYPERS + US_VENDORS + EU_ASIA + REITS_PWR

# ---------------------
# Data fetch
# ---------------------
@st.cache_data(show_spinner=False)
def fetch_ohlcv(tickers: Tuple[str, ...], period: str = "2y") -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker")
    closes, vols = [], []
    for t in tickers:
        try:
            closes.append(raw[(t, "Close")].rename(t))
            vols.append(raw[(t, "Volume")].rename(t))
        except Exception:
            pass
    if not closes:
        return pd.DataFrame(), pd.DataFrame()
    close = pd.concat(closes, axis=1).dropna(how="all"); close.index.name = "Date"
    vol   = pd.concat(vols,   axis=1).reindex(close.index).fillna(0)
    return close, vol

# ---------------------
# Indicators & transforms
# ---------------------

def theme_index(df_close: pd.DataFrame) -> pd.Series:
    return (df_close.pct_change().fillna(0).add(1).cumprod()).mean(axis=1)

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_down = down.ewm(com=n-1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(series: pd.Series, n: int = 14) -> pd.Series:
    # Proxy ATR using close-to-close true range (no H/L intraday available in yfinance group fetch reliably across tickers)
    tr = series.pct_change().abs()
    return tr.rolling(n).mean()

@dataclass
class Metrics:
    price: float
    ret1d: float
    ret5d: float
    ret20d: float
    rs60: float
    from_high: float
    above50: bool
    above200: bool
    vol_z: float
    rsi14: float
    atrp: float
    breakout20: bool


def compute_metrics(close: pd.DataFrame, vol: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    out = []
    # precomputes
    ma20 = close.rolling(20).mean(); ma50 = close.rolling(50).mean(); ma200 = close.rolling(200).mean()
    v20 = vol.rolling(20).mean(); vsd = vol.rolling(20).std().replace(0, np.nan)
    high52 = close.rolling(252).max()
    rsi14_all = close.apply(rsi)
    atrp_all = close.apply(lambda s: atr(s).rename(s.name))

    for t in close.columns:
        s = close[t].dropna(); v = vol[t].reindex(s.index).fillna(0)
        if len(s) < 250:
            continue
        price = s.iloc[-1]
        ret1 = (s.iloc[-1]/s.iloc[-2]-1)*100 if len(s) > 1 else np.nan
        ret5 = (s.iloc[-1]/s.iloc[-6]-1)*100 if len(s) > 6 else np.nan
        ret20= (s.iloc[-1]/s.iloc[-21]-1)*100 if len(s) > 21 else np.nan
        rs60 = (s.iloc[-1]/s.iloc[-61]) / (spy.reindex(s.index).iloc[-1]/spy.reindex(s.index).iloc[-61]) - 1 if len(s)>61 else np.nan
        from_high = (s.iloc[-1]/high52[t].loc[s.index].iloc[-1]-1)*100
        above50 = bool(s.iloc[-1] > ma50[t].loc[s.index].iloc[-1])
        above200= bool(s.iloc[-1] > ma200[t].loc[s.index].iloc[-1])
        volz = float(((v.iloc[-1] - v20[t].loc[s.index].iloc[-1]) / vsd[t].loc[s.index].iloc[-1]) if vsd[t].loc[s.index].iloc[-1] else 0)
        rsi14v = float(rsi14_all[t].loc[s.index].iloc[-1])
        atrp = float(atrp_all[t].loc[s.index].iloc[-1]*100)
        breakout20 = bool(s.iloc[-1] >= s.rolling(20).max().iloc[-1] and volz > 1)
        out.append({
            "Ticker": t,
            "Price": round(price,2),
            "1d%": round(ret1,2),
            "5d%": round(ret5,2),
            "20d%": round(ret20,2),
            "RS60%": round(rs60*100,2) if pd.notna(rs60) else np.nan,
            "%from52wHigh": round(from_high*100,2),
            "Above50": above50,
            "Above200": above200,
            "VolZ20": round(volz,2),
            "RSI14": round(rsi14v,1),
            "ATR%": round(atrp,2),
            "Breakout20": breakout20,
        })
    df = pd.DataFrame(out).set_index("Ticker")
    return df.sort_values("RS60%", ascending=False)

# Composite score (tunable)

def score_table(df: pd.DataFrame, w_rs=0.35, w_dist=0.25, w_trend=0.2, w_vol=0.2) -> pd.DataFrame:
    trend = df["Above50"].astype(int)*0.5 + df["Above200"].astype(int)*0.5
    dist = -df["%from52wHigh"].clip(-100,100) # closer to high is better
    z = (df[["VolZ20"]] - df[["VolZ20"]].mean())/df[["VolZ20"]].std(ddof=0)
    score = w_rs*df["RS60%"].fillna(0) + w_dist*dist.fillna(0) + w_trend*(trend*100) + w_vol*z["VolZ20"].fillna(0)
    out = df.copy()
    out["Score"] = score.round(2)
    return out.sort_values("Score", ascending=False)

# ---------------------
# Sidebar controls
# ---------------------
with st.sidebar:
    st.title("AI + Datacenter Tradeboard")
    st.write("Decision‑oriented signals using only price/volume.")
    preset = st.selectbox("Universe preset", ["Core","US Vendors","US Hypers","Add REITs/Power","Custom"], index=0)
    period = st.selectbox("History", ["1y","2y","5y"], index=1)
    custom = st.text_input("Add tickers (comma separated)", value="")
    st.markdown("**Scoring weights** (sum is arbitrary; relative weights matter)")
    w_rs = st.slider("Relative strength (60d)", 0.0, 1.0, 0.35, 0.05)
    w_dist = st.slider("Near 52w High", 0.0, 1.0, 0.25, 0.05)
    w_trend = st.slider("Trend (50/200DMA)", 0.0, 1.0, 0.20, 0.05)
    w_vol = st.slider("Volume Z‑score", 0.0, 1.0, 0.20, 0.05)

# Select universe
if preset == "Core":
    universe = UNIVERSE_DEFAULT
elif preset == "US Vendors":
    universe = US_VENDORS
elif preset == "US Hypers":
    universe = US_HYPERS
elif preset == "Add REITs/Power":
    universe = UNIVERSE_DEFAULT + REITS_PWR
else:
    universe = []

if custom.strip():
    add = [t.strip().upper() for t in custom.split(',') if t.strip()]
    universe = list(dict.fromkeys(universe + add))

# ---------------------
# Data load
# ---------------------
close, vol = fetch_ohlcv(tuple(universe + ["SPY"]), period=period)
if close.empty:
    st.warning("No data. Try different universe or history.")
    st.stop()
spy = close.pop("SPY")

# ---------------------
# Theme overview
# ---------------------
idx = theme_index(close)
rel = idx / (spy/spy.iloc[0])

k1,k2,k3,k4 = st.columns(4)
k1.metric("Theme total return", f"{(idx.iloc[-1]-1)*100:,.1f}%")
k2.metric("Theme vs SPY", f"{(rel.iloc[-1]-1)*100:,.1f}%")
k3.metric("Breadth >50DMA", f"{(close.iloc[-1] > close.rolling(50).mean().iloc[-1]).mean()*100:,.1f}%")
k4.metric("New 20d highs", f"{int((close.iloc[-1] >= close.rolling(20).max().iloc[-1]).sum())}")

st.subheader("Theme index vs SPY")
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx.index, y=idx, mode="lines", name="Theme EW"))
fig.add_trace(go.Scatter(x=spy.index, y=spy/spy.iloc[0], mode="lines", name="SPY"))
fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", y=1.02, x=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------------
# Signals table & candidates
# ---------------------
base = compute_metrics(close, vol, spy)
scored = score_table(base, w_rs=w_rs, w_dist=w_dist, w_trend=w_trend, w_vol=w_vol)

st.subheader("Signals (sortable)")
st.dataframe(scored, use_container_width=True)

# Candidate lists
longs = scored.query("((Above50 and Above200 and `RS60%` > 0 and `%from52wHigh` > -5) or Breakout20)").head(8)
trims = scored.query("(`RS60%` < 0) or (Above200 == False)").sort_values("Score").head(8)

st.subheader("Long candidates")
if longs.empty:
    st.write("None meet criteria today.")
else:
    st.dataframe(longs[["Price","1d%","5d%","20d%","RS60%","%from52wHigh","VolZ20","RSI14","ATR%","Breakout20","Score"]], use_container_width=True)

st.subheader("Trim/Short candidates")
if trims.empty:
    st.write("None meet criteria today.")
else:
    st.dataframe(trims[["Price","1d%","5d%","20d%","RS60%","%from52wHigh","VolZ20","RSI14","ATR%","Breakout20","Score"]], use_container_width=True)

# ---------------------
# Pairs & Ratios
# ---------------------
rat1, rat2, rat3 = st.tabs(["NVDA vs Vendor basket","Vendors vs Hypers","REITs vs Power"]) 

with rat1:
    needed = ["NVDA","AMD","AVGO","MRVL","MU"]
    avail = [t for t in needed if t in close.columns]
    if "NVDA" in avail and len(avail) > 1:
        vendors = (close[[t for t in avail if t != "NVDA"]].pct_change().add(1).cumprod()).mean(axis=1)
        ratio = (close["NVDA"]/close["NVDA"].iloc[0]) / (vendors/vendors.iloc[0])
        f = go.Figure(); f.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="NVDA / Vendors EW"))
        f.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(f, use_container_width=True)
    else:
        st.write("Insufficient data for this ratio.")

with rat2:
    v = [t for t in US_VENDORS if t in close.columns]
    h = [t for t in US_HYPERS if t in close.columns]
    if len(v)>=2 and len(h)>=2:
        vend = (close[v].pct_change().add(1).cumprod()).mean(axis=1)
        hyp  = (close[h].pct_change().add(1).cumprod()).mean(axis=1)
        ratio = (vend/vend.iloc[0]) / (hyp/hyp.iloc[0])
        f = go.Figure(); f.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="Vendors / Hypers"))
        f.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(f, use_container_width=True)
    else:
        st.write("Insufficient data for this ratio.")

with rat3:
    r = [t for t in ["EQIX","DLR"] if t in close.columns]
    p = [t for t in ["VST","NEE","DUK"] if t in close.columns]
    if len(r)>=1 and len(p)>=1:
        reits = (close[r].pct_change().add(1).cumprod()).mean(axis=1)
        power = (close[p].pct_change().add(1).cumprod()).mean(axis=1)
        ratio = (reits/reits.iloc[0]) / (power/power.iloc[0])
        f = go.Figure(); f.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="REITs / Power"))
        f.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(f, use_container_width=True)
    else:
        st.write("Insufficient data for this ratio.")

st.caption("Signals from price/volume only. Use as a fast read; layer in fundamentals, positioning, and policy.")
