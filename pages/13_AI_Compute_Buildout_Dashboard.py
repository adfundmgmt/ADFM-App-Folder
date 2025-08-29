# ai_tradeboard.py

"""
AI + Datacenter Tradeboard — no logins, no keys
Source: Yahoo Finance via yfinance (daily prices & volume)

Focus: a **Weekly Playbook** that is short and readable.
- Longs / Trims / Breakouts, each with compact reasons: RS60, distance to 52w high, trend arrows (↑/↓), VolZ.
- Theme KPIs: Theme return, Theme vs SPY, % above 50DMA, 20d highs count.
- Compact ratio panels: NVDA/Vendors, Vendors/Hypers, REITs/Power.

Run
pip install streamlit yfinance pandas numpy plotly
streamlit run ai_tradeboard.py
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI + Datacenter Tradeboard", layout="wide")

# ---------------------
# Universe presets (no custom option)
# ---------------------
US_HYPERS   = ["AMZN","MSFT","GOOGL","META","ORCL","AAPL"]
US_VENDORS  = ["NVDA","AMD","AVGO","MRVL","SMCI","MU"]
EU_ASIA     = ["TSM","ASML","BABA","TCEHY"]
REITS_PWR   = ["EQIX","DLR","VST","NEE","DUK"]
UNIV_DEFAULT = US_HYPERS + US_VENDORS + EU_ASIA + REITS_PWR

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
# Indicators
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

def atr_pct(series: pd.Series, n: int = 14) -> pd.Series:
    tr = series.pct_change().abs()
    return tr.rolling(n).mean()*100

# metrics table

def compute_metrics(close: pd.DataFrame, vol: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    ma50 = close.rolling(50).mean(); ma200 = close.rolling(200).mean()
    v20 = vol.rolling(20).mean(); vsd = vol.rolling(20).std().replace(0, np.nan)
    high52 = close.rolling(252).max()
    rsi14 = close.apply(rsi)
    atrp  = close.apply(atr_pct)

    rows = []
    for t in close.columns:
        s = close[t].dropna(); v = vol[t].reindex(s.index).fillna(0)
        if len(s) < 61:
            continue
        pct = lambda r: (r-1)*100
        # RS60 as scalar with guards
        try:
            rs60_ratio = (s.iloc[-1]/s.iloc[-61]) / (spy.reindex(s.index).iloc[-1]/spy.reindex(s.index).iloc[-61])
            rs60_pct = float(pct(rs60_ratio))
        except Exception:
            rs60_pct = np.nan
        # 52w high distance (use last valid high scalar)
        try:
            hval = high52[t].reindex(s.index).dropna().iloc[-1]
            dist52w_pct = float((s.iloc[-1]/hval - 1)*100) if hval and np.isfinite(hval) else np.nan
        except Exception:
            dist52w_pct = np.nan
        rows.append({
            "ticker": t,
            "price": round(float(s.iloc[-1]),2),
            "d5": round(float(pct(s.iloc[-1]/s.iloc[-6])) if len(s)>6 else np.nan,2),
            "d20": round(float(pct(s.iloc[-1]/s.iloc[-21])) if len(s)>21 else np.nan,2),
            "rs60": round(rs60_pct,2) if pd.notna(rs60_pct) else np.nan,
            "dist52w": round(dist52w_pct,2) if pd.notna(dist52w_pct) else np.nan,
            "above50": bool(s.iloc[-1] > ma50[t].reindex(s.index).iloc[-1]),
            "above200": bool(s.iloc[-1] > ma200[t].reindex(s.index).iloc[-1]),
            "volz": round(float(((v.iloc[-1] - v20[t].reindex(s.index).iloc[-1]) / (vsd[t].reindex(s.index).iloc[-1] or np.nan))),2) if len(s)>20 else 0.0,
            "rsi14": round(float(rsi14[t].reindex(s.index).iloc[-1]),1),
            "atrp": round(float(atrp[t].reindex(s.index).iloc[-1]),2),
            "break20": bool(len(s)>=20 and s.iloc[-1] >= s.rolling(20).max().iloc[-1] and ((v.iloc[-1]-v20[t].reindex(s.index).iloc[-1]) / (vsd[t].reindex(s.index).iloc[-1] or np.nan))>1),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    return df.set_index("ticker")

# ranking (kept internal; we won't print the numeric score)

def rank_table(df: pd.DataFrame, w_rs=0.4, w_dist=0.25, w_trend=0.2, w_vol=0.15) -> pd.DataFrame:
    trend = df["above50"].astype(int)*0.5 + df["above200"].astype(int)*0.5
    dist = -df["dist52w"].clip(-100,100)
    z = (df[["volz"]] - df[["volz"]].mean())/df[["volz"]].std(ddof=0)
    score = w_rs*df["rs60"].fillna(0) + w_dist*dist.fillna(0) + w_trend*(trend*100) + w_vol*z["volz"].fillna(0)
    out = df.copy(); out["score"] = score
    return out.sort_values("score", ascending=False)

# compact reason string

def reason(row: pd.Series) -> str:
    trend = ("↑" if row["above50"] else "↓") + ("↑" if row["above200"] else "↓")
    bits: List[str] = [f"RS60 {row['rs60']:+.1f}%", f"{row['dist52w']:.1f}% from high", f"trend {trend}"]
    if row["volz"] > 1: bits.append(f"VolZ {row['volz']:.1f}")
    if row["break20"]: bits.append("20d breakout")
    return " · ".join(bits)

# ---------------------
# Sidebar controls (no Custom)
# ---------------------
with st.sidebar:
    st.title("AI + Datacenter Tradeboard")
    st.write("Weekly, digestible signals. Price/volume only.")
    preset = st.selectbox("Universe", ["Core","US Vendors","US Hypers","Add REITs/Power"], index=0)
    period = st.selectbox("History", ["1y","2y","5y"], index=1)
    st.markdown("Weights for ranking")
    w_rs = st.slider("Relative strength (60d)", 0.0, 1.0, 0.40, 0.05)
    w_dist = st.slider("Near 52w High", 0.0, 1.0, 0.25, 0.05)
    w_trend = st.slider("Trend (50/200DMA)", 0.0, 1.0, 0.20, 0.05)
    w_vol = st.slider("Volume Z-score", 0.0, 1.0, 0.15, 0.05)

if preset == "Core":
    universe = UNIV_DEFAULT
elif preset == "US Vendors":
    universe = US_VENDORS
elif preset == "US Hypers":
    universe = US_HYPERS
else:  # Add REITs/Power
    universe = UNIV_DEFAULT + REITS_PWR

# ---------------------
# Data
# ---------------------
close, vol = fetch_ohlcv(tuple(universe + ["SPY"]), period=period)
if close.empty:
    st.warning("No data. Try different universe or history.")
    st.stop()
spy = close.pop("SPY")

# ---------------------
# KPIs
# ---------------------
idx = theme_index(close)
rel = idx / (spy/spy.iloc[0])

c1,c2,c3,c4 = st.columns(4)
c1.metric("Theme total return", f"{(idx.iloc[-1]-1)*100:,.1f}%")
c2.metric("Theme vs SPY", f"{(rel.iloc[-1]-1)*100:,.1f}%")
c3.metric("50DMA breadth", f"{(close.iloc[-1] > close.rolling(50).mean().iloc[-1]).mean()*100:,.1f}%")
c4.metric("20d highs", f"{int((close.iloc[-1] >= close.rolling(20).max().iloc[-1]).sum())}")

# ---------------------
# Weekly playbook
# ---------------------
base = compute_metrics(close, vol, spy)
if base.empty:
    st.warning("Insufficient data for signals")
    st.stop()
scored = rank_table(base, w_rs=w_rs, w_dist=w_dist, w_trend=w_trend, w_vol=w_vol)

st.subheader("Weekly playbook")
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("**Longs**")
    longs = scored[(scored["above50"] & scored["above200"] & (scored["rs60"]>0) & (scored["dist52w"]>-5)) | (scored["break20"])]
    for t, row in longs.head(6).iterrows():
        st.markdown(f"**{t}** · {reason(row)}")
    if longs.empty:
        st.write("None")

with colB:
    st.markdown("**Trims**")
    trims = scored[(~scored["above200"]) | (scored["rs60"]<0)]
    for t, row in trims.head(6).iterrows():
        st.markdown(f"**{t}** · {reason(row)}")
    if trims.empty:
        st.write("None")

with colC:
    st.markdown("**Breakouts (20d high + volume)**")
    b = scored[scored["break20"]].sort_values(["volz","rs60"], ascending=False)
    for t, row in b.head(6).iterrows():
        st.markdown(f"**{t}** · {reason(row)}")
    if b.empty:
        st.write("None")

st.markdown("---")

# ---------------------
# Compact charts
# ---------------------
st.subheader("Theme vs SPY and key ratios")
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx.index, y=idx, mode="lines", name="Theme EW"))
fig.add_trace(go.Scatter(x=spy.index, y=spy/spy.iloc[0], mode="lines", name="SPY"))
fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", y=1.02, x=0))
st.plotly_chart(fig, use_container_width=True)

rat1, rat2, rat3 = st.columns(3)

with rat1:
    needed = ["NVDA","AMD","AVGO","MRVL","MU"]
    avail = [t for t in needed if t in close.columns]
    if "NVDA" in avail and len(avail) > 1:
        vendors = (close[[t for t in avail if t != "NVDA"]].pct_change().add(1).cumprod()).mean(axis=1)
        ratio = (close["NVDA"]/close["NVDA"].iloc[0]) / (vendors/vendors.iloc[0])
        f = go.Figure(); f.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="NVDA/Vendors"))
        f.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(f, use_container_width=True)

with rat2:
    v = [t for t in US_VENDORS if t in close.columns]
    h = [t for t in US_HYPERS if t in close.columns]
    if len(v)>=2 and len(h)>=2:
        vend = (close[v].pct_change().add(1).cumprod()).mean(axis=1)
        hyp  = (close[h].pct_change().add(1).cumprod()).mean(axis=1)
        ratio = (vend/vend.iloc[0]) / (hyp/hyp.iloc[0])
        f = go.Figure(); f.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="Vendors/Hypers"))
        f.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(f, use_container_width=True)

with rat3:
    r = [t for t in ["EQIX","DLR"] if t in close.columns]
    p = [t for t in ["VST","NEE","DUK"] if t in close.columns]
    if len(r)>=1 and len(p)>=1:
        reits = (close[r].pct_change().add(1).cumprod()).mean(axis=1)
        power = (close[p].pct_change().add(1).cumprod()).mean(axis=1)
        ratio = (reits/reits.iloc[0]) / (power/power.iloc[0])
        f = go.Figure(); f.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="REITs/Power"))
        f.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(f, use_container_width=True)

# ---------------------
# Optional full table (collapsed)
# ---------------------
with st.expander("Full signals table"):
    view = scored.copy()
    view.rename(columns={"d5":"5d%","d20":"20d%","rs60":"RS60%","dist52w":"%from52wHigh","volz":"VolZ20","atrp":"ATR%"}, inplace=True)
    st.dataframe(view[["price","5d%","20d%","RS60%","%from52wHigh","VolZ20","rsi14","ATR%","break20","above50","above200"]], use_container_width=True)

st.caption("Use as a weekly bias/timing sheet. Layer fundamentals, positioning, and policy.")
