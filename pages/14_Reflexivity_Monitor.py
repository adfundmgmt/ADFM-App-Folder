# reflexivity_monitor.py
# Streamlit page: Reflexivity Monitor (Soros)
# Purpose: quantify when price action is feeding back into fundamentals and produce a reflexivity intensity index

import os
import io
import math
import json
import time
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional libraries. Your app already uses yfinance and may use fredapi. If not, install them in your env.
import yfinance as yf

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

# -----------------------------
# Helpers
# -----------------------------

def _zscore(x: pd.Series, clip=3.0):
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z

@st.cache_data(show_spinner=False)
def load_prices(tickers, start="2005-01-01"):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.ffill().dropna(how="all")
    return data

@st.cache_data(show_spinner=False)
def load_fred_series(series_map: dict, start="2005-01-01"):
    if not FRED_AVAILABLE:
        st.warning("fredapi not available. Install fredapi and set FRED_API_KEY to enable FRED pulls.")
        return pd.DataFrame()
    api_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=api_key) if api_key else Fred()
    frames = []
    for name, sid in series_map.items():
        s = fred.get_series(sid)
        s.name = name
        s = s.to_frame()
        frames.append(s)
    df = pd.concat(frames, axis=1)
    df = df.loc[pd.to_datetime(start):]
    # Business daily index for alignment
    daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
    df = df.reindex(daily).ffill()
    return df

def net_liquidity(df):
    # Net Liquidity = WALCL - RRPONTSYD - WTREGEN
    need = ["WALCL","RRPONTSYD","WTREGEN"]
    if not set(need).issubset(df.columns):
        return pd.Series(dtype=float)
    nl = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]
    nl.name = "NetLiquidity"
    return nl

def real_yield(df):
    # Real10 = DGS10 - T10YIE
    if {"DGS10","T10YIE"}.issubset(df.columns):
        ry = df["DGS10"] - df["T10YIE"]
        ry.name = "Real10y"
        return ry
    return pd.Series(dtype=float)

def hy_oas(df):
    # High Yield OAS from FRED id BAMLH0A0HYM2
    col = "HY_OAS"
    if col in df.columns:
        return df[col].rename("HYOAS")
    return pd.Series(dtype=float)

def series_pct_change_annualized(s: pd.Series, period_days=21):
    ret = s.pct_change(period_days)
    if period_days > 0:
        ret = ((1 + ret) ** (252/period_days)) - 1
    return ret

def growth_yoy(s: pd.Series, months=12):
    # YoY growth using 21 trading days per month as a rough bridge
    return s.pct_change(int(months*21))

def rolling_lag_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=126, lag_days=21):
    # corr between r_t and dF_{t+lag}
    dF_fwd = fund_delta.shift(-lag_days)
    aligned = pd.concat([asset_ret, dF_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[asset_ret.name].rolling(window).corr(aligned[dF_fwd.name])

def rolling_lead_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=126, lead_days=21):
    # corr between dF_t and r_{t+lead}
    r_fwd = asset_ret.shift(-lead_days)
    aligned = pd.concat([fund_delta, r_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[fund_delta.name].rolling(window).corr(aligned[r_fwd.name])

def make_reflexivity_score(asset_px: pd.Series, fundamentals: pd.DataFrame, params: dict):
    # Compute price->fund and fund->price correlation spreads over multiple horizons
    r = asset_px.pct_change().rename("ret")
    scores = []
    meta = []
    for fcol in fundamentals.columns:
        f = fundamentals[fcol].rename(fcol)
        # use daily change scaled
        dF = f.diff().rename(f"d_{fcol}")
        for lag in params.get("lags", [21, 63]):
            rc = rolling_lag_corr(r, dF, window=params.get("window", 126), lag_days=lag)
            fc = rolling_lead_corr(r, dF, window=params.get("window", 126), lead_days=lag)
            spread = rc - fc
            spread.name = f"spread_{fcol}_{lag}d"
            scores.append(spread)
            meta.append({"factor": fcol, "lag": lag})
    if not scores:
        return pd.Series(dtype=float), pd.DataFrame()
    S = pd.concat(scores, axis=1)
    # Weight by sector sensitivity if provided, else equal
    w = []
    for m in meta:
        fac = m["factor"]
        w.append(params.get("factor_weights", {}).get(fac, 1.0))
    w = np.array(w, dtype=float)
    w = w / (np.sum(w) if np.sum(w) != 0 else 1.0)
    # zscore each column before weighting
    Z = S.apply(_zscore)
    composite = (Z @ w).rename("ReflexivityIntensity")
    return composite, S

# -----------------------------
# UI
# -----------------------------

st.title("Reflexivity Monitor (Soros)")
st.caption("Quantify when prices are feeding back into fundamentals. Output is a real time reflexivity intensity index for position sizing and reversal timing.")

st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2010-01-01"))

default_assets = ["SPY","QQQ","IWM","HYG","XLF","XHB","XLE","GLD","BTC-USD"]
assets = st.sidebar.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")
window = st.sidebar.number_input("Rolling window (days)", 60, 252, 126, step=21)
lags = st.sidebar.multiselect("Lags for lead/lag tests (days)", [21, 63, 126], [21,63])

st.sidebar.subheader("FRED series to pull")
# Core macro set
series_map = {
    # Liquidity and policy
    "WALCL": "WALCL",            # Fed balance sheet
    "RRPONTSYD": "RRPONTSYD",    # Overnight RRP
    "WTREGEN": "WTREGEN",        # Treasury General Account
    # Rates and inflation
    "DGS10": "DGS10",            # 10y yield
    "T10YIE": "T10YIE",          # 10y breakeven
    # Credit
    "HY_OAS": "BAMLH0A0HYM2",    # High Yield OAS
    # Money and activity
    "M2SL": "M2SL",              # M2
    "INDPRO": "INDPRO",          # Industrial production
}

custom_series = st.sidebar.text_area("Extra FRED series (name=ID per line)", value="")
if custom_series.strip():
    for line in custom_series.splitlines():
        if "=" in line:
            name, sid = [x.strip() for x in line.split("=",1)]
            series_map[name] = sid

with st.spinner("Loading prices..."):
    px = load_prices(assets, start=str(start_date))

with st.spinner("Loading FRED series..."):
    fred_df = load_fred_series(series_map, start=str(start_date))

# Build fundamental features
fund = pd.DataFrame(index=px.index)
if not fred_df.empty:
    fred_df = fred_df.reindex(px.index).ffill()
    fund["NetLiquidity"] = net_liquidity(fred_df)
    fund["Real10y"] = real_yield(fred_df)
    if "HY_OAS" in fred_df:
        fund["HY_OAS"] = fred_df["HY_OAS"]
    if "M2SL" in fred_df:
        fund["M2_YoY"] = growth_yoy(fred_df["M2SL"]) 
    if "INDPRO" in fred_df:
        fund["INDPRO_YoY"] = growth_yoy(fred_df["INDPRO"]) 

fund = fund.ffill()

# Factor weights by liquidity sensitivity. You can edit these based on your research.
base_weights = {
    "NetLiquidity": 1.25,
    "Real10y": 1.0,
    "HY_OAS": 1.25,
    "M2_YoY": 0.75,
    "INDPRO_YoY": 0.75,
}

params = {
    "lags": lags,
    "window": int(window),
    "factor_weights": base_weights,
}

st.subheader("Reflexivity intensity by asset")

tabs = st.tabs([f for f in assets if f in px.columns])
for i, tkr in enumerate([f for f in assets if f in px.columns]):
    with tabs[i]:
        s = px[tkr].dropna()
        if s.empty or fund.dropna(how="all").empty:
            st.info("Need both prices and fundamentals to compute the index.")
            continue
        ri, details = make_reflexivity_score(s, fund.dropna(how="all"), params)
        if ri.empty:
            st.info("Insufficient data for rolling correlations.")
            continue
        # Normalize to 0..100 for an easy gauge
        ri_z = _zscore(ri)
        ri_idx = 50 + 15*ri_z  # 50 baseline, 15 per z
        ri_idx = ri_idx.clip(0,100)

        col1, col2 = st.columns([2,1])
        with col1:
            st.line_chart(pd.DataFrame({"ReflexivityIntensity": ri, "Gauge": ri_idx}, index=ri.index))
        with col2:
            st.metric(label=f"{tkr} latest reflexivity", value=f"{ri_idx.dropna().iloc[-1]:.1f}", help="0 to 100. Above 65 suggests price likely driving fundamentals. Below 35 suggests fundamentals leading price. Mid range suggests neutral.")
            st.caption("Rule of thumb: > 65 self reinforcing, 35 to 65 neutral, < 35 self correcting.")

        with st.expander("Factor contributions and spreads"):
            if not details.empty:
                # Show last values by factor and lag
                last_vals = details.dropna().iloc[-1].sort_values(ascending=False)
                st.bar_chart(last_vals)
                st.dataframe(details.tail(10))

        with st.expander("Download data"):
            out = pd.concat([s.rename("price"), fund], axis=1)
            out["ReflexivityIntensity"] = ri
            csv = out.to_csv().encode()
            st.download_button("Download CSV", csv, file_name=f"{tkr}_reflexivity_monitor.csv")

st.divider()

st.subheader("How the score is built")
st.markdown(
    """
**Idea**: measure whether price returns predict future changes in macro fundamentals more than fundamentals predict future returns.

**Steps**
- Build fundamental features: Net Liquidity, Real 10y yield, HY OAS, M2 YoY, INDPRO YoY
- Compute rolling correlations of price returns vs future fundamental changes at user selected lags
- Compute rolling correlations of fundamental changes vs future price returns
- Take the spread. Positive means price is leading fundamentals. Negative means fundamentals are leading price
- Z score and weight by liquidity sensitivity to produce a composite reflexivity intensity index

**Caveats**
- Use ALFRED vintages if you want real time data without revisions
- Series have different frequencies. We daily align with forward fill to avoid lookahead
- This is a meta signal to size risk and to spot parabolic feedback loops. It is not a day trading signal
"""
)
