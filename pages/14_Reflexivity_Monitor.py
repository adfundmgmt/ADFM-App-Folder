# reflexivity_monitor.py
# Streamlit page: Reflexivity Monitor (Soros)
# Pulls FRED data via fredapi if available, else falls back to pandas_datareader.
# Purpose: quantify when price action is feeding back into fundamentals and produce a reflexivity intensity index.

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Try to import fredapi and pandas_datareader
FRED_AVAILABLE = False
FRED = None

try:
    from fredapi import Fred
    fred_key = None
    try:
        fred_key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        fred_key = None
    if fred_key is None:
        fred_key = os.getenv("FRED_API_KEY")
    FRED = Fred(api_key=fred_key) if fred_key else Fred()
    FRED_AVAILABLE = True
except Exception:
    pass

from pandas_datareader import data as pdr

# -----------------------------
# Helpers
# -----------------------------

def _zscore(x: pd.Series, clip=3.0):
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    return z.clip(-clip, clip) if clip is not None else z

@st.cache_data(show_spinner=False)
def load_prices(tickers, start="2005-01-01"):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.ffill().dropna(how="all")
    return data

@st.cache_data(show_spinner=False)
def load_fred_series(series_map: dict, start="2005-01-01"):
    frames = []
    # Try fredapi first, fallback to pandas_datareader
    if FRED_AVAILABLE:
        try:
            for name, sid in series_map.items():
                s = FRED.get_series(sid)
                s.name = name
                frames.append(s.to_frame())
            if frames:
                df = pd.concat(frames, axis=1)
                df = df.loc[pd.to_datetime(start):]
                daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
                df = df.reindex(daily).ffill()
                return df
        except Exception:
            pass
    # Fallback to pandas_datareader (no API key needed)
    try:
        for name, sid in series_map.items():
            s = pdr.DataReader(sid, "fred", start)[sid]
            s.name = name
            frames.append(s.to_frame())
        if frames:
            df = pd.concat(frames, axis=1)
            daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
            df = df.reindex(daily).ffill()
            return df
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def net_liquidity(df):
    need = ["WALCL","RRPONTSYD","WTREGEN"]
    if not set(need).issubset(df.columns):
        return pd.Series(dtype=float)
    nl = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]
    nl.name = "NetLiquidity"
    return nl

def real_yield(df):
    if {"DGS10","T10YIE"}.issubset(df.columns):
        ry = df["DGS10"] - df["T10YIE"]
        ry.name = "Real10y"
        return ry
    return pd.Series(dtype=float)

def growth_yoy(s: pd.Series, months=12):
    return s.pct_change(int(months*21))

def rolling_lag_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=126, lag_days=21):
    dF_fwd = fund_delta.shift(-lag_days)
    aligned = pd.concat([asset_ret, dF_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[asset_ret.name].rolling(window).corr(aligned[dF_fwd.name])

def rolling_lead_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=126, lead_days=21):
    r_fwd = asset_ret.shift(-lead_days)
    aligned = pd.concat([fund_delta, r_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[fund_delta.name].rolling(window).corr(aligned[r_fwd.name])

def make_reflexivity_score(asset_px: pd.Series, fundamentals: pd.DataFrame, params: dict):
    r = asset_px.pct_change().rename("ret")
    scores = []
    meta = []
    for fcol in fundamentals.columns:
        f = fundamentals[fcol].rename(fcol)
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
    w = np.array([params.get("factor_weights", {}).get(m["factor"], 1.0) for m in meta], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
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
assets_default = ["SPY","QQQ","IWM","HYG","XLF","XHB","XLE","GLD","BTC-USD"]
assets = st.sidebar.text_input("Assets (comma separated)", ",".join(assets_default)).replace(" ","").split(",")
window = st.sidebar.number_input("Rolling window (days)", 60, 252, 126, step=21)
lags = st.sidebar.multiselect("Lags for lead/lag tests (days)", [21, 63, 126], [21, 63])

st.sidebar.subheader("FRED series")
series_map = {
    "WALCL": "WALCL",
    "RRPONTSYD": "RRPONTSYD",
    "WTREGEN": "WTREGEN",
    "DGS10": "DGS10",
    "T10YIE": "T10YIE",
    "HY_OAS": "BAMLH0A0HYM2",
    "M2SL": "M2SL",
    "INDPRO": "INDPRO",
}
custom_series = st.sidebar.text_area("Extra FRED series (name=ID per line)")
if custom_series.strip():
    for line in custom_series.splitlines():
        if "=" in line:
            name, sid = [x.strip() for x in line.split("=", 1)]
            series_map[name] = sid

with st.spinner("Loading prices"):
    px = load_prices(assets, start=str(start_date))

with st.spinner("Loading FRED series"):
    fred_df = load_fred_series(series_map, start=str(start_date))

# Fundamentals
fund = pd.DataFrame(index=px.index)
if not fred_df.empty:
    fred_df = fred_df.reindex(px.index).ffill()
    tmp_nl = net_liquidity(fred_df)
    if not tmp_nl.empty:
        fund["NetLiquidity"] = tmp_nl
    tmp_ry = real_yield(fred_df)
    if not tmp_ry.empty:
        fund["Real10y"] = tmp_ry
    if "HY_OAS" in fred_df:
        fund["HY_OAS"] = fred_df["HY_OAS"]
    if "M2SL" in fred_df:
        fund["M2_YoY"] = growth_yoy(fred_df["M2SL"]) 
    if "INDPRO" in fred_df:
        fund["INDPRO_YoY"] = growth_yoy(fred_df["INDPRO"]) 

fund = fund.ffill()

base_weights = {"NetLiquidity": 1.25, "Real10y": 1.0, "HY_OAS": 1.25, "M2_YoY": 0.75, "INDPRO_YoY": 0.75}
params = {"lags": lags, "window": int(window), "factor_weights": base_weights}

st.subheader("Reflexivity intensity by asset")

valid_assets = [a for a in assets if a in px.columns]
tabs = st.tabs(valid_assets if valid_assets else ["No assets"])

for i, tkr in enumerate(valid_assets):
    with tabs[i]:
        s = px[tkr].dropna()
        if s.empty or fund.dropna(how="all").empty:
            st.info("Need both prices and fundamentals to compute the index.")
            continue
        ri, details = make_reflexivity_score(s, fund.dropna(how="all"), params)
        if ri.empty:
            st.info("Insufficient data for rolling correlations.")
            continue
        ri_z = _zscore(ri)
        ri_idx = (50 + 15*ri_z).clip(0, 100)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.line_chart(pd.DataFrame({"ReflexivityIntensity": ri, "Gauge": ri_idx}, index=ri.index))
        with col2:
            latest_val = float(ri_idx.dropna().iloc[-1]) if not ri_idx.dropna().empty else np.nan
            st.metric(label=f"{tkr} latest reflexivity", value=f"{latest_val:.1f}" if not np.isnan(latest_val) else "NA", help="0 to 100. Above 65 suggests price likely driving fundamentals. Below 35 suggests fundamentals leading price.")
            st.caption("Rule of thumb: > 65 self reinforcing, 35 to 65 neutral, < 35 self correcting.")
        with st.expander("Factor contributions and spreads"):
            if not details.empty:
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
- Uses fredapi if available, else pandas_datareader fallback (no key required)
- Series have different frequencies. We daily align with forward fill to avoid lookahead
- This is a meta signal to size risk and to spot parabolic feedback loops. It is not a day trading signal
"""
)
