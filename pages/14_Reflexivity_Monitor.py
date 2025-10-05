# reflexivity_monitor.py
# Streamlit page: Reflexivity Monitor (Soros)
# FRED pulls: try fredapi, fallback to pandas_datareader (no key needed).
# Perf: weekly option, lighter calculations, safer charts.

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.graph_objects as go

# fredapi optional
FRED_AVAILABLE, FRED = False, None
try:
    from fredapi import Fred
    fred_key = None
    try:
        fred_key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        fred_key = None
    fred_key = fred_key or os.getenv("FRED_API_KEY")
    FRED = Fred(api_key=fred_key) if fred_key else Fred()
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

# --------------- Helpers ---------------

def _zscore(x: pd.Series, clip=3.0):
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    return z.clip(-clip, clip) if clip is not None else z

@st.cache_data(show_spinner=False)
def load_prices(tickers, start="2010-01-01", freq="W-FRI"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.ffill()
    if freq:
        px = px.resample(freq).last()
    return px.dropna(how="all")

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_fred_series(series_map: dict, start="2010-01-01", freq="W-FRI"):
    frames = []
    # Try fredapi first
    if FRED_AVAILABLE:
        try:
            for name, sid in series_map.items():
                s = FRED.get_series(sid)
                s.name = name
                frames.append(s.to_frame())
        except Exception:
            frames = []
    # Fallback to pandas_datareader
    if not frames:
        for name, sid in series_map.items():
            try:
                s = pdr.DataReader(sid, "fred", start)[sid]
                s.name = name
                frames.append(s.to_frame())
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df = df.loc[pd.to_datetime(start):]
    if freq:
        df = df.resample(freq).ffill()
    else:
        daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
        df = df.reindex(daily).ffill()
    return df

def net_liquidity(df):
    need = {"WALCL","RRPONTSYD","WTREGEN"}
    if not need.issubset(df.columns):
        return pd.Series(dtype=float)
    nl = df["WALCL"] - df["RRPONTSYD"] - df["WTREGEN"]
    nl.name = "NetLiquidity"
    return nl

def real_yield(df):
    have = {"DGS10","T10YIE"}.issubset(df.columns)
    if have:
        ry = df["DGS10"] - df["T10YIE"]
        ry.name = "Real10y"
        return ry
    return pd.Series(dtype=float)

def growth_yoy(s: pd.Series, months=12, freq_weeks=True):
    if s.empty:
        return s
    if freq_weeks:
        return s.pct_change(int(months*4.33/1))  # ~weeks
    return s.pct_change(int(months*21))

def rolling_lag_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=52, lag_steps=4):
    dF_fwd = fund_delta.shift(-lag_steps)
    aligned = pd.concat([asset_ret, dF_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[asset_ret.name].rolling(window).corr(aligned[dF_fwd.name])

def rolling_lead_corr(asset_ret: pd.Series, fund_delta: pd.Series, window=52, lead_steps=4):
    r_fwd = asset_ret.shift(-lead_steps)
    aligned = pd.concat([fund_delta, r_fwd], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned[fund_delta.name].rolling(window).corr(aligned[r_fwd.name])

def make_reflexivity_score(asset_px: pd.Series, fundamentals: pd.DataFrame, params: dict):
    """
    Build reflexivity score using rolling correlations between price returns and
    *lag-matched* fundamental changes. To avoid zero-variance windows (e.g.,
    monthly series forward-filled to weekly), compute dF as a diff over the
    same step as the tested lag.
    """
    r = asset_px.pct_change().rename("ret")
    scores, meta = [], []
    window = params.get("window", 52)
    lags = params.get("lags", [4, 12])
    for fcol in fundamentals.columns:
        f = fundamentals[fcol].rename(fcol)
        for lag in lags:
            # change over the lag step to inject variance for monthly series
            dF = f.diff(int(lag)).rename(f"d_{fcol}")
            rc = rolling_lag_corr(r, dF, window=window, lag_steps=int(lag))
            fc = rolling_lead_corr(r, dF, window=window, lead_steps=int(lag))
            spread = rc - fc
            spread.name = f"spread_{fcol}_{lag}"
            scores.append(spread)
            meta.append({"factor": fcol, "lag": lag})
    if not scores:
        return pd.Series(dtype=float), pd.DataFrame()
    S = pd.concat(scores, axis=1)
    # Drop columns that are entirely NaN to prevent empty last-row errors
    S = S.dropna(axis=1, how="all")
    if S.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    w = np.array([params.get("factor_weights", {}).get(m["factor"], 1.0) for m in meta if f"spread_{m['factor']}_{m['lag']}" in S.columns], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    Z = S.apply(_zscore)
    composite = (Z @ w).rename("ReflexivityIntensity")
    return composite, S

# --------------- UI ---------------

st.title("Reflexivity Monitor (Soros)")
st.caption("Quantify when prices are feeding back into fundamentals. Output is a real time reflexivity intensity index for position sizing and reversal timing.")

st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2010-01-01"))
freq = st.sidebar.selectbox("Computation frequency", ["Weekly (fast)", "Daily (slow)"] , index=0)
use_weekly = freq.startswith("Weekly")

assets_default = ["SPY","QQQ","IWM","HYG","XLF","XHB","XLE","GLD","BTC-USD"]
assets = st.sidebar.text_input("Assets (comma separated)", ",".join(assets_default)).replace(" ","").split(",")

# Windows and lags in steps of chosen frequency
if use_weekly:
    window = st.sidebar.number_input("Rolling window (weeks)", 26, 156, 52, step=4)
    sel_lags = st.sidebar.multiselect("Lead/Lag steps", [4, 12, 26], [4, 12])
    freq_str = "W-FRI"
else:
    window = st.sidebar.number_input("Rolling window (days)", 60, 252, 126, step=21)
    sel_lags = st.sidebar.multiselect("Lead/Lag steps (days)", [21, 63, 126], [21, 63])
    freq_str = None

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
    px = load_prices(assets, start=str(start_date), freq=freq_str)

with st.spinner("Loading FRED series"):
    fred_df = load_fred_series(series_map, start=str(start_date), freq=freq_str or "W-FRI")

# Fundamentals
fund = pd.DataFrame(index=px.index)
if not fred_df.empty:
    fred_df = fred_df.reindex(px.index).ffill()
    nl = net_liquidity(fred_df)
    if not nl.empty:
        fund["NetLiquidity"] = nl
    ry = real_yield(fred_df)
    if not ry.empty:
        fund["Real10y"] = ry
    if "HY_OAS" in fred_df:
        fund["HY_OAS"] = fred_df["HY_OAS"]
    if "M2SL" in fred_df:
        fund["M2_YoY"] = growth_yoy(fred_df["M2SL"], freq_weeks=use_weekly)
    if "INDPRO" in fred_df:
        fund["INDPRO_YoY"] = growth_yoy(fred_df["INDPRO"], freq_weeks=use_weekly)

fund = fund.ffill()

base_weights = {"NetLiquidity": 1.25, "Real10y": 1.0, "HY_OAS": 1.25, "M2_YoY": 0.75, "INDPRO_YoY": 0.75}
params = {"lags": sel_lags, "window": int(window), "factor_weights": base_weights}

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
        c1, c2 = st.columns([2,1])
        with c1:
            # Plot only the 0-100 gauge with regime bands, yearly ticks
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ri.index, y=ri_idx, name="Gauge 0-100", mode="lines"))
            fig.add_hline(y=65, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=35, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_yaxes(range=[0,100], title_text="Reflexivity gauge")
            fig.update_xaxes(dtick="M12", tickformat="%Y", title_text="Year")
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            latest_val = float(ri_idx.dropna().iloc[-1]) if not ri_idx.dropna().empty else np.nan
            st.metric(label=f"{tkr} latest reflexivity", value=f"{latest_val:.1f}" if not np.isnan(latest_val) else "NA")
            st.caption("0 to 100 scale. >65 self reinforcing, 35-65 neutral, <35 self correcting.")
        with st.expander("Factor contributions (latest)"):
            if not details.empty:
                det_nonan = details.dropna(how="all")
                if det_nonan.empty:
                    st.info("No valid factor spreads yet for this asset and settings.")
                else:
                    try:
                        last_row = det_nonan.iloc[-1]
                        fig = go.Figure(data=[go.Bar(x=last_row.index, y=last_row.values)])
                        fig.update_layout(height=300, margin=dict(l=20,r=20,t=10,b=10))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.dataframe(det_nonan.tail(5))
                    st.dataframe(det_nonan.tail(5))
            else:
                st.info("No factor details available with current frequency/window.")
        with st.expander("Download data"):
            out = pd.concat([s.rename("price"), fund], axis=1)
            out["ReflexivityIntensity"] = ri
            st.download_button("Download CSV", out.to_csv().encode(), file_name=f"{tkr}_reflexivity_monitor.csv")

st.divider()

st.subheader("Notes")
st.markdown(
    """
- Weekly frequency is the default for speed and stability; switch to Daily if needed.
- FRED fetching tries fredapi first, then falls back to pandas_datareader with no key.
- Fundamentals are forward-filled and aligned to the chosen frequency to avoid lookahead.
- The score compares price→future fundamentals vs fundamentals→future price. Positive spread implies reflexive loop.
"""
)
