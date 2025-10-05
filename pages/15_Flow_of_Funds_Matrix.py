# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix (Druckenmiller) — always-on version with hard fallbacks
# Weekly-only. FRED via fredapi if present; else pandas_datareader('fred').
# Regress weekly log returns on 4/8w flow shocks; never fails due to overlap.

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# -------- fredapi (optional) --------
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


# -------- helpers --------
@st.cache_data(show_spinner=False)
def load_prices(tickers, start="2010-01-01"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.ffill()

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_fred(series_map: dict, start="2010-01-01"):
    frames = []
    if FRED_AVAILABLE:
        try:
            for name, sid in series_map.items():
                s = FRED.get_series(sid)
                s.name = name
                frames.append(s.to_frame())
        except Exception:
            frames = []
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
    return df

def weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="W-FRI")

def robust_scale(s: pd.Series):
    """MAD z-score; fallbacks: std z-score -> mean-centering -> identity (no scale)."""
    if s.isna().all():
        return s
    med = s.median()
    mad = (s - med).abs().median()
    if np.isfinite(mad) and mad > 0:
        return (s - med) / (1.4826 * mad)
    std = s.std(ddof=0)
    if np.isfinite(std) and std > 0:
        return (s - s.mean()) / (std + 1e-9)
    return s - s.mean() if np.isfinite(s.mean()) else s

def build_design(fund_df: pd.DataFrame, flow_list, horizons, min_rows=26):
    feats = {}
    for c in flow_list:
        if c not in fund_df.columns:
            continue
        for h in horizons:
            d = fund_df[c] - fund_df[c].shift(h)
            x = robust_scale(d)
            feats[f"{c}_d{h}w"] = x
    if not feats:
        return pd.DataFrame(), []
    X = pd.DataFrame(feats, index=fund_df.index)
    valid = [c for c in X.columns if X[c].notna().sum() >= min_rows]
    return X[valid], valid


# -------- UI --------
st.set_page_config(page_title="Flow of Funds Matrix (Druckenmiller)", layout="wide")
st.title("Flow of Funds Matrix (Druckenmiller)")
st.caption("Identify where incremental liquidity and relative performance co-move. Weekly regressions of asset returns on flow-of-funds shocks.")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Flows (weekly):**
- **Fed Net Liquidity (preferred):** WALCL − RRPONTSYD − TGA
- **TGA:** WTREGEN (weekly change) or WDTGAL (level), sign-flipped for drain
- **FX Reserves (optional):** sum of CHN/JPN reserve series if provided

We compute **4/8-week deltas**, robust-scale, and regress **weekly log returns**.
**Liquidity Beta** = sum of betas to sign-corrected liquidity shocks.
"""
    )
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2010-01-01"))
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")
    st.markdown("---")
    st.subheader("FRED series (auto)")
    st.code("WALCL, RRPONTSYD, WDTGAL  # preloaded; add WTREGEN / CHN_RES / JPN_RES if you have them", language="text")
    extra = st.text_area("Extra FRED (name=ID per line)", height=120)

# base FRED ids (TGA level by default)
fred_ids = {"WALCL": "WALCL", "RRPONTSYD": "RRPONTSYD", "WDTGAL": "WDTGAL"}
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            fred_ids[k] = v

# -------- data --------
with st.spinner("Loading data"):
    px_daily = load_prices(tickers, start=str(start_date))
    widx = weekly_index(start_date)
    px = px_daily.resample("W-FRI").last().reindex(widx).ffill()

    fred_raw = load_fred(fred_ids, start=str(start_date))
    fred = fred_raw.reindex(widx).ffill() if not fred_raw.empty else pd.DataFrame(index=widx)

if px.empty:
    st.error("No price data.")
    st.stop()

# build flows (with hard fallbacks)
fund = pd.DataFrame(index=widx)
have = set(fred.columns)

# TGA choice preference: WTREGEN (weekly change) > WDTGAL (level)
tga_series = fred["WTREGEN"] if "WTREGEN" in have else (fred["WDTGAL"] if "WDTGAL" in have else None)

# Preferred: WALCL & RRP present
if {"WALCL", "RRPONTSYD"}.issubset(have):
    if tga_series is not None:
        fund["Fed_NetLiq"] = fred["WALCL"] - fred["RRPONTSYD"] - tga_series
        fund["TGA_Drain"] = -tga_series
    else:
        # degrade gracefully: omit TGA
        fund["Fed_NetLiq"] = fred["WALCL"] - fred["RRPONTSYD"]
# If RRP missing, use WALCL as proxy to ensure at least one factor survives
elif "WALCL" in have:
    fund["Fed_NetLiq_proxy"] = fred["WALCL"]

# FX reserves auto-aggregate if provided
fx_cols = [c for c in fred.columns if c.upper().startswith(("CHN", "JPN")) or c.endswith("_RES")]
if fx_cols:
    fund["FX_Reserves"] = fred[fx_cols].sum(axis=1)

# choose target flow names in priority order
flow_priority = [
    ["Fed_NetLiq", "TGA_Drain", "FX_Reserves"],  # best case
    ["Fed_NetLiq", "TGA_Drain"],
    ["Fed_NetLiq"],
    ["Fed_NetLiq_proxy"],                        # last resort
]
flow_list = None
for candidate in flow_priority:
    ok = [c for c in candidate if c in fund.columns]
    if ok:
        flow_list = ok
        break

if flow_list is None:
    st.error("No usable flow series (WALCL/RRP/TGA missing and no proxy). Add at least WALCL and RRPONTSYD.")
    st.stop()

# -------- design matrix (progressive) --------
# pass 1: requested horizons
X1, valid1 = build_design(fund, flow_list, horizons, min_rows=26)

# pass 2: if empty, use 4w only
if X1.empty or len(valid1) == 0:
    X1, valid1 = build_design(fund, flow_list, [4], min_rows=20)

# pass 3: if still empty and we have only proxy, accept raw 4w diff without scaling guard
if (X1.empty or len(valid1) == 0) and ("Fed_NetLiq_proxy" in flow_list):
    d = fund["Fed_NetLiq_proxy"] - fund["Fed_NetLiq_proxy"].shift(4)
    X1 = pd.DataFrame({"Fed_NetLiq_proxy_d4w": d}, index=fund.index)
    valid1 = ["Fed_NetLiq_proxy_d4w"]

if X1.empty or len(valid1) == 0:
    st.error("Zero usable factors after differencing/scaling. Nudge start to >=2012 or ensure WALCL/RRP/TGA present.")
    st.stop()

# returns
rets = np.log(px).diff()

# effective window
eff_window = min(int(reg_window), len(widx))
if eff_window < 26:
    st.warning(f"Thin sample ({eff_window} weeks). Using all available rows; treat betas cautiously.")

# -------- per-asset regressions (per-asset overlap) --------
betas, tstats, r2s = {}, {}, {}
for asset in rets.columns:
    y = rets[asset]
    df = pd.concat([y, X1], axis=1).dropna()
    if df.empty:
        continue
    df = df.iloc[-eff_window:]
    # require at least 26 rows and >= factors*5 for minimal stability
    if df.shape[0] < max(26, (df.shape[1]-1) * 5):
        continue
    Y = df.iloc[:, 0].values.reshape(-1, 1)
    X = df.iloc[:, 1:]
    Xmat = np.column_stack([np.ones(len(df)), X.values])
    # OLS (pinv covariance)
    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta
    resid = (Y.flatten() - yhat)
    s2 = (resid @ resid) / max(1, (len(df) - Xmat.shape[1]))
    cov = s2 * np.linalg.pinv(Xmat.T @ Xmat)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    t = beta / se
    names = ["Intercept"] + list(X.columns)
    b = pd.Series(beta, index=names).drop("Intercept")
    tser = pd.Series(t, index=names).drop("Intercept")
    r2 = 1.0 - (resid @ resid) / (((Y - Y.mean()) ** 2).sum() + 1e-12)
    betas[asset], tstats[asset], r2s[asset] = b, tser, float(r2)

if not betas:
    st.error("No asset cleared the minimal overlap after fallbacks. Try start >= 2012 and 4w horizon.")
    st.stop()

B = pd.DataFrame(betas).T
T = pd.DataFrame(tstats).T
R2 = pd.Series(r2s, name="R2")

# liquidity columns present
liq_cols = [c for c in B.columns if any(k in c for k in ["Fed_NetLiq", "TGA_Drain", "FX_Reserves", "Fed_NetLiq_proxy"])]
liq_beta = B[liq_cols].sum(axis=1).rename("LiquidityBeta")

st.caption(f"Factors used: {liq_cols} • Assets: {B.shape[0]} • Per-asset window ≤ {eff_window} weeks")

# -------- heatmap --------
st.subheader("Liquidity beta heatmap (assets × flow shocks)")
hm = go.Figure(
    data=go.Heatmap(
        z=B[liq_cols].values,
        x=liq_cols,
        y=B.index,
        colorscale="RdBu",
        reversescale=True,
        zmid=0,
        colorbar=dict(title="Beta"),
    )
)
hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)

# -------- ranking --------
rank_df = pd.concat([liq_beta, R2], axis=1).sort_values("LiquidityBeta", ascending=False)
rank_df.index.name = "Asset"
st.subheader("Ranking by Liquidity Beta (higher = absorbs liquidity up)")
st.dataframe(rank_df.style.format({"LiquidityBeta": "{:.3f}", "R2": "{:.2%}"}), use_container_width=True)

# -------- detail --------
st.subheader("Detail: latest betas and recent flow shocks")
asset_pick = st.selectbox("Asset for details", list(B.index), index=0)
b_series = B.loc[asset_pick, liq_cols].sort_values(ascending=False)
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in b_series.values]
bar = go.Figure(data=[go.Bar(x=b_series.index, y=b_series.values, marker=dict(color=colors))])
bar.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10), yaxis_title="Beta")
st.plotly_chart(bar, use_container_width=True)

recent = X1.iloc[-26:][liq_cols]
line = go.Figure()
for c in recent.columns:
    line.add_trace(go.Scatter(x=recent.index, y=recent[c], name=c, mode="lines"))
line.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10))
line.update_yaxes(title_text="Scaled flow shock")
line.update_xaxes(dtick="M1", tickformat="%b %y", hoverformat="%b %d, %Y")
st.plotly_chart(line, use_container_width=True)

# -------- diagnostics --------
with st.expander("Diagnostics"):
    if not fred.empty:
        meta = []
        for c in fred.columns:
            s = fred[c].dropna()
            if not s.empty:
                meta.append([c, str(s.index.min().date()), str(s.index.max().date()), int(s.notna().sum())])
        if meta:
            diag = pd.DataFrame(meta, columns=["Series","First","Last","Obs"])
            st.dataframe(diag, use_container_width=True)
    st.write("Flow list selected:", flow_list)
    st.write("Design columns used:", liq_cols)

# -------- downloads --------
with st.expander("Download results"):
    st.download_button("Download betas (CSV)", B.to_csv().encode(), file_name="flow_matrix_betas.csv")
    st.download_button("Download t-stats (CSV)", T.to_csv().encode(), file_name="flow_matrix_tstats.csv")
    st.download_button("Download ranking (CSV)", rank_df.to_csv().encode(), file_name="flow_matrix_rank.csv")

# -------- notes --------
with st.expander("Methodology notes"):
    st.markdown(
        """
- **Fed_NetLiq** preference: WALCL − RRPONTSYD − TGA where TGA is WTREGEN (change) or WDTGAL (level). If TGA missing, we use WALCL − RRPONTSYD. If RRP missing, WALCL used as a proxy to keep the matrix alive.
- Deltas are **level changes** over 4/8 weeks; scaling via **MAD z-score** with safe fallbacks.
- Regressions are **per-asset aligned** to maximize rows; covariance uses pseudo-inverse for stability.
- **Liquidity Beta** sums betas to positive-liquidity shocks (sign-corrected).
"""
    )
