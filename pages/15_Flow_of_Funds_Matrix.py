# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix (Druckenmiller) — bulletproof overlap & alignment
# Weekly-only. FRED via fredapi if present; else pandas_datareader('fred').

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# ---------------- FRED (optional fredapi) ----------------
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

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_prices(tickers, start="2012-01-01"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.ffill()

@st.cache_data(show_spinner=False, ttl=24*3600)
def fred_series(series_id: str, start="2012-01-01") -> pd.Series:
    """Fetch a single FRED series robustly."""
    s = pd.Series(dtype=float)
    # fredapi first
    if FRED_AVAILABLE:
        try:
            s = FRED.get_series(series_id)
            s.index = pd.to_datetime(s.index)
        except Exception:
            s = pd.Series(dtype=float)
    # pandas_datareader fallback
    if s.empty:
        try:
            s = pdr.DataReader(series_id, "fred", start)[series_id]
        except Exception:
            s = pd.Series(dtype=float)
    return s.dropna()

def weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="W-FRI")

def align_to_weekly(s: pd.Series, widx: pd.DatetimeIndex) -> pd.Series:
    """Align a (daily/weekly) series to weekly-FRI using nearest within 7 days."""
    if s.empty:
        return s.reindex(widx)
    s = s.sort_index()
    # Reindex to union for asof-like nearest within tolerance
    df = pd.DataFrame({"v": s})
    df = df.reindex(df.index.union(widx)).sort_index().ffill()
    out = df.loc[widx, "v"]
    # If distance >7D at either end, those will be forward fills; drop if still NaN
    return out

def robust_scale(s: pd.Series):
    """MAD z-score -> std z-score -> mean-center -> identity."""
    if s.isna().all():
        return s
    med = s.median()
    mad = (s - med).abs().median()
    if np.isfinite(mad) and mad > 0:
        return (s - med) / (1.4826 * mad)
    std = s.std(ddof=0)
    if np.isfinite(std) and std > 0:
        return (s - s.mean()) / (std + 1e-9)
    mu = s.mean()
    return (s - mu) if np.isfinite(mu) else s

def build_design(fund_df: pd.DataFrame, flows, horizons, min_rows=12):
    feats = {}
    for c in flows:
        if c not in fund_df.columns:
            continue
        for h in horizons:
            d = fund_df[c] - fund_df[c].shift(h)
            feats[f"{c}_d{h}w"] = robust_scale(d)
    if not feats:
        return pd.DataFrame(index=fund_df.index), []
    X = pd.DataFrame(feats, index=fund_df.index)
    valid = [c for c in X.columns if X[c].notna().sum() >= min_rows]
    return X[valid], valid

# ---------------- UI ----------------
st.set_page_config(page_title="Flow of Funds Matrix (Druckenmiller)", layout="wide")
st.title("Flow of Funds Matrix (Druckenmiller)")
st.caption("Identify where incremental liquidity and relative performance co-move. Weekly regressions of asset returns on flow-of-funds shocks.")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Flows (weekly):**  
• **Fed Net Liquidity:** WALCL − RRPONTSYD − TGA (WTREGEN preferred, else WDTGAL level)  
• **TGA_Drain:** −TGA (sign flipped so higher = tighter)  
• **FX Reserves:** sum of CHN/JPN series if provided

We compute **4/8-week deltas**, robust-scale, regress **weekly log returns**.  
**Liquidity Beta** = sum of betas to liquidity-positive shocks.
"""
    )
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))  # RRP available from 2013
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")
    st.markdown("---")
    st.subheader("FRED (optional additions)")
    st.caption("Base fetch: WALCL, RRPONTSYD, WTREGEN, WDTGAL. Add CHN/JPN reserve IDs to light up FX_Reserves.")
    extra = st.text_area("Extra FRED series (name=ID per line)", value="", height=100)

# Base FRED ids we’ll attempt
fred_ids = {
    "WALCL": "WALCL",
    "RRPONTSYD": "RRPONTSYD",
    "WTREGEN": "WTREGEN",  # weekly TGA change
    "WDTGAL": "WDTGAL",    # TGA level
}
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            fred_ids[k] = v

# ---------------- Data ----------------
with st.spinner("Loading prices & flows"):
    px_daily = load_prices(tickers, start=str(start_date))
    widx = weekly_index(start_date)
    px = px_daily.resample("W-FRI").last().reindex(widx).ffill()

    # Fetch each series individually, align nearest-to-weekly
    fred_raw = {}
    for name, sid in fred_ids.items():
        s = fred_series(sid, start=str(start_date))
        if not s.empty:
            fred_raw[name] = align_to_weekly(s, widx)
    fred_df = pd.DataFrame(fred_raw, index=widx) if fred_raw else pd.DataFrame(index=widx)

if px.empty:
    st.error("No price data.")
    st.stop()

# Build flows with graceful fallbacks
fund = pd.DataFrame(index=widx)
have = set(fred_df.columns)

# TGA: prefer WTREGEN, else WDTGAL
tga = fred_df["WTREGEN"] if "WTREGEN" in have else (fred_df["WDTGAL"] if "WDTGAL" in have else None)

if "WALCL" in have and "RRPONTSYD" in have:
    if tga is not None:
        fund["Fed_NetLiq"] = fred_df["WALCL"] - fred_df["RRPONTSYD"] - tga
        fund["TGA_Drain"] = -tga
    else:
        fund["Fed_NetLiq"] = fred_df["WALCL"] - fred_df["RRPONTSYD"]
elif "WALCL" in have:
    fund["Fed_NetLiq_proxy"] = fred_df["WALCL"]

# FX Reserves auto-aggregate if provided by user
fx_cols = [c for c in fred_df.columns if c.upper().startswith(("CHN", "JPN")) or c.endswith("_RES")]
if fx_cols:
    fund["FX_Reserves"] = fred_df[fx_cols].sum(axis=1)

# Pick best available flow set
flow_priority = [
    ["Fed_NetLiq", "TGA_Drain", "FX_Reserves"],
    ["Fed_NetLiq", "TGA_Drain"],
    ["Fed_NetLiq"],
    ["Fed_NetLiq_proxy"],
]
flow_list = next((ok for cand in flow_priority if (ok := [c for c in cand if c in fund.columns])), [])

if not flow_list:
    st.error("No usable flows (WALCL/RRP/TGA missing and no proxy). Ensure at least WALCL+RRPONTSYD.")
    st.stop()

# ---------------- Design matrix (progressive) ----------------
X, valid = build_design(fund, flow_list, horizons, min_rows=12)
if X.empty or len(valid) == 0:
    X, valid = build_design(fund, flow_list, [4], min_rows=10)
if (X.empty or len(valid) == 0) and ("Fed_NetLiq_proxy" in flow_list):
    d = fund["Fed_NetLiq_proxy"] - fund["Fed_NetLiq_proxy"].shift(4)
    X = pd.DataFrame({"Fed_NetLiq_proxy_d4w": d}, index=fund.index)
    valid = ["Fed_NetLiq_proxy_d4w"]

if X.empty or len(valid) == 0:
    st.error("Still zero usable factors after fallbacks. Try start >= 2013 and keep only Fed_NetLiq.")
    st.stop()

# Weekly log returns
rets = np.log(px).diff()

# Effective max window
eff_window = min(int(reg_window), len(widx))
if eff_window < 20:
    st.warning(f"Thin sample ({eff_window} weeks). Using all available rows; interpret with caution.")

# ---------------- Per-asset regressions ----------------
betas, tstats, r2s = {}, {}, {}
min_rows = max(20, 5 * max(1, len(valid)))  # minimal stability threshold

for asset in rets.columns:
    y = rets[asset]
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty:
        continue
    df = df.iloc[-eff_window:]
    if df.shape[0] < min_rows:
        continue
    Y = df.iloc[:, 0].values.reshape(-1, 1)
    X0 = df.iloc[:, 1:]
    Xmat = np.column_stack([np.ones(len(df)), X0.values])
    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta
    resid = (Y.flatten() - yhat)
    s2 = (resid @ resid) / max(1, (len(df) - Xmat.shape[1]))
    cov = s2 * np.linalg.pinv(Xmat.T @ Xmat)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    t = beta / se
    names = ["Intercept"] + list(X0.columns)
    betas[asset] = pd.Series(beta, index=names).drop("Intercept")
    tstats[asset] = pd.Series(t, index=names).drop("Intercept")
    r2s[asset] = float(1.0 - (resid @ resid) / (((Y - Y.mean()) ** 2).sum() + 1e-12))

if not betas:
    st.error("No asset met minimal overlap. Nudge start ≥ 2013 and horizon=4w.")
    st.stop()

B = pd.DataFrame(betas).T
T = pd.DataFrame(tstats).T
R2 = pd.Series(r2s, name="R2")

liq_cols = [c for c in B.columns if any(k in c for k in ["Fed_NetLiq", "TGA_Drain", "FX_Reserves", "Fed_NetLiq_proxy"])]
liq_beta = B[liq_cols].sum(axis=1).rename("LiquidityBeta")

st.caption(f"Factors used: {liq_cols} • Assets: {B.shape[0]} • Per-asset window ≤ {eff_window} weeks")

# ---------------- Heatmap ----------------
st.subheader("Liquidity beta heatmap (assets × flow shocks)")
hm = go.Figure(data=go.Heatmap(
    z=B[liq_cols].values, x=liq_cols, y=B.index,
    colorscale="RdBu", reversescale=True, zmid=0,
    colorbar=dict(title="Beta"),
))
hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)

# ---------------- Ranking ----------------
rank_df = pd.concat([liq_beta, R2], axis=1).sort_values("LiquidityBeta", ascending=False)
rank_df.index.name = "Asset"
st.subheader("Ranking by Liquidity Beta (higher = absorbs liquidity up)")
st.dataframe(rank_df.style.format({"LiquidityBeta": "{:.3f}", "R2": "{:.2%}"}), use_container_width=True)

# ---------------- Detail ----------------
st.subheader("Detail: latest betas and recent flow shocks")
asset_pick = st.selectbox("Asset for details", list(B.index), index=0)
b_series = B.loc[asset_pick, liq_cols].sort_values(ascending=False)
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in b_series.values]
bar = go.Figure(data=[go.Bar(x=b_series.index, y=b_series.values, marker=dict(color=colors))])
bar.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10), yaxis_title="Beta")
st.plotly_chart(bar, use_container_width=True)

recent = X.iloc[-26:][liq_cols] if not X.empty else pd.DataFrame()
line = go.Figure()
for c in recent.columns:
    line.add_trace(go.Scatter(x=recent.index, y=recent[c], name=c, mode="lines"))
line.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10))
line.update_yaxes(title_text="Scaled flow shock")
line.update_xaxes(dtick="M1", tickformat="%b %y", hoverformat="%b %d, %Y")
st.plotly_chart(line, use_container_width=True)

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    if not fred_df.empty:
        meta = []
        for c in fred_df.columns:
            s = fred_df[c].dropna()
            if not s.empty:
                meta.append([c, str(s.index.min().date()), str(s.index.max().date()), int(s.notna().sum())])
        if meta:
            diag = pd.DataFrame(meta, columns=["Series","First","Last","Obs"])
            st.dataframe(diag, use_container_width=True)
    st.write("Flow list selected:", flow_list)
    st.write("Design columns used:", valid)

# ---------------- Downloads ----------------
with st.expander("Download results"):
    st.download_button("Download betas (CSV)", B.to_csv().encode(), file_name="flow_matrix_betas.csv")
    st.download_button("Download t-stats (CSV)", T.to_csv().encode(), file_name="flow_matrix_tstats.csv")
    st.download_button("Download ranking (CSV)", rank_df.to_csv().encode(), file_name="flow_matrix_rank.csv")
