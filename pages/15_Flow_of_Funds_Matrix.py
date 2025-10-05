# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix (Druckenmiller) — allocator-ready UI
# Robust data pipeline + readable outputs (annualized betas, compass, quadrant map)

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# ---------------- FRED (optional via fredapi) ----------------
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
    s = pd.Series(dtype=float)
    if FRED_AVAILABLE:
        try:
            s = FRED.get_series(series_id)
            s.index = pd.to_datetime(s.index)
        except Exception:
            s = pd.Series(dtype=float)
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
    if s.empty: return s.reindex(widx)
    df = pd.DataFrame({"v": s.sort_index()})
    df = df.reindex(df.index.union(widx)).sort_index().ffill()
    return df.loc[widx, "v"]

def robust_scale(s: pd.Series):
    if s.isna().all(): return s
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
        if c not in fund_df.columns: continue
        for h in horizons:
            d = fund_df[c] - fund_df[c].shift(h)
            feats[f"{c}_d{h}w"] = robust_scale(d)
    if not feats: return pd.DataFrame(index=fund_df.index), []
    X = pd.DataFrame(feats, index=fund_df.index)
    valid = [c for c in X.columns if X[c].notna().sum() >= min_rows]
    return X[valid], valid

# ---------------- UI ----------------
st.set_page_config(page_title="Flow of Funds Matrix (Druckenmiller)", layout="wide")
st.title("Flow of Funds Matrix (Druckenmiller)")
st.caption("Identify where incremental liquidity and relative performance co-move. Regress weekly log returns on 4/8-week flow-of-funds shocks; annualized betas for intuition.")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")
    st.markdown("---")
    st.subheader("FRED IDs")
    st.caption("Preloaded: WALCL, RRPONTSYD, WTREGEN (weekly change), WDTGAL (level). Add CHN/JPN reserve IDs if you have them.")
    extra = st.text_area("Extra FRED series (name=ID per line)", height=110)

# Base FRED ids
fred_ids = {"WALCL":"WALCL", "RRPONTSYD":"RRPONTSYD", "WTREGEN":"WTREGEN", "WDTGAL":"WDTGAL"}
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=",1)]
            fred_ids[k] = v

# ---------------- Data ----------------
with st.spinner("Loading data"):
    px_d = load_prices(tickers, start=str(start_date))
    widx = weekly_index(start_date)
    px = px_d.resample("W-FRI").last().reindex(widx).ffill()

    fred_raw = {}
    for name, sid in fred_ids.items():
        s = fred_series(sid, start=str(start_date))
        if not s.empty:
            fred_raw[name] = align_to_weekly(s, widx)
    fred_df = pd.DataFrame(fred_raw, index=widx) if fred_raw else pd.DataFrame(index=widx)

if px.empty:
    st.error("No price data."); st.stop()

# Build flows gracefully
fund = pd.DataFrame(index=widx)
have = set(fred_df.columns)
tga = fred_df["WTREGEN"] if "WTREGEN" in have else (fred_df["WDTGAL"] if "WDTGAL" in have else None)

if "WALCL" in have and "RRPONTSYD" in have:
    fund["Fed_NetLiq"] = fred_df["WALCL"] - fred_df["RRPONTSYD"] - (tga if tga is not None else 0)
    if tga is not None:
        fund["TGA_Drain"] = -tga
elif "WALCL" in have:
    fund["Fed_NetLiq_proxy"] = fred_df["WALCL"]

fx_cols = [c for c in fred_df.columns if c.upper().startswith(("CHN","JPN")) or c.endswith("_RES")]
if fx_cols:
    fund["FX_Reserves"] = fred_df[fx_cols].sum(axis=1)

flow_priority = [
    ["Fed_NetLiq","TGA_Drain","FX_Reserves"],
    ["Fed_NetLiq","TGA_Drain"],
    ["Fed_NetLiq"],
    ["Fed_NetLiq_proxy"],
]
flow_list = next((ok for cand in flow_priority if (ok := [c for c in cand if c in fund.columns])), [])
if not flow_list:
    st.error("No usable flows (need WALCL/RRPONTSYD or at least WALCL proxy)."); st.stop()

# Design matrix (progressive fallbacks)
X, valid = build_design(fund, flow_list, horizons, min_rows=12)
if X.empty or len(valid)==0:
    X, valid = build_design(fund, flow_list, [4], min_rows=10)
if (X.empty or len(valid)==0) and ("Fed_NetLiq_proxy" in flow_list):
    d = fund["Fed_NetLiq_proxy"] - fund["Fed_NetLiq_proxy"].shift(4)
    X = pd.DataFrame({"Fed_NetLiq_proxy_d4w": robust_scale(d)}, index=fund.index); valid = ["Fed_NetLiq_proxy_d4w"]
if X.empty or len(valid)==0:
    st.error("No usable factors even after fallbacks. Try start ≥ 2013 and keep only Fed_NetLiq."); st.stop()

# Weekly log returns
rets = np.log(px).diff()
eff_window = min(int(reg_window), len(widx))
if eff_window < 20:
    st.warning(f"Thin sample ({eff_window} weeks). Using all available rows; interpret with caution.")

# Per-asset OLS (per-asset overlap), then annualize betas (×52)
betas, tstats, r2s = {}, {}, {}
min_rows = max(20, 5*max(1,len(valid)))

for asset in rets.columns:
    y = rets[asset]
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty: continue
    df = df.iloc[-eff_window:]
    if df.shape[0] < min_rows: continue
    Y = df.iloc[:,0].values.reshape(-1,1)
    X0 = df.iloc[:,1:]
    Xmat = np.column_stack([np.ones(len(df)), X0.values])
    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta
    resid = (Y.flatten() - yhat)
    s2 = (resid @ resid) / max(1,(len(df)-Xmat.shape[1]))
    cov = s2 * np.linalg.pinv(Xmat.T @ Xmat)
    se = np.sqrt(np.clip(np.diag(cov),1e-12,None))
    t = beta / se
    names = ["Intercept"] + list(X0.columns)
    b = pd.Series(beta, index=names).drop("Intercept") * 52.0  # annualize
    tser = pd.Series(t, index=names).drop("Intercept")
    r2 = 1.0 - (resid @ resid) / (((Y - Y.mean())**2).sum() + 1e-12)
    betas[asset], tstats[asset], r2s[asset] = b, tser, float(r2)

if not betas:
    st.error("No asset met minimal overlap. Try start ≥ 2013 and 4w horizon."); st.stop()

B = pd.DataFrame(betas).T      # assets × factors (ANNUALIZED betas)
T = pd.DataFrame(tstats).T
R2 = pd.Series(r2s, name="R2")

liq_cols = [c for c in B.columns if any(k in c for k in ["Fed_NetLiq","TGA_Drain","FX_Reserves","Fed_NetLiq_proxy"])]
Bv = B[liq_cols].copy()
liq_beta = Bv.sum(axis=1).rename("LiquidityBeta")

# ---------------- Top annotations ----------------
st.caption(
    f"Factors used: {list(Bv.columns)} • Assets: {Bv.shape[0]} • Per-asset window ≤ {eff_window} weeks"
)
pulse = float(liq_beta.mean()) if not liq_beta.empty else 0.0
st.markdown(f"**Liquidity Pulse (avg beta):** {pulse:+.2f}  → positive favors cyclicals/growth; negative favors defensives.")

# ---------------- Liquidity Compass (rank + tilt) ----------------
tilt = np.where(liq_beta > 0.10, "Overweight risk",
        np.where(liq_beta < -0.10, "Defensive / trim", "Neutral"))
compass = pd.DataFrame({"LiquidityBeta": liq_beta, "R2": R2.reindex(liq_beta.index), "Tilt": tilt})
compass = compass.sort_values("LiquidityBeta", ascending=False)
st.subheader("Liquidity Compass — where to tilt gross exposure")
st.dataframe(compass.style.format({"LiquidityBeta":"{:.2f}", "R2":"{:.2%}"}), use_container_width=True)

# ---------------- Heatmap (diverging, centered at 0) ----------------
st.subheader("Liquidity beta heatmap (annualized sensitivity)")
colorscale = [[0.0, "#1a9850"], [0.5, "#f7f7f7"], [1.0, "#d73027"]]  # green (risk-on) → red (risk-off)
hm = go.Figure(data=go.Heatmap(
    z=Bv.values, x=Bv.columns, y=Bv.index,
    colorscale=colorscale, zmid=0, colorbar=dict(title="Annualized β")
))
hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)

# ---------------- Quadrant Map (allocator map) ----------------
st.subheader("Allocator map — Fed_NetLiq vs TGA_Drain betas (size = R², color = net LiquidityBeta)")
xcol = next((c for c in Bv.columns if c.startswith("Fed_NetLiq")), None)
ycol = next((c for c in Bv.columns if c.startswith("TGA_Drain")), None)
if xcol is not None and ycol is not None:
    xs = Bv[xcol]; ys = Bv[ycol]; rsq = R2.reindex(Bv.index).fillna(0)
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=Bv.index,
        textposition="top center",
        marker=dict(size=10 + 60*rsq.clip(0,0.2), color=liq_beta, colorscale=colorscale, cmin=-abs(liq_beta).max(), cmax=abs(liq_beta).max(), showscale=True),
    ))
    scatter.update_xaxes(title=f"{xcol} (β, annualized)")
    scatter.update_yaxes(title=f"{ycol} (β, annualized)")
    scatter.update_layout(height=500, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(scatter, use_container_width=True)
else:
    st.info("Need both Fed_NetLiq_* and TGA_Drain_* factors for the quadrant map.")

# ---------------- Liquidity Pulse (recent) ----------------
st.subheader("Liquidity Pulse — recent net momentum (standardized)")
# Build a single composite flow shock (sum of available positive-liquidity shocks)
pos_cols = [c for c in X.columns if any(k in c for k in ["Fed_NetLiq","FX_Reserves","TGA_Drain","Fed_NetLiq_proxy"])]
pulse_series = X[pos_cols].sum(axis=1).dropna()
pulse_recent = pulse_series.iloc[-26:]
line = go.Figure()
line.add_trace(go.Scatter(x=pulse_recent.index, y=pulse_recent, mode="lines", name="Net liquidity momentum"))
line.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
line.update_yaxes(title_text="Σ standardized flow shocks")
line.update_xaxes(dtick="M1", tickformat="%b %y", hoverformat="%b %d, %Y")
st.plotly_chart(line, use_container_width=True)

# ---------------- Downloads ----------------
with st.expander("Download results"):
    st.download_button("Betas (annualized) CSV", B.to_csv().encode(), file_name="flow_betas_annualized.csv")
    st.download_button("Liquidity Compass CSV", compass.to_csv().encode(), file_name="flow_liquidity_compass.csv")

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    st.write("Design columns used:", list(X.columns))
    if 'fred_df' in locals() and not fred_df.empty:
        meta = []
        for c in fred_df.columns:
            s = fred_df[c].dropna()
            if not s.empty:
                meta.append([c, str(s.index.min().date()), str(s.index.max().date()), int(s.notna().sum())])
        if meta:
            st.dataframe(pd.DataFrame(meta, columns=["Series","First","Last","Obs"]), use_container_width=True)

# ---------------- Notes ----------------
with st.expander("Methodology notes"):
    st.markdown(
        """
- Betas are **annualized** (weekly × 52) so magnitudes are intuitive. Example: β=+0.15 ⇒ +1σ liquidity = +15% annualized relative impact.
- **LiquidityBeta** = sum of annualized betas to liquidity-positive shocks (Fed_NetLiq, FX_Reserves, TGA_Drain already sign-corrected).
- Quadrant map shows where assets sit on **Fed_NetLiq (x)** vs **TGA_Drain (y)** axes; bubble size = R² (fit quality).
- Liquidity Pulse is a **single composite** of recent standardized shocks (last ~26w) to track regime turns.
"""
    )
