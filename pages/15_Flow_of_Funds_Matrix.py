# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix (Druckenmiller) — robust overlap & fallbacks
# Weekly-only. FRED: fredapi if available else pandas_datareader("fred").

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# ---------------- Optional fredapi ----------------
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
def load_prices(tickers, start="2005-01-01", freq="W-FRI"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.ffill()
    if freq:
        px = px.resample(freq).last()
    return px.dropna(how="all")

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_fred(series_map: dict, start="2005-01-01"):
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

def zscore_safe(s: pd.Series):
    if s.isna().all():
        return s
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        # fall back to mean-centering (keeps sign info, avoids all-NaN)
        mu = s.mean()
        return (s - mu)
    return (s - s.mean()) / (std + 1e-9)

def build_weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    idx = pd.date_range(pd.to_datetime(start), end, freq="W-FRI")
    return idx

def design_from_flows(fund_df: pd.DataFrame, flows: list[str], horizons: list[int], min_rows=30):
    feats = {}
    for c in flows:
        if c not in fund_df.columns:
            continue
        for h in horizons:
            d = fund_df[c] - fund_df[c].shift(h)
            feats[f"{c}_d{h}w"] = zscore_safe(d)
    X = pd.DataFrame(feats, index=fund_df.index)
    # prune thin factors
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
• **Fed Net Liquidity:** WALCL − RRPONTSYD − TGA  
• **TGA (drain):** WDTGAL level (or WTREGEN if provided), sign-flipped  
• **FX Reserves:** sum of CHN/JPN reserve series if supplied

We build **4w/8w deltas** of flows, standardize robustly, and regress **weekly log returns**.  
**Liquidity Beta** = sum of betas to positive-liquidity shocks (sign-corrected).
"""
    )
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2010-01-01"))  # safer default vs RRP start in 2013
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    use_flows = st.multiselect(
        "Flows to include",
        ["Fed_NetLiq", "FX_Reserves", "TGA_Drain"],
        default=["Fed_NetLiq", "TGA_Drain", "FX_Reserves"],
    )
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    st.markdown("---")
    st.subheader("Assets")
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    assets = st.text_input("Tickers (comma separated)", ",".join(default_assets)).replace(" ", "").split(",")
    st.markdown("---")
    st.subheader("FRED series")
    st.caption("Defaults include WALCL, RRPONTSYD, WDTGAL (TGA level). Add FX reserves if you know the IDs.")
    st.code("WALCL  RRPONTSYD  WDTGAL  # (preloaded)\n# Optional:\n# CHN_RES = ...\n# JPN_RES = ...\n# WTREGEN = ...  (weekly TGA change, if you prefer)", language="text")
    extra = st.text_area("Extra FRED series (name=ID per line)", value="", height=120)

# Base FRED map: WALCL, RRP, TGA level by default; user can add WTREGEN/FX reserves
fred_map = {"WALCL": "WALCL", "RRPONTSYD": "RRPONTSYD", "WDTGAL": "WDTGAL"}
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            fred_map[k] = v

# ---------------- Data ----------------
with st.spinner("Loading data"):
    px_w = load_prices(assets, start=str(start_date), freq=None)  # daily
    # weekly target index
    widx = build_weekly_index(start=start_date)
    # resample prices to weekly on target index
    px = px_w.resample("W-FRI").last().reindex(widx).ffill()
    fred_raw = load_fred(fred_map, start=str(start_date))

if px.empty:
    st.error("No price data loaded.")
    st.stop()

fund = pd.DataFrame(index=widx)
if fred_raw.empty:
    st.warning("No FRED data loaded — proceeding with NetLiq fallback if possible.")
else:
    fred = fred_raw.reindex(widx).ffill()
    have = set(fred.columns)

    # TGA preference: WTREGEN (if provided) else WDTGAL level
    tga = None
    if "WTREGEN" in have:
        tga = fred["WTREGEN"]
    elif "WDTGAL" in have:
        tga = fred["WDTGAL"]

    # Build Fed_NetLiq (graceful if TGA missing)
    if {"WALCL", "RRPONTSYD"}.issubset(have):
        if tga is not None:
            fund["Fed_NetLiq"] = fred["WALCL"] - fred["RRPONTSYD"] - tga
            fund["TGA_Drain"] = -tga
        else:
            # degrade gracefully: TGA assumed ~0 (documented)
            fund["Fed_NetLiq"] = fred["WALCL"] - fred["RRPONTSYD"]
    # FX reserves: auto-sum any *_RES or CHN*/JPN* entries
    fx_cols = [c for c in fred.columns if c.upper().startswith(("CHN", "JPN")) or c.endswith("_RES")]
    if fx_cols:
        fund["FX_Reserves"] = fred[fx_cols].sum(axis=1)

# Keep only requested & available flows
flow_cols = [c for c in ["Fed_NetLiq","FX_Reserves","TGA_Drain"] if (c in fund.columns and c in use_flows)]
if not flow_cols:
    st.error("No usable flow series. Ensure WALCL/RRPONTSYD loaded and keep Fed_NetLiq selected.")
    st.stop()

# ---------------- Design matrix (progressive fallbacks) ----------------
# Pass 1: requested horizons (4 & 8) with stricter min rows
X1, valid1 = design_from_flows(fund, flow_cols, horizons, min_rows=40)

# If empty or too few columns, fallback to 4w only, lower min rows
if X1.empty or len(valid1) == 0:
    X1, valid1 = design_from_flows(fund, flow_cols, [4], min_rows=26)

# If still empty, Fed_NetLiq only (4w), very lax min rows
if (X1.empty or len(valid1) == 0) and "Fed_NetLiq" in flow_cols:
    X1, valid1 = design_from_flows(fund, ["Fed_NetLiq"], [4], min_rows=20)

if X1.empty or len(valid1) == 0:
    st.error("Zero usable factors after differencing/z-scoring. Try later start date (>=2010) and select Fed_NetLiq.")
    st.stop()

# Returns (weekly log), aligned to widx
rets = np.log(px).diff()

# Effective window limited by intersection length
avail = len(widx)
eff_window = min(int(reg_window), avail)
if eff_window < 30:
    st.warning(f"Thin sample ({eff_window} weeks). Using all available rows; interpret betas cautiously.")

# ---------------- Per-asset regression with per-asset alignment ----------------
betas, tstats, r2s = {}, {}, {}
for asset in rets.columns:
    y = rets[asset]
    df = pd.concat([y, X1], axis=1).dropna()
    if df.empty:
        continue
    # use last eff_window rows for this asset specifically
    df = df.iloc[-eff_window:]
    if df.shape[0] < max(30, max(1, df.shape[1]-1) * 6):  # rows vs factors
        continue
    Y = df.iloc[:, 0].values.reshape(-1, 1)
    X = df.iloc[:, 1:]
    Xmat = np.column_stack([np.ones(len(df)), X.values])
    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta
    resid = (Y.flatten() - yhat)
    # robust covariance via pinv
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
    st.error("No asset had sufficient overlap even after fallbacks. Nudge start date later (>=2012) and keep only Fed_NetLiq with 4w horizon.")
    st.stop()

B = pd.DataFrame(betas).T  # assets x factors
T = pd.DataFrame(tstats).T
R2 = pd.Series(r2s, name="R2")

# Liquidity beta = sum of betas to liquidity-positive shocks
liq_cols = [c for c in B.columns if any(k in c for k in ["Fed_NetLiq", "FX_Reserves", "TGA_Drain"])]
liq_beta = B[liq_cols].sum(axis=1).rename("LiquidityBeta")

st.caption(f"Factors in model: {len(liq_cols)} • Assets regressed: {B.shape[0]} • Window (per-asset) ≤ {eff_window} weeks")

# ---------------- Heatmap ----------------
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

recent = X1.iloc[-26:][liq_cols]
line = go.Figure()
for c in recent.columns:
    line.add_trace(go.Scatter(x=recent.index, y=recent[c], name=c, mode="lines"))
line.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10))
line.update_yaxes(title_text="Z-scored flow shock")
line.update_xaxes(dtick="M1", tickformat="%b %y", hoverformat="%b %d, %Y")
st.plotly_chart(line, use_container_width=True)

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    st.write("Flow columns requested:", use_flows)
    st.write("Flow columns available:", flow_cols)
    st.write("Factors used (post-prune):", liq_cols)
    if fred_raw is not None and not fred_raw.empty:
        meta = []
        for c in fred_raw.columns:
            s = fred_raw[c].dropna()
            if not s.empty:
                meta.append([c, str(s.index.min().date()), str(s.index.max().date()), int(s.notna().sum())])
        if meta:
            diag = pd.DataFrame(meta, columns=["Series","First","Last","Obs"])
            st.dataframe(diag, use_container_width=True)

# ---------------- Downloads ----------------
with st.expander("Download results"):
    st.download_button("Download betas (CSV)", B.to_csv().encode(), file_name="flow_matrix_betas.csv")
    st.download_button("Download t-stats (CSV)", T.to_csv().encode(), file_name="flow_matrix_tstats.csv")
    st.download_button("Download ranking (CSV)", rank_df.to_csv().encode(), file_name="flow_matrix_rank.csv")

# ---------------- Notes ----------------
with st.expander("Methodology notes"):
    st.markdown(
        """
- **Net Liquidity** = WALCL − RRPONTSYD − TGA (TGA via WDTGAL level by default; WTREGEN if provided). If TGA missing, we compute Fed_NetLiq without it (documented).
- Deltas are **level changes** over 4/8 weeks; z-scoring is robust (falls back to mean-centering if variance collapses).
- Regressions are **per-asset aligned** to maximize usable rows; covariance uses pseudo-inverse for stability.
- **Liquidity Beta** = sum of betas to liquidity-positive shocks (Fed_NetLiq, FX_Reserves, TGA_Drain already sign-corrected).
"""
    )
