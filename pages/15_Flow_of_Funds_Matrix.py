# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix (Druckenmiller)
# Weekly-only. FRED -> fredapi if available, else pandas_datareader("fred").
# Regress weekly returns of major assets on 4/8 week flow-of-funds deltas.
# Output: heatmap of betas, ranked table, and quick visuals.

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    px = yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.ffill()
    if freq:
        px = px.resample(freq).last()
    return px.dropna(how="all")

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_fred(series_map: dict, start="2005-01-01", freq="W-FRI"):
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
    df = df.loc[pd.to_datetime(start):]
    if freq:
        df = df.resample(freq).ffill()
    else:
        daily = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="B")
        df = df.reindex(daily).ffill()
    return df

def zscore(s: pd.Series):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def ols_beta(y: pd.Series, X: pd.DataFrame):
    # returns betas, t-stats, R^2 using last valid window
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < max(40, X.shape[1] * 8):
        return None, None, np.nan
    Y = df.iloc[:, 0].values.reshape(-1, 1)
    Xmat = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].values])
    # OLS (no regularization)
    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta
    resid = (Y.flatten() - yhat)
    s2 = (resid @ resid) / (len(df) - Xmat.shape[1])
    cov = s2 * np.linalg.inv(Xmat.T @ Xmat)
    se = np.sqrt(np.diag(cov))
    tstats = beta / se
    r2 = 1.0 - (resid @ resid) / (((Y - Y.mean()) ** 2).sum())
    # drop intercept from outputs
    names = ["Intercept"] + list(df.columns[1:])
    betas = pd.Series(beta, index=names).drop("Intercept")
    tser = pd.Series(tstats, index=names).drop("Intercept")
    return betas, tser, float(r2)

# ---------------- UI ----------------
st.set_page_config(page_title="Flow of Funds Matrix (Druckenmiller)", layout="wide")
st.title("Flow of Funds Matrix (Druckenmiller)")
st.caption("Identify where incremental liquidity and relative performance co-move. Weekly regressions of asset returns on flow-of-funds shocks.")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Purpose:** Macro allocator's dashboard for where liquidity is flowing.

**Flows used (weekly):**
- **Fed Net Liquidity:** WALCL − RRPONTSYD − WTREGEN  
- **FX Reserves (CHN, JPN):** summed USD changes (optional if available)  
- **Treasury cash (TGA proxy):** WTREGEN (sign flipped = drain)  

We build **4w and 8w deltas** of each, standardize, and regress asset returns on these shocks.
**Liquidity beta** = sum of positive-flow betas (sign-corrected). Higher = more elastic to liquidity.

"""
    )
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2005-01-01"))
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    use_flows = st.multiselect(
        "Flows to include",
        ["Fed_NetLiq", "FX_Reserves", "TGA_Drain"],
        default=["Fed_NetLiq", "FX_Reserves", "TGA_Drain"],
    )
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    st.markdown("---")
    st.subheader("Assets")
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    assets = st.text_input("Tickers (comma separated)", ",".join(default_assets)).replace(" ", "").split(",")
    st.markdown("---")
    st.subheader("FRED series")
    st.caption("Defaults include WALCL, RRPONTSYD, WTREGEN; add FX reserves if you know the IDs.")
    st.code("WALCL  RRPONTSYD  WTREGEN  # (preloaded)\n# Add e.g.\n# CHN_RES = ?????\n# JPN_RES = ?????", language="text")
    extra = st.text_area("Extra FRED series (name=ID per line)", value="", height=120)

# Base FRED map
fred_map = {
    "WALCL": "WALCL",
    "RRPONTSYD": "RRPONTSYD",
    "WTREGEN": "WTREGEN",  # TGA proxy (weekly)
}
# Parse user-provided FRED ids
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            fred_map[k] = v

# ---------------- Data ----------------
with st.spinner("Loading data"):
    px = load_prices(assets, start=str(start_date), freq="W-FRI")
    fred = load_fred(fred_map, start=str(start_date), freq="W-FRI")

if px.empty:
    st.error("No price data loaded.")
    st.stop()

fund = pd.DataFrame(index=px.index)
if not fred.empty:
    fred = fred.reindex(px.index).ffill()
    # Core flows
    if {"WALCL", "RRPONTSYD", "WTREGEN"}.issubset(fred.columns):
        fund["Fed_NetLiq"] = fred["WALCL"] - fred["RRPONTSYD"] - fred["WTREGEN"]
        # TGA drain (higher WTREGEN = less market liquidity) -> use negative sign so "more drain" is negative liquidity
        fund["TGA_Drain"] = -fred["WTREGEN"]
    # Optional FX reserves (user must provide series like CHN_RES, JPN_RES)
    fx_cols = [c for c in fred.columns if c.upper().startswith(("CHN", "JPN")) or c.endswith("_RES")]
    if fx_cols:
        fund["FX_Reserves"] = fred[fx_cols].sum(axis=1)

# Only keep selected flows
flow_cols = [c for c in ["Fed_NetLiq","FX_Reserves","TGA_Drain"] if (c in fund.columns and c in use_flows)]
if not flow_cols:
    st.error("No flow series available. Add FX reserve IDs or include Fed/TGA.")
    st.stop()

# Build flow deltas (4w/8w), standardized
flows = {}
for c in flow_cols:
    for h in horizons:
        d = fund[c] - fund[c].shift(h)
        flows[f"{c}_d{h}w"] = zscore(d)

X = pd.DataFrame(flows).dropna(how="all")
# Asset returns (weekly)
rets = np.log(px).diff()
rets = rets.reindex(X.index).dropna(how="all")

# Align for rolling regression window
def last_window(df, n):
    return df.iloc[-n:] if len(df) >= n else df

Xw = last_window(X, int(reg_window))
retsw = last_window(rets, int(reg_window))
common_idx = Xw.index.intersection(retsw.index)
Xw = Xw.loc[common_idx]
retsw = retsw.loc[common_idx]

# ---------------- Regressions ----------------
betas = {}
tstats = {}
r2s = {}

for col in retsw.columns:
    b, t, r2 = ols_beta(retsw[col], Xw)
    if b is None:
        continue
    betas[col] = b
    tstats[col] = t
    r2s[col] = r2

if not betas:
    st.warning("Insufficient overlap for regression window. Try a shorter window or later start date.")
    st.stop()

B = pd.DataFrame(betas).T  # assets x factors
T = pd.DataFrame(tstats).T
R2 = pd.Series(r2s, name="R2")

# Define "liquidity beta" as sum of betas to positive-liquidity shocks only
# We treat: Fed_NetLiq_d* > 0 as positive liquidity; FX_Reserves_d* > 0 positive; TGA_Drain_d* > 0 positive (since we flipped sign)
liquidity_factor_cols = [c for c in B.columns if any(key in c for key in ["Fed_NetLiq", "FX_Reserves", "TGA_Drain"])]
liq_beta = B[liquidity_factor_cols].sum(axis=1).rename("LiquidityBeta")

# ---------------- UI: Heatmap + Ranking ----------------
st.subheader("Liquidity beta heatmap (assets × flow shocks)")
hm = go.Figure(
    data=go.Heatmap(
        z=B[liquidity_factor_cols].values,
        x=liquidity_factor_cols,
        y=B.index,
        colorscale="RdBu",
        reversescale=True,
        zmid=0,
        colorbar=dict(title="Beta"),
    )
)
hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)

# Ranked table
rank_df = pd.concat([liq_beta, R2], axis=1).sort_values("LiquidityBeta", ascending=False)
rank_df.index.name = "Asset"

st.subheader("Ranking by Liquidity Beta (higher = more elastic to liquidity up)")
st.dataframe(rank_df.style.format({"LiquidityBeta": "{:.3f}", "R2": "{:.2%}"}), use_container_width=True)

# ---------------- Quick Visuals ----------------
st.subheader("Detail: latest betas and recent flows")

# Bar of an asset's betas
asset_pick = st.selectbox("Asset for details", list(B.index), index=0)
b_series = B.loc[asset_pick, liquidity_factor_cols].sort_values(ascending=False)
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in b_series.values]
bar = go.Figure(data=[go.Bar(x=b_series.index, y=b_series.values, marker=dict(color=colors))])
bar.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10), yaxis_title="Beta")
st.plotly_chart(bar, use_container_width=True)

# Recent standardized flow shocks (last ~26w)
recent = X.iloc[-26:][liquidity_factor_cols]
line = go.Figure()
for c in recent.columns:
    line.add_trace(go.Scatter(x=recent.index, y=recent[c], name=c, mode="lines"))
line.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10))
line.update_yaxes(title_text="Z-scored flow shock")
line.update_xaxes(dtick="M1", tickformat="%b %y", hoverformat="%b %d, %Y")
st.plotly_chart(line, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download results"):
    out = {
        "betas": B,
        "tstats": T,
        "r2": R2,
        "rank": rank_df,
        "flows_design_matrix": X,
        "returns": rets,
    }
    # package to CSVs in a zip-like multi-file? Here: provide individual downloads
    st.download_button("Download betas (CSV)", B.to_csv().encode(), file_name="flow_matrix_betas.csv")
    st.download_button("Download t-stats (CSV)", T.to_csv().encode(), file_name="flow_matrix_tstats.csv")
    st.download_button("Download ranking (CSV)", rank_df.to_csv().encode(), file_name="flow_matrix_rank.csv")

# ---------------- Notes ----------------
with st.expander("Methodology notes"):
    st.markdown(
        """
- **Net Liquidity** = WALCL − RRPONTSYD − WTREGEN. We flip WTREGEN sign for the **TGA_Drain** factor so higher values mean **less** liquidity.
- We build **4w and 8w level deltas** of each flow series, then **z-score** them to comparable units.
- Weekly asset **log returns** are regressed on these shocks over the last configurable window (default 156w).
- **Liquidity Beta** = sum of betas to positive-liquidity shocks (Fed_NetLiq, FX_Reserves, TGA_Drain already sign-corrected).  
  Higher beta ⇒ asset tends to outperform when liquidity rises (Druckenmiller tilt).
- Provide CHN/JPN FX reserve FRED IDs in the sidebar to light up the FX factor fully.
"""
    )
