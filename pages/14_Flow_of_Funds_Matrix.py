# 14_Flow_of_Funds_Compass.py
# Compact flow-of-funds model with:
# - Liquidity Composite L(t) from core FRED flows
# - Features: Level Z = Lz, Slope Z = ΔLz over a selectable horizon
# - OLS on asset returns with [Lz, ΔLz]
# - Regime metrics, OW/UW, pairs, contribution bars, sensitivity compass
# - VIX term structure overlay (VIX3M/VIX) on the regime chart
# - Resolution toggle: Weekly or Daily
# - 20-asset default universe
# NOTE: No pandas_datareader. FRED pulled via CSV endpoints.

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ────────────────────────────── Page ──────────────────────────────
st.set_page_config(page_title="Flow of Funds Compass", layout="wide")
st.title("Flow of Funds Compass")

# ────────────────────────────── Utils ─────────────────────────────
@st.cache_data(ttl=6*3600, show_spinner=False)
def load_prices(tickers, start="2012-01-01", interval="1d"):
    try:
        px = yf.download(tickers, start=start, auto_adjust=True, progress=False, interval=interval)["Close"]
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()
    if isinstance(px, pd.Series):
        px = px.to_frame()
    bad = [c for c in px.columns if px[c].dropna().empty]
    if bad:
        st.warning(f"No data for: {', '.join(map(str, bad))}. Excluding.")
        px = px.drop(columns=bad)
    return px.ffill()

@st.cache_data(ttl=24*3600, show_spinner=False)
def fred_series(series_id: str, start="2012-01-01") -> pd.Series:
    """Fetch a FRED series without pandas_datareader using the public CSV endpoints."""
    # Primary endpoint
    url1 = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    # Reliable fallback
    url2 = f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv"
    for url in (url1, url2):
        try:
            df = pd.read_csv(url, parse_dates=["DATE"])
            # Standard FRED CSV has DATE and either VALUE or the series_id as column name
            value_col = "VALUE" if "VALUE" in df.columns else series_id if series_id in df.columns else None
            if value_col is None:
                continue
            s = pd.to_numeric(df[value_col], errors="coerce")
            out = pd.Series(s.values, index=pd.to_datetime(df["DATE"]), name=series_id).dropna()
            if start:
                out = out[out.index >= pd.to_datetime(start)]
            return out
        except Exception:
            continue
    return pd.Series(dtype=float)

def weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="W-FRI")

def daily_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="B")

def align_to_index(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if s.empty:
        return pd.Series(index=idx, dtype=float)
    df = pd.DataFrame({"v": s.sort_index()})
    df = df.reindex(df.index.union(idx)).sort_index().ffill()
    return df.loc[idx, "v"]

def rolling_zscore(s: pd.Series, w: int, minp: int | None = None):
    minp = minp or max(8, w // 3)
    r = s.rolling(w, min_periods=minp)
    return (s - r.mean()) / (r.std(ddof=0) + 1e-12)

def ols_two_factors(y: pd.Series, X: pd.DataFrame):
    # X has columns ["L", "DL"]
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < 30:
        return None
    Y = df.iloc[:, 0].values.reshape(-1, 1)
    X0 = df[["L", "DL"]].values
    Xmat = np.column_stack([np.ones(len(df)), X0])
    beta = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta
    resid = (Y.flatten() - yhat)
    dof = max(1, len(df) - Xmat.shape[1])
    s2 = float((resid @ resid) / dof)
    cov = s2 * np.linalg.pinv(Xmat.T @ Xmat)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    t = beta / se
    denom = ((Y - Y.mean()) ** 2).sum() + 1e-12
    r2 = float(1.0 - (resid @ resid) / denom)
    names = ["Intercept", "L", "DL"]
    out = {
        "beta": pd.Series(beta, index=names),
        "t": pd.Series(t, index=names),
        "r2": r2,
        "rows": int(len(df)),
        "last_pred": float(np.array([1.0, X["L"].iloc[-1], X["DL"].iloc[-1]]) @ beta)
    }
    return out

# ───────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    resolution = st.selectbox("Resolution", ["Weekly", "Daily"], index=0)
    # Expanded default universe to 20 assets
    default_assets = ["SPY","QQQ","IWM","EFA","EEM",
                      "TLT","IEF","HYG","LQD",
                      "XLE","XLK","XLF","XLI","XLU",
                      "GLD","SLV","USO","UNG","UUP","VNQ","BTC-USD"]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets[:20])).replace(" ","").split(",")
    reg_window = st.number_input("Regression window (periods)", 52, 520, 156, step=13)
    z_win = st.number_input("Z-score window (periods)", 26, 104, 52, step=13)
    slope_lag = st.selectbox("Slope horizon for ΔL", [2,4,8,13,26], index=1)
    st.markdown("---")
    st.subheader("Liquidity Composite Weights")
    w_walcl = st.slider("WALCL (Fed balance sheet)", 0.0, 2.0, 1.0, 0.1)
    w_rrp   = st.slider("RRPONTSYD (Reverse repo, negative)", 0.0, 2.0, 1.0, 0.1)
    w_tga   = st.slider("TGA (Treasury cash, negative)", 0.0, 2.0, 0.8, 0.1)
    w_fx    = st.slider("FX reserves proxy (optional)", 0.0, 2.0, 0.5, 0.1)
    method = st.selectbox("Composite method", ["Z-sum", "PCA-1"], index=0)
    st.caption("Z-sum uses standardized levels; PCA-1 uses the first principal component of standardized flows.")
    st.markdown("---")
    st.subheader("Confidence Weights")
    wt_t = st.slider("Weight of |t|-mean", 0.0, 1.0, 0.55, 0.05)
    wt_r2 = st.slider("Weight of R²", 0.0, 1.0, 0.25, 0.05)
    wt_stab = st.slider("Weight of Stability", 0.0, 1.0, 0.20, 0.05)
    st.markdown("---")
    st.subheader("VIX Overlay")
    show_vix = st.toggle("Show VIX term structure overlay", value=True)
    st.caption("Overlay VIX3M/VIX - 1 on the regime chart. Backwardation < 0, contango > 0.")

st.caption("Two-factor spec on returns: Liquidity Level Z and Liquidity Slope Z. Overlay VIX term structure for risk context. Toggle weekly or daily resolution.")

# ───────────────────────────── Data ──────────────────────────────
with st.spinner("Loading data"):
    if resolution == "Weekly":
        idx = weekly_index(start_date)
        yf_interval = "1d"  # price source still daily; we will resample to weekly
    else:
        idx = daily_index(start_date)
        yf_interval = "1d"

    # Prices
    px_raw = load_prices(tickers, start=str(start_date), interval=yf_interval)
    if px_raw.empty:
        st.error("No price data. Check tickers.")
        st.stop()

    if resolution == "Weekly":
        px = px_raw.resample("W-FRI").last().reindex(idx).ffill()
    else:
        px = px_raw.reindex(idx).ffill()

    rets = np.log(px).diff()

    # VIX and VIX3M for overlay
    vix_df = pd.DataFrame()
    if show_vix:
        vix_src = load_prices(["^VIX","^VIX3M"], start=str(start_date), interval=yf_interval)
        if not vix_src.empty:
            if resolution == "Weekly":
                vix_df = vix_src.resample("W-FRI").last().reindex(idx).ffill()
            else:
                vix_df = vix_src.reindex(idx).ffill()
        else:
            st.warning("VIX data unavailable. Overlay disabled.")
            show_vix = False

    # FRED series
    fred_ids = {
        "WALCL":"WALCL",          # Fed balance sheet (weekly Wed)
        "RRPONTSYD":"RRPONTSYD",  # Reverse repo (daily)
        "WTREGEN":"WTREGEN",      # TGA (daily, preferred)
        "WDTGAL":"WDTGAL",        # TGA alt (daily)
        # Optional FX reserve proxies
        "CHNR":"CHNRORGDPM",
        "JPNR":"JPNRGSBP",
    }

    fred_raw = {}
    for name, sid in fred_ids.items():
        s = fred_series(sid, start=str(start_date))
        if not s.empty:
            fred_raw[name] = align_to_index(s, idx)

    fred_df = pd.DataFrame(fred_raw, index=idx)
    if fred_df.empty:
        st.error("No FRED data returned. Try a later start date.")
        st.stop()

# Build core flows
flows = {}
have = set(fred_df.columns)
tga = fred_df["WTREGEN"] if "WTREGEN" in have else (fred_df["WDTGAL"] if "WDTGAL" in have else None)

if "WALCL" in have:
    flows["WALCL"] = fred_df["WALCL"]
if "RRPONTSYD" in have:
    flows["RRPONTSYD"] = fred_df["RRPONTSYD"]
if tga is not None:
    flows["TGA"] = tga

fx_cols = [c for c in fred_df.columns if c.upper().startswith(("CHN","JPN")) or c.upper().endswith("RGSBP")]
if fx_cols:
    flows["FXRES"] = fred_df[fx_cols].sum(axis=1)

if not flows:
    st.error("No usable flow series present.")
    st.stop()

flows_df = pd.DataFrame(flows, index=idx).ffill()

# Standardize levels for composite
Z = flows_df.apply(lambda s: rolling_zscore(s, w=int(z_win)), axis=0)

# Signed weights
w_map = {
    "WALCL": +w_walcl,
    "RRPONTSYD": -w_rrp,
    "TGA": -w_tga,
    "FXRES": +w_fx
}
weights = pd.Series({k: w_map.get(k, 0.0) for k in Z.columns})

# Composite L(t)
if method == "Z-sum":
    L = (Z * weights.reindex(Z.columns).fillna(0.0)).sum(axis=1)
else:
    Zc = Z.dropna(how="all", axis=1).fillna(0.0)
    if Zc.shape[1] == 0:
        L = (Z * weights.reindex(Z.columns).fillna(0.0)).sum(axis=1)
    else:
        M = Zc.values
        cov = np.cov(M.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, -1]
        L = pd.Series(M @ pc1, index=Zc.index)
        Z = Zc.copy()
        weights = pd.Series(pc1, index=Zc.columns)

# Level Z and Slope Z
Lz = rolling_zscore(L, w=int(z_win))
DL = L.diff(int(slope_lag))
DLz = rolling_zscore(DL, w=int(z_win))

features = pd.DataFrame({"L": Lz, "DL": DLz}, index=idx)

# Trim to regression window
eff_w = min(int(reg_window), len(idx))
features_w = features.iloc[-eff_w:]
rets_w = rets.reindex(features_w.index)

# ───────────────────────────── Model ─────────────────────────────
betas, tstats, r2s, rows_used, preds, stab = {}, {}, {}, {}, {}, {}
for asset in rets_w.columns:
    y = rets_w[asset]
    res = ols_two_factors(y, features_w)
    if res is None:
        continue
    betas[asset] = res["beta"]
    tstats[asset] = res["t"]
    r2s[asset] = res["r2"]
    rows_used[asset] = res["rows"]
    preds[asset] = res["last_pred"]

    # Stability across halves
    df = pd.concat([y, features_w], axis=1).dropna()
    mid = len(df)//2
    def part_beta(df_part):
        out = ols_two_factors(df_part.iloc[:,0], df_part.iloc[:,1:])
        if out is None:
            return np.nan, np.nan
        b = out["beta"]
        return float(b["L"]), float(b["DL"])
    b1L, b1D = part_beta(df.iloc[:mid])
    b2L, b2D = part_beta(df.iloc[mid:])
    if np.any(~np.isfinite([b1L,b1D,b2L,b2D])):
        stab[asset] = np.nan
    else:
        denom = abs(b1L)+abs(b2L)+abs(b1D)+abs(b2D)+1e-6
        stab[asset] = float(np.clip(1.0 - (abs(b1L-b2L)+abs(b1D-b2D))/denom, 0.0, 1.0))

if not betas:
    st.error("No assets had sufficient overlap for regression.")
    st.stop()

B = pd.DataFrame(betas).T[["L","DL"]]
Tstats = pd.DataFrame(tstats).T[["L","DL"]]
R2 = pd.Series(r2s, name="R2")
Rows = pd.Series(rows_used, name="Rows")
Pred = pd.Series(preds, name="FlowNow")
Stability = pd.Series(stab, name="Stability").clip(0.0, 1.0)

# Confidence blend
abs_t_mean = Tstats.abs().mean(axis=1).rename("|t|_mean").fillna(0.0)
t_conf = np.clip(abs_t_mean, 0.0, 3.0) / 3.0
r2_conf = R2.fillna(0.0).clip(0.0, 0.20) / 0.20
stab_conf = Stability.fillna(0.0).clip(0.0, 1.0)

w_sum = max(1e-6, wt_t + wt_r2 + wt_stab)
Confidence = ((wt_t/w_sum)*t_conf + (wt_r2/w_sum)*r2_conf + (wt_stab/w_sum)*stab_conf).rename("Confidence")

# Tilt: exposures to current state
state_L = float(features_w["L"].iloc[-1]) if not features_w.empty else np.nan
state_DL = float(features_w["DL"].iloc[-1]) if not features_w.empty else np.nan
TiltRaw = (B["L"]*state_L + B["DL"]*state_DL).rename("TiltRaw")
Tilt = (TiltRaw * Confidence).rename("TiltScore")

playbook = pd.concat([B, Tstats.add_prefix("t_"), R2, Rows, Stability, Confidence, Tilt, Pred], axis=1)
playbook = playbook.sort_values("TiltScore", ascending=False)

# ────────────────────────── Header metrics ───────────────────────
if show_vix and not vix_df.empty:
    vix = vix_df["^VIX"]
    vix3m = vix_df["^VIX3M"] if "^VIX3M" in vix_df.columns else vix_df.get("VIX3M", pd.Series(index=idx))
    ts = (vix3m / vix - 1.0).rename("TS")
    ts_now = float(ts.iloc[-1]) if ts.notna().any() else np.nan
    vix_now = float(vix.iloc[-1]) if vix.notna().any() else np.nan
else:
    ts = pd.Series(index=idx, dtype=float)
    ts_now, vix_now = np.nan, np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidity Level z", f"{state_L:+.2f}")
c2.metric("Liquidity Slope z", f"{state_DL:+.2f}")
c3.metric("Avg Confidence", f"{Confidence.mean():.2f}")
if show_vix:
    c4.metric("VIX3M/VIX - 1", f"{ts_now:+.3f}")
else:
    c4.metric("Coverage", f"{len(playbook)} assets")

def regime_text(Lz_now, DLz_now, ts_now_val):
    base = "Mixed/neutral"
    if np.isfinite(Lz_now) and np.isfinite(DLz_now):
        if Lz_now >= 1.0 and DLz_now > 0:
            base = "Liquidity improving"
        elif Lz_now <= -1.0 and DLz_now < 0:
            base = "Liquidity deteriorating"
    if np.isfinite(ts_now_val):
        if ts_now_val < 0:
            return base + " with VIX backwardation"
        elif ts_now_val > 0.1:
            return base + " with VIX contango"
    return base

st.caption(f"Regime: {regime_text(state_L, state_DL, ts_now)}. Use OW/UW or pairs accordingly. Invalidate if TiltScore flips sign or |t|-mean collapses.")

# ─────────────────────── OW / UW and Pairs ───────────────────────
top_n = min(5, len(playbook))
ow = playbook.head(top_n)
uw = playbook.tail(top_n).iloc[::-1]

def fmt_tbl(df):
    cols = ["L","DL","t_L","t_DL","R2","Stability","Confidence","TiltScore","FlowNow","Rows"]
    view = df.reindex(columns=cols)
    try:
        return view.style.format({
            "L":"{:+.2f}","DL":"{:+.2f}",
            "t_L":"{:+.2f}","t_DL":"{:+.2f}",
            "R2":"{:.1%}","Stability":"{:.0%}",
            "Confidence":"{:.2f}","TiltScore":"{:+.3f}",
            "FlowNow":"{:+.3f}","Rows":"{:d}"
        }).background_gradient(subset=["TiltScore"], cmap="RdYlGn")
    except Exception:
        return view

st.subheader("Actionable Playbook")
cL, cR = st.columns(2)
with cL:
    st.markdown("**Top Overweights**")
    st.dataframe(fmt_tbl(ow), use_container_width=True)
with cR:
    st.markdown("**Top Underweights**")
    st.dataframe(fmt_tbl(uw), use_container_width=True)

def make_pairs(long_df, short_df, n=3):
    pairs = []
    for la, ls in zip(long_df.index[:n], short_df.index[:n]):
        pairs.append([la, ls, float(long_df.loc[la,"TiltScore"] - short_df.loc[ls,"TiltScore"])])
    return pd.DataFrame(pairs, columns=["Long","Short","NetScore"])
pairs = make_pairs(ow, uw, n=min(3, top_n))
st.markdown("**Pairs**")
try:
    st.dataframe(pairs.style.format({"NetScore":"{:+.3f}"}), use_container_width=True)
except Exception:
    st.dataframe(pairs, use_container_width=True)

# ───────────── Liquidity composite visuals with VIX overlay ─────────────
st.subheader("Liquidity Composite, Slope, and VIX Term Structure")
bars_periods = 26

# Contribution bars for composite
contrib = pd.DataFrame(index=Z.index, columns=Z.columns, dtype=float)
for c in Z.columns:
    contrib[c] = Z[c] * (weights.get(c, 0.0))
contrib = contrib.iloc[-bars_periods:]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.06,
                    specs=[[{"secondary_y": True}],[{"type":"bar"}]])

# Level and slope lines
fig.add_hline(y=0, line=dict(color="#cccccc", width=1, dash="dot"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=Lz.index, y=Lz, mode="lines", name="Level Z", line=dict(width=1.6)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=DLz.index, y=DLz, mode="lines", name="Slope Z", line=dict(width=1.2, dash="dot")), row=1, col=1, secondary_y=False)

# VIX TS overlay
if show_vix and not ts.empty:
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="VIX3M/VIX - 1", line=dict(width=1.2)), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="VIX TS", row=1, col=1, secondary_y=True)

# Contribution stacked bars
for c in contrib.columns:
    fig.add_trace(go.Bar(x=contrib.index, y=contrib[c], name=c, showlegend=True), row=2, col=1)

fig.update_yaxes(title_text="Z-scores", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Weighted level contribution", row=2, col=1)
fig.update_layout(height=560, margin=dict(l=40, r=10, t=40, b=30), barmode="relative", hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
st.plotly_chart(fig, use_container_width=True)

# ───────────────── Sensitivity compass ──────────────────────────
st.subheader("Sensitivity Compass (β to Level vs β to Slope)")
comp = B.copy()
comp["Confidence"] = Confidence
comp["Tilt"] = Tilt
comp = comp.sort_values("Tilt", ascending=False)

scatter = go.Figure()
scatter.add_hline(y=0, line=dict(color="#cccccc", dash="dot"))
scatter.add_vline(x=0, line=dict(color="#cccccc", dash="dot"))
scatter.add_trace(
    go.Scatter(
        x=comp["L"], y=comp["DL"],
        mode="markers+text",
        text=comp.index,
        textposition="top center",
        marker=dict(
            size=(5 + 20*comp["Confidence"].clip(0,1)).astype(float),
            opacity=0.85,
        ),
        name="Assets",
        customdata=np.stack([comp["Tilt"], comp["Confidence"], comp.index], axis=1),
        hovertemplate="<b>%{customdata[2]}</b><br>β_L=%{x:.2f}  β_ΔL=%{y:.2f}<br>Tilt=%{customdata[0]:+.3f}<br>Conf=%{customdata[1]:.2f}<extra></extra>"
    )
)
scatter.update_xaxes(title="β to Liquidity Level (Z)")
scatter.update_yaxes(title="β to Liquidity Slope (Z)")
scatter.update_layout(height=480, margin=dict(l=40, r=10, t=40, b=40), showlegend=False)
st.plotly_chart(scatter, use_container_width=True)

# ───────────────── Downloads ─────────────────────────────────────
with st.expander("Download data"):
    st.download_button("Playbook CSV", playbook.to_csv().encode(), file_name="flow_of_funds_playbook.csv")
    out_feats = pd.DataFrame({"Lz": Lz, "DLz": DLz})
    if show_vix and not ts.empty:
        out_feats["VIX_TS"] = ts
    st.download_button("Composite features CSV", out_feats.to_csv().encode(), file_name="liquidity_features.csv")

st.caption("© 2025 AD Fund Management LP")
