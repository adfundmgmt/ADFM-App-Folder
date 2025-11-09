# 14_Flow_of_Funds_Compass.py
# Resilient, no-stop flow-of-funds: mirror-first, non-FRED where possible, cache, graceful degrade.

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ───────── Page ─────────
st.set_page_config(page_title="Flow of Funds Compass", layout="wide")
st.title("Flow of Funds Compass")

# ───────── Settings & secrets ─────────
FRED_KEY = st.secrets.get("FRED_API_KEY", "")
FRED_MIRROR = st.secrets.get("FRED_MIRROR_URL", "").rstrip("/")  # e.g. https://raw.githubusercontent.com/yourorg/adfm-data/main/fred
CACHE_DIR = "/mnt/data/ff_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ───────── Utilities ─────────
def weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="W-FRI")

def daily_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="B")

def align_to_index(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(index=idx, dtype=float)
    df = pd.DataFrame({"v": s.sort_index()})
    df = df.reindex(df.index.union(idx)).sort_index().ffill()
    return df.loc[idx, "v"]

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

def rolling_zscore(s: pd.Series, w: int, minp: int | None = None):
    minp = minp or max(8, w // 3)
    r = s.rolling(w, min_periods=minp)
    return (s - r.mean()) / (r.std(ddof=0) + 1e-12)

def ols_two_factors(y: pd.Series, X: pd.DataFrame):
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
    return {
        "beta": pd.Series(beta, index=names),
        "t": pd.Series(t, index=names),
        "r2": r2,
        "rows": int(len(df)),
        "last_pred": float(np.array([1.0, X["L"].iloc[-1], X["DL"].iloc[-1]]) @ beta),
    }

# ───────── Robust series fetchers ─────────
def cache_path(series_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{series_id}.parquet")

def load_from_cache(series_id: str) -> pd.Series:
    p = cache_path(series_id)
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            s.name = series_id
            return s.dropna()
        except Exception:
            return pd.Series(dtype=float)
    return pd.Series(dtype=float)

def save_to_cache(series_id: str, s: pd.Series):
    try:
        df = s.dropna().to_frame(series_id)
        df.to_parquet(cache_path(series_id))
    except Exception:
        pass

def fred_api_observations(series_id: str, start: str, api_key: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "observation_start": start, "file_type": "json", "api_key": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    dates = pd.to_datetime([o["date"] for o in obs])
    vals = pd.to_numeric([o.get("value", ".") for o in obs], errors="coerce")
    return pd.Series(vals, index=dates, name=series_id).dropna()

def fred_csv(series_id: str) -> pd.Series:
    headers = {"User-Agent": "ADFM-Streamlit/1.0"}
    urls = [
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            date_col = "DATE" if "DATE" in df.columns else "date"
            value_col = "VALUE" if "VALUE" in df.columns else (series_id if series_id in df.columns else None)
            if value_col is None:
                continue
            s = pd.to_numeric(df[value_col], errors="coerce")
            return pd.Series(s.values, index=pd.to_datetime(df[date_col]), name=series_id).dropna()
        except Exception:
            continue
    return pd.Series(dtype=float)

def fred_mirror(series_id: str) -> pd.Series:
    if not FRED_MIRROR:
        return pd.Series(dtype=float)
    url = f"{FRED_MIRROR}/{series_id}.csv"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        date_col = "DATE" if "DATE" in df.columns else "date"
        value_col = "VALUE" if "VALUE" in df.columns else (series_id if series_id in df.columns else None)
        if value_col is None:
            # accept two-column generic mirror: date,value
            if df.shape[1] >= 2:
                value_col = df.columns[1]
        s = pd.to_numeric(df[value_col], errors="coerce")
        return pd.Series(s.values, index=pd.to_datetime(df[date_col]), name=series_id).dropna()
    except Exception:
        return pd.Series(dtype=float)

def fetch_tga_fiscaldata(start: str) -> pd.Series:
    # Treasury Operating Cash Balance (TGA) from FiscalData API
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/operating_cash_balance"
    params = {
        "format": "json",
        "fields": "record_date,close_today_bal",
        "filter": f"record_date:gte:{start}",
        "sort": "record_date",
        "page[size]": "10000",
    }
    try:
        r = requests.get(base, params=params, timeout=25)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.Series(dtype=float)
        dates = pd.to_datetime([d["record_date"] for d in data])
        vals = pd.to_numeric([d["close_today_bal"] for d in data], errors="coerce")
        return pd.Series(vals, index=dates, name="WTREGEN").dropna()
    except Exception:
        return pd.Series(dtype=float)

def get_series(series_id: str, start: str) -> pd.Series:
    # Special handling: prefer non-FRED source for TGA
    if series_id in ("WTREGEN", "WDTGAL", "TGA"):
        s = fetch_tga_fiscaldata(start)
        if not s.empty:
            save_to_cache("WTREGEN", s)
            return s
        # fallback chain for WTREGEN via FRED
        series_try = ["WTREGEN", "WDTGAL"]
        for sid in series_try:
            s = fred_mirror(sid)
            if s.empty and FRED_KEY:
                try:
                    s = fred_api_observations(sid, start, FRED_KEY)
                except Exception:
                    s = pd.Series(dtype=float)
            if s.empty:
                s = fred_csv(sid)
            if not s.empty:
                save_to_cache(sid, s)
                return s
        return load_from_cache("WTREGEN")

    # Generic chain for other FRED series
    s = fred_mirror(series_id)
    if s.empty and FRED_KEY:
        try:
            s = fred_api_observations(series_id, start, FRED_KEY)
        except Exception:
            s = pd.Series(dtype=float)
    if s.empty:
        s = fred_csv(series_id)
    if s.empty:
        s = load_from_cache(series_id)
    else:
        save_to_cache(series_id, s)
    return s

# ───────── Sidebar ─────────
with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    resolution = st.selectbox("Resolution", ["Weekly", "Daily"], index=0)
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

st.caption("Two-factor spec on returns: Liquidity Level Z and Liquidity Slope Z. Mirror-first, cached fallback, no hard dependency on FRED at runtime.")

# ───────── Data ─────────
with st.spinner("Loading data"):
    if resolution == "Weekly":
        idx = weekly_index(start_date)
        yf_interval = "1d"
    else:
        idx = daily_index(start_date)
        yf_interval = "1d"

    # Prices
    px_raw = load_prices(tickers, start=str(start_date), interval=yf_interval)
    if px_raw.empty:
        st.error("No price data. Check tickers.")
        st.stop()
    px = px_raw.resample("W-FRI").last().reindex(idx).ffill() if resolution == "Weekly" else px_raw.reindex(idx).ffill()
    rets = np.log(px).diff()

    # VIX overlay
    vix_df = pd.DataFrame()
    if show_vix:
        vix_src = load_prices(["^VIX","^VIX3M"], start=str(start_date), interval=yf_interval)
        vix_df = (vix_src.resample("W-FRI").last().reindex(idx).ffill() if resolution == "Weekly" else vix_src.reindex(idx).ffill()) if not vix_src.empty else pd.DataFrame()

    # Flow series: try to get as many as possible; never stop on failure
    raw = {}
    for sid in ["WALCL", "RRPONTSYD", "WTREGEN"]:
        s = get_series(sid, str(start_date))
        if not s.empty:
            raw[sid] = align_to_index(s, idx)

    # Optional FX reserve proxies if you have a mirror (leave empty otherwise)
    fx_sum = pd.Series(index=idx, dtype=float)
    # Example: If you mirror CHNRORGDPM and JPNRGSBP under your mirror, they will load via get_series
    fx_parts = []
    for sid in ["CHNRORGDPM", "JPNRGSBP"]:
        s = get_series(sid, str(start_date))
        if not s.empty:
            fx_parts.append(align_to_index(s, idx))
    if fx_parts:
        fx_sum = pd.concat(fx_parts, axis=1).sum(axis=1)
        raw["FXRES"] = fx_sum

fred_df = pd.DataFrame(raw, index=idx)
used_cols = list(fred_df.columns)
missing_cols = [c for c in ["WALCL","RRPONTSYD","WTREGEN","FXRES"] if c not in used_cols]
msg = f"Flows loaded: {', '.join(used_cols) if used_cols else 'none'}."
if missing_cols:
    msg += f" Missing: {', '.join(missing_cols)}. Using what is available; results degrade gracefully."
st.info(msg)

# Build flows; if none, keep app functional
flows = {}
if "WALCL" in fred_df:
    flows["WALCL"] = fred_df["WALCL"]
if "RRPONTSYD" in fred_df:
    flows["RRPONTSYD"] = fred_df["RRPONTSYD"]
if "WTREGEN" in fred_df:
    flows["TGA"] = fred_df["WTREGEN"]
if "FXRES" in fred_df:
    flows["FXRES"] = fred_df["FXRES"]

flows_df = pd.DataFrame(flows, index=idx).ffill()

# If absolutely nothing, show overlay-only view
if flows_df.empty:
    st.warning("No flow series available from mirror, API, CSV, or cache. Showing VIX overlay only.")
    if not vix_df.empty and "^VIX" in vix_df and "^VIX3M" in vix_df:
        ts = (vix_df["^VIX3M"] / vix_df["^VIX"] - 1.0).rename("TS")
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color="#cccccc", dash="dot"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="VIX3M/VIX - 1"))
        fig.update_layout(height=360, margin=dict(l=40, r=10, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

# Composite
Z = flows_df.apply(lambda s: rolling_zscore(s, w=int(z_win)), axis=0)
w_map = {"WALCL": +w_walcl, "RRPONTSYD": -w_rrp, "TGA": -w_tga, "FXRES": +w_fx}
weights = pd.Series({k: w_map.get(k, 0.0) for k in Z.columns})

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

Lz = rolling_zscore(L, w=int(z_win))
DL = L.diff(int(slope_lag))
DLz = rolling_zscore(DL, w=int(z_win))
features = pd.DataFrame({"L": Lz, "DL": DLz}, index=idx)

# Window and regressions
eff_w = min(int(reg_window), len(idx))
features_w = features.iloc[-eff_w:]
rets_w = np.log(px).diff().reindex(features_w.index)

betas, tstats, r2s, rows_used, preds, stab = {}, {}, {}, {}, {}, {}
for asset in rets_w.columns:
    out = ols_two_factors(rets_w[asset], features_w)
    if out is None:
        continue
    betas[asset] = out["beta"]
    tstats[asset] = out["t"]
    r2s[asset] = out["r2"]
    rows_used[asset] = out["rows"]
    preds[asset] = out["last_pred"]

    df = pd.concat([rets_w[asset], features_w], axis=1).dropna()
    mid = len(df)//2
    def part_beta(dfp):
        res = ols_two_factors(dfp.iloc[:,0], dfp.iloc[:,1:])
        if res is None: return np.nan, np.nan
        b = res["beta"]; return float(b["L"]), float(b["DL"])
    b1L, b1D = part_beta(df.iloc[:mid]); b2L, b2D = part_beta(df.iloc[mid:])
    if np.any(~np.isfinite([b1L,b1D,b2L,b2D])):
        stab[asset] = np.nan
    else:
        denom = abs(b1L)+abs(b2L)+abs(b1D)+abs(b2D)+1e-6
        stab[asset] = float(np.clip(1.0 - (abs(b1L-b2L)+abs(b1D-b2D))/denom, 0.0, 1.0))

if not betas:
    st.warning("Insufficient overlap for regressions with current window. Reduce window or add assets.")
    st.stop()

B = pd.DataFrame(betas).T[["L","DL"]]
Tstats = pd.DataFrame(tstats).T[["L","DL"]]
R2 = pd.Series(r2s, name="R2")
Rows = pd.Series(rows_used, name="Rows")
Pred = pd.Series(preds, name="FlowNow")
Stability = pd.Series(stab, name="Stability").clip(0.0, 1.0)

abs_t_mean = Tstats.abs().mean(axis=1).rename("|t|_mean").fillna(0.0)
t_conf = np.clip(abs_t_mean, 0.0, 3.0) / 3.0
r2_conf = R2.fillna(0.0).clip(0.0, 0.20) / 0.20
stab_conf = Stability.fillna(0.0).clip(0.0, 1.0)
w_sum = max(1e-6, wt_t + wt_r2 + wt_stab)
Confidence = ((wt_t/w_sum)*t_conf + (wt_r2/w_sum)*r2_conf + (wt_stab/w_sum)*stab_conf).rename("Confidence")

state_L = float(features_w["L"].iloc[-1]) if not features_w.empty else np.nan
state_DL = float(features_w["DL"].iloc[-1]) if not features_w.empty else np.nan
TiltRaw = (B["L"]*state_L + B["DL"]*state_DL).rename("TiltRaw")
Tilt = (TiltRaw * Confidence).rename("TiltScore")

playbook = pd.concat([B, Tstats.add_prefix("t_"), R2, Rows, Stability, Confidence, Tilt, Pred], axis=1)
playbook = playbook.sort_values("TiltScore", ascending=False)

# Header metrics
if show_vix and not vix_df.empty and "^VIX" in vix_df and "^VIX3M" in vix_df:
    ts = (vix_df["^VIX3M"] / vix_df["^VIX"] - 1.0).rename("TS")
    ts_now = float(ts.iloc[-1]) if ts.notna().any() else np.nan
else:
    ts = pd.Series(index=idx, dtype=float); ts_now = np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidity Level z", f"{state_L:+.2f}")
c2.metric("Liquidity Slope z", f"{state_DL:+.2f}")
c3.metric("Avg Confidence", f"{Confidence.mean():.2f}")
c4.metric("VIX3M/VIX - 1", f"{ts_now:+.3f}" if np.isfinite(ts_now) else "n/a")

# OW/UW and Pairs
top_n = min(5, len(playbook))
ow = playbook.head(top_n); uw = playbook.tail(top_n).iloc[::-1]

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
    st.markdown("**Top Overweights**"); st.dataframe(fmt_tbl(ow), use_container_width=True)
with cR:
    st.markdown("**Top Underweights**"); st.dataframe(fmt_tbl(uw), use_container_width=True)

def make_pairs(long_df, short_df, n=3):
    rows = []
    for la, ls in zip(long_df.index[:n], short_df.index[:n]):
        rows.append([la, ls, float(long_df.loc[la,"TiltScore"] - short_df.loc[ls,"TiltScore"])])
    return pd.DataFrame(rows, columns=["Long","Short","NetScore"])

pairs = make_pairs(ow, uw, n=min(3, top_n))
st.markdown("**Pairs**")
try:
    st.dataframe(pairs.style.format({"NetScore":"{:+.3f}"}), use_container_width=True)
except Exception:
    st.dataframe(pairs, use_container_width=True)

# Liquidity composite visuals
st.subheader("Liquidity Composite, Slope, and VIX Term Structure")
bars_periods = 26
Z_for_plot = Z.copy()
weights_for_plot = weights.copy()

contrib = pd.DataFrame(index=Z_for_plot.index, columns=Z_for_plot.columns, dtype=float)
for c in Z_for_plot.columns:
    contrib[c] = Z_for_plot[c] * (weights_for_plot.get(c, 0.0))
contrib = contrib.iloc[-bars_periods:]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.06,
                    specs=[[{"secondary_y": True}], [{"type":"bar"}]])

fig.add_hline(y=0, line=dict(color="#cccccc", width=1, dash="dot"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=Lz.index, y=Lz, mode="lines", name="Level Z", line=dict(width=1.6)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=DLz.index, y=DLz, mode="lines", name="Slope Z", line=dict(width=1.2, dash="dot")), row=1, col=1, secondary_y=False)

if not ts.empty:
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="VIX3M/VIX - 1", line=dict(width=1.2)), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="VIX TS", row=1, col=1, secondary_y=True)

for c in contrib.columns:
    fig.add_trace(go.Bar(x=contrib.index, y=contrib[c], name=c, showlegend=True), row=2, col=1)

fig.update_yaxes(title_text="Z-scores", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Weighted level contribution", row=2, col=1)
fig.update_layout(height=560, margin=dict(l=40, r=10, t=40, b=30), barmode="relative", hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
st.plotly_chart(fig, use_container_width=True)

# Downloads
with st.expander("Download data"):
    st.download_button("Playbook CSV", playbook.to_csv().encode(), file_name="flow_of_funds_playbook.csv")
    out_feats = pd.DataFrame({"Lz": Lz, "DLz": DLz})
    if not ts.empty:
        out_feats["VIX_TS"] = ts
    st.download_button("Composite features CSV", out_feats.to_csv().encode(), file_name="liquidity_features.csv")

st.caption("© 2025 AD Fund Management LP")
