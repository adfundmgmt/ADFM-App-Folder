# 14_Flow_of_Funds_Compass.py
# Clean, resilient Flow-of-Funds model with NO pandas_datareader and NO hard FRED key requirement.
# - WALCL, RRPONTSYD via FRED CSV endpoints (no key) with retry + local cache
# - TGA via U.S. Treasury FiscalData API (no key)
# - Writable cache in /tmp (or override via st.secrets["CACHE_DIR"] / env ADFM_CACHE_DIR)
# - Graceful degrade: build composite from whatever flows are available; never st.stop() unless prices fail

import os, io, math, json, tempfile, requests
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────── Page ───────────────────────────
st.set_page_config(page_title="Flow of Funds Compass", layout="wide")
st.title("Flow of Funds Compass")

# ─────────────────────── Writable cache dir ───────────────────────
def _pick_cache_dir() -> Path | None:
    for p in [
        st.secrets.get("CACHE_DIR", ""),
        os.getenv("ADFM_CACHE_DIR", ""),
        os.path.join(tempfile.gettempdir(), "ff_cache"),
    ]:
        if not p:
            continue
        try:
            path = Path(p)
            path.mkdir(parents=True, exist_ok=True)
            t = path / ".rwtest"
            t.write_text("ok", encoding="utf-8")
            t.unlink(missing_ok=True)
            return path
        except Exception:
            continue
    return None

CACHE_DIR = _pick_cache_dir()

def _cache_path(series_id: str) -> Path | None:
    return None if CACHE_DIR is None else CACHE_DIR / f"{series_id}.csv"

def load_from_cache(series_id: str) -> pd.Series:
    p = _cache_path(series_id)
    if not p or not p.exists():
        return pd.Series(dtype=float, name=series_id)
    try:
        df = pd.read_csv(p)
        s = pd.to_numeric(df["value"], errors="coerce")
        out = pd.Series(s.values, index=pd.to_datetime(df["date"]), name=series_id).dropna()
        return out
    except Exception:
        return pd.Series(dtype=float, name=series_id)

def save_to_cache(series_id: str, s: pd.Series) -> None:
    p = _cache_path(series_id)
    if not p:
        return
    try:
        pd.DataFrame({
            "date": pd.to_datetime(s.index).astype(str),
            "value": s.astype(float).values
        }).to_csv(p, index=False)
    except Exception:
        pass

# ───────────────────────── Helpers ─────────────────────────
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

def rolling_z(s: pd.Series, w: int, minp: int | None = None):
    minp = minp or max(8, w // 3)
    r = s.rolling(w, min_periods=minp)
    return (s - r.mean()) / (r.std(ddof=0) + 1e-12)

@st.cache_data(ttl=6*3600, show_spinner=False)
def load_prices(tickers, start="2012-01-01", interval="1d"):
    try:
        df = yf.download(tickers, start=start, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"].copy()
        else:
            df = df[["Close"]].rename(columns={"Close": tickers[0]})
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        bad = [c for c in df.columns if df[c].dropna().empty]
        if bad:
            df = df.drop(columns=bad)
        return df.sort_index().ffill()
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()

def ols_two_factor(y: pd.Series, X: pd.DataFrame):
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

# ─────────────── Robust flow fetchers (no key required) ───────────────
UA = {"User-Agent": "ADFM-Streamlit/1.0"}

def _fred_try_csv(series_id: str) -> pd.Series:
    urls = [
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            date_col = "DATE" if "DATE" in df.columns else "date"
            if series_id in df.columns:
                val_col = series_id
            elif "VALUE" in df.columns:
                val_col = "VALUE"
            else:
                # fallback: second column
                val_col = df.columns[1]
            s = pd.to_numeric(df[val_col], errors="coerce")
            out = pd.Series(s.values, index=pd.to_datetime(df[date_col]), name=series_id).dropna()
            return out
        except Exception:
            continue
    return pd.Series(dtype=float, name=series_id)

@st.cache_data(ttl=12*3600, show_spinner=False)
def get_fred_series(series_id: str, start: str) -> pd.Series:
    # cache-first
    s = load_from_cache(series_id)
    if not s.empty:
        s = s[s.index >= pd.to_datetime(start)]
        if not s.empty:
            return s
    # fetch
    s = _fred_try_csv(series_id)
    if not s.empty:
        save_to_cache(series_id, s)
        s = s[s.index >= pd.to_datetime(start)]
    return s

@st.cache_data(ttl=12*3600, show_spinner=False)
def get_tga_fiscaldata(start: str) -> pd.Series:
    # Treasury Operating Cash Balance (TGA)
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
            return pd.Series(dtype=float, name="WTREGEN")
        dates = pd.to_datetime([d["record_date"] for d in data])
        vals = pd.to_numeric([d["close_today_bal"] for d in data], errors="coerce")
        s = pd.Series(vals, index=dates, name="WTREGEN").dropna()
        save_to_cache("WTREGEN", s)
        return s
    except Exception:
        # fallback to cached if any
        return load_from_cache("WTREGEN")

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    resolution = st.selectbox("Resolution", ["Weekly", "Daily"], index=0)
    default_assets = ["SPY","QQQ","IWM","EFA","EEM",
                      "TLT","IEF","HYG","LQD",
                      "XLE","XLK","XLF","XLI","XLU",
                      "GLD","SLV","USO","UNG","UUP","VNQ","BTC-USD"]
    tickers = st.text_input(
        "Assets (comma separated)", ",".join(default_assets[:20])
    ).replace(" ","").split(",")
    reg_window = st.number_input("Regression window (periods)", 52, 520, 156, step=13)
    z_win = st.number_input("Z-score window", 26, 104, 52, step=13)
    slope_lag = st.selectbox("Slope horizon for ΔL", [2,4,8,13,26], index=1)
    st.markdown("---")
    st.subheader("Liquidity weights")
    w_walcl = st.slider("WALCL (+)", 0.0, 2.0, 1.0, 0.1)
    w_rrp   = st.slider("RRPONTSYD (−)", 0.0, 2.0, 1.0, 0.1)
    w_tga   = st.slider("TGA (−)", 0.0, 2.0, 0.8, 0.1)
    st.markdown("---")
    st.subheader("Confidence blend")
    wt_t = st.slider("|t|-mean", 0.0, 1.0, 0.55, 0.05)
    wt_r2 = st.slider("R²", 0.0, 1.0, 0.25, 0.05)
    wt_stab = st.slider("Stability", 0.0, 1.0, 0.20, 0.05)
    st.markdown("---")
    show_vix = st.toggle("Show VIX term structure (VIX3M/VIX - 1)", value=True)

st.caption("Two-factor on returns: Liquidity Level Z and Liquidity Slope Z. Mirrorless, keyless, cached fetchers; degrades gracefully when some flows are missing.")

# ───────────────────────── Data ─────────────────────────
with st.spinner("Loading data"):
    idx = weekly_index(start_date) if resolution == "Weekly" else daily_index(start_date)
    yf_interval = "1d"  # daily fetch then resample

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
        vix_src = load_prices(["^VIX","^VIX3M"], start=str(start_date), interval="1d")
        if not vix_src.empty:
            vix_df = vix_src.resample("W-FRI").last().reindex(idx).ffill() if resolution == "Weekly" else vix_src.reindex(idx).ffill()

    # Core flows
    raw = {}
    walcl = get_fred_series("WALCL", str(start_date))
    if not walcl.empty: raw["WALCL"] = align_to_index(walcl, idx)
    rrp = get_fred_series("RRPONTSYD", str(start_date))
    if not rrp.empty: raw["RRPONTSYD"] = align_to_index(rrp, idx)
    tga = get_tga_fiscaldata(str(start_date))
    if not tga.empty: raw["WTREGEN"] = align_to_index(tga, idx)

fred_df = pd.DataFrame(raw, index=idx)
used = list(fred_df.columns)
missing = [c for c in ["WALCL","RRPONTSYD","WTREGEN"] if c not in used]
st.info(("Flows loaded: " + (", ".join(used) if used else "none")) + (f" | Missing: {', '.join(missing)}" if missing else ""))

# Build flows used in composite
flows = {}
if "WALCL" in fred_df:    flows["WALCL"] = fred_df["WALCL"]
if "RRPONTSYD" in fred_df:flows["RRPONTSYD"] = fred_df["RRPONTSYD"]
if "WTREGEN" in fred_df:  flows["TGA"] = fred_df["WTREGEN"]

flows_df = pd.DataFrame(flows, index=idx).ffill()

if flows_df.empty:
    st.warning("No flow series available (all fetches failed and no cache). Displaying VIX term structure only.")
    if show_vix and not vix_df.empty and "^VIX" in vix_df and "^VIX3M" in vix_df:
        ts = (vix_df["^VIX3M"] / vix_df["^VIX"] - 1.0).rename("TS")
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color="#cccccc", dash="dot"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="VIX3M/VIX - 1"))
        fig.update_layout(height=360, margin=dict(l=40, r=10, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ───────────────────── Composite & features ─────────────────────
Z = flows_df.apply(lambda s: rolling_z(s, w=int(z_win)), axis=0)
weights = pd.Series({
    "WALCL": +w_walcl if "WALCL" in Z.columns else 0.0,
    "RRPONTSYD": -w_rrp if "RRPONTSYD" in Z.columns else 0.0,
    "TGA": -w_tga if "TGA" in Z.columns else 0.0,
}).reindex(Z.columns).fillna(0.0)

L = (Z * weights).sum(axis=1)
Lz = rolling_z(L, w=int(z_win))
DL = L.diff(int(slope_lag))
DLz = rolling_z(DL, w=int(z_win))
features = pd.DataFrame({"L": Lz, "DL": DLz}, index=idx)

# Trim to regression window and run models
eff_w = min(int(reg_window), len(idx))
features_w = features.iloc[-eff_w:]
rets_w = rets.reindex(features_w.index)

betas, tstats, r2s, rows_used, preds, stab = {}, {}, {}, {}, {}, {}
for asset in rets_w.columns:
    out = ols_two_factor(rets_w[asset], features_w)
    if out is None:
        continue
    betas[asset] = out["beta"]
    tstats[asset] = out["t"]
    r2s[asset] = out["r2"]
    rows_used[asset] = out["rows"]
    preds[asset] = out["last_pred"]

    # stability across halves
    df = pd.concat([rets_w[asset], features_w], axis=1).dropna()
    mid = len(df)//2
    def part_beta(dfp):
        r = ols_two_factor(dfp.iloc[:,0], dfp.iloc[:,1:])
        if r is None: return np.nan, np.nan
        b = r["beta"]; return float(b["L"]), float(b["DL"])
    b1L,b1D = part_beta(df.iloc[:mid]); b2L,b2D = part_beta(df.iloc[mid:])
    if np.any(~np.isfinite([b1L,b1D,b2L,b2D])):
        stab[asset] = np.nan
    else:
        denom = abs(b1L)+abs(b2L)+abs(b1D)+abs(b2D)+1e-6
        stab[asset] = float(np.clip(1.0 - (abs(b1L-b2L)+abs(b1D-b2D))/denom, 0.0, 1.0))

if not betas:
    st.warning("Insufficient overlap for regressions with current window. Reduce window or adjust tickers.")
    st.stop()

B = pd.DataFrame(betas).T[["L","DL"]]
Tstats = pd.DataFrame(tstats).T[["L","DL"]]
R2 = pd.Series(r2s, name="R2")
Rows = pd.Series(rows_used, name="Rows")
Pred = pd.Series(preds, name="FlowNow")
Stability = pd.Series(stab, name="Stability").clip(0.0, 1.0)

# Confidence + Tilt
abs_t_mean = Tstats.abs().mean(axis=1).rename("|t|_mean").fillna(0.0)
t_conf   = np.clip(abs_t_mean, 0.0, 3.0) / 3.0
r2_conf  = R2.fillna(0.0).clip(0.0, 0.20) / 0.20
stab_conf= Stability.fillna(0.0).clip(0.0, 1.0)

w_sum = max(1e-6, wt_t + wt_r2 + wt_stab)
Confidence = ((wt_t/w_sum)*t_conf + (wt_r2/w_sum)*r2_conf + (wt_stab/w_sum)*stab_conf).rename("Confidence")

state_L = float(features_w["L"].iloc[-1]) if not features_w.empty else np.nan
state_DL = float(features_w["DL"].iloc[-1]) if not features_w.empty else np.nan
TiltRaw = (B["L"]*state_L + B["DL"]*state_DL).rename("TiltRaw")
Tilt = (TiltRaw * Confidence).rename("TiltScore")

playbook = pd.concat([B, Tstats.add_prefix("t_"), R2, Rows, Stability, Confidence, Tilt, Pred], axis=1)
playbook = playbook.sort_values("TiltScore", ascending=False)

# ───────────────────── Header metrics ─────────────────────
if show_vix and not vix_df.empty and "^VIX" in vix_df and "^VIX3M" in vix_df:
    ts = (vix_df["^VIX3M"] / vix_df["^VIX"] - 1.0).rename("TS")
    ts_now = float(ts.iloc[-1]) if ts.notna().any() else np.nan
else:
    ts = pd.Series(index=idx, dtype=float); ts_now = np.nan

c1,c2,c3,c4 = st.columns(4)
c1.metric("Liquidity Level z", f"{state_L:+.2f}")
c2.metric("Liquidity Slope z", f"{state_DL:+.2f}")
c3.metric("Avg Confidence", f"{Confidence.mean():.2f}")
c4.metric("VIX3M/VIX - 1", f"{ts_now:+.3f}" if np.isfinite(ts_now) else "n/a")

# ───────────────────── Playbook & Pairs ─────────────────────
st.subheader("Actionable Playbook")
left, right = st.columns(2)

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

top_n = min(5, len(playbook))
ow = playbook.head(top_n); uw = playbook.tail(top_n).iloc[::-1]

with left:
    st.markdown("**Top Overweights**")
    st.dataframe(fmt_tbl(ow), use_container_width=True)
with right:
    st.markdown("**Top Underweights**")
    st.dataframe(fmt_tbl(uw), use_container_width=True)

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

# ───────────────── Composite & VIX overlay ─────────────────
st.subheader("Liquidity Composite, Slope, and VIX Term Structure")

bars_periods = 26
contrib = pd.DataFrame(index=Z.index, columns=Z.columns, dtype=float)
for c in Z.columns:
    contrib[c] = Z[c] * weights.get(c, 0.0)
contrib = contrib.iloc[-bars_periods:]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.65, 0.35], specs=[[{"secondary_y": True}], [{"type":"bar"}]]
)
fig.add_hline(y=0, line=dict(color="#cccccc", dash="dot"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=Lz.index, y=Lz, mode="lines", name="Level Z", line=dict(width=1.6)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=DLz.index, y=DLz, mode="lines", name="Slope Z", line=dict(width=1.2, dash="dot")), row=1, col=1, secondary_y=False)

if not ts.empty:
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="VIX3M/VIX - 1", line=dict(width=1.2)), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="VIX TS", row=1, col=1, secondary_y=True)

for c in contrib.columns:
    fig.add_trace(go.Bar(x=contrib.index, y=contrib[c], name=c, showlegend=True), row=2, col=1)

fig.update_yaxes(title_text="Z-scores", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Weighted level contribution", row=2, col=1)
fig.update_layout(
    height=560, margin=dict(l=40, r=10, t=44, b=36),
    barmode="relative", hovermode="x unified",
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
)
st.plotly_chart(fig, use_container_width=True)

# ───────────────── Downloads ─────────────────
with st.expander("Download data"):
    st.download_button("Playbook CSV", playbook.to_csv().encode(), file_name="flow_of_funds_playbook.csv")
    out_feats = pd.DataFrame({"Lz": Lz, "DLz": DLz})
    if not ts.empty:
        out_feats["VIX_TS"] = ts
    st.download_button("Composite features CSV", out_feats.to_csv().encode(), file_name="liquidity_features.csv")

st.caption("© 2025 AD Fund Management LP")
