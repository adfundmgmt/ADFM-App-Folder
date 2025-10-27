############################################################
# Market Stress Composite — Dual (FRED Daily + Live Intraday)
# Built for AD Fund Management LP
# Changes: Live timestamped composite, tighter defaults, freshness banner,
#          delta metrics, dual plots, proxy-only fallback kept.
############################################################

# ---- Py3.12 compat shim for pandas_datareader ----
import sys, types
import packaging.version
try:
    import distutils.version
except Exception:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = packaging.version.Version
    _d.version = _dv
    sys.modules.update({"distutils": _d, "distutils.version": _dv})

# ---- Imports ----
import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---- Config defaults tuned for actionability ----
TITLE = "Market Stress Composite — FRED vs Live"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

FRED = {
    "vix": "VIXCLS",          # VIX close
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M
    "hy_oas": "BAMLH0A0HYM2", # HY OAS
    "cp_3m": "DCPF3M",        # 3M AA CP
    "tbill_3m": "DTB3",       # 3M T-bill
    "spx": "SP500",           # S&P 500 index
}

# Intraday proxies for the Live composite
PX = {
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "TNX": "^TNX",
    "IRX": "^IRX",
    "HYG": "HYG",
    "LQD": "LQD",
    "BIL": "BIL",
    "SHY": "SHY",
}

DEFAULT_LOOKBACK = "3y"            # plotting window
DEFAULT_SMOOTH = 1                 # no smoothing by default
PCTL_WINDOW_YEARS_FRED = 3         # faster percentiles for daily composite
REGIME_HI, REGIME_LO = 70, 30
MAX_STALE_BDAYS = 0                # do not accept yesterday for FRED composite live read
MAX_PROXY_AGE_MIN = 2              # proxy freshness for Live composite
LIVE_INTRADAY_WINDOW_DAYS = 2      # minutes for last 2 trading days
LIVE_RESAMPLE = "5T"               # 5 minute bars for plotting
Z_WINDOW_DAYS = 60                 # for delta z scores

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y","2y","3y","5y","10y"],
                            index=["1y","2y","3y","5y","10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])

    st.subheader("Normalization")
    smooth = st.number_input("Smoothing window (days, daily composite)", 1, 60, DEFAULT_SMOOTH, 1)
    pctl_years = st.number_input("Percentile window for daily composite (years)", 2, 10, PCTL_WINDOW_YEARS_FRED, 1)
    z_days = st.number_input("Z score window for deltas (trading days)", 20, 252, Z_WINDOW_DAYS, 5)

    st.subheader("Weights")
    w_vix   = st.slider("VIX", 0.0, 1.0, 0.25, 0.05)
    w_hy    = st.slider("HY OAS", 0.0, 1.0, 0.25, 0.05)
    w_curve = st.slider("Yield Curve (inverted)", 0.0, 1.0, 0.20, 0.05)
    w_fund  = st.slider("Funding (CP − T-bill or BIL/SHY)", 0.0, 1.0, 0.15, 0.05)
    w_dd    = st.slider("SPX Drawdown", 0.0, 1.0, 0.15, 0.05)

    st.subheader("Live composite controls")
    use_live = st.checkbox("Compute Live composite from intraday proxies", value=True)
    auto_refresh = st.checkbox("Auto refresh every 60 seconds", value=False)
    if auto_refresh:
        st.autorefresh(interval=60_000, key="autorf")
    st.caption("Sources: FRED via pandas-datareader, intraday proxies via Yahoo Finance")

# ---- Time helpers ----
NY_TZ = "America/New_York"

def today_naive() -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp.now(tz=NY_TZ).date())

def to_naive_date_index(obj):
    if isinstance(obj, pd.Series):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
        return obj
    elif isinstance(obj, pd.DataFrame):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
        return obj
    return obj

def as_naive_date(ts_like) -> pd.Timestamp:
    t = pd.to_datetime(ts_like)
    return pd.Timestamp(t.date())

def is_fresh(ts, max_age_min=MAX_PROXY_AGE_MIN):
    if ts is None or pd.isna(ts):
        return False
    now = pd.Timestamp.now(tz=NY_TZ)
    t = ts if getattr(ts, "tzinfo", None) else pd.Timestamp(ts).tz_localize(NY_TZ)
    return (now - t).total_seconds() / 60.0 <= max_age_min

# ---- Data loaders ----
@st.cache_data(ttl=900, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        s = s.ffill().dropna()
        return to_naive_date_index(s)
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=30, show_spinner=False)
def yf_hist(tickers, period="max", interval="1d"):
    try:
        df = yf.download(tickers=tickers, period=period, interval=interval,
                         auto_adjust=False, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def yf_last_minute(tickers):
    try:
        data = {}
        for tk in tickers:
            h = yf.Ticker(tk).history(period="5d", interval="1m")
            if not h.empty:
                # enforce tz aware index
                if h.index.tz is None:
                    h.index = h.index.tz_localize(NY_TZ)
                else:
                    h.index = h.index.tz_convert(NY_TZ)
                data[tk] = h["Close"].copy()
        return pd.concat(data, axis=1) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def latest_quote(df_1m: pd.DataFrame, symbol: str):
    if df_1m.empty or symbol not in df_1m.columns:
        return np.nan, None
    s = df_1m[symbol].dropna()
    if s.empty:
        return np.nan, None
    return float(s.iloc[-1]), s.index[-1]

def rolling_percentile_daily(s: pd.Series, window_days: int) -> pd.Series:
    s = to_naive_date_index(s)
    w = max(60, int(window_days * 252 / 365))
    out = s.rolling(w, min_periods=max(20, w // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    return to_naive_date_index(out)

def pad_range(lo: float, hi: float, pad: float = 0.06):
    if pd.isna(lo) or pd.isna(hi):
        return None
    d = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1.0)
    return lo - d * pad, hi + d * pad

# ---- Load FRED history (daily, tz naive) ----
today = today_naive()
start_all = as_naive_date(today - pd.DateOffset(years=12))

vix_f   = fred_series(FRED["vix"], start_all, today)
hy_f    = fred_series(FRED["hy_oas"], start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"], start_all, today)
spx_f   = fred_series(FRED["spx"], start_all, today)

fund_f = to_naive_date_index(cp3m_f - tb3m_f)

series_list = [vix_f, hy_f, yc_f, fund_f, spx_f]
have_any_fred = any((not s.dropna().empty) for s in series_list)

# ---- Daily base panel on business days ----
if have_any_fred:
    bidx_start = min(s.index.min() for s in series_list if not s.dropna().empty)
    bidx_start = as_naive_date(bidx_start)
    bidx = pd.bdate_range(bidx_start, today)
    def to_bidx(s):
        s = to_naive_date_index(s)
        return s.reindex(bidx).ffill()

    vix_b, hy_b, yc_b, fund_b, spx_b = map(to_bidx, [vix_f, hy_f, yc_f, fund_f, spx_f])

    panel_d = pd.DataFrame({"VIX": vix_b, "HY_OAS": hy_b, "T10Y3M": yc_b, "FUND": fund_b, "SPX": spx_b}).dropna(how="all")
    panel_d = to_naive_date_index(panel_d)

    # freshness masks for daily data
    def fresh_mask_bdays(original: pd.Series) -> pd.Series:
        original = to_naive_date_index(original)
        obs = original.reindex(panel_d.index)
        pos = np.arange(len(panel_d.index))
        has = obs.notna().values
        last_pos = pd.Series(np.where(has, pos, np.nan), index=panel_d.index).ffill().values
        age = pos - last_pos
        return pd.Series((~np.isnan(age)) & (age <= MAX_STALE_BDAYS), index=panel_d.index)

    m_vix_d  = fresh_mask_bdays(vix_f)
    m_hy_d   = fresh_mask_bdays(hy_f)
    m_yc_d   = fresh_mask_bdays(yc_f)
    m_fund_d = fresh_mask_bdays(fund_f)
    m_dd_d   = fresh_mask_bdays(spx_f)

    mode_daily = "FRED Daily"
else:
    panel_d = pd.DataFrame()
    m_vix_d = m_hy_d = m_yc_d = m_fund_d = m_dd_d = pd.Series(dtype=bool)
    mode_daily = "Unavailable"

# ---- Proxies: daily and minute ----
px_daily = yf_hist(list(PX.values()), period="max", interval="1d")
px_last  = yf_last_minute(list(PX.values()))

# ---- Build a Live panel on timestamped index ----
def build_live_panel(px_1m: pd.DataFrame) -> pd.DataFrame:
    if px_1m.empty:
        return pd.DataFrame()

    df = px_1m.copy()
    # last N days window
    cutoff = pd.Timestamp.now(tz=NY_TZ) - pd.Timedelta(days=LIVE_INTRADAY_WINDOW_DAYS)
    df = df.loc[df.index >= cutoff].copy()
    if df.empty:
        return pd.DataFrame()

    # compute proxies at minute level
    out = pd.DataFrame(index=df.index.unique().sort_values())
    def col(name): return PX.get(name)

    if col("VIX") in df.columns:
        out["VIX"] = df[col("VIX")]
    if col("SPX") in df.columns:
        out["SPX"] = df[col("SPX")]
    if col("TNX") in df.columns and col("IRX") in df.columns:
        out["T10Y3M"] = df[col("TNX")] - df[col("IRX")]
    if col("HYG") in df.columns and col("LQD") in df.columns:
        out["HYG_LQD_RATIO"] = df[col("HYG")] / df[col("LQD")]
        out["HY_OAS_PROXY"] = -out["HYG_LQD_RATIO"]
    if col("BIL") in df.columns and col("SHY") in df.columns:
        out["BIL_SHY_RATIO"] = df[col("BIL")] / df[col("SHY")]
        out["FUND"] = out["BIL_SHY_RATIO"]

    # resample to 5m bars for stability
    out5 = out.resample(LIVE_RESAMPLE).last().ffill()
    return out5

panel_live = build_live_panel(px_last) if use_live else pd.DataFrame()

# ---- Percentiles for daily composite ----
def compute_daily_scores(panel_daily: pd.DataFrame, pctl_years: int) -> pd.DataFrame:
    if panel_daily.empty:
        return pd.DataFrame()
    window_days = int(pctl_years * 365)
    # transforms
    roll_max = panel_daily["SPX"].dropna().cummax()
    drawdown = 100.0 * (panel_daily["SPX"] / roll_max - 1.0)
    stress_dd = -drawdown.clip(upper=0)
    inv_curve = -panel_daily["T10Y3M"]

    pct_vix   = rolling_percentile_daily(panel_daily["VIX"], window_days)
    pct_curve = rolling_percentile_daily(inv_curve, window_days)
    pct_fund  = rolling_percentile_daily(panel_daily["FUND"], window_days)
    pct_dd    = rolling_percentile_daily(stress_dd, window_days)

    if "HY_OAS" in panel_daily.columns and not panel_daily["HY_OAS"].dropna().empty:
        pct_hy = rolling_percentile_daily(panel_daily["HY_OAS"], window_days)
    else:
        pct_hy = rolling_percentile_daily(panel_daily["HY_OAS_PROXY"], window_days) if "HY_OAS_PROXY" in panel_daily else pd.Series(index=panel_daily.index, dtype=float)

    scores = pd.concat([pct_vix, pct_hy, pct_curve, pct_fund, pct_dd], axis=1)
    scores.columns = ["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]
    return to_naive_date_index(scores)

scores_d_all = compute_daily_scores(panel_d, pctl_years) if not panel_d.empty else pd.DataFrame()

# ---- Build daily composite ----
def composite_from_scores(scores: pd.DataFrame, masks: pd.DataFrame, weights: np.ndarray) -> tuple[pd.Series, pd.Series]:
    if scores.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    weights = weights if weights.sum() > 0 else np.ones(5)
    W = (weights / weights.sum()).reshape(1, -1)
    M = masks.values
    X = scores[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]].values
    active_w = (W * M).sum(axis=1)
    comp = np.where(active_w > 0, 100.0 * (X * W * M).sum(axis=1) / active_w, np.nan)
    comp_s = pd.Series(comp, index=scores.index, name="Composite")
    coverage = pd.Series(active_w / W.sum(), index=scores.index, name="ActiveWeightShare")
    return comp_s, coverage

def align_mask(mask_like: pd.Series, target_index: pd.Index) -> pd.Series:
    m = to_naive_date_index(mask_like)
    return m.reindex(target_index).astype(float).fillna(0.0)

weights_vec = np.array([w_vix, w_hy, w_curve, w_fund, w_dd], dtype=float)

if not scores_d_all.empty:
    start_lb = as_naive_date(today - pd.DateOffset(years=years))
    scores_d = scores_d_all.loc[start_lb:].copy()
    # optional smoothing
    if smooth > 1:
        scores_d = scores_d.rolling(smooth, min_periods=1).mean()
        scores_d = to_naive_date_index(scores_d)

    masks_d = pd.concat([
        align_mask(m_vix_d, scores_d.index),
        align_mask(m_hy_d, scores_d.index),
        align_mask(m_yc_d, scores_d.index),
        align_mask(m_fund_d, scores_d.index),
        align_mask(m_dd_d, scores_d.index),
    ], axis=1)
    masks_d.columns = ["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]

    comp_d, coverage_d = composite_from_scores(scores_d, masks_d, weights_vec)
else:
    scores_d = pd.DataFrame()
    comp_d = pd.Series(dtype=float)
    coverage_d = pd.Series(dtype=float)

# ---- Live composite on timestamped index ----
def compute_live_composite(panel_live_ts: pd.DataFrame,
                           panel_daily_hist: pd.DataFrame,
                           weights: np.ndarray) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Map intraday levels into daily-based percentiles, compute live composite minute by minute.
    For each factor, build a reference distribution from the last pctl_years of daily data.
    """
    if panel_live_ts.empty or panel_daily_hist.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)

    # Build reference vectors for percentiles from daily history
    ref = {}
    # VIX
    if "VIX" in panel_daily_hist:
        ref["VIX"] = panel_daily_hist["VIX"].dropna()
    # Curve inverse
    if "T10Y3M" in panel_daily_hist:
        ref["CurveInv"] = (-panel_daily_hist["T10Y3M"]).dropna()
    # Funding
    if "FUND" in panel_daily_hist:
        ref["Fund"] = panel_daily_hist["FUND"].dropna()
    # Drawdown stress from daily closes
    if "SPX" in panel_daily_hist:
        dd = 100.0 * (panel_daily_hist["SPX"] / panel_daily_hist["SPX"].cummax() - 1.0)
        ref["DD"] = (-dd.clip(upper=0)).dropna()
    # HY
    if "HY_OAS" in panel_daily_hist and not panel_daily_hist["HY_OAS"].dropna().empty:
        ref["HY"] = panel_daily_hist["HY_OAS"].dropna()
    elif "HY_OAS_PROXY" in panel_daily_hist:
        ref["HY"] = panel_daily_hist["HY_OAS_PROXY"].dropna()

    # Build live factor series
    live = pd.DataFrame(index=panel_live_ts.index)
    if "VIX" in panel_live_ts:
        live["VIX"] = panel_live_ts["VIX"]
    if "T10Y3M" in panel_live_ts:
        live["CurveInv"] = -panel_live_ts["T10Y3M"]
    if "FUND" in panel_live_ts:
        live["Fund"] = panel_live_ts["FUND"]
    # Live drawdown from intraday SPX vs intraday running max
    if "SPX" in panel_live_ts:
        spx_m = panel_live_ts["SPX"].dropna()
        dd_m = 100.0 * (spx_m / spx_m.cummax() - 1.0)
        live["DD"] = (-dd_m.clip(upper=0))
    # HY proxy intraday
    if "HY_OAS_PROXY" in panel_live_ts:
        live["HY"] = panel_live_ts["HY_OAS_PROXY"]

    # Convert each live factor to a percentile vs its daily reference
    def pct_vs_ref(xval, ref_series):
        # handle nan
        if pd.isna(xval) or ref_series.empty:
            return np.nan
        # percentile rank of xval within ref
        return (ref_series.rank(pct=True).iloc[(np.abs(ref_series - xval)).idxmin()] 
                if False else (ref_series <= xval).mean())

    pcols = []
    for fac, ref_series in ref.items():
        if fac in live.columns:
            p = live[fac].apply(lambda v: pct_vs_ref(v, ref_series))
            pcols.append((fac, p))
    if not pcols:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)

    scores_live = pd.DataFrame({f"{fac}_p": p for fac, p in pcols})
    # Align to the canonical 5 factors order, keep missing as NaN
    for name in ["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]:
        if name not in scores_live.columns:
            scores_live[name] = np.nan
    scores_live = scores_live[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]]

    # Freshness mask per timestamp, only accept factors with quotes inside MAX_PROXY_AGE_MIN
    def series_fresh(symbol):
        val, ts = latest_quote(px_last, PX[symbol]) if PX.get(symbol) else (np.nan, None)
        return is_fresh(ts)

    freshness_flags = {
        "VIX_m": series_fresh("VIX"),
        "HY_m": series_fresh("HYG") and series_fresh("LQD"),
        "Curve_m": series_fresh("TNX") and series_fresh("IRX"),
        "Fund_m": series_fresh("BIL") and series_fresh("SHY"),
        "DD_m": series_fresh("SPX"),
    }
    # Build mask as 1 or 0 across all timestamps, since quotes arrive continuously within the window
    masks_live = pd.DataFrame(
        {k: (1.0 if v else 0.0) for k, v in freshness_flags.items()},
        index=scores_live.index
    )

    # Composite
    weights = weights if weights.sum() > 0 else np.ones(5)
    W = (weights / weights.sum()).reshape(1, -1)
    X = scores_live[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]].values
    M = masks_live[["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]].values
    active_w = (W * M).sum(axis=1)
    comp = np.where(active_w > 0, 100.0 * (X * W * M).sum(axis=1) / active_w, np.nan)
    comp_live = pd.Series(comp, index=scores_live.index, name="Composite_Live")
    coverage_live = pd.Series(active_w / W.sum(), index=scores_live.index, name="ActiveWeightShare_Live")
    return comp_live, scores_live, coverage_live

if use_live and not panel_live.empty and not panel_d.empty:
    comp_live, scores_live, coverage_live = compute_live_composite(panel_live, panel_d.tail(int(PCTL_WINDOW_YEARS_FRED*252*1.2)), weights_vec)
else:
    comp_live, scores_live, coverage_live = pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)

# ---- Assemble daily lookback subset for plotting ----
panel_lb = panel_d.loc[as_naive_date(today - pd.DateOffset(years=years)):] if not panel_d.empty else pd.DataFrame()

# ---- Metrics and banners ----
def tag_for_level(x):
    if pd.isna(x): return "N/A"
    if x >= REGIME_HI: return "High stress"
    if x <= REGIME_LO: return "Low stress"
    return "Neutral"

# Daily composite latest
if not comp_d.dropna().empty:
    latest_d = comp_d.dropna().index[-1]
    weights_text = f"Weights: VIX {weights_vec[0]:.2f}, HY {weights_vec[1]:.2f}, Curve {weights_vec[2]:.2f}, Funding {weights_vec[3]:.2f}, Drawdown {weights_vec[4]:.2f}"
    st.info(f"Daily mode: {mode_daily}. As of {latest_d.date()} | {weights_text}")

# Live freshness banner
def freshness_banner():
    if comp_live.dropna().empty:
        st.warning("Live composite not available or insufficient proxies.")
        return
    last_ts = comp_live.dropna().index[-1]
    cov = coverage_live.reindex([last_ts]).iloc[0] if last_ts in coverage_live.index else np.nan
    def pctf(x): return "N/A" if pd.isna(x) else f"{100*x:.0f}%"
    st.success(f"Live mode: proxies up to {last_ts.tz_convert(NY_TZ).strftime('%Y-%m-%d %H:%M ET')} | Live coverage {pctf(cov)}")

if use_live:
    freshness_banner()

# ---- Headline metrics ----
c1, c2, c3, c4 = st.columns(4)
if not comp_d.dropna().empty:
    c1.metric("Daily Composite", f"{comp_d.loc[latest_d]:.0f}")
    c2.metric("Daily Regime", tag_for_level(comp_d.loc[latest_d]))
    # daily deltas
    def day_change(series, n=1):
        ser = series.dropna()
        if len(ser) < n+1:
            return np.nan
        return ser.iloc[-1] - ser.iloc[-1-n]
    d1 = day_change(comp_d, 1)
    d5 = day_change(comp_d, 5)
    c3.metric("Daily Δ 1d", "N/A" if pd.isna(d1) else f"{d1:.1f}")
    c4.metric("Daily Δ 5d", "N/A" if pd.isna(d5) else f"{d5:.1f}")

if use_live and not comp_live.dropna().empty:
    last_ts = comp_live.dropna().index[-1]
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Live Composite", f"{comp_live.iloc[-1]:.0f}")
    lc2.metric("Live Regime", tag_for_level(comp_live.iloc[-1]))
    # intraday change vs previous session close composite value if available
    if not comp_d.dropna().empty:
        last_close = comp_d.dropna().iloc[-1]
        lc3.metric("Live − Daily Close", f"{(comp_live.iloc[-1] - last_close):.1f}")
    else:
        lc3.metric("Live − Daily Close", "N/A")

# ---- Charts ----
# Figure 1: Daily composite and regime bands
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                     specs=[[{}], [{"secondary_y": True}]],
                     subplot_titles=("Daily Composite (0 to 100)", "SPX and Drawdown (daily)"))

if not comp_d.dropna().empty:
    fig1.add_trace(go.Scatter(x=comp_d.index, y=comp_d, name="Composite Daily",
                              line=dict(color="#000000", width=2)), row=1, col=1)
    fig1.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
    fig1.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
    fig1.update_yaxes(title="Score", range=[0,100], row=1, col=1)

if not panel_lb.empty:
    spx_rebased = panel_lb["SPX"] / panel_lb["SPX"].iloc[0] * 100
    dd_series = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)
    lr = pad_range(spx_rebased.min(), spx_rebased.max())
    rr = pad_range(max(0.0, dd_series.min()), dd_series.max())
    fig1.add_trace(go.Scatter(x=panel_lb.index, y=spx_rebased, name="SPX (rebased=100)",
                              line=dict(color="#7f7f7f")), row=2, col=1, secondary_y=False)
    fig1.add_trace(go.Scatter(x=panel_lb.index, y=dd_series, name="Drawdown (%)",
                              line=dict(color="#ff7f0e")), row=2, col=1, secondary_y=True)
    if lr: fig1.update_yaxes(title="Index", range=lr, row=2, col=1, secondary_y=False)
    if rr: fig1.update_yaxes(title="Drawdown %", range=rr, row=2, col=1, secondary_y=True)

fig1.update_xaxes(tickformat="%b-%d-%y", row=1, col=1)
fig1.update_xaxes(tickformat="%b-%d-%y", title="Date", row=2, col=1)
fig1.update_layout(template="plotly_white", height=720,
                   legend=dict(orientation="h", x=0, y=1.15, xanchor="left"),
                   margin=dict(l=60, r=40, t=60, b=60))
st.plotly_chart(fig1, use_container_width=True)

# Figure 2: Live composite vs time, with coverage overlay
if use_live and not comp_live.dropna().empty:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         specs=[[{}], [{}]],
                         subplot_titles=("Live Composite (5 min)", "Live Coverage Share"))
    fig2.add_trace(go.Scatter(x=comp_live.index, y=comp_live, name="Composite Live",
                              line=dict(color="#1f77b4", width=2)))
    fig2.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
    fig2.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
    fig2.update_yaxes(title="Score", range=[0,100], row=1, col=1)

    if not coverage_live.dropna().empty:
        fig2.add_trace(go.Scatter(x=coverage_live.index, y=100*coverage_live, name="Coverage %",
                                  line=dict(color="#2ca02c", width=2)), row=2, col=1)
        fig2.update_yaxes(title="Coverage %", range=[0,100], row=2, col=1)
    fig2.update_xaxes(title="Time (ET)", tickformat="%b-%d %H:%M")
    fig2.update_layout(template="plotly_white", height=600,
                       legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
                       margin=dict(l=60, r=40, t=60, b=60))
    st.plotly_chart(fig2, use_container_width=True)

# ---- Delta table and z scores ----
st.subheader("Change metrics")
def z_score(s: pd.Series, win: int):
    s = s.dropna()
    if s.empty:
        return np.nan
    tail = s.tail(win)
    if tail.std(ddof=0) == 0 or np.isnan(tail.std(ddof=0)):
        return np.nan
    return (s.iloc[-1] - tail.mean()) / tail.std(ddof=0)

colA, colB, colC, colD, colE, colF = st.columns(6)
if not comp_d.dropna().empty:
    d1 = comp_d.diff(1).iloc[-1] if len(comp_d) > 1 else np.nan
    d5 = comp_d.diff(5).iloc[-1] if len(comp_d) > 5 else np.nan
    z = z_score(comp_d, int(z_days))
    colA.metric("Daily Δ 1d", "N/A" if pd.isna(d1) else f"{d1:.1f}")
    colB.metric("Daily Δ 5d", "N/A" if pd.isna(d5) else f"{d5:.1f}")
    colC.metric("Daily z over window", "N/A" if pd.isna(z) else f"{z:.2f}")
else:
    colA.metric("Daily Δ 1d", "N/A")
    colB.metric("Daily Δ 5d", "N/A")
    colC.metric("Daily z over window", "N/A")

if use_live and not comp_live.dropna().empty:
    # Live deltas: last vs 30 min ago, last vs session start
    last = comp_live.iloc[-1]
    comp_30m = comp_live.asfreq("1T").ffill().dropna()
    t30 = comp_30m.index.max() - pd.Timedelta(minutes=30)
    near_30 = comp_30m.index[comp_30m.index.get_indexer([t30], method="nearest")][0] if len(comp_30m) else None
    d30 = last - comp_30m.loc[near_30] if near_30 in comp_30m.index else np.nan
    first_today = comp_live.resample("1D").first().iloc[-1] if len(comp_live) else np.nan
    dstart = last - first_today if not pd.isna(first_today) else np.nan
    colD.metric("Live Δ 30m", "N/A" if pd.isna(d30) else f"{d30:.1f}")
    colE.metric("Live Δ today", "N/A" if pd.isna(dstart) else f"{dstart:.1f}")
else:
    colD.metric("Live Δ 30m", "N/A")
    colE.metric("Live Δ today", "N/A")
colF.metric("Coverage note", "See banner above")

# ---- Download ----
with st.expander("Download Data"):
    # Daily export
    if not panel_lb.empty:
        export_idx = panel_lb.index
        if not scores_d.empty:
            export_scores = (100.0 * scores_d.rename(columns={
                "VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct","DD_p":"DD_pct"
            })).reindex(export_idx)
        else:
            export_scores = pd.DataFrame(index=export_idx)
        export_comp_d = comp_d.rename("Composite_Daily").reindex(export_idx) if not comp_d.empty else pd.Series(dtype=float)
        out_daily = pd.concat(
            [
                panel_lb.filter(items=[c for c in ["VIX","HY_OAS","HY_OAS_PROXY","T10Y3M","FUND","SPX"] if c in panel_lb.columns]),
                export_scores, export_comp_d
            ],
            axis=1
        )
        out_daily.index.name = "Date"
        st.download_button("Download Daily CSV", out_daily.to_csv(), file_name="market_stress_daily.csv", mime="text/csv")
    # Live export
    if use_live and not panel_live.empty:
        out_live = pd.concat([panel_live, scores_live, comp_live.rename("Composite_Live"), coverage_live.rename("Coverage_Live")], axis=1)
        out_live.index.name = "Timestamp_ET"
        st.download_button("Download Live CSV", out_live.to_csv(), file_name="market_stress_live.csv", mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
