# Cross-Asset Correlations & Vol — pastel, insight-first, simplified
# Universe locked, minimal controls, actionable outputs only.

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
import streamlit as st

# =========================
# App config and style
# =========================
st.set_page_config(page_title="Cross-Asset Correlations & Vol (Simplified)", layout="wide")

plt.rcParams.update({
    "figure.figsize": (8, 3),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
})

st.title("Cross-Asset Correlations & Volatility — Simplified")
st.caption("Data: Yahoo Finance via yfinance. Pastel red↔green, insight-first. Universe locked.")

# =========================
# Universe
# =========================
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],
    "Corporate Credit": ["LQD", "HYG"],
    "Rates": ["^TNX", "^TYX"],
    "Commodities": ["CL=F", "GLD", "HG=F"],
    "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],
}
ORDER = sum(GROUPS.values(), [])
IV_TICKERS = ["^VIX", "^OVX"]

# =========================
# Sidebar: minimal controls + reading guide
# =========================
st.sidebar.header("Quick guide")
st.sidebar.markdown(
"""
- Matrix: last 21 sessions, lower triangle only. Colors are signed.
- Bold = |ρ| above your threshold. Absolute view removes direction.
- Panels: 21D rolling corr with 6M band, 2017+ focus.
- Vol: 21D realized, annualized percent. Implied overlays where available.
"""
)

rho_thresh = st.sidebar.slider("|ρ| threshold for highlights", 0.3, 0.9, 0.5, 0.05)
show_abs = st.sidebar.checkbox("Show absolute correlations", value=False)

# =========================
# Helpers
# =========================
def pastelize_cmap(name="RdYlGn", lighten=0.65, n=256):
    base = plt.cm.get_cmap(name, n)
    colors = base(np.linspace(0, 1, n))
    colors[:, :3] = 1.0 - lighten * (1.0 - colors[:, :3])  # pull toward white
    return ListedColormap(colors)

PASTEL_RdYlGn = pastelize_cmap("RdYlGn", lighten=0.65)
PASTEL_Greens = pastelize_cmap("Greens", lighten=0.65)

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers, start, end, interval="1d"):
    try:
        data = yf.download(
            tickers=tickers, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False, group_by="ticker", threads=True
        )
    except Exception as e:
        st.error(f"Yahoo Finance error: {e}")
        return pd.DataFrame()

    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            try:
                if t not in data.columns.get_level_values(0): 
                    continue
                df = data[t].copy()
                col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
                frames.append(df[col].rename(t).to_frame())
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
    else:
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else data.columns[0])
        wide = data[[col]]
        if len(tickers) == 1:
            wide = wide.rename(columns={col: tickers[0]})
    wide.index.name = "Date"
    return wide.sort_index()

def pct_returns(prices):
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

def realized_vol(returns, window=21, annualization=252):
    return returns.rolling(window).std() * math.sqrt(annualization)

def zscores(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return s
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def percentile_of_last(series):
    s = pd.Series(series).dropna()
    return float(s.rank(pct=True).iloc[-1] * 100.0) if not s.empty else np.nan

def current_value(series):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def focus_2017(index):
    start = pd.Timestamp("2017-01-01")
    return index[index >= start]

def clip_limits(array_like, pmin=1, pmax=99):
    a = pd.Series(np.asarray(array_like).astype(float)).dropna()
    if a.empty:
        return None
    lo, hi = np.percentile(a, [pmin, pmax])
    span = hi - lo
    return (lo - 0.05*span, hi + 0.05*span)

def add_regime_shading(ax):
    spans = [("2020-02-15", "2020-06-30"), ("2022-03-01", "2023-11-01")]
    for a, b in spans:
        ax.axvspan(pd.to_datetime(a), pd.to_datetime(b), color="#cccccc", alpha=0.18)

# =========================
# Data
# =========================
end_date = datetime.now().date()
start_date = (datetime.now() - timedelta(days=365*12)).date()
ALL = list(dict.fromkeys(ORDER + IV_TICKERS))

with st.spinner("Downloading market data..."):
    PRICES = fetch_prices(ALL, str(start_date), str(end_date), interval="1d")

if PRICES.empty:
    st.error("No data downloaded. Check connectivity.")
    st.stop()

BASE = PRICES[[c for c in ORDER if c in PRICES.columns]].copy()
IV   = PRICES[[c for c in IV_TICKERS if c in PRICES.columns]].copy()
RET = pct_returns(BASE)
WIN = 21
REALVOL = realized_vol(RET, window=WIN)

last_date = BASE.index.max()
st.info(f"Data last date: {last_date.strftime('%Y-%m-%d')}" if pd.notna(last_date) else "No dates available.")

# =========================
# Quick Takeaways block
# =========================
def roll_corr(ret_df, a, b, window=21):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a, b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(window).corr(df[b])

takeaways = []
# SPX vs 10Y
rc_er = roll_corr(RET, "^GSPC", "^TNX", WIN)
if not rc_er.empty:
    takeaways.append(f"SPX vs US 10Y 21D corr: {rc_er.iloc[-1]:.2f} ({percentile_of_last(rc_er):.0f}th pct)")

# SPX realized vs VIX spread
if "^GSPC" in REALVOL.columns and "^VIX" in IV.columns:
    rv = REALVOL["^GSPC"] * 100
    vix = IV["^VIX"]
    if not rv.dropna().empty and not vix.dropna().empty:
        common = rv.dropna().to_frame("rv").join(vix.dropna().to_frame("vix"), how="inner")
        if not common.empty:
            spread = common["vix"] - common["rv"]
            takeaways.append(f"VIX minus SPX realized: {spread.iloc[-1]:.1f} pts ({percentile_of_last(spread):.0f}th pct)")

# FX vol snapshot
fx_cols = [c for c in REALVOL.columns if c.endswith("=X")]
if fx_cols:
    fx_pick = (REALVOL[fx_cols[0]] * 100).dropna()
    if not fx_pick.empty:
        takeaways.append(f"FX realized vol now: {fx_pick.iloc[-1]:.1f}% ({percentile_of_last(fx_pick):.0f}th pct)")

st.subheader("Quick takeaways")
st.markdown("- " + "\n- ".join(takeaways) if takeaways else "No quick stats available.")

# =========================
# 1) Correlation Matrix — lower triangle
# =========================
st.subheader("Cross-Asset Correlation Matrix (1M)")

if len(RET) >= WIN:
    corr_raw = RET.tail(WIN).corr().replace([-np.inf, np.inf], np.nan)
else:
    corr_raw = pd.DataFrame()

if corr_raw.empty:
    st.warning("Not enough data to compute the 1M correlation matrix.")
else:
    # Original group order
    order = [t for t in ORDER if t in corr_raw.columns]
    corr_signed = corr_raw.loc[order, order]
    corr = corr_raw.abs().loc[order, order] if show_abs else corr_signed.copy()

    n = corr.shape[0]
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask_diag  = np.eye(n, dtype=bool)
    disp = corr.copy().astype(float)
    disp.values[mask_upper] = np.nan
    disp.values[mask_diag]  = np.nan

    cmap = PASTEL_Greens if show_abs else PASTEL_RdYlGn

    def _highlight(x):
        css = np.full(x.shape, "", dtype=object)
        cond = (np.abs(x.values) >= rho_thresh) & ~np.isnan(x.values)
        css[cond] = "font-weight:700; border:1px solid #777;"
        return pd.DataFrame(css, index=x.index, columns=x.columns)

    sty = (
        disp.style
            .background_gradient(cmap=cmap, vmin=(0 if show_abs else -1), vmax=1)
            .format(na_rep="", formatter=lambda v: "" if pd.isna(v) else f"{int(round(v*100,0))}%")
            .apply(_highlight, axis=None)
            .set_properties(**{"text-align":"center"})
    )
    rows = disp.shape[0]
    st.dataframe(sty, use_container_width=True, height=min(110 + 40*rows, 1600))
    st.caption(f"Lower triangle only. Bold cells meet |ρ| ≥ {rho_thresh:.2f}. Pastel red=negative, green=positive. Absolute mode removes direction.")

    # Top pairs now
    def pair_snapshot(ret_df, a, b):
        r = ret_df[[a, b]].dropna()
        if r.empty:
            return None
        rc = r[a].rolling(WIN).corr(r[b]).dropna()
        if rc.empty:
            return None
        return rc.iloc[-1], float(rc.rank(pct=True).iloc[-1] * 100.0)

    pairs = []
    tickers = [t for t in order]
    for i in range(1, len(tickers)):
        for j in range(0, i):
            a, b = tickers[i], tickers[j]
            ps = pair_snapshot(RET, a, b)
            if ps is None:
                continue
            now, pct = ps
            pairs.append({"A": a, "B": b, "ρ": now, "|ρ|": abs(now), "Percentile(5y)": pct})

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        filt = pairs_df[pairs_df["|ρ|"] >= rho_thresh].copy().sort_values("|ρ|", ascending=False)

        def _render(df):
            if df.empty:
                return df
            out = df.copy()
            out["Corr"] = out["ρ"].map(lambda x: f"{x: .2f}")
            out["Percentile(5y)"] = out["Percentile(5y)"].map(lambda x: f"{x:.0f}th")
            return out[["A","B","Corr","Percentile(5y)"]].rename(columns={"A":"Leg A","B":"Leg B"})

        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            st.markdown("**Top positive pairs (now)**")
            pos = filt.sort_values("ρ", ascending=False).head(6)
            st.dataframe(_render(pos), use_container_width=True, height=262)
        with c2:
            st.markdown("**Top negative pairs (now)**")
            neg = filt.sort_values("ρ", ascending=True).head(6)
            st.dataframe(_render(neg), use_container_width=True, height=262)
        with c3:
            st.markdown("**Stats**")
            st.metric("Pairs ≥ threshold", f"{len(filt)}/{len(pairs_df)}")
            st.metric("Median |ρ| (all)", f"{pairs_df['|ρ|'].median():.2f}")
            st.metric("Max |ρ| now", f"{pairs_df['|ρ|'].max():.2f}")

# =========================
# 2) High-signal correlation panels
# =========================
st.subheader("Key correlation panels")

PANEL_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US 10Y",      title="Equity vs Rates"),
    dict(a="^GSPC", b="HYG",  la="SPX", lb="HY (HYG)",     title="Equity vs HY Credit"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="Oil (WTI)",    title="Equity vs Oil"),
]

def corr_with_context(ax, rc):
    rc = rc.loc[focus_2017(rc.index)]
    if rc.empty:
        return None, None
    mean_rc = rc.rolling(126).mean()
    std_rc  = rc.rolling(126).std()
    ax.fill_between(rc.index, (mean_rc-std_rc).values, (mean_rc+std_rc).values, alpha=0.15)
    ax.plot(mean_rc.index, mean_rc.values, linewidth=1.5, label="6M mean")
    ax.plot(rc.index, rc.values, linewidth=1.0, label="21D corr")
    ax.axhline(0, linewidth=1)
    lims = clip_limits(rc.values, 1, 99)
    ax.set_ylim(max(-1, lims[0]) if lims else -1, min(1, lims[1]) if lims else 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.margins(y=0.05); add_regime_shading(ax)
    return rc.iloc[-1], percentile_of_last(rc)

cols = st.columns(3)
for spec, col in zip(PANEL_SPECS, cols):
    with col:
        fig, ax = plt.subplots()
        rc = roll_corr(RET, spec["a"], spec["b"], WIN)
        curr, pct = corr_with_context(ax, rc)
        ax.set_title(spec["title"]); ax.legend(loc="lower left", fontsize=8); plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        m1, m2 = st.columns(2)
        with m1: st.metric("Current corr", f"{curr:.1%}" if curr is not None and not np.isnan(curr) else "NA")
        with m2: st.metric("Percentile", f"{pct:.0f}th" if pct is not None and not np.isnan(pct) else "NA")

# =========================
# 3) Volatility monitor
# =========================
st.subheader("Vol monitor")

def iqr_band(ax, s):
    s = pd.Series(s).dropna()
    if s.empty:
        return
    q25, q75 = np.nanpercentile(s, [25, 75])
    ax.axhspan(q25, q75, color="lightgray", alpha=0.25)

def vol_panel(series, label, title, overlay=None):
    fig, ax = plt.subplots()
    s = series.loc[focus_2017(series.index)] if series is not None else pd.Series(dtype=float)
    if s is None or s.empty:
        st.info(f"{label} unavailable.")
        return
    ax.plot(s.index, s.values, label=label)
    if overlay is not None and not overlay.dropna().empty:
        o = overlay.loc[focus_2017(overlay.index)]
        ax.plot(o.index, o.values, label=overlay.name)
    iqr_band(ax, s)
    lims = clip_limits(s.values, 1, 99)
    if lims: 
        ax.set_ylim(lims)
    ax.set_title(title); ax.set_ylabel("Percent"); ax.margins(y=0.05); add_regime_shading(ax)
    ax.legend(fontsize=8, ncol=2); plt.tight_layout(); st.pyplot(fig, use_container_width=True)
    m1, m2 = st.columns(2)
    with m1: st.metric("Current", f"{current_value(s):.1f}%")
    with m2: st.metric("Percentile", f"{percentile_of_last(s):.0f}th")

c1, c2, c3 = st.columns(3)
with c1:
    if "^GSPC" in REALVOL.columns:
        vix = IV["^VIX"] if "^VIX" in IV.columns else None
        if vix is not None: vix.name = "VIX"
        vol_panel(REALVOL["^GSPC"] * 100, "SPX 1M Realized", "Equity vol, implied vs realized", overlay=vix)
with c2:
    if "CL=F" in REALVOL.columns:
        ovx = IV["^OVX"] if "^OVX" in IV.columns else None
        if ovx is not None: ovx.name = "OVX"
        vol_panel(REALVOL["CL=F"] * 100, "WTI 1M Realized", "Oil vol, implied vs realized", overlay=ovx)
with c3:
    fx_cols = [c for c in REALVOL.columns if c.endswith("=X")]
    if fx_cols:
        # Pick USDJPY if available, else first FX
        fx = "USDJPY=X" if "USDJPY=X" in REALVOL.columns else fx_cols[0]
        vol_panel(REALVOL[fx] * 100, f"{fx} 1M Realized", "FX vol, realized proxy")

# =========================
# 4) Standardized vol snapshot (table)
# =========================
st.subheader("Vol snapshot — standardized, ranked")

z_panel = {}
if "^VIX" in IV.columns:  z_panel["^VIX"] = IV["^VIX"]
if "^OVX" in IV.columns:  z_panel["^OVX"] = IV["^OVX"]

mapping = [("^GSPC", "SPX Realized"),
           ("^TNX", "US10Y Realized"),
           ("LQD", "IG Realized"),
           ("HYG", "HY Realized"),
           ("CL=F", "Oil Realized"),
           ("GLD", "Gold Realized"),
           ("EURUSD=X", "EURUSD Realized"),
           ("USDJPY=X", "USDJPY Realized"),
           ("GBPUSD=X", "GBPUSD Realized")]

for sym, label in mapping:
    if sym in REALVOL.columns:
        z_panel[label] = REALVOL[sym] * 100

if len(z_panel) >= 2:
    zdf = pd.DataFrame(z_panel).dropna(how="all").apply(zscores, axis=0)
    zdf = zdf.loc[focus_2017(zdf.index)]
    if not zdf.empty:
        latest = zdf.tail(1).T.rename(columns={zdf.index[-1]: "Current Z"})
        latest["Percentile"] = pd.Series({c: percentile_of_last(zdf[c]) for c in zdf.columns})
        latest = latest.sort_values("Current Z", ascending=False)
        st.dataframe(
            latest.style.format({"Current Z": "{:.2f}", "Percentile": "{:.0f}%"}).background_gradient(
                subset=["Current Z"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=min(60 + 28*len(latest), 700),
        )
else:
    st.info("Not enough series for standardized snapshot.")
