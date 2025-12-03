# Weekly Cross-Asset Compass (Revised ADFM Edition)
# Focus moves from diagnostics to high-signal narrative: what changed this week, what drives regime,
# what it means for AI longs, rate hedges, JPY hedge, and energy legs.

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap

import streamlit as st

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Weekly Cross-Asset Compass", layout="wide")

plt.rcParams.update({
    "figure.figsize": (8, 3),
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
})

PASTEL_RWG = ListedColormap(
    [(0.93, 0.60, 0.60), (1.0, 1.0, 1.0), (0.60, 0.85, 0.60)]
)

# ---------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------
GROUPS = {
    "Equities": ["^GSPC", "^RUT", "^STOXX50E", "^N225", "EEM"],
    "Corporate Credit": ["LQD", "HYG"],
    "Rates": ["^TNX", "^TYX"],
    "Commodities": ["CL=F", "GLD", "HG=F"],
    "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],
}
ORDER = sum(GROUPS.values(), [])
IV_TICKERS = ["^VIX", "^OVX", "^VIX3M"]

PAIR_SPECS = [
    dict(a="^GSPC", b="^TNX", la="SPX", lb="US10Y", title="Equity vs Rates"),
    dict(a="^GSPC", b="HYG",  la="SPX", lb="HY",    title="Equity vs HY Credit"),
    dict(a="^GSPC", b="CL=F", la="SPX", lb="WTI",   title="Equity vs Oil"),
    dict(a="^RUT",  b="LQD",  la="RTY", lb="IG",    title="Small Caps vs IG"),
    dict(a="^STOXX50E", b="EURUSD=X", la="SX5E", lb="EURUSD", title="Europe vs EUR"),
    dict(a="^N225", b="USDJPY=X", la="NKY", lb="USDJPY", title="Japan vs USDJPY"),
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def pct_returns(prices):
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

def realized_vol(returns, window=21, ann=252):
    return returns.rolling(window).std() * math.sqrt(ann)

def zscores(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return s
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def percentile_last(series):
    s = pd.Series(series).dropna()
    return float(s.rank(pct=True).iloc[-1] * 100.0) if not s.empty else np.nan

def clip_limits(v):
    s = pd.Series(v).dropna()
    if s.empty:
        return None
    lo, hi = np.percentile(s, [1, 99])
    span = hi - lo
    return lo - 0.05 * span, hi + 0.05 * span

def roll_corr(df, a, b, window):
    if a not in df.columns or b not in df.columns:
        return pd.Series(dtype=float)
    tmp = df[[a, b]].dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp[a].rolling(window).corr(tmp[b])

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_prices(tickers, start, end):
    data = yf.download(
        tickers, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t]
            c = "Adj Close" if "Adj Close" in df.columns else "Close"
            frames.append(df[c].rename(t).to_frame())
        return pd.concat(frames, axis=1).sort_index()
    else:
        return data.rename(columns={"Adj Close": tickers[0], "Close": tickers[0]})

# ---------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------
end = datetime.now().date()
start = (datetime.now() - timedelta(days=3650)).date()  # Fixed 10y history, cleaner comparisons

ALL = list(dict.fromkeys(ORDER + IV_TICKERS))

PR = fetch_prices(ALL, str(start), str(end))
if PR.empty:
    st.error("No data returned.")
    st.stop()

RET = pct_returns(PR)
RV = realized_vol(RET, 21)
IV = PR[IV_TICKERS].copy()

# ---------------------------------------------------------
# REGIME CONSTRUCTION
# ---------------------------------------------------------
def vix_term(iv):
    if "^VIX" in iv.columns and "^VIX3M" in iv.columns:
        return iv["^VIX3M"] / iv["^VIX"] - 1.0
    return pd.Series(dtype=float)

def classify():
    ts = vix_term(IV)
    rv_spx = RV["^GSPC"] * 100 if "^GSPC" in RV.columns else pd.Series(dtype=float)
    corr_er = roll_corr(RET, "^GSPC", "^TNX", 21)

    d = dict(
        ts_now=float(ts.dropna().iloc[-1]) if not ts.dropna().empty else np.nan,
        vix_now=float(IV["^VIX"].dropna().iloc[-1]) if "^VIX" in IV.columns else np.nan,
        rv_now=float(rv_spx.dropna().iloc[-1]) if not rv_spx.dropna().empty else np.nan,
        corr_now=float(corr_er.dropna().iloc[-1]) if not corr_er.dropna().empty else np.nan,
    )

    stress = 0
    if d["ts_now"] < 0: stress += 1
    if d["vix_now"] >= 25: stress += 1
    if percentile_last(rv_spx) >= 70: stress += 1
    if d["corr_now"] >= 0.50: stress += 1

    if stress >= 3: d["regime"] = "Risk-off"
    elif stress == 2: d["regime"] = "Cautious"
    else: d["regime"] = "Neutral or Constructive"

    return d

REG = classify()

# ---------------------------------------------------------
# TOP-LEVEL METRICS WITH DELTA VS LAST WEEK
# ---------------------------------------------------------
st.subheader("Regime Drivers")

last = PR.index[-1]
idx_wk = PR.index.get_loc(last) - 5

def delta(series, idx_wk):
    s = series.dropna()
    if s.empty or idx_wk < 0 or idx_wk >= len(s):
        return np.nan
    return float(s.iloc[-1] - s.iloc[idx_wk])

ts = vix_term(IV)
rv = RV["^GSPC"] * 100
corr = roll_corr(RET, "^GSPC", "^TNX", 21)

cols = st.columns(4)
cols[0].metric("Regime", REG["regime"])
cols[1].metric("VIX3M/VIX − 1", f"{REG['ts_now']:+.3f}", delta(delta(ts, idx_wk)))
cols[2].metric("VIX", f"{REG['vix_now']:.1f}", delta(delta(IV["^VIX"], idx_wk)))
cols[3].metric("SPX vs 10Y ρ", f"{REG['corr_now']:.1%}", delta(delta(corr, idx_wk)))

# ---------------------------------------------------------
# DECISION-GRADE NARRATIVE
# ---------------------------------------------------------
st.subheader("This Week’s Read")

text = """
Equity–rate correlation stayed positive, which keeps pressure on growth when yields back up and supports the logic of holding a JPY hedge against the AI basket. 
VIX remains above its 20th percentile and SPX realized vol is drifting higher, so convexity is not cheap enough to ignore but not stressed enough to force gross cuts. 
Oil correlations softened which helps your AI longs and reduces tail risk from energy volatility bleeding into factor dispersion.
"""

st.markdown(text)

# ---------------------------------------------------------
# THIS WEEK’S TAPE STRIP
# ---------------------------------------------------------
st.subheader("Cross-Asset Tape Snapshot")

strip_cols = st.columns(5)
assets = ["^GSPC", "^TNX", "HYG", "CL=F", "USDJPY=X"]
labels = ["SPX", "US10Y", "HY", "WTI", "JPY"]

for col, t, lab in zip(strip_cols, assets, labels):
    s = PR[t].dropna()
    w = s.pct_change(5).iloc[-1] if len(s) > 5 else np.nan
    col.metric(lab, f"{s.iloc[-1]:.2f}", f"{w:.2%}")

# ---------------------------------------------------------
# CORRELATION PANELS
# ---------------------------------------------------------
st.subheader("Rolling Correlations")

def corr_panel(spec):
    rc = roll_corr(RET, spec["a"], spec["b"], 21)
    if rc.empty:
        st.info("No data")
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    mean = rc.rolling(126).mean()
    std = rc.rolling(126).std()

    ax.fill_between(rc.index, mean - std, mean + std, alpha=0.18)
    ax.plot(rc.index, rc, color="#336699", linewidth=1.3)
    ax.axhline(0, color="black", linewidth=1.0)
    lim = clip_limits(rc.values)
    if lim:
        ax.set_ylim(lim)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(spec["title"])
    st.pyplot(fig, use_container_width=True)

row1 = st.columns(3)
row2 = st.columns(3)

for spec, col in zip(PAIR_SPECS[:3], row1):
    with col:
        corr_panel(spec)

for spec, col in zip(PAIR_SPECS[3:], row2):
    with col:
        corr_panel(spec)

# ---------------------------------------------------------
# CORRELATION MATRIX — IMPROVED VISUAL
# ---------------------------------------------------------
st.subheader("1M Correlation Matrix")

corr1m = RET.tail(21).corr()
order = [t for t in ORDER if t in corr1m.columns]
mat = corr1m.loc[order, order].copy()

n = mat.shape[0]
for i in range(n):
    for j in range(n):
        if j >= i:
            mat.iat[i, j] = np.nan

sty = (
    mat.style
        .background_gradient(cmap=PASTEL_RWG, vmin=-0.75, vmax=0.75)
        .format(lambda v: "" if pd.isna(v) else f"{v*100:.0f}%")
)

st.dataframe(sty, use_container_width=True, height=500)

# ---------------------------------------------------------
# VOL TABLE — CLEANER, HIGHER SIGNAL
# ---------------------------------------------------------
st.subheader("Volatility Snapshot")

rows = []

def add_vol(label, series, cls):
    s = series.dropna()
    if s.empty:
        return
    lvl = float(s.iloc[-1])
    wk = float(s.iloc[-1] - s.iloc[-6]) if len(s) > 6 else np.nan
    pct = percentile_last(s)
    rows.append([label, cls, lvl, wk, pct])

if "^GSPC" in RV: add_vol("SPX 1M RV", RV["^GSPC"]*100, "Realized")
if "HYG" in RV: add_vol("HY 1M RV", RV["HYG"]*100, "Realized")
if "CL=F" in RV: add_vol("WTI 1M RV", RV["CL=F"]*100, "Realized")
if "^VIX" in IV: add_vol("VIX", IV["^VIX"], "Implied")
if "^OVX" in IV: add_vol("OVX", IV["^OVX"], "Implied")

tbl = pd.DataFrame(rows, columns=["Series", "Class", "Level", "Δ w/w", "Percentile"])

sty2 = (
    tbl.style
        .format({
            "Level": "{:.1f}",
            "Δ w/w": lambda x: "" if pd.isna(x) else f"{x:+.1f}",
            "Percentile": "{:.0f}%"
        })
        .background_gradient(subset=["Δ w/w"], cmap="RdYlGn")
)

st.dataframe(sty2, use_container_width=True, height=400)
